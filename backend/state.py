from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _normalize_sign(value: float, tolerance: float = 1e-6) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _line_side(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> int:
    """Return -1/0/1 depending on which side of the line (start->end) the point lies."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    orientation = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return _normalize_sign(orientation)


@dataclass
class TrackState:
    last_sign: Optional[int] = None
    last_seen: float = field(default_factory=time.time)
    last_position: Optional[Tuple[float, float]] = None
    last_cross_timestamp: float = 0.0


LOGGER = logging.getLogger(__name__)


class LineCounterState:
    """Shared state for line position, track memory, and directional counts."""

    def __init__(self, state_file: Optional[Path | str] = None) -> None:
        self._lock = threading.Lock()
        self._line: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.5, 0.2),
            (0.5, 0.8),
        )
        self._counts = {"right_to_left": 0, "left_to_right": 0}
        self._tracks: Dict[int, TrackState] = {}
        self._listeners: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.cross_cooldown_seconds = 0.75
        self.track_ttl_seconds = 2.0
        self._state_file = Path(state_file) if state_file else Path(__file__).resolve().parent / "state.json"
        self._ensure_state_dir()
        self._load_state()

    # Event loop + listener helpers -------------------------------------------------
    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def register_listener(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._listeners.append(queue)
        return queue

    def unregister_listener(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)

    def _notify_listeners(self) -> None:
        loop = self._loop
        if loop is None:
            return
        snapshot = self.to_dict()
        with self._lock:
            listeners = list(self._listeners)
        for queue in listeners:
            asyncio.run_coroutine_threadsafe(queue.put(snapshot), loop)

    # Line management ----------------------------------------------------------------
    def update_line(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        notify: bool = True,
    ) -> None:
        start = (_clamp(start[0], 0.0, 1.0), _clamp(start[1], 0.0, 1.0))
        end = (_clamp(end[0], 0.0, 1.0), _clamp(end[1], 0.0, 1.0))
        with self._lock:
            self._line = (start, end)
        self._persist_state()
        if notify:
            self._notify_listeners()

    def get_line(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        with self._lock:
            return self._line

    def line_pixels(self, frame_size: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        width, height = frame_size
        with self._lock:
            start, end = self._line
        start_px = (int(start[0] * width), int(start[1] * height))
        end_px = (int(end[0] * width), int(end[1] * height))
        return start_px, end_px

    # Track processing ---------------------------------------------------------------
    def handle_detection(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        frame_size: Tuple[int, int],
    ) -> None:
        width, height = frame_size
        line_start_px, line_end_px = self.line_pixels((width, height))
        current_sign = _line_side(centroid, line_start_px, line_end_px)

        counts_changed = False
        now = time.time()
        with self._lock:
            track = self._tracks.setdefault(int(track_id), TrackState())
            track.last_seen = now
            track.last_position = centroid

            previous_sign = track.last_sign
            if (
                previous_sign is not None
                and previous_sign != 0
                and current_sign != 0
                and previous_sign != current_sign
                and now - track.last_cross_timestamp > self.cross_cooldown_seconds
            ):
                if previous_sign < current_sign:
                    self._counts["right_to_left"] += 1
                else:
                    self._counts["left_to_right"] += 1
                track.last_cross_timestamp = now
                counts_changed = True

            if current_sign != 0 or track.last_sign is None:
                track.last_sign = current_sign

        if counts_changed:
            self._persist_state()
            self._notify_listeners()

    def prune_stale_tracks(self) -> None:
        now = time.time()
        with self._lock:
            stale_ids = [
                track_id
                for track_id, state in self._tracks.items()
                if now - state.last_seen > self.track_ttl_seconds
            ]
            for track_id in stale_ids:
                self._tracks.pop(track_id, None)

    # Counts + snapshots -------------------------------------------------------------
    def get_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset_counts(self) -> None:
        with self._lock:
            self._counts = {"right_to_left": 0, "left_to_right": 0}
        self._persist_state()
        self._notify_listeners()

    def to_dict(self) -> Dict[str, object]:
        line_start, line_end = self.get_line()
        payload = {
            "line": {
                "start": {"x": line_start[0], "y": line_start[1]},
                "end": {"x": line_end[0], "y": line_end[1]},
            },
            "counts": self.get_counts(),
            "updated_at": time.time(),
        }
        return payload

    # Persistence --------------------------------------------------------------------
    def _ensure_state_dir(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to ensure state directory exists")

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            with self._state_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to load persisted state")
            return

        line = data.get("line")
        counts = data.get("counts")
        with self._lock:
            if line:
                start = line.get("start", {})
                end = line.get("end", {})
                self._line = (
                    (_clamp(float(start.get("x", 0.5)), 0.0, 1.0), _clamp(float(start.get("y", 0.2)), 0.0, 1.0)),
                    (_clamp(float(end.get("x", 0.5)), 0.0, 1.0), _clamp(float(end.get("y", 0.8)), 0.0, 1.0)),
                )
            if counts:
                self._counts = {
                    "right_to_left": int(counts.get("right_to_left", 0)),
                    "left_to_right": int(counts.get("left_to_right", 0)),
                }

    def _persist_state(self) -> None:
        try:
            with self._lock:
                payload = {
                    "line": {
                        "start": {"x": self._line[0][0], "y": self._line[0][1]},
                        "end": {"x": self._line[1][0], "y": self._line[1][1]},
                    },
                    "counts": dict(self._counts),
                }
            with self._state_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to persist state")
