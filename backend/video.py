from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .state import LineCounterState

LOGGER = logging.getLogger(__name__)


class VideoProcessor:
    """Handle camera input, run YOLOv8 tracking, and prepare annotated frames."""

    def __init__(
        self,
        state: LineCounterState,
        *,
        source: int | str = 0,
        model_path: str = "yolov8n.pt",
        target_size: Tuple[int, int] = (960, 540),
    ) -> None:
        self.state = state
        self.source = source
        self.model_path = model_path
        self.target_size = target_size

        self._model = YOLO(self.model_path)
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._latest_frame_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._latest_timestamp: float = 0.0

    # Thread lifecycle ----------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # Frame access --------------------------------------------------------------------
    def latest_frame(self) -> Optional[bytes]:
        with self._latest_frame_lock:
            return self._latest_frame

    def mjpeg_generator(self) -> Iterable[bytes]:
        boundary = b"--frame"
        while self._running.is_set():
            frame = self.latest_frame()
            if frame:
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)

    # Internal loop -------------------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            results_stream = self._model.track(
                source=self.source,
                stream=True,
                persist=True,
                classes=[0],  # 0 == person for COCO
                verbose=False,
            )
            for result in results_stream:
                if not self._running.is_set():
                    break

                frame = self._prepare_frame(result)
                if frame is None:
                    continue

                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    LOGGER.warning("Failed to encode frame")
                    continue

                with self._latest_frame_lock:
                    self._latest_frame = encoded.tobytes()
                    self._latest_timestamp = time.time()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Video processing loop terminated unexpectedly")
            self._running.clear()

    def _prepare_frame(self, result) -> Optional[np.ndarray]:
        if result is None or result.orig_img is None:
            return None

        frame = result.orig_img.copy()
        original_height, original_width = frame.shape[:2]

        scale_x = scale_y = 1.0
        if self.target_size:
            target_width, target_height = self.target_size
            frame = cv2.resize(frame, (target_width, target_height))
            scale_x = target_width / float(original_width)
            scale_y = target_height / float(original_height)

        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.xyxy is not None:
            ids = boxes.id
            if ids is not None:
                ids = ids.int().cpu().tolist()
            else:
                ids = [None] * len(boxes.xyxy)

            coordinates = boxes.xyxy.cpu().numpy()
            for idx, (box, track_id) in enumerate(zip(coordinates, ids)):
                x1, y1, x2, y2 = box
                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if track_id is None:
                    # Assign a pseudo ID based on index to keep logic consistent.
                    track_id = int(idx)

                self.state.handle_detection(
                    int(track_id),
                    centroid,
                    (frame.shape[1], frame.shape[0]),
                )

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                label = f"ID {track_id}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), max(int(y1) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        self.state.prune_stale_tracks()

        line_start, line_end = self.state.line_pixels((frame.shape[1], frame.shape[0]))
        cv2.line(frame, line_start, line_end, (0, 255, 255), 2)

        counts = self.state.get_counts()
        cv2.putText(
            frame,
            f"Right->Left: {counts['right_to_left']}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Left->Right: {counts['left_to_right']}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        return frame
