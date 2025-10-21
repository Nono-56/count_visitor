from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# アプリの設定値はこのデータクラスで集中管理する
@dataclass
class AppConfig:
    # Webカメラ番号または動画パス。0 は既定のカメラ、"rtsp://..." なども指定可能。
    source: str | int = 0
    # Ultralytics YOLO のモデル名もしくはファイルパス。軽量な yolov8n.pt を既定にする。
    model: str = "yolov8n.pt"
    # カウント領域左端の正規化 x 座標 (0.0-1.0)。
    zone_left: float = 0.4
    # カウント領域右端の正規化 x 座標 (0.0-1.0)。
    zone_right: float = 0.6
    # 描画時のフレーム幅。0 を指定すると入力映像の解像度を維持する。
    width: int = 960
    # YOLO 推論を走らせるデバイス (例: "cpu", "cuda:0")。None の場合は自動判定。
    device: Optional[str] = None
    # 検出の信頼度しきい値 (0.0-1.0)。小さいほど検出は増えるが誤検出も増える。
    conf: float = 0.5
    # 同じ人物がライン通過後に再カウントされないようにする待ち時間（秒）。
    cross_cooldown_seconds: float = 0.75
    # トラッキング情報を保持しておく最大秒数。見失ったトラックはこの秒数後に破棄。
    track_ttl_seconds: float = 2.0
    # ライン描画の太さ（ピクセル単位）。
    line_thickness_px: int = 2
    # ライン端点のドラッグ用ハンドル半径（ピクセル単位）。
    handle_radius_px: int = 12
    # ハンドルを掴める当たり判定半径（ピクセル単位）。
    handle_hit_radius_px: int = 24


# アプリのデフォルト設定。ここを変更すれば全体の初期値をまとめて調整できる。
DEFAULT_CONFIG = AppConfig()
# OpenCV の表示ウィンドウ名。
WINDOW_NAME = "count_visitor"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_source(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    try:
        return int(value)
    except ValueError:
        return value

def _parse_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"正規化座標 '{value}' は 0.0～1.0 の小数で指定してください。"
        ) from exc
    if not (0.0 <= ratio <= 1.0):
        raise argparse.ArgumentTypeError("正規化座標は 0.0 ～ 1.0 の範囲で指定してください。")
    return ratio


@dataclass
class TrackState:
    last_zone: Optional[str] = None
    entry_side: Optional[str] = None
    last_seen: float = field(default_factory=time.time)
    last_cross_timestamp: float = 0.0


class LineCrossCounter:
    def __init__(
        self,
        zone_left: float,
        zone_right: float,
        *,
        cross_cooldown_seconds: float = 0.75,
        track_ttl_seconds: float = 2.0,
    ) -> None:
        self._zone_normalized = self._normalize_zone(zone_left, zone_right)
        self._counts = {"right_to_left": 0, "left_to_right": 0}
        self._tracks: Dict[int, TrackState] = {}
        self.cross_cooldown_seconds = cross_cooldown_seconds
        self.track_ttl_seconds = track_ttl_seconds

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    @property
    def zone_normalized(self) -> Tuple[float, float]:
        return self._zone_normalized

    def update_zone(self, left: float, right: float) -> None:
        self._zone_normalized = self._normalize_zone(left, right)
        self._tracks.clear()

    def reset_counts(self) -> None:
        self._counts = {"right_to_left": 0, "left_to_right": 0}
        self._tracks.clear()

    def zone_pixels(self, frame_size: Tuple[int, int]) -> Tuple[int, int]:
        width = max(frame_size[0], 1)
        left, right = self._zone_normalized
        max_index = max(width - 1, 0)
        return (int(round(left * max_index)), int(round(right * max_index)))

    def observe(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        frame_size: Tuple[int, int],
    ) -> None:
        self._prune_stale_tracks()

        left_px, right_px = self.zone_pixels(frame_size)
        x, _ = centroid
        if x < left_px:
            current_zone = "left"
        elif x > right_px:
            current_zone = "right"
        else:
            current_zone = "inside"
        now = time.time()

        track = self._tracks.setdefault(int(track_id), TrackState())
        track.last_seen = now

        previous_zone = track.last_zone

        if (
            current_zone == "inside"
            and track.entry_side is None
            and previous_zone in ("left", "right")
        ):
            track.entry_side = previous_zone

        if current_zone in ("left", "right"):
            if (
                track.entry_side
                and track.entry_side != current_zone
                and now - track.last_cross_timestamp > self.cross_cooldown_seconds
            ):
                if track.entry_side == "right" and current_zone == "left":
                    self._counts["right_to_left"] += 1
                elif track.entry_side == "left" and current_zone == "right":
                    self._counts["left_to_right"] += 1
                track.last_cross_timestamp = now
            track.entry_side = None

        track.last_zone = current_zone

    def _prune_stale_tracks(self) -> None:
        now = time.time()
        stale_ids = [
            track_id
            for track_id, state in list(self._tracks.items())
            if now - state.last_seen > self.track_ttl_seconds
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)

    @staticmethod
    def _normalize_zone(left: float, right: float) -> Tuple[float, float]:
        left_clamped = _clamp01(left)
        right_clamped = _clamp01(right)
        if left_clamped <= right_clamped:
            return (left_clamped, right_clamped)
        return (right_clamped, left_clamped)


def draw_overlay(
    frame: np.ndarray,
    counter: LineCrossCounter,
    counts: Dict[str, int],
    config: AppConfig,
) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]
    left_px, right_px = counter.zone_pixels((width, height))
    left_px = max(0, min(left_px, max(width - 1, 0)))
    right_px = max(0, min(right_px, max(width - 1, 0)))
    if right_px < left_px:
        left_px, right_px = right_px, left_px

    handle_radius = max(3, config.handle_radius_px)
    thickness = max(1, config.line_thickness_px)

    overlay = display.copy()
    cv2.rectangle(
        overlay,
        (left_px, 0),
        (right_px, max(height - 1, 0)),
        (0, 255, 255),
        -1,
    )
    cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
    cv2.line(
        display,
        (left_px, 0),
        (left_px, max(height - 1, 0)),
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        display,
        (right_px, 0),
        (right_px, max(height - 1, 0)),
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    handle_y = height // 2
    cv2.circle(display, (left_px, handle_y), handle_radius, (255, 140, 0), -1, cv2.LINE_AA)
    cv2.circle(display, (right_px, handle_y), handle_radius, (255, 140, 0), -1, cv2.LINE_AA)

    cv2.putText(
        display,
        f"Right -> Left: {counts['right_to_left']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"Left -> Right: {counts['left_to_right']}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "Drag handles to adjust area | R: reset counts | Q/Esc: exit",
        (10, max(height - 20, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return display


class LineEditor:
    def __init__(self, counter: LineCrossCounter, config: AppConfig, window_name: str = WINDOW_NAME) -> None:
        self.counter = counter
        self.config = config
        self.window_name = window_name
        self.frame_size: Tuple[int, int] = (1, 1)
        self.active_handle: Optional[str] = None
        self.dragging = False
        self.handle_hit_radius = max(4, config.handle_hit_radius_px)

    def register(self) -> None:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_size = (max(width, 1), max(height, 1))

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Optional[object]) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            handle = self._nearest_handle(x, y)
            if handle:
                self.active_handle = handle
                self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.active_handle:
            self._update_zone(self.active_handle, x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.active_handle = None
            self.dragging = False

    def _nearest_handle(self, x: int, y: int) -> Optional[str]:
        width, height = self.frame_size
        left_px, right_px = self.counter.zone_pixels((width, height))
        handle_y = height // 2
        left_dist = np.linalg.norm(np.array([left_px, handle_y]) - np.array([x, y]))
        right_dist = np.linalg.norm(np.array([right_px, handle_y]) - np.array([x, y]))
        threshold = self.handle_hit_radius
        if left_dist <= threshold or right_dist <= threshold:
            return "left" if left_dist <= right_dist else "right"
        return None

    def _update_zone(self, handle: str, x: int) -> None:
        width = max(self.frame_size[0], 1)
        normalized_x = _clamp01(x / width)
        left, right = self.counter.zone_normalized
        if handle == "left":
            self.counter.update_zone(normalized_x, right)
        else:
            self.counter.update_zone(left, normalized_x)



def parse_args(defaults: AppConfig) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Webカメラ映像で人の移動方向をカウントするツール"
    )
    parser.add_argument(
        "--source",
        default=str(defaults.source),
        help="映像ソース (例: 0, 1, video.mp4)",
    )
    parser.add_argument(
        "--model",
        default=defaults.model,
        help="YOLOv8 モデルパスまたはモデル名",
    )
    parser.add_argument(
        "--zone-left",
        default=defaults.zone_left,
        type=_parse_ratio,
        help="カウント領域左端の正規化 x 座標 (0.0-1.0)",
    )
    parser.add_argument(
        "--zone-right",
        default=defaults.zone_right,
        type=_parse_ratio,
        help="カウント領域右端の正規化 x 座標 (0.0-1.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=defaults.width,
        help="表示時の幅。0 を指定するとリサイズしません。",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="YOLO 推論を実行するデバイス (例: cpu, cuda:0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=defaults.conf,
        help="検出の信頼度しきい値",
    )
    parser.add_argument(
        "--cross-cooldown",
        type=float,
        default=defaults.cross_cooldown_seconds,
        help="同一トラックが連続でカウントされないためのクールダウン秒数",
    )
    parser.add_argument(
        "--track-ttl",
        type=float,
        default=defaults.track_ttl_seconds,
        help="トラックの寿命 (秒)。この時間観測されないとリセットされます。",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=defaults.line_thickness_px,
        help="ライン描画の太さ (px)",
    )
    parser.add_argument(
        "--handle-radius",
        type=int,
        default=defaults.handle_radius_px,
        help="ハンドル表示の半径 (px)",
    )
    parser.add_argument(
        "--handle-hit-radius",
        type=int,
        default=defaults.handle_hit_radius_px,
        help="ハンドルのドラッグ判定半径 (px)",
    )

    args = parser.parse_args()

    return AppConfig(
        source=_parse_source(args.source),
        model=args.model,
        zone_left=args.zone_left,
        zone_right=args.zone_right,
        width=args.width,
        device=args.device,
        conf=args.conf,
        cross_cooldown_seconds=args.cross_cooldown,
        track_ttl_seconds=args.track_ttl,
        line_thickness_px=args.line_thickness,
        handle_radius_px=args.handle_radius,
        handle_hit_radius_px=args.handle_hit_radius,
    )


def main() -> None:
    config = parse_args(DEFAULT_CONFIG)

    model = YOLO(config.model)
    counter = LineCrossCounter(
        config.zone_left,
        config.zone_right,
        cross_cooldown_seconds=config.cross_cooldown_seconds,
        track_ttl_seconds=config.track_ttl_seconds,
    )

    stop_requested = False

    line_editor = LineEditor(counter, config)
    line_editor.register()

    def _handle_signal(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    results_generator = model.track(
        source=config.source,
        stream=True,
        classes=[0],  # 0: person (COCO)
        verbose=False,
        persist=True,
        conf=config.conf,
        device=config.device,
    )

    try:
        for result in results_generator:
            if stop_requested:
                break
            frame = result.orig_img
            if frame is None:
                continue

            original_height, original_width = frame.shape[:2]
            scale_x = scale_y = 1.0
            if config.width and frame.shape[1] != config.width:
                target_width = config.width
                target_height = int(original_height * (target_width / original_width))
                scale_x = target_width / original_width
                scale_y = target_height / original_height
                frame = cv2.resize(frame, (target_width, target_height))

            line_editor.set_frame_size(frame.shape[1], frame.shape[0])

            boxes = getattr(result, "boxes", None)
            if boxes is not None and boxes.xyxy is not None:
                coords = boxes.xyxy.cpu().numpy()
                if scale_x != 1.0 or scale_y != 1.0:
                    coords = coords.copy()
                    coords[:, [0, 2]] *= scale_x
                    coords[:, [1, 3]] *= scale_y
                if boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                else:
                    ids = list(range(len(coords)))
                for idx, box in enumerate(coords):
                    x1, y1, x2, y2 = box.tolist()
                    track_id = ids[idx] if idx < len(ids) else idx
                    centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    counter.observe(
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
                    cv2.circle(
                        frame,
                        (int(centroid[0]), int(centroid[1])),
                        4,
                        (255, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (int(x1), max(int(y1) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            frame_with_overlay = draw_overlay(frame, counter, counter.counts, config)
            cv2.imshow(line_editor.window_name, frame_with_overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                counter.reset_counts()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if not hasattr(sys, "getwindowsversion"):
        # On non-Windows platforms ensure Ctrl+C terminates promptly
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
