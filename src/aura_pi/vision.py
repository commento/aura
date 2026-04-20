from __future__ import annotations

from dataclasses import dataclass
import platform
import threading
import time

import cv2
import numpy as np


class VideoSourceError(RuntimeError):
    pass


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_ms: float


class VideoSource:
    def __init__(self, width: int, height: int, fps: int, source: str, device_index: int = 0):
        self.width = width
        self.height = height
        self.fps = fps
        self.source = source
        self.device_index = device_index
        self._picam = None
        self._cap = None
        self._thread = None
        self._stop_event = threading.Event()
        self._latest_packet: FramePacket | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.source == "picamera2":
            try:
                from picamera2 import Picamera2
            except ImportError as exc:  # pragma: no cover - depends on target hardware
                raise VideoSourceError("Picamera2 non disponibile. Installa python3-picamera2.") from exc

            self._picam = Picamera2()
            config = self._picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": float(self.fps)},
            )
            self._picam.configure(config)
            self._picam.start()
            return

        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.device_index, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            raise VideoSourceError(f"Impossibile aprire la sorgente video index={self.device_index}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def read(self) -> FramePacket:
        if self._picam is not None:
            frame = self._picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = self._normalize_frame(frame)
            return FramePacket(frame=frame, timestamp_ms=cv2.getTickCount() / cv2.getTickFrequency() * 1000.0)

        if self._cap is None:
            raise VideoSourceError("Video source non avviata")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._lock:
                packet = self._latest_packet
            if packet is not None:
                return FramePacket(frame=packet.frame.copy(), timestamp_ms=packet.timestamp_ms)
            time.sleep(0.005)
        raise VideoSourceError("Lettura frame fallita")

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        if src_w <= 0 or src_h <= 0:
            return frame

        target_ratio = self.width / self.height
        source_ratio = src_w / src_h

        if abs(source_ratio - target_ratio) < 0.01:
            return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if source_ratio > target_ratio:
            new_w = int(src_h * target_ratio)
            offset_x = max(0, (src_w - new_w) // 2)
            cropped = frame[:, offset_x:offset_x + new_w]
        else:
            new_h = int(src_w / target_ratio)
            offset_y = max(0, (src_h - new_h) // 2)
            cropped = frame[offset_y:offset_y + new_h, :]

        return cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest_packet = None

    def _capture_loop(self) -> None:
        if self._cap is None:
            return
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = self._normalize_frame(frame)
            packet = FramePacket(
                frame=frame,
                timestamp_ms=cv2.getTickCount() / cv2.getTickFrequency() * 1000.0,
            )
            with self._lock:
                self._latest_packet = packet
