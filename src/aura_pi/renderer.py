from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

from .audio import AudioFeatures


@dataclass
class TrackedPerformer:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    age: int


class AuraRenderer:
    def __init__(self, aura_radius: int, aura_alpha: float, background_dim: float, trail: bool, show_labels: bool):
        self.aura_radius = aura_radius
        self.aura_alpha = aura_alpha
        self.background_dim = background_dim
        self.trail = trail
        self.show_labels = show_labels
        self.trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=24))

    def render(self, frame: np.ndarray, performers: list[TrackedPerformer], audio: AudioFeatures) -> np.ndarray:
        base = frame.copy()
        mist = np.zeros_like(frame)
        dimmed = cv2.convertScaleAbs(base, alpha=max(0.0, 1.0 - self.background_dim), beta=0)
        aura_level = self._audio_gate(audio)

        for performer in performers:
            self.trails[performer.track_id].append(performer.center)
            if aura_level <= 0.0:
                continue

            tone = self._aura_tone(aura_level)
            energy = min(1.0, 0.2 + aura_level * 0.95 + audio.peak * 0.16)
            radius = int(self.aura_radius * (0.9 + energy))
            self._draw_simple_mist(mist, performer, tone, radius, energy, aura_level)
            if self.trail:
                self._draw_whisper_trail(mist, performer.track_id, tone, aura_level)

        mist = self._soft_blur(mist, sigma=16)
        composed = cv2.addWeighted(dimmed, 1.0, mist, self.aura_alpha * 0.78, 0.0)
        return composed

    def _audio_gate(self, audio: AudioFeatures) -> float:
        threshold = 0.018
        scale = 10.0
        gate = max(0.0, (audio.rms - threshold) * scale)
        return min(1.0, gate)

    def _aura_tone(self, audio_gate: float) -> tuple[int, int, int]:
        value = 156 + int(audio_gate * 20)
        return (value, value, value)

    def _draw_simple_mist(
        self,
        image: np.ndarray,
        performer: TrackedPerformer,
        color: tuple[int, int, int],
        radius: int,
        energy: float,
        audio_gate: float,
    ) -> None:
        x, y, w, h = performer.bbox
        cx, cy = performer.center
        body_center = np.array([cx, y + h // 2], dtype=np.float32)
        pale = tuple(min(220, int(channel * 0.88 + 6)) for channel in color)

        base_rx = max(22.0, w * (0.62 + audio_gate * 0.06))
        base_ry = max(36.0, h * (0.78 + audio_gate * 0.1))
        angles = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        phase = performer.track_id * 0.73 + energy * 2.4
        wobble = 0.1 + audio_gate * 0.08
        points = []
        for angle in angles:
            radial = 1.0 + wobble * np.sin(angle * 3.0 + phase) + 0.06 * np.cos(angle * 5.0 - phase * 0.7)
            px = body_center[0] + np.cos(angle) * base_rx * radial
            py = body_center[1] + np.sin(angle) * base_ry * radial
            points.append([int(px), int(py)])

        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(image, [contour], pale, lineType=cv2.LINE_AA)

        head_center = (
            int(cx + np.sin(phase) * radius * 0.04),
            max(0, int(y + h * 0.22)),
        )
        head_axes = (
            max(12, int(radius * 0.18)),
            max(16, int(radius * 0.24)),
        )
        cv2.ellipse(image, head_center, head_axes, np.sin(phase) * 12.0, 0, 360, pale, -1, cv2.LINE_AA)

    def _draw_whisper_trail(self, image: np.ndarray, track_id: int, color: tuple[int, int, int], audio_gate: float) -> None:
        points = list(self.trails[track_id])
        if len(points) < 3:
            return
        displacement = np.linalg.norm(np.array(points[-1], dtype=np.float32) - np.array(points[-3], dtype=np.float32))
        if displacement < 8.0:
            return
        for index in range(1, len(points)):
            alpha = index / len(points)
            current = points[index]
            previous = points[index - 1]
            fade = max(88, int(132 + audio_gate * 24 - (1 - alpha) * 60))
            trail_color = tuple(min(fade, channel) for channel in color)
            mx = int((current[0] + previous[0]) / 2)
            my = int((current[1] + previous[1]) / 2)
            radius = max(3, int((7 + audio_gate * 6) * alpha))
            cv2.circle(image, current, radius, trail_color, -1, cv2.LINE_AA)
            cv2.circle(image, (mx, my), max(3, int(radius * 0.7)), trail_color, -1, cv2.LINE_AA)

    def _soft_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, sigma / 2.0), sigmaY=max(1.0, sigma / 2.0))
        return cv2.resize(blurred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
