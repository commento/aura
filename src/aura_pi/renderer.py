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
    def __init__(
        self,
        aura_radius: int,
        aura_alpha: float,
        background_dim: float,
        trail: bool,
        show_labels: bool,
        debug_boxes: bool = False,
        aura_enabled: bool = True,
    ):
        self.aura_radius = aura_radius
        self.aura_alpha = aura_alpha
        self.background_dim = background_dim
        self.trail = trail
        self.show_labels = show_labels
        self.debug_boxes = debug_boxes
        self.aura_enabled = aura_enabled
        self.trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=24))

    def render(self, frame: np.ndarray, performers: list[TrackedPerformer], audio: AudioFeatures) -> np.ndarray:
        base = frame.copy()
        if self.debug_boxes:
            for performer in performers:
                self.trails[performer.track_id].append(performer.center)
                self._draw_debug_box(base, performer)
            return base

        mist = np.zeros_like(frame)
        dimmed = cv2.convertScaleAbs(base, alpha=max(0.0, 1.0 - self.background_dim), beta=0)
        aura_level = self._audio_gate(audio)

        for performer in performers:
            self.trails[performer.track_id].append(performer.center)
            if not self.aura_enabled or aura_level <= 0.0:
                continue

            tone = self._aura_tone(aura_level)
            energy = min(1.0, 0.2 + aura_level * 0.95 + audio.peak * 0.16)
            radius = int(self.aura_radius * (0.9 + energy))
            self._draw_ar_aura(mist, frame, performer, tone, radius, energy, aura_level)
            if self.trail:
                self._draw_whisper_trail(mist, performer.track_id, tone, aura_level)

        mist = self._soft_blur(mist, sigma=16)
        composed = cv2.addWeighted(dimmed, 1.0, mist, self.aura_alpha * 0.78, 0.0)
        return composed

    def _draw_debug_box(self, image: np.ndarray, performer: TrackedPerformer) -> None:
        x, y, w, h = performer.bbox
        color = (210, 210, 210)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cx, cy = performer.center
        cv2.drawMarker(
            image,
            (int(cx), int(cy)),
            (180, 180, 180),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    def _audio_gate(self, audio: AudioFeatures) -> float:
        threshold = 0.018
        scale = 10.0
        gate = max(0.0, (audio.rms - threshold) * scale)
        return min(1.0, gate)

    def _aura_tone(self, audio_gate: float) -> tuple[int, int, int]:
        value = 156 + int(audio_gate * 20)
        return (value, value, value)

    def _draw_ar_aura(
        self,
        image: np.ndarray,
        frame: np.ndarray,
        performer: TrackedPerformer,
        color: tuple[int, int, int],
        radius: int,
        energy: float,
        audio_gate: float,
    ) -> None:
        x, y, w, h = performer.bbox
        cx, cy = performer.center
        edge_mask, x0, y0 = self._person_edge_mask(frame, performer)
        if edge_mask is None:
            return

        soft = tuple(min(168, int(channel * 0.8 + 4)) for channel in color)
        faint = tuple(min(138, int(channel * 0.66 + 3)) for channel in color)

        halo = cv2.dilate(edge_mask, np.ones((9, 9), np.uint8), iterations=1)
        halo = cv2.subtract(halo, edge_mask)
        self._apply_mask(image, halo, x0, y0, faint)

        contour = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=1)
        contour = cv2.subtract(contour, cv2.erode(edge_mask, np.ones((3, 3), np.uint8), iterations=1))
        self._apply_mask(image, contour, x0, y0, soft)

        roi_h, roi_w = edge_mask.shape[:2]
        shoulder_focus = np.zeros_like(edge_mask)
        top = max(0, int(roi_h * 0.08))
        bottom = min(roi_h, int(roi_h * 0.42))
        cv2.rectangle(shoulder_focus, (0, top), (roi_w, bottom), 255, -1)
        shoulder_glow = cv2.bitwise_and(halo, shoulder_focus)
        shoulder_glow = cv2.dilate(shoulder_glow, np.ones((7, 7), np.uint8), iterations=1)
        self._apply_mask(image, shoulder_glow, x0, y0, color)

        head_center = (int(cx), max(0, int(y + h * 0.18)))
        head_axes = (max(12, int(radius * 0.12)), max(18, int(radius * 0.18)))
        cv2.ellipse(image, head_center, head_axes, 0, 0, 360, soft, -1, cv2.LINE_AA)

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

    def _person_edge_mask(self, frame: np.ndarray, performer: TrackedPerformer) -> tuple[np.ndarray | None, int, int]:
        x, y, w, h = performer.bbox
        pad_x = max(8, int(w * 0.12))
        pad_y = max(8, int(h * 0.08))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(frame.shape[1], x + w + pad_x)
        y1 = min(frame.shape[0], y + h + pad_y)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None, x0, y0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 110)

        focus = np.zeros_like(edges)
        focus_top = max(0, int(focus.shape[0] * 0.06))
        focus_bottom = min(focus.shape[0], int(focus.shape[0] * 0.94))
        cv2.rectangle(focus, (0, focus_top), (focus.shape[1], focus_bottom), 255, -1)
        edges = cv2.bitwise_and(edges, focus)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        if cv2.countNonZero(edges) == 0:
            return None, x0, y0
        return edges, x0, y0

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray, x0: int, y0: int, color: tuple[int, int, int]) -> None:
        if cv2.countNonZero(mask) == 0:
            return
        roi = image[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        colored = np.zeros_like(roi)
        colored[:] = color
        masked = cv2.bitwise_and(colored, colored, mask=mask)
        cv2.add(roi, masked, dst=roi)

    def _soft_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, sigma / 2.0), sigmaY=max(1.0, sigma / 2.0))
        return cv2.resize(blurred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
