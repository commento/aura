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
        audio_threshold: float = 0.018,
        audio_scale: float = 10.0,
        space_warp: bool = False,
        warp_strength: float = 0.0,
    ):
        self.aura_radius = aura_radius
        self.aura_alpha = aura_alpha
        self.background_dim = background_dim
        self.trail = trail
        self.show_labels = show_labels
        self.debug_boxes = debug_boxes
        self.aura_enabled = aura_enabled
        self.audio_threshold = audio_threshold
        self.audio_scale = audio_scale
        self.space_warp = space_warp
        self.warp_strength = warp_strength
        self.trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=24))
        self.aura_states: dict[int, float] = defaultdict(float)
        self.anchor_states: dict[int, float] = defaultdict(float)
        self.scene_energy = 0.0
        self._plume_layer: np.ndarray | None = None
        self._last_seed_layer: np.ndarray | None = None

    def render(self, frame: np.ndarray, performers: list[TrackedPerformer], audio: AudioFeatures) -> np.ndarray:
        base = frame.copy()
        if self.debug_boxes:
            for performer in performers:
                self.trails[performer.track_id].append(performer.center)
                self._draw_debug_box(base, performer)
            return base

        mist = np.zeros_like(frame)
        self._ensure_plume_layer(frame)
        dimmed = cv2.convertScaleAbs(base, alpha=max(0.0, 1.0 - self.background_dim), beta=0)
        aura_level = self._audio_gate(audio)
        if aura_level <= 0.001:
            self.scene_energy = 0.0
        else:
            self.scene_energy = self._ease(self.scene_energy, aura_level, attack=0.08, release=0.12)

        for performer in performers:
            self.trails[performer.track_id].append(performer.center)
            if not self.aura_enabled:
                continue

            match_strength = min(1.0, max(0.0, (performer.age - 1) / 10.0))
            target_presence = 0.0 if aura_level <= 0.001 else self.scene_energy * match_strength
            anchor = self.anchor_states[performer.track_id]
            if aura_level <= 0.001:
                anchor = 0.0
            else:
                anchor_target = 0.14 + match_strength * 0.36
                anchor = self._ease(anchor, anchor_target, attack=0.018, release=0.08)
            self.anchor_states[performer.track_id] = anchor
            target_presence = max(target_presence, anchor)
            current_presence = self.aura_states[performer.track_id]
            if target_presence <= 0.0:
                current_presence = 0.0
            else:
                current_presence = self._ease(current_presence, target_presence, attack=0.1, release=0.2)
            self.aura_states[performer.track_id] = current_presence
            if current_presence <= 0.01:
                continue

            tone = self._aura_tone(current_presence)
            energy = min(1.0, 0.16 + current_presence * 0.92 + audio.peak * 0.08)
            radius = int(self.aura_radius * (0.82 + energy * 0.9))
            self._draw_ar_aura(mist, frame, performer, tone, radius, energy, current_presence)
            if self.trail:
                self._draw_whisper_trail(mist, performer.track_id, tone, current_presence)

        active_track_ids = {performer.track_id for performer in performers}
        for track_id in list(self.aura_states.keys()):
            if track_id in active_track_ids:
                continue
            faded = self._ease(self.aura_states[track_id], 0.0, attack=0.0, release=0.04)
            if faded <= 0.01:
                del self.aura_states[track_id]
                if track_id in self.anchor_states:
                    del self.anchor_states[track_id]
            else:
                self.aura_states[track_id] = faded

        self._draw_group_fusion(mist, performers)
        plume = self._update_plume_layer(mist)
        cv2.add(mist, plume, dst=mist)
        mist = self._soft_blur(mist, sigma=20)
        warped = self._apply_space_warp(dimmed, performers) if self.space_warp and self.warp_strength > 0.0 else dimmed
        warped = self._apply_presence_glitch(warped, performers)
        composed = cv2.addWeighted(warped, 1.0, mist, self.aura_alpha * 0.66, 0.0)
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
        rms_energy = max(0.0, (audio.rms - self.audio_threshold) * self.audio_scale)
        peak_energy = max(0.0, (audio.peak - self.audio_threshold * 1.1) * (self.audio_scale * 0.35))
        gate = rms_energy * 0.85 + peak_energy * 0.15
        return min(1.0, gate)

    def _aura_tone(self, audio_gate: float) -> tuple[int, int, int]:
        blue_lift = int(audio_gate * 8)
        base = 118 + int(audio_gate * 16)
        return (base, min(150, base + 2), min(158, base + 6 + blue_lift))

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
            self._draw_presence_fallback(image, performer, color, audio_gate)
            return

        soft = tuple(min(150, int(channel * 0.84 + 2)) for channel in color)
        faint = tuple(min(126, int(channel * 0.68 + 2)) for channel in color)
        roi_h, roi_w = edge_mask.shape[:2]

        upper_focus = np.zeros_like(edge_mask)
        upper_top = max(0, int(roi_h * 0.04))
        upper_bottom = min(roi_h, int(roi_h * 0.56))
        cv2.rectangle(upper_focus, (0, upper_top), (roi_w, upper_bottom), 255, -1)
        edge_mask = cv2.bitwise_and(edge_mask, upper_focus)
        if cv2.countNonZero(edge_mask) == 0:
            self._draw_presence_fallback(image, performer, color, audio_gate)
            return

        halo = cv2.dilate(edge_mask, np.ones((9, 9), np.uint8), iterations=1)
        halo = cv2.subtract(halo, edge_mask)
        self._apply_mask(image, halo, x0, y0, faint)

        contour = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
        contour = cv2.subtract(contour, cv2.erode(edge_mask, np.ones((2, 2), np.uint8), iterations=1))
        self._apply_mask(image, contour, x0, y0, soft)

        shoulder_focus = np.zeros_like(edge_mask)
        top = max(0, int(roi_h * 0.12))
        bottom = min(roi_h, int(roi_h * 0.48))
        cv2.rectangle(shoulder_focus, (0, top), (roi_w, bottom), 255, -1)
        shoulder_glow = cv2.bitwise_and(halo, shoulder_focus)
        shoulder_glow = cv2.dilate(shoulder_glow, np.ones((9, 9), np.uint8), iterations=1)
        self._apply_mask(image, shoulder_glow, x0, y0, soft)

        mist_band = cv2.dilate(shoulder_glow, np.ones((15, 15), np.uint8), iterations=1)
        mist_band = cv2.GaussianBlur(mist_band, (0, 0), sigmaX=6.0, sigmaY=6.0)
        self._apply_mask(image, mist_band, x0, y0, faint)

        head_center = (int(cx), max(0, int(y + h * 0.18)))
        head_axes = (max(14, int(radius * 0.14)), max(20, int(radius * 0.2)))
        cv2.ellipse(image, head_center, head_axes, 0, 0, 360, soft, -1, cv2.LINE_AA)

        collar_center = (int(cx), int(y + h * 0.32))
        collar_axes = (max(16, int(w * 0.34)), max(8, int(h * 0.08)))
        cv2.ellipse(image, collar_center, collar_axes, 0, 0, 360, faint, -1, cv2.LINE_AA)

        chest_center = (int(cx), int(y + h * 0.4))
        chest_axes = (max(18, int(w * 0.24)), max(8, int(h * 0.06)))
        cv2.ellipse(image, chest_center, chest_axes, 0, 0, 360, faint, -1, cv2.LINE_AA)

        veil_center = (int(cx), int(y + h * 0.28))
        veil_axes = (max(20, int(w * 0.3)), max(14, int(h * 0.12)))
        cv2.ellipse(image, veil_center, veil_axes, 0, 0, 360, faint, -1, cv2.LINE_AA)

    def _draw_presence_fallback(
        self,
        image: np.ndarray,
        performer: TrackedPerformer,
        color: tuple[int, int, int],
        audio_gate: float,
    ) -> None:
        x, y, w, h = performer.bbox
        cx, _ = performer.center
        glow = tuple(min(146, int(channel * 0.88 + 3)) for channel in color)
        shoulder_center = (int(cx), int(y + h * 0.32))
        shoulder_axes = (max(20, int(w * 0.42)), max(14, int(h * (0.14 + audio_gate * 0.05))))
        cv2.ellipse(image, shoulder_center, shoulder_axes, 0, 0, 360, glow, -1, cv2.LINE_AA)

        head_center = (int(cx), max(0, int(y + h * 0.18)))
        head_axes = (max(12, int(w * 0.15)), max(16, int(h * 0.16)))
        cv2.ellipse(image, head_center, head_axes, 0, 0, 360, glow, -1, cv2.LINE_AA)

    def _draw_group_fusion(self, image: np.ndarray, performers: list[TrackedPerformer]) -> None:
        if len(performers) < 2:
            return

        for index, performer in enumerate(performers):
            presence_a = self.aura_states.get(performer.track_id, 0.0)
            if presence_a <= 0.08:
                continue
            ax, ay = performer.center
            for other in performers[index + 1:]:
                presence_b = self.aura_states.get(other.track_id, 0.0)
                if presence_b <= 0.08:
                    continue
                bx, by = other.center
                distance = float(np.hypot(ax - bx, ay - by))
                if distance > max(220.0, (performer.bbox[2] + other.bbox[2]) * 0.9):
                    continue

                shared = min(presence_a, presence_b)
                fusion_strength = max(0.0, 1.0 - distance / 220.0) * shared
                if fusion_strength <= 0.04:
                    continue

                mid_x = int((ax + bx) / 2)
                shoulder_y = int(min(performer.bbox[1] + performer.bbox[3] * 0.3, other.bbox[1] + other.bbox[3] * 0.3))
                axes = (
                    max(24, int(distance * 0.32)),
                    max(12, int((performer.bbox[3] + other.bbox[3]) * 0.06)),
                )
                tone = self._aura_tone(fusion_strength)
                fusion_color = tuple(min(132, int(channel * 0.74)) for channel in tone)
                cv2.ellipse(image, (mid_x, shoulder_y), axes, 0, 0, 360, fusion_color, -1, cv2.LINE_AA)

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

    def _ease(self, current: float, target: float, attack: float, release: float) -> float:
        if target > current:
            return current + (target - current) * attack
        return current + (target - current) * release

    def _apply_space_warp(self, frame: np.ndarray, performers: list[TrackedPerformer]) -> np.ndarray:
        active = [
            (
                performer,
                max(
                    self.aura_states.get(performer.track_id, 0.0),
                    self.anchor_states.get(performer.track_id, 0.0),
                    self.scene_energy * 0.75,
                ),
            )
            for performer in performers
            if max(
                self.aura_states.get(performer.track_id, 0.0),
                self.anchor_states.get(performer.track_id, 0.0),
                self.scene_energy * 0.75,
            ) > 0.04
        ]
        if not active and self.scene_energy <= 0.01:
            return frame

        h, w = frame.shape[:2]
        scale = 0.25
        grid_w = max(8, int(w * scale))
        grid_h = max(8, int(h * scale))
        yy, xx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        xx *= w / grid_w
        yy *= h / grid_h
        scene_presence = float(
            np.clip(
                max(
                    self.scene_energy,
                    np.mean([presence for _, presence in active]) if active else 0.0,
                ),
                0.0,
                1.0,
            )
        )

        center_x = w * 0.5
        center_y = h * 0.5
        if active:
            weighted_x = 0.0
            weighted_y = 0.0
            total = 0.0
            for performer, presence in active:
                px, py = performer.center
                weighted_x += px * presence
                weighted_y += (performer.bbox[1] + performer.bbox[3] * 0.28) * presence
                total += presence
            if total > 0.0:
                target_x = weighted_x / total
                target_y = weighted_y / total
                center_x = center_x * 0.82 + target_x * 0.18
                center_y = center_y * 0.82 + target_y * 0.18

        norm_x = (xx - center_x) / max(w * 0.5, 1.0)
        norm_y = (yy - center_y) / max(h * 0.5, 1.0)
        radius = np.sqrt(norm_x * norm_x + norm_y * norm_y)
        edge_falloff = np.clip((radius - 0.22) / 0.95, 0.0, 1.0)

        fisheye_strength = float(self.warp_strength) * (0.4 + scene_presence * 0.6)
        stretch = 1.0 + edge_falloff * edge_falloff * fisheye_strength * 0.28
        map_x_small = center_x + norm_x * stretch * (w * 0.5)
        map_y_small = center_y + norm_y * stretch * (h * 0.5)

        swirl = edge_falloff * edge_falloff * fisheye_strength * 0.02
        map_x_small += -norm_y * swirl * w
        map_y_small += norm_x * swirl * h

        map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_CUBIC)
        map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    def _apply_presence_glitch(self, frame: np.ndarray, performers: list[TrackedPerformer]) -> np.ndarray:
        output = frame.copy()
        for performer in performers:
            presence = self.aura_states.get(performer.track_id, 0.0)
            if presence <= 0.05:
                continue

            x, y, w, h = performer.bbox
            roi_w = max(72, int(w * 0.86))
            roi_h = max(80, int(h * 0.66))
            cx = int(performer.center[0])
            cy = int(y + h * 0.22)
            x0 = max(0, cx - roi_w // 2)
            y0 = max(0, cy - roi_h // 2)
            x1 = min(output.shape[1], x0 + roi_w)
            y1 = min(output.shape[0], y0 + roi_h)
            roi = output[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            glitched_roi = self._glitch_presence_roi(roi, presence, performer.track_id)
            blend = min(0.62, 0.22 + presence * 0.34)
            cv2.addWeighted(glitched_roi, blend, roi, 1.0 - blend, 0.0, dst=roi)
        return output

    def _glitch_presence_roi(self, roi: np.ndarray, presence: float, track_id: int) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 4 or w < 4:
            return roi

        glitched = roi.copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        threshold = int(76 + presence * 28)
        active = gray > threshold

        phase = track_id * 0.61 + presence * 4.0
        max_shift = max(3, int(4 + presence * 14))
        for col in range(w):
            column_energy = float(np.mean(active[:, col]))
            if column_energy < 0.08:
                continue
            shift = int(np.sin(col * 0.16 + phase) * max_shift * column_energy)
            if shift != 0:
                glitched[:, col] = np.roll(glitched[:, col], -shift, axis=0)

        row_start = max(0, int(h * 0.06))
        row_end = min(h, int(h * 0.78))
        for row in range(row_start, row_end):
            indices = np.flatnonzero(active[row])
            if indices.size < max(8, int(w * 0.08)):
                continue
            left = int(indices[0])
            right = int(indices[-1]) + 1
            segment = glitched[row, left:right]
            luminance = np.mean(segment, axis=1)
            order = np.argsort(luminance)
            if (row + track_id) % 2 == 0:
                glitched[row, left:right] = segment[order]
            else:
                glitched[row, left:right] = segment[order[::-1]]

        channel_offset = max(1, int(2 + presence * 7))
        glitched[:, :, 0] = np.roll(glitched[:, :, 0], -channel_offset, axis=1)
        glitched[:, :, 2] = np.roll(glitched[:, :, 2], channel_offset, axis=0)
        return cv2.GaussianBlur(glitched, (0, 0), sigmaX=0.45, sigmaY=0.45)

    def _ensure_plume_layer(self, frame: np.ndarray) -> None:
        if self._plume_layer is None or self._plume_layer.shape != frame.shape:
            self._plume_layer = np.zeros_like(frame)
        if self._last_seed_layer is None or self._last_seed_layer.shape != frame.shape:
            self._last_seed_layer = np.zeros_like(frame)

    def _update_plume_layer(self, mist: np.ndarray) -> np.ndarray:
        assert self._plume_layer is not None
        assert self._last_seed_layer is not None
        plume = self._plume_layer
        seed_layer = self._extract_upper_seed(mist)

        shifted = np.zeros_like(plume)
        rise_px = 8
        if rise_px < plume.shape[0]:
            shifted[:-rise_px] = plume[rise_px:]

        spread = cv2.GaussianBlur(shifted, (0, 0), sigmaX=5.5, sigmaY=6.5)
        spread = cv2.convertScaleAbs(spread, alpha=0.82, beta=0)

        seed = cv2.GaussianBlur(seed_layer, (0, 0), sigmaX=4.0, sigmaY=4.0)
        seed = cv2.convertScaleAbs(seed, alpha=0.22, beta=0)
        cv2.add(spread, seed, dst=spread)

        self._last_seed_layer = seed_layer
        self._plume_layer = spread
        return spread

    def _extract_upper_seed(self, mist: np.ndarray) -> np.ndarray:
        h, w = mist.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        top = 0
        bottom = max(1, int(h * 0.42))
        cv2.rectangle(mask, (0, top), (w, bottom), 255, -1)
        masked = cv2.bitwise_and(mist, mist, mask=mask)
        return masked
