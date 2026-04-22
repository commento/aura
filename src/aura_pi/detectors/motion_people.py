from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import Detection


class MotionPeopleDetector:
    def __init__(self, min_area: int, history: int, var_threshold: int, learning_rate: float, max_detections: int = 6):
        self.min_area = min_area
        self.learning_rate = learning_rate
        self.max_detections = max_detections
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.face_detector = self._load_face_detector()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = self.background.apply(frame, learningRate=self.learning_rate)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / max(w, 1)
            if aspect_ratio < 1.05:
                continue
            if not self._looks_like_person(frame, x, y, w, h):
                continue

            detections.append(Detection(x=x, y=y, w=w, h=h, score=min(area / 20000.0, 1.0)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if self.face_detector is not None:
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
            )

        for fx, fy, fw, fh in faces[: self.max_detections]:
            px = max(0, int(fx - fw * 0.9))
            py = max(0, int(fy - fh * 0.7))
            pw = min(frame.shape[1] - px, int(fw * 2.8))
            ph = min(frame.shape[0] - py, int(fh * 4.2))
            candidate = Detection(x=px, y=py, w=pw, h=ph, score=0.55)
            if not self._overlaps_existing(candidate, detections, threshold=0.18):
                detections.append(candidate)

        if len(detections) < min(4, self.max_detections):
            detections.extend(self._hog_people(frame, detections))

        detections = self._non_max_suppression(detections, threshold=0.35)
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: self.max_detections]

    def _looks_like_person(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.1)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(frame.shape[1], x + w + pad_x)
        y1 = min(frame.shape[0], y + h + pad_y)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.face_detector is not None:
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(36, 36),
            )
            if len(faces) > 0:
                return True

        roi_height, roi_width = roi.shape[:2]
        if roi_height < 96 or roi_width < 48:
            return False

        resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_LINEAR)
        rects, _ = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        return len(rects) > 0

    def _hog_people(self, frame: np.ndarray, existing: list[Detection]) -> list[Detection]:
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            candidate = Detection(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                score=float(min(max(weight, 0.35), 0.8)),
            )
            if self._overlaps_existing(candidate, existing + detections, threshold=0.18):
                continue
            detections.append(candidate)
            if len(detections) >= self.max_detections:
                break
        return detections

    def _overlaps_existing(self, candidate: Detection, existing: list[Detection], threshold: float) -> bool:
        candidate_bbox = (candidate.x, candidate.y, candidate.w, candidate.h)
        for detection in existing:
            if self._iou(candidate_bbox, (detection.x, detection.y, detection.w, detection.h)) >= threshold:
                return True
        return False

    def _non_max_suppression(self, detections: list[Detection], threshold: float) -> list[Detection]:
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            if self._overlaps_existing(detection, kept, threshold):
                continue
            kept.append(detection)
        return kept

    def _iou(self, bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def _load_face_detector(self):
        cascade_name = "haarcascade_frontalface_default.xml"
        candidates: list[Path] = []

        data_module = getattr(cv2, "data", None)
        if data_module is not None:
            haarcascades = getattr(data_module, "haarcascades", None)
            if haarcascades:
                candidates.append(Path(haarcascades) / cascade_name)

        candidates.extend(
            [
                Path("/usr/share/opencv4/haarcascades") / cascade_name,
                Path("/usr/share/opencv/haarcascades") / cascade_name,
                Path("/usr/local/share/opencv4/haarcascades") / cascade_name,
            ]
        )

        for candidate in candidates:
            if not candidate.exists():
                continue
            detector = cv2.CascadeClassifier(str(candidate))
            if detector.empty():
                continue
            return detector
        return None
