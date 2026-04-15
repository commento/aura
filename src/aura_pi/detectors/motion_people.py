from __future__ import annotations

import cv2
import numpy as np

from .base import Detection


class MotionPeopleDetector:
    def __init__(self, min_area: int, history: int, var_threshold: int, learning_rate: float):
        self.min_area = min_area
        self.learning_rate = learning_rate
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
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

        detections.sort(key=lambda item: item.score, reverse=True)
        if detections:
            return detections[:6]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        fallback: list[Detection] = []
        for fx, fy, fw, fh in faces[:4]:
            px = max(0, int(fx - fw * 0.9))
            py = max(0, int(fy - fh * 0.7))
            pw = min(frame.shape[1] - px, int(fw * 2.8))
            ph = min(frame.shape[0] - py, int(fh * 4.2))
            fallback.append(Detection(x=px, y=py, w=pw, h=ph, score=0.55))
        return fallback

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
