from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
import platform

import cv2

from .audio import AudioAnalyzer, AudioFeatures
from .config import AppConfig
from .detectors import Detection, HailoPersonDetector, MotionPeopleDetector
from .recorder import FfmpegRecorder
from .renderer import AuraRenderer, TrackedPerformer
from .vision import VideoSource


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    age: int = 0
    missing_frames: int = 0
    hits: int = 1


class PerformerTracker:
    def __init__(self, max_distance: float, max_missing_frames: int):
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}

    def update(self, detections: list[Detection]) -> list[TrackedPerformer]:
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()

        for det_index, detection in enumerate(detections):
            best_track_id = None
            best_score = float("-inf")
            for track_id, track in self._tracks.items():
                if track_id in assigned_tracks:
                    continue
                dist = hypot(detection.center[0] - track.center[0], detection.center[1] - track.center[1])
                if dist >= self.max_distance:
                    continue
                overlap = self._iou((detection.x, detection.y, detection.w, detection.h), track.bbox)
                score = overlap * 2.0 + (1.0 - dist / max(self.max_distance, 1.0))
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None:
                track = self._tracks[best_track_id]
                track.bbox = self._smooth_bbox(track.bbox, (detection.x, detection.y, detection.w, detection.h))
                track.center = self._bbox_center(track.bbox)
                track.age += 1
                track.missing_frames = 0
                track.hits += 1
                assigned_tracks.add(best_track_id)
                assigned_detections.add(det_index)

        for det_index, detection in enumerate(detections):
            if det_index in assigned_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=(detection.x, detection.y, detection.w, detection.h),
                center=detection.center,
                age=1,
                hits=1,
            )

        for track_id in list(self._tracks.keys()):
            if track_id not in assigned_tracks:
                self._tracks[track_id].missing_frames += 1
            if self._tracks[track_id].missing_frames > self.max_missing_frames:
                del self._tracks[track_id]

        return [
            TrackedPerformer(
                track_id=track.track_id,
                bbox=track.bbox,
                center=track.center,
                age=track.age,
            )
            for track in self._tracks.values()
            if track.hits >= 2 and track.missing_frames <= min(self.max_missing_frames, 20)
        ]

    def _smooth_bbox(
        self,
        previous: tuple[int, int, int, int],
        current: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        alpha = 0.65
        return tuple(
            int(previous[index] * (1.0 - alpha) + current[index] * alpha)
            for index in range(4)
        )

    def _bbox_center(self, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def _iou(
        self,
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
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


class AuraPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.video = VideoSource(
            width=config.video.width,
            height=config.video.height,
            fps=config.video.fps,
            source=config.video.source,
            device_index=config.video.device_index,
        )
        self.detector = self._build_detector()
        self.tracker = PerformerTracker(
            max_distance=config.tracker.max_distance,
            max_missing_frames=config.tracker.max_missing_frames,
        )
        self.audio = AudioAnalyzer(
            sample_rate=config.audio.sample_rate,
            block_size=config.audio.block_size,
            device=config.audio.device,
        )
        self.renderer = AuraRenderer(
            aura_radius=config.render.aura_radius,
            aura_alpha=config.render.aura_alpha,
            background_dim=config.render.background_dim,
            trail=config.render.trail,
            show_labels=config.render.show_labels,
            debug_boxes=config.render.debug_boxes,
            aura_enabled=config.render.aura_enabled,
            audio_threshold=config.render.audio_threshold,
            audio_scale=config.render.audio_scale,
        )
        self.recorder = None
        self.archive_recorder = None
        self.audio_recording_path: Path | None = None
        if config.recording.enabled:
            self.recorder = FfmpegRecorder(
                ffmpeg_bin=config.recording.ffmpeg_bin,
                output_path=config.recording.output_path,
                width=config.video.width,
                height=config.video.height,
                fps=config.video.fps,
                video_codec=config.recording.video_codec,
                pixel_format=config.recording.pixel_format,
                crf=config.recording.crf,
                preset=config.recording.preset,
                audio_enabled=config.recording.audio_enabled,
                audio_input_format=config.recording.audio_input_format,
                audio_input_device=config.recording.audio_input_device,
            )
        if config.archive_recording.enabled:
            self.archive_recorder = FfmpegRecorder(
                ffmpeg_bin=config.recording.ffmpeg_bin,
                output_path=config.archive_recording.output_path,
                width=config.video.width,
                height=config.video.height,
                fps=config.video.fps,
                video_codec=config.recording.video_codec,
                pixel_format=config.recording.pixel_format,
                crf=config.recording.crf,
                preset=config.recording.preset,
                audio_enabled=config.archive_recording.audio_enabled,
                audio_input_format=config.recording.audio_input_format,
                audio_input_device=config.recording.audio_input_device,
            )
        if config.audio.enabled and (self.recorder is not None or self.archive_recorder is not None):
            base_output = self.recorder.output_file if self.recorder is not None else self.archive_recorder.output_file
            self.audio_recording_path = base_output.with_suffix(".wav")

    def run(self) -> None:
        self.video.start()
        if self.config.audio.enabled:
            if self.audio_recording_path is not None:
                self.audio.start_recording(str(self.audio_recording_path))
            self.audio.start()
        self._setup_window()
        paused = False
        last_output = None

        try:
            while True:
                packet = None
                if not paused or last_output is None:
                    packet = self.video.read()
                    detections = self.detector.detect(packet.frame)
                    performers = self.tracker.update(detections)
                    audio_features = self.audio.read() if self.config.audio.enabled else AudioFeatures()
                    last_output = self.renderer.render(packet.frame, performers, audio_features)

                if self.recorder is not None and last_output is not None and not paused:
                    self.recorder.write(last_output)
                if self.archive_recorder is not None and packet is not None and not paused:
                    self.archive_recorder.write(packet.frame)

                cv2.imshow(self.config.video.window_name, last_output)

                key = cv2.waitKey(30 if paused else 1) & 0xFF
                if key in (27, ord("q"), ord("s")):
                    break
                if key == ord("p"):
                    paused = not paused
        finally:
            self.video.stop()
            self.audio.stop()
            cv2.destroyAllWindows()
            detector_close = getattr(self.detector, "close", None)
            if callable(detector_close):
                detector_close()
            if self.recorder is not None:
                self.recorder.close()
                audio_path = None
                if (
                    self.audio_recording_path is not None
                    and self.config.recording.audio_enabled
                    and self.audio.recorded_duration >= 0.5
                ):
                    audio_path = str(self.audio_recording_path)
                elif self.config.recording.audio_enabled and self.config.audio.enabled:
                    print(
                        "[Aura Pi] Audio realtime non rilevato o troppo breve; salvo il video senza audio."
                    )
                self.recorder.finalize(audio_path)
            if self.archive_recorder is not None:
                self.archive_recorder.close()
                archive_audio_path = None
                if (
                    self.audio_recording_path is not None
                    and self.config.archive_recording.audio_enabled
                    and self.audio.recorded_duration >= 0.5
                ):
                    archive_audio_path = str(self.audio_recording_path)
                self.archive_recorder.finalize(archive_audio_path)

    def _setup_window(self) -> None:
        cv2.namedWindow(self.config.video.window_name, cv2.WINDOW_NORMAL)
        if hasattr(cv2, "WND_PROP_ASPECT_RATIO") and hasattr(cv2, "WINDOW_FREERATIO"):
            cv2.setWindowProperty(
                self.config.video.window_name,
                cv2.WND_PROP_ASPECT_RATIO,
                cv2.WINDOW_FREERATIO,
            )
        use_native_fullscreen = self.config.video.fullscreen_preview and platform.system() != "Darwin"
        if use_native_fullscreen:
            cv2.setWindowProperty(
                self.config.video.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            try:
                cv2.moveWindow(self.config.video.window_name, 0, 0)
            except cv2.error:
                pass
        else:
            cv2.resizeWindow(
                self.config.video.window_name,
                self.config.video.width,
                self.config.video.height,
            )

    def _build_detector(self):
        detector_type = self.config.detector.type
        if detector_type in {"motion", "motion_person", "motion_people"}:
            return MotionPeopleDetector(
                min_area=self.config.detector.min_area,
                history=self.config.detector.history,
                var_threshold=self.config.detector.var_threshold,
                learning_rate=self.config.detector.learning_rate,
            )
        if detector_type in {"hailo", "hailo_person"}:
            hailo_detector = HailoPersonDetector(
                model_path=self.config.detector.model_path,
                labels_path=self.config.detector.labels_path,
                score_threshold=self.config.detector.score_threshold,
                max_detections=self.config.detector.max_detections,
                target_label=self.config.detector.target_label,
            )
            if hailo_detector.is_ready:
                return hailo_detector
            print(
                "[Aura Pi] Hailo detector non pronto, fallback a motion_person:",
                hailo_detector.init_error or "motivo sconosciuto",
            )
            return MotionPeopleDetector(
                min_area=self.config.detector.min_area,
                history=self.config.detector.history,
                var_threshold=self.config.detector.var_threshold,
                learning_rate=self.config.detector.learning_rate,
            )
        raise ValueError(f"Detector non supportato: {detector_type}")
