from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VideoConfig:
    width: int
    height: int
    fps: int
    source: str
    device_index: int
    fullscreen_preview: bool
    window_name: str


@dataclass
class DetectorConfig:
    type: str
    min_area: int
    history: int
    var_threshold: int
    learning_rate: float
    interval: int = 1
    score_threshold: float = 0.4
    max_detections: int = 6
    model_path: str | None = None
    labels_path: str | None = None
    target_label: str = "person"


@dataclass
class TrackerConfig:
    max_distance: float
    max_missing_frames: int


@dataclass
class AudioConfig:
    enabled: bool
    device: str | int | None
    sample_rate: int
    block_size: int


@dataclass
class RenderConfig:
    background_dim: float
    aura_radius: int
    aura_alpha: float
    trail: bool
    show_labels: bool
    debug_boxes: bool = False
    aura_enabled: bool = True
    audio_threshold: float = 0.018
    audio_scale: float = 10.0
    space_warp: bool = False
    warp_strength: float = 0.0


@dataclass
class RecordingConfig:
    enabled: bool
    output_path: str
    ffmpeg_bin: str
    video_codec: str
    pixel_format: str
    crf: int
    preset: str
    audio_enabled: bool
    audio_input_format: str
    audio_input_device: str


@dataclass
class ArchiveRecordingConfig:
    enabled: bool = False
    output_path: str = "output/archive_%Y%m%d_%H%M%S.mp4"
    audio_enabled: bool = True


@dataclass
class AppConfig:
    project: dict[str, Any]
    video: VideoConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    audio: AudioConfig
    render: RenderConfig
    recording: RecordingConfig
    archive_recording: ArchiveRecordingConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path) -> AppConfig:
    raw = _read_yaml(Path(path))
    detector_raw = raw["detector"]
    archive_raw = raw.get("archive_recording", {})
    return AppConfig(
        project=raw.get("project", {}),
        video=VideoConfig(**raw["video"]),
        detector=DetectorConfig(
            type=detector_raw["type"],
            min_area=detector_raw.get("min_area", 5000),
            history=detector_raw.get("history", 250),
            var_threshold=detector_raw.get("var_threshold", 64),
            learning_rate=detector_raw.get("learning_rate", 0.001),
            interval=detector_raw.get("interval", 1),
            score_threshold=detector_raw.get("score_threshold", 0.4),
            max_detections=detector_raw.get("max_detections", 6),
            model_path=detector_raw.get("model_path"),
            labels_path=detector_raw.get("labels_path"),
            target_label=detector_raw.get("target_label", "person"),
        ),
        tracker=TrackerConfig(**raw["tracker"]),
        audio=AudioConfig(**raw["audio"]),
        render=RenderConfig(**raw["render"]),
        recording=RecordingConfig(**raw["recording"]),
        archive_recording=ArchiveRecordingConfig(
            enabled=archive_raw.get("enabled", False),
            output_path=archive_raw.get("output_path", "output/archive_%Y%m%d_%H%M%S.mp4"),
            audio_enabled=archive_raw.get("audio_enabled", True),
        ),
    )
