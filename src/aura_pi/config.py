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
class AppConfig:
    project: dict[str, Any]
    video: VideoConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    audio: AudioConfig
    render: RenderConfig
    recording: RecordingConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path) -> AppConfig:
    raw = _read_yaml(Path(path))
    return AppConfig(
        project=raw.get("project", {}),
        video=VideoConfig(**raw["video"]),
        detector=DetectorConfig(**raw["detector"]),
        tracker=TrackerConfig(**raw["tracker"]),
        audio=AudioConfig(**raw["audio"]),
        render=RenderConfig(**raw["render"]),
        recording=RecordingConfig(**raw["recording"]),
    )
