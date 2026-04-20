from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detection


class HailoPersonDetector:
    """
    Skeleton detector for Raspberry Pi 5 + AI HAT+.

    This adapter keeps the rest of the pipeline unchanged: whatever Hailo runtime
    you install on the Pi only needs to provide person detections that we convert
    to the project's Detection dataclass.

    The actual Hailo runtime wiring is intentionally isolated here because the
    exact Python package and model export path can vary with the installed stack.
    """

    def __init__(
        self,
        model_path: str | None,
        labels_path: str | None = None,
        score_threshold: float = 0.4,
        max_detections: int = 6,
        target_label: str = "person",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.labels_path = Path(labels_path) if labels_path else None
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.target_label = target_label
        self._runtime_name = "unconfigured"
        self._infer = None
        self._init_error: str | None = None
        self._labels = self._load_labels(self.labels_path)
        self._init_runtime()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._infer is None:
            raise RuntimeError(
                "Hailo runtime non inizializzato. Installa lo stack Hailo sul Raspberry Pi "
                "e configura detector.model_path con un modello person compatibile."
            )

        raw_predictions = self._infer(frame)
        detections = self._convert_predictions(raw_predictions, frame.shape[1], frame.shape[0])
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: self.max_detections]

    def _init_runtime(self) -> None:
        if self.model_path is None:
            self._init_error = "model_path mancante"
            return

        # Placeholder for the most common deployment path on Raspberry Pi / Hailo.
        # When the Hailo Python runtime is available on the Pi, replace this wiring
        # with the installed package entrypoints for loading a HEF and invoking inference.
        try:
            import hailo_platform  # type: ignore  # pragma: no cover
        except Exception:
            self._runtime_name = "missing_hailo_runtime"
            self._init_error = "runtime hailo_platform non disponibile"
            self._infer = None
            return

        self._runtime_name = "hailo_platform"
        self._infer = self._build_stub_infer(model_path=self.model_path, runtime=hailo_platform)
        self._init_error = "adapter modello Hailo non ancora implementato"

    def _build_stub_infer(self, model_path: Path, runtime) :
        def _not_implemented(frame: np.ndarray):
            raise NotImplementedError(
                "Rilevato runtime Hailo ma l'adapter specifico del modello non e' ancora "
                f"implementato per {model_path.name}. Completa questo metodo sul Raspberry Pi."
            )

        return _not_implemented

    def _convert_predictions(self, predictions, frame_width: int, frame_height: int) -> list[Detection]:
        detections: list[Detection] = []
        for item in predictions or []:
            label = self._prediction_label(item)
            if label != self.target_label:
                continue

            score = float(item.get("score", 0.0))
            if score < self.score_threshold:
                continue

            x, y, w, h = self._prediction_bbox(item, frame_width, frame_height)
            if w <= 0 or h <= 0:
                continue

            detections.append(Detection(x=x, y=y, w=w, h=h, score=score, label=label))
        return detections

    def _prediction_label(self, item: dict) -> str:
        if "label" in item:
            return str(item["label"])
        class_id = item.get("class_id")
        if class_id is None:
            return self.target_label
        return self._labels.get(int(class_id), str(class_id))

    def _prediction_bbox(self, item: dict, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        bbox = item.get("bbox", {})
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        # Accept normalized coordinates as well as pixel coordinates.
        if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0:
            x1 *= frame_width
            x2 *= frame_width
        if 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
            y1 *= frame_height
            y2 *= frame_height

        x = max(0, int(x1))
        y = max(0, int(y1))
        w = max(0, int(x2 - x1))
        h = max(0, int(y2 - y1))
        return x, y, w, h

    def _load_labels(self, labels_path: Path | None) -> dict[int, str]:
        if labels_path is None or not labels_path.exists():
            return {}
        labels: dict[int, str] = {}
        for index, raw_line in enumerate(labels_path.read_text(encoding="utf-8").splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                labels[int(key.strip())] = value.strip()
            else:
                labels[index] = line
        return labels

    @property
    def is_ready(self) -> bool:
        return self._infer is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error
