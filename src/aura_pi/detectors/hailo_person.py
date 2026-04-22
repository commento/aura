from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np

from .base import Detection


class HailoPersonDetector:
    """
    Minimal Hailo adapter kept intentionally small until a verified official
    decoder is available for the exact runtime/model combination on the Pi.
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
        self._resources: list[object] = []
        self._init_runtime()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._infer is None:
            raise RuntimeError(
                "Hailo runtime non inizializzato. Installa lo stack Hailo sul Raspberry Pi "
                "e configura detector.model_path con un modello compatibile."
            )

        raw_predictions = self._infer(frame)
        detections = self._convert_predictions(raw_predictions, frame.shape[1], frame.shape[0])
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: self.max_detections]

    def _init_runtime(self) -> None:
        if self.model_path is None:
            self._init_error = "model_path mancante"
            return

        runtime = self._import_runtime()
        if runtime is None:
            self._runtime_name = "missing_hailo_runtime"
            self._init_error = (
                "runtime Hailo non disponibile nell'ambiente Python corrente. "
                "Sul Raspberry spesso i moduli sono installati a livello di sistema: "
                "esegui senza venv oppure ricrea la venv con --system-site-packages."
            )
            return

        self._infer = self._build_runtime_infer(model_path=self.model_path, runtime=runtime)
        if self._infer is None and self._init_error is None:
            self._init_error = (
                f"runtime {self._runtime_name} rilevato ma adapter modello non verificato "
                f"per {self.model_path.name}"
            )

    def _import_runtime(self):
        for module_name in ("hailo_platform", "hailo", "gsthailo"):
            try:
                runtime = importlib.import_module(module_name)
            except Exception:
                continue
            self._runtime_name = module_name
            return runtime
        return None

    def _build_runtime_infer(self, model_path: Path, runtime):
        if self._runtime_name != "hailo_platform":
            return None

        try:
            hef = runtime.HEF(str(model_path))
            vdevice = runtime.VDevice()
            configure_params = runtime.ConfigureParams.create_from_hef(
                hef,
                interface=runtime.HailoStreamInterface.PCIe,
            )
            network_group = vdevice.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()

            input_params_factory = getattr(runtime.InputVStreamParams, "make_from_network_group", None)
            if input_params_factory is None:
                input_params_factory = runtime.InputVStreamParams.make
            output_params_factory = getattr(runtime.OutputVStreamParams, "make_from_network_group", None)
            if output_params_factory is None:
                output_params_factory = runtime.OutputVStreamParams.make

            input_vstream_params = input_params_factory(
                network_group,
                quantized=False,
                format_type=runtime.FormatType.UINT8,
            )
            output_vstream_params = output_params_factory(
                network_group,
                quantized=False,
                format_type=runtime.FormatType.FLOAT32,
            )

            input_info = hef.get_input_vstream_infos()[0]
            output_infos = list(hef.get_output_vstream_infos())

            activation = network_group.activate(network_group_params)
            activation.__enter__()
            try:
                infer_pipeline = runtime.InferVStreams(
                    network_group,
                    input_vstream_params,
                    output_vstream_params,
                    tf_nms_format=True,
                )
            except TypeError:
                infer_pipeline = runtime.InferVStreams(
                    network_group,
                    input_vstream_params,
                    output_vstream_params,
                )
            infer_pipeline.__enter__()

            self._resources.extend([infer_pipeline, activation, vdevice])
            input_name = getattr(input_info, "name", next(iter(input_vstream_params.keys())))
            input_h, input_w, input_c = self._resolve_input_shape(getattr(input_info, "shape", ()))

            def _infer(frame: np.ndarray):
                batch = self._preprocess_frame(frame, input_w, input_h, input_c)
                outputs = infer_pipeline.infer({input_name: batch})
                return self._parse_hailo_outputs(outputs, output_infos)

            return _infer
        except Exception as exc:
            self.close()
            self._init_error = f"inizializzazione Hailo fallita: {exc}"
            return None

    def _resolve_input_shape(self, shape) -> tuple[int, int, int]:
        dims = tuple(int(value) for value in shape)
        if len(dims) >= 3:
            return dims[-3], dims[-2], dims[-1]
        return 640, 640, 3

    def _preprocess_frame(self, frame: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if channels == 3:
            if resized.ndim == 2:
                resized = np.repeat(resized[..., np.newaxis], 3, axis=2)
            elif resized.ndim == 3 and resized.shape[2] == 1:
                resized = np.repeat(resized, 3, axis=2)
            elif resized.ndim == 3 and resized.shape[2] > 3:
                resized = resized[:, :, :3]
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        elif channels == 1:
            if resized.ndim == 3 and resized.shape[2] >= 3:
                resized = cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2GRAY)
            elif resized.ndim == 3 and resized.shape[2] == 1:
                resized = resized[:, :, 0]
            resized = resized[..., np.newaxis]
        return np.expand_dims(np.ascontiguousarray(resized), axis=0).astype(np.uint8)

    def _parse_hailo_outputs(self, outputs, output_infos) -> list[dict]:
        parsed: list[dict] = []
        if isinstance(outputs, dict):
            items = list(outputs.items())
        else:
            items = [("output", outputs)]

        for output_index, (name, value) in enumerate(items):
            info = output_infos[output_index] if output_index < len(output_infos) else None
            parsed.extend(self._parse_output_tensor(value, info=info, fallback_name=name))
        return parsed

    def _parse_output_tensor(self, value, info=None, fallback_name: str = "output") -> list[dict]:
        if isinstance(value, list):
            return self._parse_nms_list(value)

        array = np.asarray(value)
        if array.size == 0:
            return []
        if array.dtype == object:
            detections: list[dict] = []
            for item in array.tolist():
                detections.extend(self._parse_output_tensor(item, info=info, fallback_name=fallback_name))
            return detections
        if array.ndim >= 1 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3:
            return self._parse_nms_list([array[class_id] for class_id in range(array.shape[0])])
        if array.ndim == 2 and array.shape[-1] >= 5:
            return self._parse_detection_rows(array, info=info, fallback_name=fallback_name)
        return []

    def _parse_nms_list(self, per_class_outputs: list) -> list[dict]:
        detections: list[dict] = []
        for class_id, class_output in enumerate(per_class_outputs):
            rows = np.asarray(class_output)
            if rows.size == 0:
                continue
            if rows.ndim == 1:
                rows = np.expand_dims(rows, axis=0)
            if rows.ndim > 2:
                rows = rows.reshape(-1, rows.shape[-1])
            label = self._output_label(class_id=class_id)
            detections.extend(self._parse_detection_rows(rows, label=label))
        return detections

    def _parse_detection_rows(self, rows: np.ndarray, label: str | None = None, info=None, fallback_name: str = "output") -> list[dict]:
        detections: list[dict] = []
        for row in rows:
            if len(row) < 5:
                continue
            x1, y1, x2, y2, score = self._decode_bbox_row(row)
            class_id = self._row_class_id(row)
            resolved_label = label or self._output_label(info=info, fallback_name=fallback_name, class_id=class_id)
            detections.append(
                {
                    "label": resolved_label,
                    "class_id": class_id,
                    "score": float(score),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
        return detections

    def _decode_bbox_row(self, row) -> tuple[float, float, float, float, float]:
        values = [float(item) for item in row[:6]]
        if len(values) >= 6:
            y1, x1, y2, x2, score, _ = values[:6]
            return x1, y1, x2, y2, score
        y1, x1, y2, x2, score = values[:5]
        return x1, y1, x2, y2, score

    def _output_label(self, info=None, fallback_name: str = "output", class_id: int = 0) -> str:
        if class_id in self._labels:
            return self._labels[class_id]
        if info is not None and hasattr(info, "name"):
            return str(info.name)
        return fallback_name

    def _row_class_id(self, row) -> int:
        if len(row) < 6:
            return 0
        candidate = float(row[5])
        rounded = int(round(candidate))
        if abs(candidate - rounded) < 0.05 and rounded >= 0:
            return rounded
        return 0

    def _convert_predictions(self, predictions, frame_width: int, frame_height: int) -> list[Detection]:
        detections: list[Detection] = []
        for item in predictions or []:
            label = self._prediction_label(item)
            if not self._matches_target(item, label):
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

    def _is_target_label(self, label: str) -> bool:
        normalized_target = self.target_label.strip().lower()
        normalized_label = label.strip().lower()
        if normalized_target.isdigit():
            return normalized_label == normalized_target
        if normalized_label == normalized_target:
            return True
        return normalized_target in normalized_label or normalized_label in normalized_target

    def _matches_target(self, item: dict, label: str) -> bool:
        if self._is_target_label(label):
            return True
        class_id = item.get("class_id")
        if class_id is None:
            return False
        normalized_target = self.target_label.strip().lower()
        if normalized_target.isdigit():
            return int(class_id) == int(normalized_target)
        return normalized_target == "person" and int(class_id) == 0

    def _prediction_bbox(self, item: dict, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        bbox = item.get("bbox", {})
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

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

    def close(self) -> None:
        while self._resources:
            resource = self._resources.pop()
            close = getattr(resource, "__exit__", None)
            if close is None:
                close = getattr(resource, "release", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
                continue
            try:
                close(None, None, None)
            except Exception:
                pass

    def __del__(self):
        self.close()
