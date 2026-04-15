from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    label: str = "person"

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...
