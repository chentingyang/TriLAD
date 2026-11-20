from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WindowSample:
    window_id: int
    start_idx: int
    end_idx: int
    values: list[list[float]]
    feature_names: list[str]
    gt_labels: list[int]
    image_path: str | None = None


@dataclass
class DetectionOutput:
    window_id: int
    start_idx: int
    end_idx: int
    anomaly_ranges_local: list[dict[str, int]]
    anomaly_score: float
    rationale: str
    raw_response: str
    parsed_ok: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

