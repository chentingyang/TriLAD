"""Few-shot pool construction for the champion trunk (test_gt only)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from TriLAD.few_shot import FewShotConfig, FewShotPoolBuilder
from TriLAD.schemas import WindowSample

PoolSource = Literal["test_gt"]
ReferenceIntervalMode = Literal["pseudo"]
ExampleMode = Literal["anomaly", "normal", "dual"]


def build_few_shot_pool(
    config: FewShotConfig,
    *,
    pool_source: PoolSource,
    windows: list[WindowSample],
    labels: np.ndarray,
    values: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    normal_csv: str | None = None,
    normalize_mode: str = "none",
    normalize_fit_csv: str | None = None,
    window_size: int = 0,
    stride: int = 0,
    down_len: int = 1,
    **_unused,
) -> tuple[list[WindowSample], ExampleMode]:
    """Build the anomaly few-shot pool from oracle GT on the attack/test CSV."""
    del values, feature_names, normal_csv, normalize_mode, normalize_fit_csv
    del window_size, stride, down_len, _unused
    if not config.enabled:
        return [], "anomaly"
    if pool_source != "test_gt":
        raise ValueError(
            f"Unsupported few_shot_pool_source={pool_source!r}; "
            "champion trunk only supports test_gt"
        )
    pool = FewShotPoolBuilder(config).build(windows, labels)
    return pool, "anomaly"


def anomaly_ranges_for_reference(
    sample: WindowSample,
    *,
    pool_source: PoolSource,
    example_mode: ExampleMode,
    pseudo_labels: np.ndarray | None = None,
    reference_interval_mode: ReferenceIntervalMode | str = "pseudo",
) -> list[dict[str, int]]:
    """Return local anomaly ranges for a few-shot reference (GT / pseudo)."""
    from TriLAD.few_shot import labels_to_ranges

    del pool_source, pseudo_labels
    mode = (reference_interval_mode or "pseudo").strip().lower()
    if mode != "pseudo":
        raise ValueError(
            f"Unsupported reference_interval_mode={mode!r}; "
            "champion trunk only supports pseudo"
        )
    if example_mode == "normal":
        return []
    return labels_to_ranges(sample.gt_labels)
