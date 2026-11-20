"""Eval window loading and prompt_max_intervals helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from utils.data_pipeline import downsample, load_csv, make_windows
from TriLAD.few_shot import labels_to_ranges
from TriLAD.schemas import WindowSample


def compute_auto_prompt_max_intervals_from_windows(windows: list[WindowSample]) -> int:
    """Max GT anomaly-interval count across windows (used when auto=True)."""
    if not windows:
        return 1
    return max((len(labels_to_ranges(w.gt_labels)) for w in windows), default=1)


def build_eval_windows(
    *,
    csv_path: str | Path,
    window_size: int = 1000,
    stride: int = 500,
    down_len: int = 1,
    max_windows: int = 0,
    normalize_mode: str = "none",
    normalize_fit_csv: str | Path | None = None,
) -> list[WindowSample]:
    """Load CSV and cut sliding windows (optional head truncate via max_windows)."""
    values, labels, feature_names = load_csv(
        csv_path,
        normalize_mode=normalize_mode,
        normalize_fit_csv=normalize_fit_csv,
    )
    values = downsample(values, down_len=down_len, is_label=False).astype(np.float32)
    labels = downsample(labels, down_len=down_len, is_label=True).astype(np.int32)
    windows = make_windows(
        values=values,
        labels=labels,
        feature_names=feature_names,
        window_size=window_size,
        stride=stride,
    )
    if max_windows > 0:
        return windows[:max_windows]
    return windows


def materialize_prompt_max_intervals(cfg: dict[str, Any]) -> dict[str, Any]:
    """If prompt_max_intervals_auto, resolve to an explicit integer cap."""
    out = deepcopy(cfg)
    if not out.get("prompt_max_intervals_auto", False):
        out["prompt_max_intervals_auto"] = False
        if "prompt_max_intervals" not in out:
            out["prompt_max_intervals"] = 5
        return out

    windows = build_eval_windows(
        csv_path=out["csv_path"],
        window_size=int(out.get("window_size", 1000)),
        stride=int(out.get("stride", 500)),
        down_len=int(out.get("down_len", 1)),
        max_windows=int(out.get("max_windows") or 0),
        normalize_mode=str(out.get("normalize_values") or "none"),
        normalize_fit_csv=out.get("normalize_fit_csv"),
    )
    computed = compute_auto_prompt_max_intervals_from_windows(windows)
    out["prompt_max_intervals"] = computed
    out["prompt_max_intervals_auto"] = False
    out["_materialized_prompt_max_intervals_from_auto"] = computed
    return out
