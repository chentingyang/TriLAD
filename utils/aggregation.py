from __future__ import annotations

import numpy as np

from TriLAD.schemas import DetectionOutput
from TriLAD.parsing import merge_ranges


def _effective_vote_thresholds(covers: np.ndarray, vote_threshold: float, vote_mode: str) -> np.ndarray:
    """Per-point vote threshold."""
    if vote_mode == "adaptive":
        # SWaT stride=500/window=1000 => overlap=2 for most points.
        # Use majority (0.5) when only two windows overlap; stricter threshold when >=3.
        th = np.full(covers.shape, float(vote_threshold), dtype=np.float64)
        th[covers <= 2] = min(0.5, float(vote_threshold))
        return th
    return np.full(covers.shape, float(vote_threshold), dtype=np.float64)


def window_outputs_to_point_labels(
    outputs: list[DetectionOutput],
    total_len: int,
    vote_threshold: float = 0.5,
    *,
    vote_mode: str = "binary",
    score_gate_threshold: float = 0.0,
    merge_range_gap: int = 0,
) -> np.ndarray:
    """
    Merge window-level ranges into point labels.

    vote_mode:
      - binary: each covering window with ranges adds +1 vote (original behavior)
      - score_weighted: add anomaly_score per vote (still normalized by covers)
      - adaptive: use 0.5 threshold when covers<=2, else vote_threshold
    score_gate_threshold: skip a window's ranges when anomaly_score < threshold (0 disables)
    merge_range_gap: merge local ranges within a window before voting
    """
    if vote_threshold < 0 or vote_threshold > 1:
        raise ValueError("vote_threshold must be within [0, 1]")
    if vote_mode not in {"binary", "score_weighted", "adaptive"}:
        raise ValueError(f"Unsupported vote_mode: {vote_mode}")

    votes = np.zeros(total_len, dtype=np.float64)
    covers = np.zeros(total_len, dtype=np.int32)

    for out in outputs:
        window_len = max(0, out.end_idx - out.start_idx)
        score = max(0.0, min(1.0, float(out.anomaly_score)))
        if score_gate_threshold > 0.0 and score < score_gate_threshold:
            continue

        local_ranges = out.anomaly_ranges_local
        if merge_range_gap > 0 and local_ranges:
            local_ranges = merge_ranges(
                local_ranges,
                max_gap=int(merge_range_gap),
                window_length=window_len,
            )

        weight = score if vote_mode == "score_weighted" else 1.0
        covers[out.start_idx : out.end_idx] += 1
        for rg in local_ranges:
            g_start = max(out.start_idx, out.start_idx + int(rg["start"]))
            g_end = min(out.end_idx - 1, out.start_idx + int(rg["end"]))
            if g_end >= g_start:
                votes[g_start : g_end + 1] += weight

    ratio = np.divide(votes, np.maximum(covers, 1), dtype=np.float64)
    if vote_mode == "adaptive":
        th = _effective_vote_thresholds(covers, vote_threshold, vote_mode)
        return (ratio >= th).astype(np.int32)
    return (ratio >= vote_threshold).astype(np.int32)
