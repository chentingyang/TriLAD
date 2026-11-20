"""Post-process VLM ranges: keep high-contrast cores + optional pad.

MSL champion: contrast / level-shift / sharp gates only (no peak/core crop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class PreciseHitPostConfig:
    contrast_zthr: float = 2.0
    min_level_shift: float = 0.25
    min_sharp: float = 0.005
    max_len: int = 600
    pad: int = 0
    min_len: int = 16
    max_ranges: int = 2


# Tuned offline on champion preds. Keep in sync with scripts.champion.MSL_CHAMPION_PRECISE_HIT_POST.
DEFAULT_PRECISE_HIT_POST = PreciseHitPostConfig(
    contrast_zthr=2.0,
    min_level_shift=0.25,
    min_sharp=0.01,
    max_len=800,
    pad=16,
    min_len=16,
    max_ranges=2,
)


def _range_pair(r: Any) -> tuple[int, int]:
    if isinstance(r, dict):
        return int(r["start"]), int(r["end"])
    return int(r[0]), int(r[1])


def contrast_score(x: np.ndarray, a: int, b: int) -> float:
    a, b = int(a), int(b)
    if b < a or a < 0 or b >= len(x):
        return 0.0
    seg = x[a : b + 1]
    mask = np.ones(len(x), dtype=bool)
    mask[a : b + 1] = False
    if not mask.any():
        return 0.0
    base = float(np.median(x[mask]))
    mad = float(np.median(np.abs(x[mask] - base))) + 1e-6
    return float(np.mean(np.abs(seg - base) / mad))


def level_shift(x: np.ndarray, a: int, b: int) -> float:
    seg = x[a : b + 1]
    mask = np.ones(len(x), dtype=bool)
    mask[a : b + 1] = False
    if not mask.any():
        return 0.0
    return abs(float(np.median(seg) - np.median(x[mask])))


def sharp_score(x: np.ndarray, a: int, b: int) -> float:
    if b <= a:
        return 0.0
    return float(np.mean(np.abs(np.diff(x[a : b + 1]))))


def _merge(ranges: list[dict[str, int]]) -> list[dict[str, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda r: r["start"])
    out = [dict(ranges[0])]
    for r in ranges[1:]:
        if r["start"] <= out[-1]["end"] + 1:
            out[-1]["end"] = max(out[-1]["end"], r["end"])
        else:
            out.append(dict(r))
    return out


def refine_window_ranges(
    values_1d: Iterable[float] | np.ndarray,
    ranges: list[Any] | None,
    cfg: PreciseHitPostConfig | None = None,
) -> list[dict[str, int]]:
    cfg = cfg or DEFAULT_PRECISE_HIT_POST
    x = np.asarray(values_1d, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1) if x.size == x.shape[0] else x[:, 0]
    L = len(x)
    kept: list[tuple[float, dict[str, int]]] = []
    for r in ranges or []:
        a, b = _range_pair(r)
        a, b = max(0, a), min(L - 1, b)
        if b - a + 1 < cfg.min_len or b - a + 1 > cfg.max_len:
            continue
        score = contrast_score(x, a, b)
        shift = level_shift(x, a, b)
        sharp = sharp_score(x, a, b)
        if score < cfg.contrast_zthr or shift < cfg.min_level_shift or sharp < cfg.min_sharp:
            continue
        a2, b2 = max(0, a - cfg.pad), min(L - 1, b + cfg.pad)
        # rank by shift*sharp*log contrast — favor separable morphology
        rank = shift * (sharp + 1e-6) * np.log1p(score)
        kept.append((float(rank), {"start": int(a2), "end": int(b2)}))
    kept.sort(key=lambda t: t[0], reverse=True)
    top = [r for _, r in kept[: max(1, cfg.max_ranges)]]
    return _merge(top)
