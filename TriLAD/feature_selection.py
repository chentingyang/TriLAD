from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureSelectionResult:
    indices: list[int]
    scores: list[float]
    method: str
    k: int


def _safe_2d(window_data: np.ndarray) -> np.ndarray:
    arr = np.asarray(window_data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"window_data must be 2D, got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("window_data needs at least 2 time steps")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def score_variance(window_data: np.ndarray) -> np.ndarray:
    return np.var(window_data, axis=0)


def score_mad(window_data: np.ndarray) -> np.ndarray:
    med = np.median(window_data, axis=0)
    return np.median(np.abs(window_data - med), axis=0)


def score_derivative_energy(window_data: np.ndarray) -> np.ndarray:
    diff = np.diff(window_data, axis=0)
    return np.sum(diff * diff, axis=0)


def _top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    n = scores.shape[0]
    if k <= 0 or k >= n:
        return list(range(n))
    order = np.argsort(-scores, kind="stable")
    return order[:k].tolist()


def select_correlation_clusters(window_data: np.ndarray, k: int) -> list[int]:
    arr = _safe_2d(window_data)
    n = arr.shape[1]
    if k <= 0 or k >= n:
        return list(range(n))

    var = score_variance(arr)
    corr = np.abs(np.corrcoef(arr.T))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    threshold = 0.85

    unused = set(range(n))
    clusters: list[list[int]] = []
    while unused:
        seed = max(unused, key=lambda i: var[i])
        cluster = [seed]
        unused.remove(seed)
        to_remove: list[int] = []
        for j in list(unused):
            if corr[seed, j] >= threshold:
                cluster.append(j)
                to_remove.append(j)
        for j in to_remove:
            unused.remove(j)
        clusters.append(cluster)

    clusters.sort(key=lambda cl: max(var[i] for i in cl), reverse=True)
    picked: list[int] = []
    for cl in clusters:
        rep = max(cl, key=lambda i: var[i])
        picked.append(rep)
        if len(picked) >= k:
            break

    if len(picked) < k:
        rest = _top_k_indices(var, k)
        for i in rest:
            if i not in picked:
                picked.append(i)
            if len(picked) >= k:
                break

    return picked[:k]


def compute_grid_shape(num_panels: int, max_cols: int = 7) -> tuple[int, int]:
    if num_panels <= 0:
        return (1, 1)
    if num_panels <= max_cols * max_cols:
        cols = int(math.ceil(math.sqrt(num_panels)))
        rows = int(math.ceil(num_panels / cols))
        return (rows, cols)
    cols = max_cols
    rows = int(math.ceil(num_panels / cols))
    return (rows, cols)


def select_features(
    window_data: np.ndarray,
    *,
    method: str = "none",
    k: int = 0,
    seed: int = 0,
    baseline_mean: np.ndarray | None = None,
    baseline_std: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> FeatureSelectionResult:
    del baseline_mean, baseline_std, feature_names
    arr = _safe_2d(window_data)
    n = arr.shape[1]
    mode = (method or "none").strip().lower()
    effective_k = n if k <= 0 else min(k, n)

    if mode in {"none", "all"}:
        idx = list(range(n))
        return FeatureSelectionResult(
            indices=idx,
            scores=[1.0] * len(idx),
            method=mode,
            k=len(idx),
        )

    if mode == "first_k":
        idx = list(range(effective_k))
        return FeatureSelectionResult(
            indices=idx,
            scores=[1.0] * len(idx),
            method=mode,
            k=len(idx),
        )

    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, effective_k, replace=False).tolist()
        idx.sort()
        return FeatureSelectionResult(
            indices=idx,
            scores=[0.0] * len(idx),
            method=mode,
            k=len(idx),
        )

    if mode == "cluster":
        idx = select_correlation_clusters(arr, effective_k)
        sc = score_variance(arr)
        return FeatureSelectionResult(
            indices=idx,
            scores=[float(sc[i]) for i in idx],
            method=mode,
            k=len(idx),
        )

    score_fns = {
        "variance": score_variance,
        "mad": score_mad,
        "derivative": score_derivative_energy,
    }
    if mode in score_fns:
        sc = score_fns[mode](arr)
        idx = _top_k_indices(sc, effective_k)
        return FeatureSelectionResult(
            indices=idx,
            scores=[float(sc[i]) for i in idx],
            method=mode,
            k=len(idx),
        )

    raise ValueError(
        f"Unknown feature_select_mode={method!r}. Choose from: none, all, first_k, "
        "random, variance, mad, derivative, cluster"
    )


def subset_window_values(
    values: list[list[float]],
    feature_names: list[str],
    indices: list[int],
) -> tuple[list[list[float]], list[str]]:
    idx = sorted(set(indices))
    sub_names = [feature_names[i] for i in idx]
    sub_values = [[row[i] for i in idx] for row in values]
    return sub_values, sub_names
