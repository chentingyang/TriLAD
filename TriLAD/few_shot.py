from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from TriLAD.schemas import WindowSample

MAX_FEW_SHOT_RATIO = 0.2
FAST_STATS_DUAL_VARIANTS = frozenset({"lite_dual"})
DEFAULT_FEW_SHOT_FEATURE_ALIGN = "similarity_mad_subset"
_CHAMPION_POOL_STRATEGIES = frozenset({"diverse_segment", "one_per_segment", "random_per_segment"})


@dataclass
class FewShotConfig:
    enabled: bool = True
    ratio: float = 0.1
    per_window: int = 1
    strategy: str = "diverse_segment"
    sampler: str = "similarity"
    seed: int = 0
    pool_budget: str = "windows"
    pool_ratio_scope: str = "all_windows"
    dual_normal_strategy: str | None = None

    def capped_ratio(self) -> float:
        return max(0.0, min(float(self.ratio), MAX_FEW_SHOT_RATIO))

    def uses_point_budget(self) -> bool:
        return (self.pool_budget or "windows").strip().lower() == "points"

    def uses_typed_ratio_scope(self) -> bool:
        return (self.pool_ratio_scope or "all_windows").strip().lower() == "typed_windows"


@dataclass
class FewShotReference:
    sample: WindowSample
    anomaly_ranges_local: list[dict[str, int]]
    image_b64: str | None = None
    vision_feature_indices: list[int] | None = None
    example_type: str | None = None
    pairing_meta: dict[str, Any] | None = None

    @property
    def window_id(self) -> int:
        return self.sample.window_id

    def resolved_example_type(self, global_mode: str = "anomaly") -> str:
        if self.example_type:
            return self.example_type
        return global_mode


def labels_to_ranges(labels: list[int]) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    i = 0
    n = len(labels)
    while i < n:
        if int(labels[i]) == 1:
            j = i + 1
            while j < n and int(labels[j]) == 1:
                j += 1
            ranges.append({"start": i, "end": j - 1})
            i = j
        else:
            i += 1
    return ranges


def spans_from_binary_labels(labels: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(labels)
    while i < n:
        if int(labels[i]) == 1:
            j = i + 1
            while j < n and int(labels[j]) == 1:
                j += 1
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return spans


def spans_from_normal_binary_labels(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous GT-normal (label=0) point spans."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(labels)
    while i < n:
        if int(labels[i]) == 0:
            j = i + 1
            while j < n and int(labels[j]) == 0:
                j += 1
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return spans


def _pool_window_cap(
    typed_count: int,
    all_window_count: int,
    ratio: float,
    *,
    typed_scope: bool,
) -> int:
    if typed_scope:
        return max(1, math.ceil(typed_count * ratio))
    return max(1, int(all_window_count * ratio))


def window_has_anomaly(sample: WindowSample) -> bool:
    return any(int(x) == 1 for x in sample.gt_labels)


def _select_one_per_segment(
    windows: list[WindowSample],
    labels: np.ndarray,
    *,
    require_anomaly: bool,
    rng: random.Random,
) -> list[int]:
    spans = (
        spans_from_binary_labels(labels)
        if require_anomaly
        else spans_from_normal_binary_labels(labels)
    )

    def _ok(w: WindowSample) -> bool:
        return window_has_anomaly(w) if require_anomaly else not window_has_anomaly(w)

    selected: list[int] = []
    for lo, hi in spans:
        ids = [
            w.window_id
            for w in windows
            if w.end_idx - 1 >= lo and w.start_idx <= hi and _ok(w)
        ]
        if ids:
            selected.append(rng.choice(ids))
    return selected


def _select_random_per_segment_typed(
    windows: list[WindowSample],
    labels: np.ndarray,
    max_pool: int,
    *,
    require_anomaly: bool,
    rng: random.Random,
) -> list[int]:
    """Per GT span round-robin random pick; filter by anomaly or normal windows."""
    spans = (
        spans_from_binary_labels(labels)
        if require_anomaly
        else spans_from_normal_binary_labels(labels)
    )
    span_candidates: list[list[int]] = []

    def _ok(w: WindowSample) -> bool:
        return window_has_anomaly(w) if require_anomaly else not window_has_anomaly(w)

    for lo, hi in spans:
        ids = [
            w.window_id
            for w in windows
            if w.end_idx - 1 >= lo and w.start_idx <= hi and _ok(w)
        ]
        if ids:
            span_candidates.append(ids)
    if not span_candidates:
        fallback = [w.window_id for w in windows if _ok(w)]
        rng.shuffle(fallback)
        return fallback[:max_pool]
    selected: set[int] = set()
    changed = True
    while changed and len(selected) < max_pool:
        changed = False
        for ids in span_candidates:
            if len(selected) >= max_pool:
                break
            pick = rng.choice(ids)
            if pick not in selected:
                selected.add(pick)
                changed = True
    return sorted(selected)[:max_pool]


def _select_diverse_window_ids(
    candidate_ids: list[int],
    windows: list[WindowSample],
    max_pool: int,
    rng: random.Random,
) -> list[int]:
    if not candidate_ids:
        return []
    by_id = {w.window_id: w for w in windows}
    remaining = [wid for wid in candidate_ids if wid in by_id]
    rng.shuffle(remaining)
    if len(remaining) <= max_pool:
        return remaining[:max_pool]
    picked: list[int] = [remaining[0]]
    while len(picked) < max_pool and remaining:
        best_id = None
        best_score = -1.0
        for wid in remaining:
            if wid in picked:
                continue
            w = by_id[wid]
            q = _window_embedding(w)
            dists = [
                _cosine_similarity(q, _window_embedding(by_id[pid]))
                for pid in picked
                if pid in by_id
            ]
            score = min(dists) if dists else 1.0
            if score > best_score:
                best_score = score
                best_id = wid
        if best_id is None:
            break
        picked.append(best_id)
        remaining = [wid for wid in remaining if wid not in picked]
    return picked[:max_pool]


def _assert_anomaly_segment_coverage(
    windows: list[WindowSample], labels: np.ndarray, selected_ids: set[int]
) -> None:
    spans = spans_from_binary_labels(labels)
    by_id = {w.window_id: w for w in windows}
    for lo, hi in spans:
        covered = any(
            wid in selected_ids
            and wid in by_id
            and by_id[wid].end_idx - 1 >= lo
            and by_id[wid].start_idx <= hi
            for wid in selected_ids
        )
        if not covered:
            raise ValueError(
                f"Few-shot pool misses anomaly segment [{lo}, {hi}]"
            )


class FewShotPoolBuilder:
    def __init__(self, config: FewShotConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed + 2026)

    @staticmethod
    def _trim_ids_to_point_budget(
        selected_ids: list[int],
        windows: list[WindowSample],
        total_len: int,
        target_points: int,
    ) -> list[int]:
        by_id = {w.window_id: w for w in windows}
        covered = np.zeros(total_len, dtype=bool)
        out: list[int] = []
        for wid in selected_ids:
            w = by_id.get(wid)
            if w is None:
                continue
            out.append(wid)
            covered[w.start_idx : w.end_idx] = True
            if int(covered.sum()) >= target_points:
                break
        if not out and selected_ids:
            first = selected_ids[0]
            if first in by_id:
                out = [first]
        return out

    def _select_random_per_segment(
        self, windows: list[WindowSample], labels: np.ndarray, max_pool: int
    ) -> list[int]:
        return _select_random_per_segment_typed(
            windows,
            labels,
            max_pool,
            require_anomaly=True,
            rng=self._rng,
        )

    def _select_by_strategy(
        self,
        windows: list[WindowSample],
        labels: np.ndarray,
        max_pool: int,
        *,
        strategy: str,
    ) -> list[int]:
        if strategy == "random_per_segment":
            return self._select_random_per_segment(windows, labels, max_pool)
        if strategy == "one_per_segment":
            return _select_one_per_segment(
                windows, labels, require_anomaly=True, rng=self._rng
            )[:max_pool]
        if strategy == "diverse_segment":
            per_seg = _select_one_per_segment(
                windows, labels, require_anomaly=True, rng=self._rng
            )
            return _select_diverse_window_ids(per_seg, windows, max_pool, self._rng)
        raise ValueError(f"Unsupported few-shot pool strategy: {strategy!r}")

    def build(
        self, windows: list[WindowSample], labels: np.ndarray
    ) -> list[WindowSample]:
        if not self.config.enabled:
            return []
        ratio = self.config.capped_ratio()
        if ratio <= 0:
            return []
        if self.config.uses_point_budget():
            target_points = max(1, int(len(labels) * ratio))
            max_pool = len(windows)
        else:
            target_points = None
            anom_windows = [w for w in windows if window_has_anomaly(w)]
            max_pool = _pool_window_cap(
                len(anom_windows),
                len(windows),
                ratio,
                typed_scope=self.config.uses_typed_ratio_scope(),
            )
        anom_windows = [w for w in windows if window_has_anomaly(w)]
        if not anom_windows:
            return []
        strategy = (self.config.strategy or "diverse_segment").strip().lower()
        if strategy not in _CHAMPION_POOL_STRATEGIES:
            raise ValueError(f"Unsupported few-shot pool strategy: {strategy!r}")
        selected_ids = self._select_by_strategy(windows, labels, max_pool, strategy=strategy)
        if target_points is not None:
            selected_ids = self._trim_ids_to_point_budget(
                selected_ids, windows, len(labels), target_points
            )
        by_id = {w.window_id: w for w in windows}
        pool = [by_id[wid] for wid in selected_ids if wid in by_id]
        if pool:
            try:
                _assert_anomaly_segment_coverage(windows, labels, set(selected_ids))
            except ValueError as exc:
                # Ratio cap can be smaller than the GT segment count (SWaT w=1000).
                print(f"[WARN] {exc}; keep ratio-capped pool n={len(pool)}", flush=True)
        return pool


def _window_embedding(
    sample: WindowSample, feature_indices: list[int] | None = None
) -> np.ndarray:
    arr = np.asarray(sample.values, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(2, dtype=np.float32)
    if feature_indices is not None:
        idx = [i for i in feature_indices if 0 <= i < arr.shape[1]]
        if idx:
            arr = arr[:, idx]
    feat_mean = np.nanmean(arr, axis=0)
    feat_std = np.nanstd(arr, axis=0)
    anom_ratio = float(np.mean(np.asarray(sample.gt_labels, dtype=np.float32)))
    return np.concatenate(
        [feat_mean, feat_std, np.asarray([anom_ratio], dtype=np.float32)], axis=0
    ).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _rank_references_by_similarity(
    window: WindowSample,
    references: list[FewShotReference],
    *,
    feature_indices: list[int] | None,
    top_k: int,
) -> list[FewShotReference]:
    q = _window_embedding(window, feature_indices)
    scored = [
        (idx, _cosine_similarity(q, _window_embedding(ref.sample, feature_indices)))
        for idx, ref in enumerate(references)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [references[idx] for idx, _ in scored[:top_k]]


class DualExampleFewShotSampler:
    """Dual few-shot: one NORMAL + one ANOMALY reference per target window (independent picks)."""

    def __init__(
        self,
        anomaly_refs: list[FewShotReference],
        normal_refs: list[FewShotReference],
        *,
        select_for_window=None,
    ) -> None:
        self.anomaly_refs = anomaly_refs
        self.normal_refs = normal_refs
        self._select_for_window = select_for_window

    def _subset_for(self, window: WindowSample) -> list[int] | None:
        if self._select_for_window is None:
            return None
        return self._select_for_window(window).indices

    def _clone_with_type(self, ref: FewShotReference, example_type: str) -> FewShotReference:
        return FewShotReference(
            sample=ref.sample,
            anomaly_ranges_local=ref.anomaly_ranges_local,
            image_b64=ref.image_b64,
            vision_feature_indices=ref.vision_feature_indices,
            example_type=example_type,
        )

    def sample_for_window(self, window: WindowSample) -> list[FewShotReference]:
        subset = self._subset_for(window)
        out: list[FewShotReference] = []
        anom = _rank_references_by_similarity(
            window, self.anomaly_refs, feature_indices=subset, top_k=1
        )
        if anom:
            out.append(self._clone_with_type(anom[0], "anomaly"))
        if self.normal_refs:
            ranked = _rank_references_by_similarity(
                window, self.normal_refs, feature_indices=subset, top_k=1
            )
            if ranked:
                out.append(self._clone_with_type(ranked[0], "normal"))
        return out


def build_eval_mask(
    total_len: int,
    windows: list[WindowSample],
    reference_window_ids: set[int],
) -> np.ndarray:
    mask = np.ones(total_len, dtype=bool)
    if not reference_window_ids:
        return mask
    for win in windows:
        if win.window_id in reference_window_ids:
            mask[win.start_idx : win.end_idx] = False
    return mask


def build_normal_reference_pool(
    windows: list[WindowSample],
    config: FewShotConfig,
    *,
    labels: np.ndarray | None = None,
    labels_len: int | None = None,
    diverse_from_all_normal: bool = False,
) -> list[WindowSample]:
    """GT-normal reference pool (typed or global window cap; champion segment strategies).

    diverse_from_all_normal only changes diverse_segment: pick a diverse subset of
    every normal window up to the ratio cap. A fully normal training series is one
    span, so the per-segment candidate list would otherwise stay at a single window.
    """
    if not config.enabled:
        return []
    ratio = config.capped_ratio()
    if ratio <= 0:
        return []
    normal_wins = [w for w in windows if not window_has_anomaly(w)]
    if not normal_wins:
        return []
    rng = random.Random(config.seed + 3031)
    if config.uses_point_budget():
        total_len = labels_len if labels_len is not None else max(
            (w.end_idx for w in windows), default=0
        )
        target_points = max(1, int(total_len * ratio))
        shuffled = list(normal_wins)
        rng.shuffle(shuffled)
        selected_ids = FewShotPoolBuilder._trim_ids_to_point_budget(
            [w.window_id for w in shuffled], windows, total_len, target_points
        )
        by_id = {w.window_id: w for w in windows}
        return [by_id[wid] for wid in selected_ids if wid in by_id]
    max_pool = _pool_window_cap(
        len(normal_wins),
        len(windows),
        ratio,
        typed_scope=config.uses_typed_ratio_scope(),
    )
    strategy = (
        config.dual_normal_strategy or config.strategy or "diverse_segment"
    ).strip().lower()
    if strategy not in _CHAMPION_POOL_STRATEGIES:
        raise ValueError(f"Unsupported normal pool strategy: {strategy!r}")
    if labels is None:
        raise ValueError("Normal reference pool requires GT labels for champion strategies.")
    if strategy == "one_per_segment":
        selected_ids = _select_one_per_segment(
            windows, labels, require_anomaly=False, rng=rng
        )
    elif strategy == "diverse_segment":
        if diverse_from_all_normal:
            candidates = [w.window_id for w in normal_wins]
        else:
            candidates = _select_one_per_segment(
                windows, labels, require_anomaly=False, rng=rng
            )
        selected_ids = _select_diverse_window_ids(candidates, windows, max_pool, rng)
    else:
        selected_ids = _select_random_per_segment_typed(
            windows, labels, max_pool, require_anomaly=False, rng=rng
        )
    by_id = {w.window_id: w for w in windows}
    return [by_id[wid] for wid in selected_ids if wid in by_id][:max_pool]


def _changepoint_count(y: np.ndarray) -> int:
    d = np.diff(np.asarray(y, dtype=np.float64))
    if d.size == 0:
        return 0
    thr = 2.0 * float(np.nanstd(d))
    return int(np.sum(np.abs(d) > thr))


def window_mad_stats_summary(
    sample: WindowSample,
    feature_indices: list[int],
    *,
    compact: bool = False,
    max_features: int | None = None,
) -> dict[str, Any]:
    arr = np.asarray(sample.values, dtype=np.float64)
    feats = feature_indices[: max_features or len(feature_indices)]
    rows: list[dict[str, Any]] = []
    for fi in feats:
        if fi < 0 or fi >= arr.shape[1]:
            continue
        col = arr[:, fi]
        med = float(np.median(col))
        mad = float(np.median(np.abs(col - med)))
        rows.append(
            {
                "name": sample.feature_names[fi],
                "median": round(med, 4),
                "mad": round(mad, 4),
                "changepoints": _changepoint_count(col),
            }
        )
    out: dict[str, Any] = {"window_id": sample.window_id, "features": rows}
    if compact:
        out["compact"] = True
    return out


def references_to_prompt_examples(
    references: list[FewShotReference],
    *,
    example_mode: str = "dual",
    reference_interval_mode: str = "pseudo",
    dual_variant: str = "lite_dual",
) -> list[dict[str, Any]]:
    del dual_variant
    # lite_dual already injects compact stats + image_url. Dumping full-channel
    # values and PNG base64 into this JSON bills them as text and blows the
    # 262k context (SWaT w=1000 ~340k). Keep metadata only; images stay on the
    # FewShotReference object for the multimodal image_url path.
    examples: list[dict[str, Any]] = []
    for ref in references:
        ex_type = ref.resolved_example_type(example_mode)
        payload: dict[str, Any] = {
            "window_id": ref.sample.window_id,
            "example_type": ex_type,
            "feature_names": ref.sample.feature_names,
            "value_note": "numeric series omitted; use reference image and/or compact stats",
        }
        if reference_interval_mode == "pseudo" and ex_type != "normal":
            payload["anomaly_ranges_local"] = ref.anomaly_ranges_local
        if ref.vision_feature_indices is not None:
            payload["vision_feature_indices"] = ref.vision_feature_indices
        examples.append(payload)
    return examples
