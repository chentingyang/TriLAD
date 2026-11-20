"""Per-window SWaT coupling hints bound to MAD-selected features."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from hints.swat_tag_kb import (
    CROSS_STAGE_NOTES,
    KNOWN_SWAT_TAGS,
    STAGE_COUPLING_GROUPS,
    tag_meta,
)

HintVersion = Literal["v1", "v2", "v2.1"]

_DECISION_HINTS_V1: tuple[str, ...] = (
    "Use only tags in input.features.",
    "Prefer intervals with >=2 tags from same coupling_checks group.",
    "Keep lite_dual baseline+pattern; no index copy.",
)

_DECISION_HINTS_V2: tuple[str, ...] = (
    "Coupling hints are soft; keep lite_dual intervals when chart shows sustained deviation.",
    "Do not return empty ranges only because a coupling group is incomplete.",
    "Use tag names from input.features only.",
)

_DECISION_HINTS_V2_1: tuple[str, ...] = (
    *_DECISION_HINTS_V2,
    "Trim interval endpoints to the tightest sustained-deviation core; avoid fuzzy ramp margins.",
    "Prefer one strong coupling group over many weak partial groups.",
)


def _resolve_version(version: str) -> HintVersion:
    v = str(version).strip().lower()
    if v in {"v2.1", "v21", "2.1"}:
        return "v2.1"
    if v == "v2":
        return "v2"
    return "v1"


def _normalize_features(features: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in features or []:
        name = str(raw).strip().upper()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _mad_rank_map(features: list[str], mad_scores: list[float] | None) -> dict[str, int]:
    if not mad_scores or len(mad_scores) != len(features):
        return {name: i + 1 for i, name in enumerate(features)}
    order = sorted(range(len(features)), key=lambda i: float(mad_scores[i]), reverse=True)
    return {features[i]: rank + 1 for rank, i in enumerate(order)}


def _score_group(visible: list[str], mad_rank: dict[str, int]) -> float:
    if not visible:
        return 0.0
    return float(sum(1.0 / max(1, mad_rank.get(n, 9)) for n in visible))


def build_window_swat_hint(
    features: list[str],
    *,
    mad_scores: list[float] | None = None,
    max_coupling_groups: int | None = None,
    include_cross_stage: bool | None = None,
    include_mad_rank_top: int = 0,
    compact: bool = False,
    version: HintVersion = "v1",
) -> dict[str, Any]:
    """
    Build a short JSON-serializable hint for the current window's MAD features.

    All coupling checks reference only tags present in `features`.
  v2: drop weak_singletons/cross_stage; add recall guardrails for lite_dual.
    """
    ver = _resolve_version(version)
    if max_coupling_groups is None:
        max_coupling_groups = 1 if ver == "v2.1" else 2
    if include_cross_stage is None:
        include_cross_stage = ver == "v1"

    names = _normalize_features(features)
    mad_rank = _mad_rank_map(names, mad_scores)

    tag_rows: list[dict[str, Any]] = []
    for name in names:
        meta = tag_meta(name)
        if compact:
            tag_rows.append(meta["name"])
        else:
            row = {"name": meta["name"], "stage": meta["stage"], "role": meta["role"]}
            if name not in KNOWN_SWAT_TAGS:
                row["known_tag"] = False
            tag_rows.append(row)

    stage_counts = Counter(
        s
        for n in tag_rows
        for s in [tag_meta(n if isinstance(n, str) else n["name"])["stage"]]
        if s != "unknown"
    )
    active_stages = [s for s in ("P1", "P2", "P3", "P4", "P5", "P6") if stage_counts.get(s, 0) > 0]

    selected_set = set(names)
    coupling_candidates: list[dict[str, Any]] = []
    covered: set[str] = set()

    for stage in active_stages:
        for group in STAGE_COUPLING_GROUPS.get(stage, []):
            members = list(group.get("members") or [])
            visible = [m for m in members if m in selected_set]
            if len(visible) < 2:
                continue
            coupling_candidates.append(
                {
                    "stage": stage,
                    "group_id": str(group.get("group_id", "")),
                    "features": visible,
                    "check": str(group.get("check", "")),
                    "_score": _score_group(visible, mad_rank) + 0.1 * len(visible),
                }
            )
            covered.update(visible)

    coupling_candidates.sort(key=lambda g: float(g.get("_score") or 0.0), reverse=True)
    cap = max(0, int(max_coupling_groups))
    coupling_checks = []
    for g in coupling_candidates[:cap]:
        row = {k: v for k, v in g.items() if k != "_score"}
        if compact:
            row = {
                "features": row.get("features"),
                "check": row.get("check"),
            }
        coupling_checks.append(row)

    decision_hints = list(
        _DECISION_HINTS_V2_1
        if ver == "v2.1"
        else (_DECISION_HINTS_V2 if ver == "v2" else _DECISION_HINTS_V1)
    )
    if include_mad_rank_top > 0:
        top_n = max(1, int(include_mad_rank_top))
        ranked = sorted(names, key=lambda n: mad_rank.get(n, 99))
        decision_hints.append(
            "Prefer intervals covering higher MAD-rank tags when chart shows sustained deviation."
        )
    else:
        ranked = []

    hint: dict[str, Any] = {
        "hint_version": ver,
        "tags": tag_rows,
        "coupling_checks": coupling_checks,
        "decision_hints": decision_hints[:2] if compact else decision_hints,
    }
    if not compact:
        hint["active_stages"] = active_stages
    if include_mad_rank_top > 0:
        hint["mad_rank_top"] = [
            {"tag": n, "rank": mad_rank.get(n, 99)} for n in ranked[:top_n]
        ]

    if ver == "v1":
        weak_singletons: list[dict[str, str]] = []
        for name in names:
            if name in covered:
                continue
            meta = tag_meta(name)
            weak_singletons.append(
                {
                    "name": name,
                    "stage": meta["stage"],
                    "note": "Single tag; weaker unless sustained on chart.",
                }
            )
        if weak_singletons:
            hint["weak_singletons"] = weak_singletons[:3]

        if include_cross_stage and len(active_stages) >= 2:
            active_set = set(active_stages)
            for edge in CROSS_STAGE_NOTES:
                a, b = edge["stages"]
                if a in active_set and b in active_set:
                    hint["cross_stage_note"] = str(edge["note"])
                    break

    return hint
