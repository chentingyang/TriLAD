"""Per-window MSL hint for the champion univar branch (var_0-only)."""

from __future__ import annotations

from typing import Any


_DECISION_HINTS_UNIVAR: tuple[str, ...] = (
    "Univariate MSL primary telemetry only (usually var_0); command/one-hot dims are not shown.",
    "Empty anomaly_ranges_local is valid when the curve stays near its in-window baseline.",
    "Flag sustained level/shape shifts (≥20 consecutive indices); reject isolated spikes <15 indices.",
    "Large windows are often only partially anomalous: output the tight local sub-interval, never the whole window by default.",
    "Require lite_dual anomaly-pattern match on this single curve; stats-only wobble without morph similarity → [].",
    "At most 2 intervals per window; anomaly_score=0.0 when []; score≥0.7 only with clear sustained morph match.",
)


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


def build_window_msl_hint(
    features: list[str],
    *,
    mad_scores: list[float] | None = None,
    max_coupling_groups: int | None = None,
    include_cross_subsystem: bool | None = None,
    compact: bool = False,
    version: str = "univar",
) -> dict[str, Any]:
    """
    Champion MSL hint: univariate primary telemetry only.

    ``version`` is accepted for call-site compatibility but always resolves to univar.
    Coupling / multi-channel hint variants have been removed.
    """
    del mad_scores, max_coupling_groups, include_cross_subsystem, version  # unused (champion-only)
    names = _normalize_features(features)
    channel_rows: list[Any] = []
    for name in names:
        if compact:
            channel_rows.append(name)
        else:
            channel_rows.append(
                {
                    "name": name,
                    "role": "primary_telemetry",
                    "note": "Univariate target series after dropping command/one-hot dims.",
                }
            )
    decision_hints = list(_DECISION_HINTS_UNIVAR)
    return {
        "hint_version": "univar",
        "dataset": "MSL",
        "mode": "univariate_primary",
        "channels": channel_rows,
        "coupling_checks": [],
        "decision_hints": decision_hints[:2] if compact else decision_hints,
    }
