"""Per-window SMD hint for server-machine multivariate telemetry (MAD subset)."""
from __future__ import annotations

from typing import Any


_DECISION_HINTS_V1: tuple[str, ...] = (
    "SMD = server-machine KPI telemetry (CPU/mem/disk/net-like channels); anomalies are service faults, not ICS attacks.",
    "Empty anomaly_ranges_local is valid when MAD channels stay near the in-window baseline.",
    "Prefer sustained co-moving shifts across ≥2 MAD channels; veto isolated single-channel micro-spikes.",
    "Segment lengths vary widely (few points → hundreds): keep short cores when the chart shows a brief burst; "
    "do not expand to the whole window.",
    "Require lite_dual anomaly-pattern morph match; stats-only wobble without pattern similarity → [].",
    "At most 2–3 intervals per window; anomaly_score=0.0 when []; score≥0.7 only with clear sustained morph match.",
)

_DECISION_HINTS_V2: tuple[str, ...] = (
    "SMD = server KPI faults; often ONE dominant MAD channel shifts (co-move optional, not mandatory).",
    "Empty only when no sustained deviant core; do not miss clear mid-window bursts (≥5–12 indices).",
    "Veto 1–3 point spikes and edge-to-edge whole-window dumps; keep tight cores.",
    "Morph-match the lite_dual anomaly few-shot; single-channel sustained shift is enough if pattern matches.",
    "At most 3 intervals; anomaly_score≥0.6 for clear cores.",
)


def _normalize_features(features: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in features or []:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def build_window_smd_hint(
    features: list[str],
    *,
    mad_scores: list[float] | None = None,
    compact: bool = False,
    hint_version: str = "v1",
) -> dict[str, Any]:
    """Soft per-window hint for SMD MAD feature subset."""
    names = _normalize_features(features)
    scores = list(mad_scores or [])
    channel_rows: list[Any] = []
    for i, name in enumerate(names):
        score = float(scores[i]) if i < len(scores) else None
        if compact:
            channel_rows.append(name)
        else:
            row: dict[str, Any] = {
                "name": name,
                "role": "mad_selected_kpi",
                "note": "Server KPI channel selected by MAD for this window.",
            }
            if score is not None:
                row["mad_score"] = round(score, 6)
            channel_rows.append(row)

    ver = (hint_version or "v1").strip().lower()
    # v2 / recall used by champion smd_hint_recall; smd_hint_v2 is a deprecated CLI alias.
    if ver in {"v2", "recall", "smd_hint_v2", "smd_hint_recall"}:
        hints = list(_DECISION_HINTS_V2)
        coupling = []
        if names:
            coupling.append(
                {
                    "group": "mad_primary_or_co_move",
                    "members": names[: min(4, len(names))],
                    "check": (
                        "Prefer ≥2 MAD KPIs shifting together; OR one top-MAD channel with sustained "
                        "level/slope shift (≥8–15 indices) matching anomaly few-shot."
                    ),
                }
            )
        mode = "multivariate_mad_kpi_recall"
        hv = "smd_v2"
    else:
        hints = list(_DECISION_HINTS_V1)
        coupling = []
        if len(names) >= 2:
            coupling.append(
                {
                    "group": "mad_co_move",
                    "members": names[: min(4, len(names))],
                    "check": "Prefer intervals where ≥2 listed MAD KPIs shift together for ≥15–20 indices.",
                }
            )
        mode = "multivariate_mad_kpi"
        hv = "smd_v1"

    return {
        "hint_version": hv,
        "dataset": "SMD",
        "mode": mode,
        "channels": channel_rows,
        "coupling_checks": coupling[:1] if compact else coupling,
        "decision_hints": hints[:3] if compact else hints,
    }
