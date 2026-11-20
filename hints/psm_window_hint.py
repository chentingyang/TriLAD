"""Per-window PSM hint for eBay application-server multivariate metrics (MAD subset).

PSM champion trunk: cover_bal (psm_hint_cover_bal / psm_hint).
"""
from __future__ import annotations

from typing import Any


# Frozen PSM champion decision hints (cover_bal).
_DECISION_HINTS_COVER_BAL: tuple[str, ...] = (
    "PSM cover-balance: prioritize PA + Affi. Point may be slightly lower. "
    "Keep TRUE sustained faults as ONE full contiguous core (good for PA/Affi-R).",
    "Raise Affi-P softly: veto 1–8 pt spikes and fragmented shorts; prefer exactly 1 interval (max 2); "
    "do not invent weak mid-window wobble.",
    "Critical: if a clear sustained MAD core exists (co-move OR strong top-MAD morph match), "
    "you MUST report it with full local span — never empty a clear-fault window.",
    "anomaly_score≥0.70 for kept cores; empty only when the MAD panel is clean / no clear core.",
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


def build_window_psm_hint(
    features: list[str],
    *,
    mad_scores: list[float] | None = None,
    compact: bool = False,
    hint_version: str = "cover_bal",
) -> dict[str, Any]:
    """Soft per-window hint for PSM MAD feature subset (champion = cover_bal)."""
    names = _normalize_features(features)
    scores = list(mad_scores or [])
    ver = (hint_version or "cover_bal").strip().lower()
    # All PSM aliases resolve to champion cover_bal.
    _ = ver  # kept for API compatibility
    force_compact = bool(compact)

    channel_rows: list[Any] = []
    for i, name in enumerate(names):
        score = float(scores[i]) if i < len(scores) else None
        if force_compact:
            channel_rows.append(name)
        else:
            row: dict[str, Any] = {
                "name": name,
                "role": "mad_selected_app_metric",
                "note": "eBay app-server metric selected by MAD for this window.",
            }
            if score is not None:
                row["mad_score"] = round(score, 6)
            channel_rows.append(row)

    hints = list(_DECISION_HINTS_COVER_BAL)
    mode = "multivariate_mad_app_server_cover_bal"
    hv = "psm_cover_bal"
    check = (
        "Cover full clear sustained core for PA/Affi-R; soft Affi-P (veto 1–8 pt, prefer 1 core); "
        "never empty a clear-fault window; score≥0.70."
    )

    coupling: list[dict[str, Any]] = []
    if names:
        coupling.append(
            {
                "group": "mad_co_move_or_primary",
                "members": names[: min(4, len(names))],
                "check": check,
            }
        )

    return {
        "mode": mode,
        "hint_version": hv,
        "channels": channel_rows,
        "coupling_groups": coupling,
        "decision_hints": hints,
    }
