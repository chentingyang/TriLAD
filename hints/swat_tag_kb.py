"""Static SWaT tag metadata and stage coupling groups for per-window hints."""

from __future__ import annotations

import re
from typing import Any

# All 51 process tags from SWaT attack CSV (excluding Timestamp, Attack).
SWAT_PROCESS_TAGS: tuple[str, ...] = (
    "FIT101", "LIT101", "MV101", "P101", "P102",
    "AIT201", "AIT202", "AIT203", "FIT201", "MV201",
    "P201", "P203", "P204", "P205", "P206",
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304",
    "P301", "P302",
    "AIT401", "AIT402", "FIT401", "LIT401", "P402", "P403", "UV401",
    "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504",
    "P501", "PIT501", "PIT502", "PIT503",
    "FIT601", "P602",
)

_ROLE_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("DPIT", "dp_pressure"),
    ("PIT", "pressure"),
    ("FIT", "flow"),
    ("LIT", "level"),
    ("AIT", "analyzer"),
    ("MV", "valve"),
    ("UV", "uv"),
    ("P", "pump"),
)


def _stage_from_tag(name: str) -> str:
    m = re.search(r"(\d)", name)
    if not m:
        return "unknown"
    n = int(m.group(1))
    return f"P{n}" if 1 <= n <= 6 else "unknown"


def _role_from_tag(name: str) -> str:
    upper = name.strip().upper()
    for prefix, role in _ROLE_BY_PREFIX:
        if upper.startswith(prefix):
            return role
    return "other"


def _label_en(stage: str, role: str) -> str:
    labels = {
        ("P1", "level"): "raw water tank level",
        ("P1", "flow"): "raw water flow",
        ("P1", "valve"): "inlet valve",
        ("P1", "pump"): "raw water pump",
        ("P2", "analyzer"): "dosing water quality",
        ("P2", "flow"): "dosing flow",
        ("P2", "valve"): "dosing valve",
        ("P2", "pump"): "dosing pump",
        ("P3", "dp_pressure"): "UF differential pressure",
        ("P3", "flow"): "UF flow",
        ("P3", "level"): "UF tank level",
        ("P3", "valve"): "UF valve",
        ("P3", "pump"): "UF pump",
        ("P4", "analyzer"): "dechlorination water quality",
        ("P4", "flow"): "RO feed flow",
        ("P4", "level"): "RO feed tank level",
        ("P4", "uv"): "UV dechlorination",
        ("P4", "pump"): "RO feed pump",
        ("P5", "analyzer"): "RO water quality",
        ("P5", "flow"): "RO flow",
        ("P5", "pressure"): "RO pressure",
        ("P5", "pump"): "RO pump",
        ("P6", "flow"): "backwash flow",
        ("P6", "pump"): "backwash pump",
    }
    return labels.get((stage, role), f"{stage} {role}")


def tag_meta(name: str) -> dict[str, str]:
    """Return stage/role/label for a SWaT tag (known or inferred)."""
    upper = name.strip().upper()
    stage = _stage_from_tag(upper)
    role = _role_from_tag(upper)
    return {
        "name": upper,
        "stage": stage,
        "role": role,
        "label": _label_en(stage, role),
    }


# Coupling groups: only tags in `members` that appear in the current window are surfaced.
STAGE_COUPLING_GROUPS: dict[str, list[dict[str, Any]]] = {
    "P1": [
        {
            "group_id": "intake_core",
            "members": ["LIT101", "FIT101", "MV101", "P101", "P102"],
            "check": "Intake: level+flow+valve+pump co-move.",
        },
    ],
    "P2": [
        {
            "group_id": "dosing_core",
            "members": ["AIT201", "AIT202", "AIT203", "FIT201", "MV201", "P201", "P203", "P205"],
            "check": "Dosing: analyzer+flow+valve/pump co-move.",
        },
    ],
    "P3": [
        {
            "group_id": "uf_core",
            "members": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302", "P301", "P302"],
            "check": "UF: DP+flow+level+valve/pump co-move.",
        },
    ],
    "P4": [
        {
            "group_id": "dechlor_feed",
            "members": ["AIT401", "AIT402", "FIT401", "LIT401", "UV401", "P402", "P403"],
            "check": "Dechlor feed: quality+flow+level co-move.",
        },
    ],
    "P5": [
        {
            "group_id": "ro_core",
            "members": [
                "AIT501", "AIT502", "AIT503", "AIT504",
                "FIT501", "FIT502", "FIT503", "FIT504",
                "PIT501", "PIT502", "PIT503", "P501",
            ],
            "check": "RO: quality+flow+pressure co-move.",
        },
    ],
    "P6": [
        {
            "group_id": "backwash",
            "members": ["FIT601", "P602"],
            "check": "Backwash: flow+pump together; brief spikes may be normal.",
        },
    ],
}

CROSS_STAGE_NOTES: list[dict[str, Any]] = [
    {
        "stages": ("P1", "P2"),
        "note": "P1→P2 lag-consistent co-shifts.",
    },
    {
        "stages": ("P2", "P3"),
        "note": "P2→P3 flow/level may lag dosing changes.",
    },
    {
        "stages": ("P4", "P5"),
        "note": "P4 feed shifts may precede P5 RO changes.",
    },
    {
        "stages": ("P3", "P6"),
        "note": "P6 backwash: brief P3 spikes OK if not sustained.",
        "transient_ok": True,
    },
]

KNOWN_SWAT_TAGS: frozenset[str] = frozenset(SWAT_PROCESS_TAGS)
