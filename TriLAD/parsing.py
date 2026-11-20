from __future__ import annotations

import json
import re
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def parse_detection_response(raw: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(raw, (str, bytes, bytearray)):
        return {"anomaly_ranges_local": [], "anomaly_score": 0.0, "rationale": ""}, False
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {"anomaly_ranges_local": [], "anomaly_score": 0.0, "rationale": ""}, False
        try:
            return json.loads(match.group(0)), False
        except Exception:
            return {"anomaly_ranges_local": [], "anomaly_score": 0.0, "rationale": ""}, False


def normalize_ranges(anomaly_ranges_local: list[dict[str, Any]], window_length: int) -> list[dict[str, int]]:
    clean: list[dict[str, int]] = []
    for rg in anomaly_ranges_local or []:
        st = _safe_int(rg.get("start", 0))
        ed = _safe_int(rg.get("end", 0))
        st = max(0, min(window_length - 1, st))
        ed = max(0, min(window_length - 1, ed))
        if ed < st:
            st, ed = ed, st
        clean.append({"start": st, "end": ed})
    clean.sort(key=lambda x: x["start"])
    return clean


def merge_ranges(
    ranges: list[dict[str, int]],
    *,
    max_gap: int = 0,
    window_length: int | None = None,
) -> list[dict[str, int]]:
    """Merge overlapping or nearby intervals (gap <= max_gap)."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x["start"])
    merged: list[dict[str, int]] = [dict(sorted_ranges[0])]
    for rg in sorted_ranges[1:]:
        prev = merged[-1]
        gap = int(rg["start"]) - int(prev["end"]) - 1
        if gap <= max_gap:
            prev["end"] = max(int(prev["end"]), int(rg["end"]))
        else:
            merged.append(dict(rg))
    if window_length is not None:
        for rg in merged:
            rg["start"] = max(0, min(window_length - 1, int(rg["start"])))
            rg["end"] = max(0, min(window_length - 1, int(rg["end"])))
    return merged

