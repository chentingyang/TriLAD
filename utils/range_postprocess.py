"""Local interval post-processing on window-level detection outputs."""

from __future__ import annotations

from copy import deepcopy

from TriLAD.schemas import DetectionOutput


def shrink_local_ranges(
    ranges: list[dict[str, int]],
    shrink: int,
    *,
    min_len: int = 1,
) -> list[dict[str, int]]:
    """Trim shrink indices from both ends of each local range (inclusive)."""
    if shrink <= 0:
        return [dict(rg) for rg in ranges]
    out: list[dict[str, int]] = []
    for rg in ranges:
        s, e = int(rg["start"]), int(rg["end"])
        span = e - s + 1
        if span <= 2 * shrink:
            if span >= min_len:
                out.append({"start": s, "end": e})
            continue
        ns, ne = s + shrink, e - shrink
        if ne >= ns and (ne - ns + 1) >= min_len:
            out.append({"start": ns, "end": ne})
    return out


def apply_endpoint_shrink(outputs: list[DetectionOutput], shrink: int) -> list[DetectionOutput]:
    """Return copies of outputs with shrunk anomaly_ranges_local."""
    if shrink <= 0:
        return outputs
    shrunk: list[DetectionOutput] = []
    for out in outputs:
        dup = deepcopy(out)
        dup.anomaly_ranges_local = shrink_local_ranges(out.anomaly_ranges_local or [], shrink)
        shrunk.append(dup)
    return shrunk
