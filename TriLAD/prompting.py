from __future__ import annotations

import json
import math

from TriLAD.schemas import WindowSample


SYSTEM_PROMPT = (
    "You are an anomaly detection expert for industrial multivariate time series. "
    "Always return strict JSON and never output markdown."
)


def _reasoning_mode_key(reasoning_mode: str) -> str:
    return (reasoning_mode or "strict").strip().lower()


_DUAL_RATIONALE_LINE = (
    "In rationale, cite normal-baseline mismatch and anomaly-pattern similarity separately."
)


_VALID_PROMPT_OUTPUT_MODES = frozenset({"full", "slim", "no_anomaly_features"})


def _normalize_prompt_output_mode(prompt_output_mode: str) -> str:
    mode = (prompt_output_mode or "full").strip().lower()
    return mode if mode in _VALID_PROMPT_OUTPUT_MODES else "full"


def _is_slim_completion(prompt_output_mode: str) -> bool:
    """Slim completion schema only (ranges+score); does not alter prompt protocol."""
    return _normalize_prompt_output_mode(prompt_output_mode) == "slim"


def _omit_anomalous_features(prompt_output_mode: str) -> bool:
    return _normalize_prompt_output_mode(prompt_output_mode) in {"slim", "no_anomaly_features"}


def _filter_rationale_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        low = line.lower()
        if "rationale" in low:
            continue
        out.append(line)
    return out


def _build_output_schema(*, prompt_output_mode: str = "full") -> dict:
    mode = _normalize_prompt_output_mode(prompt_output_mode)
    schema = {
        "anomaly_ranges_local": [{"start": 0, "end": 0}],
        "anomaly_score": "float in [0,1]",
    }
    if mode == "slim":
        return schema
    schema["rationale"] = "short explanation"
    if mode == "full":
        schema["anomalous_features"] = ["feature_name_1", "feature_name_2"]
    return schema


def _adjust_vision_note_for_output_mode(vision_note: str, prompt_output_mode: str) -> str:
    mode = _normalize_prompt_output_mode(prompt_output_mode)
    if mode != "no_anomaly_features":
        return vision_note
    return (
        vision_note.replace(
            "Report anomalous_features using the exact feature names shown.",
            "Name suspicious features in rationale using the exact feature names shown.",
        ).replace(
            "report anomalous_features using those exact names.",
            "name suspicious features in rationale using those exact names.",
        )
    )


def _reasoning_protocol(reasoning_mode: str) -> list[str]:
    mode = _reasoning_mode_key(reasoning_mode)
    base = [
        "Pass 1 (scan): identify candidate abnormal segments and suspicious features.",
        "Pass 2 (recheck): remove weak candidates that can be explained by normal fluctuation/noise.",
    ]
    if mode == "cot-lite-pass":
        return base + ["Keep rationale concise but evidence-based."]
    if mode == "cot-lite":
        return base + ["Keep rationale concise but evidence-based."]
    if mode == "strict":
        return base
    raise ValueError(f"Unsupported prompt reasoning mode: {reasoning_mode!r}")


def _dual_reference_few_shot_protocol(reasoning_mode: str) -> list[str]:
    base = [
        "Few-shot paradigm: DUAL REFERENCE (independent normal + anomaly picks).",
        "Step A (normal anchor): from NORMAL few_shot_examples, infer baseline level bands, "
        "cross-feature coupling, and acceptable noise.",
        "Step B (anomaly template): from ANOMALY few_shot_examples, learn anomaly morphology, "
        "interval width, and labeling style.",
        "Step C (target decision): flag target intervals that BOTH deviate from the normal baseline "
        "AND resemble anomaly signatures; do not copy reference indices.",
        "Step D (filter): drop candidates explainable as normal jitter or lacking multi-feature support.",
    ]
    mode = _reasoning_mode_key(reasoning_mode)
    if mode == "cot-lite":
        return base + [_DUAL_RATIONALE_LINE]
    if mode == "cot-lite-pass":
        return base + _reasoning_protocol("strict") + [_DUAL_RATIONALE_LINE]
    raise ValueError(f"Unsupported prompt reasoning mode for dual ICL: {reasoning_mode!r}")


def _few_shot_paradigm_meta(
    example_mode: str,
    reasoning_mode: str,
    *,
    dual_variant: str = "lite_dual",
    compact_protocol: bool = False,
) -> dict:
    mode = (example_mode or "dual").strip().lower()
    dual_var = (dual_variant or "lite_dual").strip().lower()
    if mode != "dual":
        raise ValueError(f"Unsupported few-shot example mode: {example_mode!r}")
    if dual_var != "lite_dual":
        raise ValueError(f"Unsupported dual ICL variant: {dual_variant!r}")
    return {
        "paradigm": "lite_dual",
        "task": "Compact dual reference: stats baseline + anomaly chart; minimal rationale.",
        "few_shot_note": "Independent similarity picks; 2-line decision protocol.",
        "decision_protocol": _dual_reference_few_shot_protocol(reasoning_mode),
        "extra_rules": ["Do not copy reference indices."],
    }


def _prompt_profile_block(name: str) -> dict:
    """Per-profile prompt overlay: domain context, extra rules, decision steps."""
    n = (name or "").strip().lower()
    if n == "swat_window_hint_v2":
        return {
            "domain_context": "",
            "extra_rules": [
                "window_swat_hint is optional soft context; lite_dual baseline+pattern remains primary.",
                "Incomplete coupling groups must NOT force empty anomaly_ranges_local.",
            ],
            "decision_protocol_append": [
                "Apply lite_dual first; use coupling_checks only to refine boundaries/rationale.",
            ],
        }
    if n == "msl_univar_v1":
        return {
            "domain_context": (
                "NASA MSL univariate primary telemetry (var_0 only; command/one-hot dims dropped). "
                "Point anomalies ~10%; windows—especially large ones—are often only partially anomalous."
            ),
            "extra_rules": [
                "Judge only the single visible curve; ignore multi-channel / coupling requirements.",
                "window_msl_hint has mode=univariate_primary and empty coupling_checks — that is expected.",
                "Empty ranges when the curve stays near its in-window baseline; flag sustained local shifts only.",
                "Do not label the entire large window unless the curve is anomalous almost everywhere.",
                "Require lite_dual morph match on this curve; spikey jitter without pattern match → [].",
            ],
            "decision_protocol_append": [
                "Pass 2: keep only sustained sub-intervals (≥20 indices) with anomaly-template similarity.",
                "Trim to the tightest deviant core; prefer partial localization over whole-window spans.",
            ],
        }
    if n == "smd_hint":
        return {
            "domain_context": (
                "SMD (Server Machine Dataset): multivariate KPI telemetry from a production server "
                "(~38 channels: CPU/memory/disk/network-style metrics, already scaled to [0,1]). "
                "Anomalies are service/system faults with mixed segment lengths "
                "(very short bursts to multi-hundred-point runs). "
                "Point anomaly rates are typically low (~1.5%–9.5% depending on machine)."
            ),
            "extra_rules": [
                "Judge the MAD-selected KPI channels shown in the image; use window_smd_hint as soft context.",
                "Prefer co-moving deviations across ≥2 MAD channels; veto isolated single-channel micro-spikes.",
                "Empty ranges when KPIs stay near the in-window baseline; do not invent regular short intervals.",
                "Large windows are often only partially anomalous: output tight local sub-interval(s), never the whole window by default.",
                "Segment length varies: keep short cores for brief bursts (≥8–15 indices) when morphology matches the anomaly few-shot; "
                "for long drifts keep the sustained deviant span only.",
                "Require lite_dual morph match; stats-only wobble without pattern similarity → [].",
            ],
            "decision_protocol_append": [
                "Pass 2: keep only intervals with anomaly-template similarity and multi-KPI support when available.",
                "Trim to the tightest deviant core; prefer partial localization over whole-window spans.",
            ],
        }
    if n == "smd_hint_recall":
        return {
            "domain_context": (
                "SMD server KPI telemetry (already in [0,1]). Faults often appear as a sustained shift on "
                "ONE dominant MAD channel, sometimes with weak co-movement. Mixed lengths: short bursts "
                "(~5–30 pts) and long drifts. Prefer recall of true deviant cores over empty windows."
            ),
            "extra_rules": [
                "Judge MAD-selected KPIs; use window_smd_hint as soft context.",
                "Accept a single MAD channel sustained level/slope shift if it morph-matches the anomaly few-shot; "
                "still veto isolated 1–3 point spikes and chart noise.",
                "Do NOT output edge-to-edge ranges (start≈0 and end≈W-1) unless the curve is anomalous almost everywhere.",
                "If GT-like morphology is only mid-window, return that tight sub-interval; empty is wrong when a clear core exists.",
                "Short bursts (≥5–12 indices) are valid when they match the anomaly template; long drifts: keep the sustained span only.",
                "Require lite_dual anomaly-pattern morph match; stats-only wobble without pattern similarity → [].",
                "At most 3 intervals; anomaly_score≥0.6 for clear cores (not only ≥0.7).",
            ],
            "decision_protocol_append": [
                "Pass 2: if Pass-1 was empty but one MAD channel shows a clear sustained deviation matching the anomaly few-shot, add that core.",
                "Trim to the tightest deviant core; never default to whole-window spans.",
            ],
        }
    if n == "psm_hint_cover_bal":
        return {
            "domain_context": (
                "PSM eBay app-server metrics (~[0,1], high anomaly rate). "
                "THIS prompt balances cover with Affi/PA: keep clear sustained faults fully covered "
                "(PA + Affi recall), soft-cut false short segments (Affi precision). Point may drop slightly."
            ),
            "extra_rules": [
                "Judge MAD-selected metrics with window_psm_hint.",
                "Prefer ≥2 MAD co-move; one strong top-MAD morph match is enough — MUST report it.",
                "Hard veto: 1–8 point spikes, fragmented shorts, edge-to-edge dumps.",
                "Clear fault → cover FULL local span as preferably ONE core (max 2).",
                "anomaly_score≥0.70 for kept cores; empty ONLY when panel is clean (no clear core).",
            ],
            "decision_protocol_append": [
                "Pass 2 (cover-bal): merge same-shift fragments into one full core; drop only 1–8 pt / weak wobble; "
                "never empty a window that still has a clear sustained core.",
            ],
        }
    return {"domain_context": "", "extra_rules": [], "decision_protocol_append": []}


_PROFILE_ALIASES: dict[str, list[str]] = {
    "swat_window_hint_v2": ["swat_window_hint_v2"],
    "msl_univar_v1": ["msl_univar_v1"],
    "smd_hint": ["smd_hint"],
    "smd_hint_recall": ["smd_hint_recall"],
    "psm_hint_cover_bal": ["psm_hint_cover_bal"],
}

CHAMPION_PROMPT_PROFILES: frozenset[str] = frozenset(_PROFILE_ALIASES)


def _apply_prompt_profile(
    paradigm: dict,
    profile: str,
    *,
    compact_protocol: bool = False,
    reasoning_mode: str = "strict",
) -> dict:
    """Merge prompt_profile overlays into paradigm task/rules/protocol."""
    del reasoning_mode
    p = (profile or "none").strip().lower()
    if not p or p == "none":
        out = dict(paradigm)
    else:
        keys = _PROFILE_ALIASES.get(p, [p])
        domain_parts: list[str] = []
        extra: list[str] = []
        proto_append: list[str] = []
        for key in keys:
            block = _prompt_profile_block(key)
            if block.get("domain_context"):
                domain_parts.append(str(block["domain_context"]))
            extra.extend(block.get("extra_rules") or [])
            proto_append.extend(block.get("decision_protocol_append") or [])
        out = dict(paradigm)
        if domain_parts:
            out["task"] = f"{out.get('task', '')} Domain context: {' '.join(domain_parts)}"
        if extra:
            out["extra_rules"] = list(out.get("extra_rules") or []) + extra
        if proto_append:
            out["decision_protocol"] = list(out.get("decision_protocol") or []) + proto_append
    if compact_protocol:
        out["decision_protocol"] = _filter_rationale_lines(list(out.get("decision_protocol") or []))
        out["extra_rules"] = _filter_rationale_lines(list(out.get("extra_rules") or []))
        task = str(out.get("task") or "")
        if "rationale" in task.lower():
            out["task"] = task.replace("minimal rationale.", "minimal output.").replace(
                "minimal rationale", "minimal output"
            )
    return out


def _window_stats(sample: WindowSample) -> dict[str, float]:
    flat = [v for row in sample.values for v in row]
    if not flat:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    mean = sum(flat) / len(flat)
    return {"min": float(min(flat)), "max": float(max(flat)), "mean": float(mean)}


def build_structured_payload(
    sample: WindowSample,
    max_features: int | None = None,
    max_points: int | None = None,
    value_decimals: int = 4,
    compact_values: bool = True,
    include_values: bool = True,
    vision_feature_names: list[str] | None = None,
    feature_selection_meta: dict | None = None,
    compact_protocol: bool = False,
    compact_sections: bool = False,
) -> dict:
    if vision_feature_names is not None:
        selected = list(vision_feature_names)
    elif max_features is None:
        selected = sample.feature_names
    else:
        selected = sample.feature_names[:max_features]
    idx_map = {name: i for i, name in enumerate(sample.feature_names)}

    def _safe_number(x):
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, value_decimals)

    payload = {
        "window_id": sample.window_id,
        "start_idx": sample.start_idx,
        "end_idx": sample.end_idx,
        "features": selected,
        "notes": (
            "Local 0-based indices in anomaly_ranges_local."
            if compact_sections
            else "Index in anomaly_ranges_local is local index in current window, 0 <= idx < window_length."
        ),
    }
    if feature_selection_meta:
        if compact_sections:
            payload["feature_selection"] = {
                "method": feature_selection_meta.get("method"),
                "k": feature_selection_meta.get("k"),
                "features": feature_selection_meta.get("selected_features"),
            }
        else:
            payload["feature_selection"] = feature_selection_meta
        if compact_protocol:
            payload["vision_note"] = (
                "Image shows selected features in 'features' only; other channels omitted for readability."
            )
        elif compact_sections:
            payload["vision_note"] = (
                "Image shows 'features' only; report anomalous_features using those exact names."
            )
        else:
            payload["vision_note"] = (
                "The attached image shows only the selected features listed in 'features'. "
                "Other channels exist in the full multivariate window but are omitted from the figure "
                "to improve readability. Report anomalous_features using the exact feature names shown."
            )
    if include_values:
        sampled_rows = sample.values
        sampled_indices = list(range(len(sampled_rows)))
        if max_points is not None and max_points > 0 and len(sampled_rows) > max_points:
            step = max(1, len(sampled_rows) // max_points)
            sampled_indices = list(range(0, len(sampled_rows), step))[:max_points]
            sampled_rows = [sample.values[i] for i in sampled_indices]
        payload["sampled_local_indices"] = sampled_indices
        if compact_values:
            payload["values_2d"] = [[_safe_number(row[idx_map[name]]) for name in selected] for row in sampled_rows]
            payload["value_layout"] = "values_2d[row_idx][feature_idx] where feature_idx maps to features list."
        else:
            payload["values"] = [{name: _safe_number(row[idx_map[name]]) for name in selected} for row in sampled_rows]
    else:
        payload["value_included"] = False
        payload["value_note"] = "Numeric sequence omitted in vision mode; infer from image."

    return payload


def build_user_prompt(
    sample: WindowSample,
    payload: dict,
    few_shot_examples: list[dict] | None = None,
    max_intervals: int = 5,
    reasoning_mode: str = "strict",
    min_interval_len: int = 0,
    prefer_wide_intervals: bool = False,
    few_shot_example_mode: str = "dual",
    reference_interval_mode: str = "pseudo",
    few_shot_dual_variant: str = "lite_dual",
    prompt_profile: str = "none",
    window_swat_hint: dict | None = None,
    window_msl_hint: dict | None = None,
    window_smd_hint: dict | None = None,
    window_psm_hint: dict | None = None,
    prompt_output_mode: str = "full",
    prompt_compact_protocol: bool = False,
    prompt_compact_sections: bool = False,
) -> str:
    output_mode = _normalize_prompt_output_mode(prompt_output_mode)
    slim_completion = output_mode == "slim"
    stats = _window_stats(sample)
    paradigm = _apply_prompt_profile(
        _few_shot_paradigm_meta(
            few_shot_example_mode,
            reasoning_mode,
            dual_variant=few_shot_dual_variant,
            compact_protocol=prompt_compact_protocol,
        ),
        prompt_profile,
        compact_protocol=prompt_compact_protocol,
        reasoning_mode=reasoning_mode,
    )
    interval_mode = (reference_interval_mode or "pseudo").strip().lower()
    few_shot_note = paradigm["few_shot_note"]
    base_rules = [
        "Return valid JSON only.",
        "Ranges are inclusive and sorted by start.",
        "Do not invent unavailable indices.",
        "Use anomaly_score=0.0 if no anomaly.",
        "anomaly_score must reflect confidence: 0.0 for empty ranges; "
        "0.3-0.6 for weak/sparse evidence; 0.7-0.9 for strong multi-feature evidence; "
        "avoid saturating all windows to the same high score.",
        "Anomalies are usually sparse in real industrial windows.",
        f"Do not output more than {max(1, int(max_intervals))} intervals.",
        "Avoid over-long intervals unless there is clear contiguous evidence.",
    ]

    instr = {
        "task": paradigm["task"],
        "few_shot_paradigm": paradigm["paradigm"],
        "output_schema": _build_output_schema(prompt_output_mode=output_mode),
        "rules": base_rules + paradigm["extra_rules"],
        "decision_protocol": paradigm["decision_protocol"],
        "window_summary": {
            "window_id": sample.window_id,
            "window_length": len(sample.values),
            "feature_count": len(sample.feature_names),
            "features_in_image": len(payload.get("features", [])),
            **({} if prompt_compact_sections else {"value_stats": stats}),
        },
        "few_shot_note": few_shot_note,
        "reference_interval_mode": interval_mode if few_shot_examples else "none",
        "few_shot_examples": few_shot_examples or [],
        "input": (
            {
                **payload,
                "vision_note": (
                    "Image shows selected features in 'features' only; other channels omitted for readability."
                    if slim_completion and isinstance(payload.get("vision_note"), str)
                    else _adjust_vision_note_for_output_mode(payload["vision_note"], output_mode)
                    if output_mode == "no_anomaly_features" and isinstance(payload.get("vision_note"), str)
                    else payload.get("vision_note")
                ),
            }
            if isinstance(payload.get("vision_note"), str) or slim_completion
            else payload
        ),
    }
    if slim_completion:
        instr["rules"].append(
            "Output only fields in output_schema (anomaly_ranges_local, anomaly_score); "
            "do not add any extra JSON fields."
        )
    elif output_mode == "no_anomaly_features":
        instr["rules"].append(
            "Output only fields in output_schema; do not include anomalous_features "
            "(name suspicious features in rationale instead)."
        )
    if window_swat_hint:
        instr["window_swat_hint"] = window_swat_hint
    if window_msl_hint:
        instr["window_msl_hint"] = window_msl_hint
    if window_smd_hint:
        instr["window_smd_hint"] = window_smd_hint
    if window_psm_hint:
        instr["window_psm_hint"] = window_psm_hint
    if min_interval_len > 1:
        instr["rules"].append(
            f"Each anomaly interval must span at least {int(min_interval_len)} local indices; "
            "merge adjacent micro-spikes into one contiguous range."
        )
    if prefer_wide_intervals:
        instr["rules"].extend(
            [
                "Prefer a few wide contiguous anomaly ranges over many single-point spikes.",
                "Boundaries should align with visible sustained shifts, not isolated jitter.",
            ]
        )
    return json.dumps(instr, ensure_ascii=True)
