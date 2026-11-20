from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from TriLAD.feature_selection import FeatureSelectionResult
from TriLAD.few_shot import (
    DEFAULT_FEW_SHOT_FEATURE_ALIGN,
    FAST_STATS_DUAL_VARIANTS,
    FewShotReference,
    references_to_prompt_examples,
    window_mad_stats_summary,
)
from TriLAD.vision_encoder import apply_feature_selection_to_sample
from TriLAD.llm_api import OpenAICompatibleClient
from TriLAD.prompting import SYSTEM_PROMPT, build_structured_payload, build_user_prompt
from TriLAD.parsing import merge_ranges, normalize_ranges, parse_detection_response
from hints.msl_window_hint import build_window_msl_hint
from hints.psm_window_hint import build_window_psm_hint
from hints.smd_window_hint import build_window_smd_hint
from hints.swat_window_hint import build_window_swat_hint
from TriLAD.schemas import DetectionOutput, WindowSample


class LLMWindowDetector:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        use_vision: bool = False,
        image_encoder: Any | None = None,
        temperature: float = 0.0,
        max_features: int | None = None,
        prompt_max_points: int | None = None,
        prompt_value_decimals: int = 4,
        prompt_compact_values: bool = True,
        prompt_max_intervals: int = 5,
        prompt_reasoning_mode: str = "strict",
        prompt_profile: str = "none",
        few_shot_sampler: Any | None = None,
        feature_selector: Any | None = None,
        few_shot_feature_align: str = DEFAULT_FEW_SHOT_FEATURE_ALIGN,
        few_shot_vision_mark_anomaly: bool = False,
        parse_merge_gap: int = 0,
        parse_min_interval_len: int = 0,
        prompt_min_interval_len: int = 0,
        prompt_prefer_wide_intervals: bool = False,
        few_shot_example_mode: str = "dual",
        reference_interval_mode: str = "pseudo",
        few_shot_dual_variant: str = "lite_dual",
        prompt_output_mode: str = "full",
        prompt_compact_protocol: bool = False,
        prompt_compact_sections: bool = False,
    ) -> None:
        self.client = client
        self.use_vision = use_vision
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.max_features = max_features
        self.prompt_max_points = prompt_max_points
        self.prompt_value_decimals = prompt_value_decimals
        self.prompt_compact_values = prompt_compact_values
        self.prompt_max_intervals = prompt_max_intervals
        self.prompt_reasoning_mode = prompt_reasoning_mode
        self.prompt_profile = (prompt_profile or "none").strip().lower()
        self.few_shot_sampler = few_shot_sampler
        self.feature_selector = feature_selector
        self.few_shot_feature_align = (
            few_shot_feature_align or DEFAULT_FEW_SHOT_FEATURE_ALIGN
        ).strip().lower()
        self.few_shot_vision_mark_anomaly = bool(few_shot_vision_mark_anomaly)
        self.parse_merge_gap = int(parse_merge_gap)
        self.parse_min_interval_len = int(parse_min_interval_len)
        self.prompt_min_interval_len = int(prompt_min_interval_len)
        self.prompt_prefer_wide_intervals = bool(prompt_prefer_wide_intervals)
        self.few_shot_example_mode = (few_shot_example_mode or "dual").strip().lower()
        self.reference_interval_mode = (reference_interval_mode or "pseudo").strip().lower()
        self.few_shot_dual_variant = (few_shot_dual_variant or "lite_dual").strip().lower()
        self.prompt_output_mode = (prompt_output_mode or "full").strip().lower()
        self.prompt_compact_protocol = bool(prompt_compact_protocol)
        self.prompt_compact_sections = bool(prompt_compact_sections)

    def _build_window_swat_hint(
        self,
        vision_names: list[str] | None,
        fs_meta: dict | None,
    ) -> dict | None:
        if self.prompt_profile != "swat_window_hint_v2":
            return None
        mad_scores = (fs_meta or {}).get("scores") if fs_meta else None
        return build_window_swat_hint(
            vision_names or [],
            mad_scores=mad_scores,
            version="v2",
            include_mad_rank_top=0,
            compact=False,
        )

    def _build_window_msl_hint(
        self,
        vision_names: list[str] | None,
        fs_meta: dict | None,
    ) -> dict | None:
        del fs_meta
        if self.prompt_profile != "msl_univar_v1":
            return None
        return build_window_msl_hint(vision_names or [], version="univar")

    def _build_window_smd_hint(
        self,
        vision_names: list[str] | None,
        fs_meta: dict | None,
    ) -> dict | None:
        if self.prompt_profile not in {"smd_hint", "smd_hint_recall"}:
            return None
        mad_scores = (fs_meta or {}).get("scores") if fs_meta else None
        hint_version = "v2" if self.prompt_profile == "smd_hint_recall" else "v1"
        return build_window_smd_hint(
            vision_names or [], mad_scores=mad_scores, hint_version=hint_version
        )

    def _build_window_psm_hint(
        self,
        vision_names: list[str] | None,
        fs_meta: dict | None,
    ) -> dict | None:
        if self.prompt_profile != "psm_hint_cover_bal":
            return None
        mad_scores = (fs_meta or {}).get("scores") if fs_meta else None
        return build_window_psm_hint(
            vision_names or [], mad_scores=mad_scores, hint_version="cover_bal"
        )

    def _vision_context(self, sample: WindowSample) -> tuple[WindowSample, list[str] | None, dict | None]:
        if self.feature_selector is None:
            return sample, None, None
        sel: FeatureSelectionResult = self.feature_selector(sample)
        vision_names = [sample.feature_names[i] for i in sel.indices]
        meta = {
            "method": sel.method,
            "k": sel.k,
            "selected_indices": sel.indices,
            "selected_features": vision_names,
            "scores": sel.scores,
        }
        prompt_sample = apply_feature_selection_to_sample(sample, sel)
        return prompt_sample, vision_names, meta

    def _build_messages(self, sample: WindowSample) -> list[dict[str, Any]]:
        few_shot_examples: list[FewShotReference] = []
        if self.few_shot_sampler is not None:
            few_shot_examples = self.few_shot_sampler(sample)

        target_sel: FeatureSelectionResult | None = None
        if self.feature_selector is not None:
            target_sel = self.feature_selector(sample)

        prompt_sample, vision_names, fs_meta = self._vision_context(sample)
        payload = build_structured_payload(
            prompt_sample if self.use_vision else sample,
            max_features=self.max_features,
            max_points=self.prompt_max_points,
            value_decimals=self.prompt_value_decimals,
            compact_values=self.prompt_compact_values,
            include_values=(not self.use_vision),
            vision_feature_names=vision_names,
            feature_selection_meta=fs_meta,
            compact_protocol=self.prompt_compact_protocol,
            compact_sections=self.prompt_compact_sections,
        )
        prompt_examples = references_to_prompt_examples(
            few_shot_examples,
            example_mode=self.few_shot_example_mode,
            reference_interval_mode=self.reference_interval_mode,
            dual_variant=self.few_shot_dual_variant,
        )
        user_text = build_user_prompt(
            prompt_sample if self.use_vision else sample,
            payload,
            few_shot_examples=prompt_examples,
            max_intervals=self.prompt_max_intervals,
            reasoning_mode=self.prompt_reasoning_mode,
            min_interval_len=self.prompt_min_interval_len,
            prefer_wide_intervals=self.prompt_prefer_wide_intervals,
            few_shot_example_mode=self.few_shot_example_mode,
            reference_interval_mode=self.reference_interval_mode,
            few_shot_dual_variant=self.few_shot_dual_variant,
            prompt_profile=self.prompt_profile,
            window_swat_hint=self._build_window_swat_hint(vision_names, fs_meta),
            window_msl_hint=self._build_window_msl_hint(vision_names, fs_meta),
            window_smd_hint=self._build_window_smd_hint(vision_names, fs_meta),
            window_psm_hint=self._build_window_psm_hint(vision_names, fs_meta),
            prompt_output_mode=self.prompt_output_mode,
            prompt_compact_protocol=self.prompt_compact_protocol,
            prompt_compact_sections=self.prompt_compact_sections,
        )
        if self.use_vision and self.image_encoder is not None:
            content = [{"type": "text", "text": user_text}]
            use_fast_stats_dual = (
                self.few_shot_dual_variant in FAST_STATS_DUAL_VARIANTS
                and self.few_shot_example_mode == "dual"
                and len(few_shot_examples) >= 2
            )
            if use_fast_stats_dual:
                normal_ref = next(
                    (
                        r
                        for r in few_shot_examples
                        if r.resolved_example_type(self.few_shot_example_mode) == "normal"
                    ),
                    None,
                )
                anomaly_ref = next(
                    (
                        r
                        for r in few_shot_examples
                        if r.resolved_example_type(self.few_shot_example_mode) == "anomaly"
                    ),
                    None,
                )
                if normal_ref is not None and anomaly_ref is not None:
                    stats_indices: list[int] | None = None
                    if target_sel is not None:
                        stats_indices = list(target_sel.indices)
                    elif anomaly_ref.vision_feature_indices:
                        stats_indices = list(anomaly_ref.vision_feature_indices)
                    if stats_indices is not None:
                        stats_payload = window_mad_stats_summary(
                            normal_ref.sample,
                            stats_indices,
                            compact=True,
                        )
                        stats_payload["reference_role"] = "normal_baseline_stats"
                        content.append(
                            {"type": "text", "text": json.dumps(stats_payload, ensure_ascii=True)}
                        )
                        ref_sample = anomaly_ref.sample
                        ref_show_attack = self.few_shot_vision_mark_anomaly
                        if anomaly_ref.image_b64 is not None:
                            ref_img = anomaly_ref.image_b64
                        else:
                            ref_img = self.image_encoder(
                                ref_sample, force_show_attack_points=ref_show_attack
                            )
                        ref_payload = {
                            "reference_window_id": ref_sample.window_id,
                            "reference_example_type": "anomaly",
                            "reference_role": "anomaly_pattern_exemplar",
                            "reference_anomaly_ranges_local": anomaly_ref.anomaly_ranges_local,
                        }
                        content.append(
                            {"type": "text", "text": json.dumps(ref_payload, ensure_ascii=True)}
                        )
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{ref_img}"},
                            }
                        )

            img_b64 = self.image_encoder(
                sample,
                force_feature_indices=target_sel.indices if target_sel is not None else None,
            )
            content.append({"type": "text", "text": "Target window image for prediction:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]

    def detect_window(self, sample: WindowSample) -> DetectionOutput:
        messages = self._build_messages(sample)
        raw = self.client.chat(messages=messages, temperature=self.temperature)
        obj, strict_ok = parse_detection_response(raw)
        window_len = len(sample.values)
        ranges = normalize_ranges(obj.get("anomaly_ranges_local", []), window_length=window_len)
        if self.parse_merge_gap > 0 and ranges:
            ranges = merge_ranges(ranges, max_gap=self.parse_merge_gap, window_length=window_len)
        if self.parse_min_interval_len > 1 and ranges:
            min_len = self.parse_min_interval_len
            ranges = [rg for rg in ranges if int(rg["end"]) - int(rg["start"]) + 1 >= min_len]
        score = float(obj.get("anomaly_score", 0.0) or 0.0)
        rationale = str(obj.get("rationale", ""))
        anomalous_features = obj.get("anomalous_features", [])
        if not isinstance(anomalous_features, list):
            anomalous_features = []
        anomalous_features = [str(x) for x in anomalous_features]

        return DetectionOutput(
            window_id=sample.window_id,
            start_idx=sample.start_idx,
            end_idx=sample.end_idx,
            anomaly_ranges_local=ranges,
            anomaly_score=max(0.0, min(1.0, score)),
            rationale=rationale,
            raw_response=raw,
            parsed_ok=strict_ok,
            extra={
                "messages": messages,
                "anomalous_features": anomalous_features,
            },
        )


def load_outputs_jsonl(path: str | Path) -> list[DetectionOutput]:
    outputs: list[DetectionOutput] = []
    p = Path(path)
    if not p.exists():
        return outputs
    with open(p, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {err}") from err
            outputs.append(
                DetectionOutput(
                    window_id=int(row["window_id"]),
                    start_idx=int(row["start_idx"]),
                    end_idx=int(row["end_idx"]),
                    anomaly_ranges_local=row.get("anomaly_ranges_local", []),
                    anomaly_score=float(row.get("anomaly_score", 0.0) or 0.0),
                    rationale=str(row.get("rationale", "")),
                    raw_response=str(row.get("raw_response", "")),
                    parsed_ok=bool(row.get("parsed_ok", False)),
                )
            )
    return outputs


def _output_to_json_row(out: DetectionOutput, *, slim_output: bool = False) -> dict[str, Any]:
    row = {
        "window_id": out.window_id,
        "start_idx": out.start_idx,
        "end_idx": out.end_idx,
        "anomaly_ranges_local": out.anomaly_ranges_local,
        "anomaly_score": out.anomaly_score,
        "parsed_ok": out.parsed_ok,
        "raw_response": out.raw_response,
    }
    if not slim_output:
        row["rationale"] = out.rationale
    return row


def append_output_jsonl(
    output: DetectionOutput,
    path: str | Path,
    *,
    slim_output: bool = False,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(_output_to_json_row(output, slim_output=slim_output), ensure_ascii=True) + "\n")
        f.flush()


def dump_outputs_jsonl(
    outputs: list[DetectionOutput],
    path: str,
    *,
    slim_output: bool = False,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(_output_to_json_row(out, slim_output=slim_output), ensure_ascii=True) + "\n")
