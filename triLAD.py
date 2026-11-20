#!/usr/bin/env python3
"""TriLAD inference entry point (VLM/LLM API multivariate TSAD)."""

from __future__ import annotations

from _bootstrap_trilad import bootstrap

bootstrap()

import argparse
import difflib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_pipeline import (
    downsample,
    load_csv,
    make_windows,
)
from TriLAD.schemas import DetectionOutput, WindowSample
from TriLAD.detector import (
    LLMWindowDetector,
    append_output_jsonl,
    dump_outputs_jsonl,
)
from utils.aggregation import window_outputs_to_point_labels
from TriLAD.vision_encoder import VisionEncoderConfig, build_vision_image_encoder
from TriLAD.few_shot import (
    DEFAULT_FEW_SHOT_FEATURE_ALIGN,
    DualExampleFewShotSampler,
    FewShotConfig,
    FewShotReference,
    build_eval_mask,
    build_normal_reference_pool,
    window_has_anomaly,
)
from TriLAD.few_shot_pool_sources import (
    anomaly_ranges_for_reference,
    build_few_shot_pool,
)
from TriLAD.llm_api import OpenAICompatibleClient
from utils.metrics import affiliation_f1, point_adjust_f1, pointwise_f1


_TRAIN_WINDOW_ID_OFFSET = 1_000_000_000


def _load_train_normal_windows(
    args: argparse.Namespace,
    feature_names: list[str],
) -> tuple[list[WindowSample], np.ndarray]:
    """Window the training/normal CSV on the same scale as the test series."""
    fit_csv = args.normalize_fit_csv
    if not fit_csv:
        raise ValueError(
            "few_shot_normal_pool_source=train requires --normalize_fit_csv "
            "(the training/normal CSV)."
        )
    train_values, train_labels, train_names = load_csv(
        fit_csv,
        normalize_mode=args.normalize_values,
        normalize_fit_csv=fit_csv,
    )
    if list(train_names) != list(feature_names):
        index = {name: i for i, name in enumerate(train_names)}
        missing = [name for name in feature_names if name not in index]
        if missing:
            shown = ", ".join(missing[:8])
            raise ValueError(f"Training CSV is missing test features: {shown}")
        train_values = train_values[:, [index[name] for name in feature_names]]
    train_values = downsample(train_values, down_len=args.down_len, is_label=False)
    train_labels = downsample(train_labels, down_len=args.down_len, is_label=True).astype(np.int32)
    train_values = np.asarray(train_values, dtype=np.float32)
    train_windows = make_windows(
        values=train_values,
        labels=train_labels,
        feature_names=list(feature_names),
        window_size=args.window_size,
        stride=args.stride,
    )
    for window in train_windows:
        window.window_id += _TRAIN_WINDOW_ID_OFFSET
    print(
        f"[INFO] Train normal pool source: {fit_csv} "
        f"points={len(train_labels)} windows={len(train_windows)}"
    )
    return train_windows, train_labels


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    """CLI flags used by champion configs + operational knobs (api_key / output_dir).

    Fixed champion trunk defaults (not exposed): temperature=0, dpi=100,
    dual ICL / lite_dual / similarity / test_gt anomaly pool / typed_windows, resume=False,
    prompt_output_mode defaults to full (0707 champion). OpenRouter site/app headers via env.
    Normal few-shot windows come from the training CSV, and anomaly-pool windows
    are dropped from evaluation. Champion passes those same defaults.
    """
    parser = argparse.ArgumentParser(
        description="VLM/LLM API multivariate TS anomaly detection pipeline"
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to source CSV")
    parser.add_argument(
        "--output_dir", type=str, default="runs/adhoc", help="Output directory"
    )
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--model", type=str, default="qwen/qwen3-vl-30b-a3b-instruct")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL (default: OpenRouter).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (or use OPENROUTER_API_KEY / OPENAI_API_KEY).",
    )
    parser.add_argument("--max_windows", type=int, default=0, help="0 means all windows")
    parser.add_argument(
        "--use_vision",
        action="store_true",
        help="Use image+text VLM prompt",
    )
    parser.add_argument(
        "--show_attack_points",
        type=str2bool,
        default=False,
        help="Whether to draw attack interval shading in online plotting mode.",
    )
    parser.add_argument(
        "--normalize_values",
        type=str,
        default="none",
        choices=["none", "minmax", "standard"],
        help="Normalization for values before prompting.",
    )
    parser.add_argument(
        "--normalize_fit_csv",
        type=str,
        default=None,
        help=(
            "Training/normal CSV. Fits the scaler when normalization is on, and "
            "supplies normal few-shot windows when --few_shot_normal_pool_source=train."
        ),
    )
    parser.add_argument(
        "--down_len",
        type=int,
        default=1,
        help="Downsampling length. 1 means no downsampling.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for few-shot sampling.")
    parser.add_argument(
        "--check_model",
        type=str2bool,
        default=False,
        help="Whether to pre-check model availability via /models before inference.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent inference requests. 1 means sequential.",
    )
    parser.add_argument(
        "--few_shot_enable",
        type=str2bool,
        default=True,
        help="Enable few-shot reference sampling (champion requires True).",
    )
    parser.add_argument(
        "--few_shot_ratio",
        type=float,
        default=0.1,
        help="Max ratio of windows selected into few-shot pool (<= 0.1 recommended).",
    )
    parser.add_argument(
        "--few_shot_per_window",
        type=int,
        default=1,
        help="Number of few-shot references sampled per target window.",
    )
    parser.add_argument(
        "--few_shot_strategy",
        type=str,
        default="diverse_segment",
        choices=[
            "diverse_segment",
            "random_per_segment",
            "one_per_segment",
        ],
        help="How to construct few-shot pools (anomaly pool; normal unless dual_normal_strategy set).",
    )
    parser.add_argument(
        "--few_shot_dual_normal_strategy",
        type=str,
        default="",
        choices=[
            "",
            "diverse_segment",
            "random_per_segment",
            "one_per_segment",
        ],
        help="Dual ICL only: override normal reference pool strategy (empty = same as few_shot_strategy).",
    )
    parser.add_argument(
        "--few_shot_normal_pool_source",
        type=str,
        default="train",
        choices=["train", "attack_gt"],
        help=(
            "Normal few-shot source. train=windows from --normalize_fit_csv; "
            "attack_gt=GT-normal windows on the test/attack CSV."
        ),
    )
    parser.add_argument(
        "--exclude_anomaly_pool_from_eval",
        type=str2bool,
        default=True,
        help=(
            "Skip anomaly-pool windows at inference and drop the points they cover "
            "from metrics."
        ),
    )
    parser.add_argument(
        "--few_shot_pool_use_all_windows",
        type=str2bool,
        default=False,
        help=(
            "When max_windows>0, build few-shot pools from the full sliding-window set "
            "(not the truncated eval subset)."
        ),
    )
    parser.add_argument(
        "--few_shot_vision_mark_anomaly",
        type=str2bool,
        default=False,
        help="In vision mode, whether to highlight anomaly intervals for few-shot references.",
    )
    parser.add_argument(
        "--few_shot_feature_align",
        type=str,
        default=DEFAULT_FEW_SHOT_FEATURE_ALIGN,
        choices=["none", "similarity_mad_subset"],
        help=(
            "Few-shot vision / similarity feature alignment. "
            "similarity_mad_subset=Champion default (MAD on target, similarity on subset)."
        ),
    )
    parser.add_argument(
        "--prompt_max_intervals",
        type=int,
        default=5,
        help="Prompt-side cap hint for max anomaly intervals. Used when --prompt_max_intervals_auto=False.",
    )
    parser.add_argument(
        "--prompt_max_intervals_auto",
        type=str2bool,
        default=True,
        help="Auto-compute max anomaly interval count across current windows and use it as prompt_max_intervals.",
    )
    parser.add_argument(
        "--prompt_reasoning_mode",
        type=str,
        default="cot-lite-pass",
        choices=["cot-lite", "cot-lite-pass"],
        help="Prompt reasoning style (cot-lite or cot-lite-pass).",
    )
    parser.add_argument(
        "--prompt_profile",
        type=str,
        default="none",
        choices=[
            "none",
            "swat_window_hint_v2",
            "msl_univar_v1",
            "smd_hint",
            "smd_hint_recall",
            "psm_hint_cover_bal",
        ],
        help=(
            "Dataset-specific champion overlay: swat_window_hint_v2 | msl_univar_v1 | "
            "smd_hint | smd_hint_recall | psm_hint_cover_bal."
        ),
    )
    parser.add_argument(
        "--feature_select_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "mad",
            "first_k",
            "variance",
            "random",
            "cluster",
            "derivative",
        ],
        help=(
            "Adaptive feature subplot selection for vision mode. "
            "'none' keeps all features; others select k subplots per window "
            "(mad / first_k / variance / random / cluster / derivative)."
        ),
    )
    parser.add_argument(
        "--feature_select_k",
        type=int,
        default=8,
        help="Number of feature subplots per window image when feature_select_mode != none.",
    )
    parser.add_argument(
        "--parse_merge_gap",
        type=int,
        default=0,
        help="Merge parsed local ranges within a window when gap <= this value (0 disables).",
    )
    parser.add_argument(
        "--parse_min_interval_len",
        type=int,
        default=0,
        help="Drop parsed local ranges shorter than this length (0 disables).",
    )
    parser.add_argument(
        "--prompt_min_interval_len",
        type=int,
        default=0,
        help="Prompt hint: discourage anomaly intervals shorter than this local length.",
    )
    parser.add_argument(
        "--prompt_prefer_wide_intervals",
        type=str2bool,
        default=False,
        help="Prompt hint: prefer fewer, wider anomaly intervals over fragmented spikes.",
    )
    parser.add_argument(
        "--prompt_output_mode",
        type=str,
        default="full",
        choices=["full", "slim", "no_anomaly_features"],
        help=(
            "Completion JSON schema: full=ranges+score+rationale+anomalous_features; "
            "no_anomaly_features=ranges+score+rationale; slim=ranges+score only."
        ),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading CSV: {args.csv_path}")
    values, labels, feature_names = load_csv(
        args.csv_path,
        normalize_mode=args.normalize_values,
        normalize_fit_csv=args.normalize_fit_csv,
    )
    print(f"[INFO] Loaded {len(labels)} points, {len(feature_names)} features")

    values = downsample(values, down_len=args.down_len, is_label=False)
    labels = downsample(labels, down_len=args.down_len, is_label=True).astype(np.int32)
    values = values.astype(np.float32)

    windows = make_windows(
        values=values,
        labels=labels,
        feature_names=feature_names,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"[INFO] Total windows before truncation: {len(windows)}")
    all_windows = windows
    if args.max_windows > 0:
        eval_windows = all_windows[: args.max_windows]
        print(
            f"[INFO] Eval windows truncated by max_windows (head): "
            f"{len(eval_windows)} / {len(all_windows)}"
        )
    else:
        eval_windows = all_windows
    eval_is_subset = len(eval_windows) < len(all_windows)
    use_all_pool = bool(args.few_shot_pool_use_all_windows) and eval_is_subset
    pool_scope_windows = all_windows if use_all_pool else eval_windows
    if use_all_pool:
        print(
            f"[INFO] Few-shot pool scope: all {len(all_windows)} windows "
            f"(eval on {len(eval_windows)})"
        )
    windows = eval_windows
    if len(windows) == 0:
        raise ValueError("No windows generated. Check window_size/stride/down_len settings.")

    from utils.prompt_intervals import compute_auto_prompt_max_intervals_from_windows

    if args.prompt_max_intervals_auto:
        effective_prompt_max_intervals = compute_auto_prompt_max_intervals_from_windows(windows)
        print(
            f"[INFO] Auto prompt_max_intervals enabled: computed_max={effective_prompt_max_intervals}"
        )
    else:
        effective_prompt_max_intervals = int(args.prompt_max_intervals)
        print(
            f"[INFO] Auto prompt_max_intervals disabled: using manual value={effective_prompt_max_intervals}"
        )

    fs_mode = (args.feature_select_mode or "none").strip().lower()
    baseline_mean = None
    baseline_std = None

    vision_encoder_cfg = VisionEncoderConfig(
        feature_select_mode=fs_mode,
        feature_select_k=args.feature_select_k,
        feature_select_seed=0,
        use_fixed_7x7_grid=False,
        subplot_titles=True,
        dpi=100,
        y_margin_ratio=0.05,
        show_attack_points=args.show_attack_points,
        attack_shade_color="red",
        attack_shade_alpha=0.15,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
    )
    online_encode_image, online_select_features = build_vision_image_encoder(vision_encoder_cfg)

    fs_active = fs_mode not in {"none", "all"}

    if args.use_vision:
        print("[INFO] Vision mode: online plotting")
        if fs_active:
            print(
                f"[INFO] Feature selection: mode={fs_mode}, k={args.feature_select_k}"
            )
    elif fs_active:
        print(
            f"[INFO] Text-only mode with feature selection: mode={fs_mode}, "
            f"k={args.feature_select_k}"
        )

    client = OpenAICompatibleClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    if args.check_model:
        try:
            models = client.list_models()
            if args.model not in models:
                close = difflib.get_close_matches(args.model, models, n=5)
                raise RuntimeError(
                    f"Model {args.model!r} not in /models list. Close matches: {close}"
                )
            print(f"[INFO] Model check passed: {args.model}")
        except Exception as exc:
            raise RuntimeError(f"Model check failed: {exc}") from exc

    fs_config = FewShotConfig(
        enabled=args.few_shot_enable,
        ratio=args.few_shot_ratio,
        per_window=args.few_shot_per_window,
        strategy=args.few_shot_strategy,
        sampler="similarity",
        seed=args.seed,
        pool_budget="windows",
        pool_ratio_scope="typed_windows",
        dual_normal_strategy=(args.few_shot_dual_normal_strategy or "").strip() or None,
    )
    pool_source = "test_gt"
    few_shot_pool, _auto_example_mode = build_few_shot_pool(
        fs_config,
        pool_source=pool_source,
        windows=pool_scope_windows,
        labels=labels,
        values=values,
        feature_names=feature_names,
        normal_csv=None,
        normalize_mode=args.normalize_values,
        normalize_fit_csv=args.normalize_fit_csv,
        window_size=args.window_size,
        stride=args.stride,
        down_len=args.down_len,
    )
    few_shot_example_mode = "dual"
    dual_icl = True
    dual_variant = "lite_dual"
    ref_interval_mode = "pseudo"
    few_shot_window_ids = {w.window_id for w in few_shot_pool if w.window_id >= 0}
    anomaly_pool_ids = set(few_shot_window_ids)
    normal_source = (args.few_shot_normal_pool_source or "train").strip().lower()
    if normal_source not in {"train", "attack_gt"}:
        raise ValueError(f"Unsupported few_shot_normal_pool_source={normal_source!r}")
    normal_reference_pool: list[WindowSample] = []
    if dual_icl and fs_config.enabled:
        if normal_source == "train":
            train_windows, train_labels = _load_train_normal_windows(args, feature_names)
            normal_reference_pool = build_normal_reference_pool(
                train_windows,
                fs_config,
                labels=train_labels,
                labels_len=len(train_labels),
                diverse_from_all_normal=True,
            )
        else:
            normal_reference_pool = build_normal_reference_pool(
                pool_scope_windows, fs_config, labels=labels, labels_len=len(labels)
            )
        few_shot_window_ids |= {w.window_id for w in normal_reference_pool}

    pool_budget_note = (
        f"point_budget={int(len(labels) * fs_config.capped_ratio())}"
        if fs_config.uses_point_budget()
        else (
            f"typed_cap_anom=ceil(n_anom*{fs_config.capped_ratio():.2f}),"
            f"typed_cap_norm=ceil(n_norm*{fs_config.capped_ratio():.2f})"
            if fs_config.uses_typed_ratio_scope()
            else f"window_cap={max(1, int(len(pool_scope_windows) * fs_config.capped_ratio()))}"
        )
    )
    print(
        "[INFO] Few-shot pool: "
        f"enabled={fs_config.enabled}, pool_source={pool_source}, "
        f"strategy={fs_config.strategy}, sampler={fs_config.sampler}, "
        f"pool_budget={fs_config.pool_budget} ({pool_budget_note}), "
        f"pool_ratio_scope={fs_config.pool_ratio_scope}, "
        f"example_mode={few_shot_example_mode}, "
        f"dual_variant={dual_variant if dual_icl else 'n/a'}, "
        f"pool_size={len(few_shot_pool)}, normal_pool_size={len(normal_reference_pool)}, "
        f"dual_normal_pool_source={normal_source}, "
        f"max_ratio_cap={fs_config.capped_ratio():.4f}, "
        f"shot_per_window={fs_config.per_window}"
        + (" (dual: 1 normal + 1 anomaly)" if dual_icl else "")
        + f", reference_interval_mode={ref_interval_mode}"
    )

    def image_encoder_with_override(
        sample,
        force_show_attack_points: bool | None = None,
        *,
        force_feature_indices=None,
    ):
        show_attack = args.show_attack_points if force_show_attack_points is None else force_show_attack_points
        if show_attack != vision_encoder_cfg.show_attack_points or force_feature_indices is not None:
            cfg = VisionEncoderConfig(
                feature_select_mode=vision_encoder_cfg.feature_select_mode,
                feature_select_k=vision_encoder_cfg.feature_select_k,
                feature_select_seed=vision_encoder_cfg.feature_select_seed,
                use_fixed_7x7_grid=vision_encoder_cfg.use_fixed_7x7_grid,
                subplot_titles=vision_encoder_cfg.subplot_titles,
                dpi=vision_encoder_cfg.dpi,
                y_margin_ratio=vision_encoder_cfg.y_margin_ratio,
                show_attack_points=show_attack,
                attack_shade_color=vision_encoder_cfg.attack_shade_color,
                attack_shade_alpha=vision_encoder_cfg.attack_shade_alpha,
                baseline_mean=vision_encoder_cfg.baseline_mean,
                baseline_std=vision_encoder_cfg.baseline_std,
            )
            enc, _ = build_vision_image_encoder(cfg)
            return enc(
                sample,
                force_feature_indices=force_feature_indices,
                force_show_attack_points=show_attack,
            )
        return online_encode_image(
            sample,
            force_feature_indices=force_feature_indices,
            force_show_attack_points=force_show_attack_points,
        )

    def _append_reference_pool(
        pool_windows: list[WindowSample],
        *,
        example_mode_for_ranges: str,
        out: list[FewShotReference],
    ) -> None:
        for ref in pool_windows:
            local_ranges = anomaly_ranges_for_reference(
                ref,
                pool_source=pool_source,
                example_mode=example_mode_for_ranges,
                pseudo_labels=None,
                reference_interval_mode=ref_interval_mode,
            )
            ref_item = FewShotReference(
                sample=ref,
                anomaly_ranges_local=local_ranges,
            )
            if args.use_vision:
                mark_anom = args.few_shot_vision_mark_anomaly and example_mode_for_ranges != "normal"
                if fs_mode not in {"none", "all"}:
                    sel = online_select_features(ref)
                    ref_item.vision_feature_indices = list(sel.indices)
                ref_item.image_b64 = image_encoder_with_override(
                    ref,
                    force_show_attack_points=mark_anom,
                )
            out.append(ref_item)

    few_shot_payload_pool: list[FewShotReference] = []
    normal_payload_pool: list[FewShotReference] = []
    if fs_config.enabled and few_shot_pool:
        _append_reference_pool(
            few_shot_pool,
            example_mode_for_ranges="anomaly" if dual_icl else few_shot_example_mode,
            out=few_shot_payload_pool,
        )
    if dual_icl and normal_reference_pool:
        _append_reference_pool(
            normal_reference_pool,
            example_mode_for_ranges="normal",
            out=normal_payload_pool,
        )
    fs_align = (args.few_shot_feature_align or DEFAULT_FEW_SHOT_FEATURE_ALIGN).strip().lower()
    if fs_align != "none":
        print(f"[INFO] Few-shot feature align mode: {fs_align}")
    feature_select_cb = (
        online_select_features
        if fs_align == "similarity_mad_subset" and fs_active
        else None
    )
    if not dual_icl or not fs_config.enabled:
        raise RuntimeError("Champion pipeline requires dual ICL with few-shot enabled.")
    if not few_shot_payload_pool or not normal_payload_pool:
        raise RuntimeError(
            "dual ICL requires both anomaly few_shot_pool and normal_reference_pool to be non-empty."
        )
    fs_sampler = DualExampleFewShotSampler(
        anomaly_refs=few_shot_payload_pool,
        normal_refs=normal_payload_pool,
        select_for_window=feature_select_cb,
    )

    image_encoder = image_encoder_with_override if args.use_vision else None
    detector = LLMWindowDetector(
        client=client,
        use_vision=args.use_vision,
        image_encoder=image_encoder,
        temperature=0.0,
        max_features=None,
        prompt_max_points=None,
        prompt_value_decimals=4,
        prompt_compact_values=True,
        prompt_max_intervals=effective_prompt_max_intervals,
        prompt_reasoning_mode=args.prompt_reasoning_mode,
        prompt_profile=args.prompt_profile,
        few_shot_sampler=fs_sampler.sample_for_window,
        feature_selector=online_select_features if fs_active else None,
        few_shot_feature_align=fs_align,
        few_shot_vision_mark_anomaly=args.few_shot_vision_mark_anomaly,
        parse_merge_gap=args.parse_merge_gap,
        parse_min_interval_len=args.parse_min_interval_len,
        prompt_min_interval_len=args.prompt_min_interval_len,
        prompt_prefer_wide_intervals=args.prompt_prefer_wide_intervals,
        few_shot_example_mode=few_shot_example_mode,
        reference_interval_mode=ref_interval_mode,
        few_shot_dual_variant=dual_variant,
        prompt_output_mode=args.prompt_output_mode,
        prompt_compact_protocol=False,
        prompt_compact_sections=False,
    )

    slim_output = (args.prompt_output_mode or "full").strip().lower() == "slim"

    print("[INFO] Running model inference over windows...")
    checkpoint_path = out_dir / "window_predictions.jsonl"
    completed_by_id: dict[int, DetectionOutput] = {}
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[INFO] Fresh run: removed existing {checkpoint_path.name}")

    exclude_anomaly = bool(args.exclude_anomaly_pool_from_eval)
    if exclude_anomaly:
        infer_windows = [w for w in windows if w.window_id not in anomaly_pool_ids]
        print(
            f"[INFO] Excluding {len(windows) - len(infer_windows)} "
            "anomaly-pool windows from evaluation"
        )
        if not infer_windows:
            raise RuntimeError(
                "All windows were anomaly-pool windows; nothing left to evaluate."
            )
    else:
        infer_windows = list(windows)
    pending_windows = infer_windows
    total = len(infer_windows)
    pending_total = len(pending_windows)

    checkpoint_lock = threading.Lock()

    def _checkpoint_output(out: DetectionOutput) -> None:
        with checkpoint_lock:
            append_output_jsonl(out, checkpoint_path, slim_output=slim_output)
            completed_by_id[out.window_id] = out

    def _detect_and_checkpoint(sample) -> DetectionOutput:
        out = detector.detect_window(sample)
        _checkpoint_output(out)
        return out

    if pending_total > 0:
        if args.concurrency <= 1:
            for i, sample in enumerate(pending_windows, start=1):
                _detect_and_checkpoint(sample)
                done_total = len(completed_by_id)
                if i == 1 or i % 10 == 0 or i == pending_total:
                    print(f"[INFO] Inference progress: {done_total}/{total} windows")
        else:
            print(f"[INFO] Concurrency enabled: {args.concurrency}")
            done_pending = 0
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = {
                    ex.submit(_detect_and_checkpoint, sample): sample.window_id
                    for sample in pending_windows
                }
                for fut in as_completed(futures):
                    fut.result()
                    done_pending += 1
                    done_total = len(completed_by_id)
                    if done_pending == 1 or done_pending % 10 == 0 or done_pending == pending_total:
                        print(f"[INFO] Inference progress: {done_total}/{total} windows")

    missing = [w.window_id for w in infer_windows if w.window_id not in completed_by_id]
    if missing:
        raise RuntimeError(
            f"Inference incomplete: missing {len(missing)} windows (e.g. {missing[:5]})."
        )
    outputs = [completed_by_id[w.window_id] for w in infer_windows]

    print("[INFO] Aggregating window predictions to point-level labels...")
    # Provisional single-agg metrics only; champion scores use offline per-metric aggregation.
    pred = window_outputs_to_point_labels(
        outputs,
        total_len=len(labels),
        vote_threshold=0.1,
        vote_mode="binary",
        score_gate_threshold=0.0,
        merge_range_gap=5,
    )

    coverage_mask = build_eval_mask(
        total_len=len(values),
        windows=pool_scope_windows,
        reference_window_ids=few_shot_window_ids,
    )
    if exclude_anomaly:
        score_mask = build_eval_mask(
            total_len=len(values),
            windows=pool_scope_windows,
            reference_window_ids=anomaly_pool_ids,
        )
        if not np.any(score_mask):
            raise RuntimeError(
                "Anomaly-pool windows cover the entire series; nothing left to score."
            )
        # Kept points stay in time order. Removed spans are not scored as normal gaps.
        metric_labels = labels[score_mask]
        metric_pred = pred[score_mask]
        excluded_points = int((~score_mask).sum())
        kept_points = int(score_mask.sum())
    else:
        metric_labels = labels
        metric_pred = pred
        excluded_points = 0
        kept_points = int(len(labels))
    metrics = {
        "point_wise_f1": pointwise_f1(metric_labels, metric_pred),
        "point_adjust_f1": point_adjust_f1(metric_labels, metric_pred),
        "affi_f1": affiliation_f1(metric_labels, metric_pred),
    }
    metrics["aggregation"] = {
        "vote_threshold": 0.1,
        "vote_mode": "binary",
        "score_gate_threshold": 0.0,
        "merge_range_gap": 5,
        "note": "provisional; champion score uses offline aggregation_per_metric",
    }
    metrics["prompt_postprocess"] = {
        "parse_merge_gap": int(args.parse_merge_gap),
        "parse_min_interval_len": int(args.parse_min_interval_len),
        "prompt_min_interval_len": int(args.prompt_min_interval_len),
        "prompt_prefer_wide_intervals": bool(args.prompt_prefer_wide_intervals),
    }
    metrics["feature_selection"] = {
        "mode": fs_mode,
        "k": int(args.feature_select_k),
        "fixed_7x7_grid": False,
        "subplot_titles": True,
    }
    metrics["few_shot"] = {
        "enabled": bool(fs_config.enabled),
        "pool_source": pool_source,
        "example_mode": few_shot_example_mode,
        "paradigm": "dual_reference_independent" if dual_icl else "anomaly_pattern_matching",
        "dual_variant": dual_variant if dual_icl else None,
        "strategy": fs_config.strategy,
        "dual_normal_strategy": fs_config.dual_normal_strategy,
        "sampler": fs_config.sampler,
        "feature_align": fs_align,
        "reference_interval_mode": ref_interval_mode,
        "dual_normal_pool_source": normal_source if dual_icl else None,
        "few_shot_pool_budget": fs_config.pool_budget if fs_config.enabled else None,
        "few_shot_pool_ratio_scope": fs_config.pool_ratio_scope if fs_config.enabled else None,
        "prompt_max_intervals": int(effective_prompt_max_intervals),
        "prompt_max_intervals_auto": bool(args.prompt_max_intervals_auto),
        "prompt_reasoning_mode": args.prompt_reasoning_mode,
        "prompt_profile": args.prompt_profile,
        "pool_size": int(len(few_shot_pool)),
        "normal_pool_size": int(len(normal_reference_pool)),
        "pool_window_ids": sorted(int(wid) for wid in few_shot_window_ids),
        "anomaly_pool_window_ids": sorted(int(w.window_id) for w in few_shot_pool if w.window_id >= 0),
        "normal_pool_window_ids": sorted(int(w.window_id) for w in normal_reference_pool),
        "per_window_shot": int(fs_config.per_window),
        "eval_mask_applied": exclude_anomaly,
        "eval_excluded_points": excluded_points,
        "eval_total_points": int(len(labels)),
        "eval_kept_points": kept_points,
        "few_shot_pool_coverage_points": int((~coverage_mask).sum()),
    }

    print("[INFO] Saving outputs...")
    dump_outputs_jsonl(outputs, out_dir / "window_predictions.jsonl", slim_output=slim_output)
    np.save(out_dir / "gt_point_labels.npy", labels)
    np.save(out_dir / "pred_point_labels.npy", pred)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()}, f, indent=2)

    print(
        f"[INFO] Metrics: PA-F1={metrics['point_adjust_f1']['f1']:.4f}, "
        f"Affi-F1={metrics['affi_f1']['f1']:.4f}, "
        f"Point-F1={metrics['point_wise_f1']['f1']:.4f}"
    )
    print(f"[INFO] Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
