#!/usr/bin/env python3
"""Unified VLM-AD champion reproduction entry for SWaT / MSL / SMD / PSM.

CLI:
  python scripts/champion.py {swat|msl|smd|psm} {show|run|score} [options]

`show` prints reproduction config (including offline aggregation / precise-hit).
`run` always starts a fresh API inference, then offline-scores the new outputs.
`score` offline-scores an existing run dir (no API).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import _bootstrap_trilad as _bootstrap

_bootstrap.bootstrap()

from utils.aggregation import window_outputs_to_point_labels
from utils.metrics import affiliation_f1, evaluate_metrics, point_adjust_f1, pointwise_f1
from utils.precise_hit_postprocess import PreciseHitPostConfig, refine_window_ranges
from utils.prompt_intervals import build_eval_windows, materialize_prompt_max_intervals
from utils.range_postprocess import apply_endpoint_shrink
from TriLAD.schemas import DetectionOutput

QWEN30B = "qwen/qwen3-vl-30b-a3b-instruct"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SMD_ENTITIES = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7"]

PSM_FULL_WINDOWS = 877
PSM_TEST_CSV = "dataset/PSM/vlm/PSM_Dataset_test_example.csv"
PSM_NORMAL_CSV = "dataset/PSM/vlm/PSM_Dataset_normal_example.csv"

# ---------------------------------------------------------------------------
# SWaT
# ---------------------------------------------------------------------------
SWAT_CHAMPION_VARIANT_KEY = "swat_hint_v2_cot_lite_pass_k8"
SWAT_CHAMPION_LABEL = (
    "SWaT champion: window hint v2 + cot-lite-pass (dual Step A-D + Pass1/2 + rationale + full output schema) "
    "+ lite_dual MAD-8 + auto max_intervals + Affi shrink=10"
)
SWAT_FULL_WINDOWS = 898

_SHARED_FEWSHOT: dict[str, Any] = {
    "few_shot_strategy": "diverse_segment",
    "few_shot_dual_normal_strategy": "random_per_segment",
    "few_shot_ratio": 0.1,
    "few_shot_dual_variant": "lite_dual",
    "few_shot_pool_source": "test_gt",
    "few_shot_example_mode": "dual",
    "few_shot_pool_ratio_scope": "typed_windows",
    "few_shot_pool_budget": "windows",
    "few_shot_dual_normal_pool_source": "attack_gt",
    "parse_merge_gap": 30,
    "parse_min_interval_len": 8,
    "prompt_min_interval_len": 20,
    "prompt_prefer_wide_intervals": False,
}

SWAT_CHAMPION_OVERRIDES: dict[str, Any] = {
    **_SHARED_FEWSHOT,
    "csv_path": "dataset/SWaT/SWaT_Dataset_attack_example.csv",
    "normalize_fit_csv": "dataset/SWaT/SWaT_Dataset_normal_example.csv",
    "window_size": 1000,
    "stride": 500,
    "down_len": 1,
    "seed": 0,
    "feature_select_mode": "mad",
    "feature_select_k": 8,
    "prompt_profile": "swat_window_hint_v2",
    "prompt_reasoning_mode": "cot-lite-pass",
    "prompt_output_mode": "full",
    "show_attack_points": True,
    "prompt_max_intervals_auto": True,
    "prompt_max_intervals": 5,
    "few_shot_pool_use_all_windows": True,
    "few_shot_sampler": "similarity",
    "few_shot_feature_align": "similarity_mad_subset",
    "few_shot_reference_interval_mode": "pseudo",
}

SWAT_CHAMPION_AGGREGATION_PER_METRIC: dict[str, dict[str, Any]] = {
    "point": {
        "vote_threshold": 0.51,
        "vote_mode": "score_weighted",
        "merge_range_gap": 200,
        "score_gate_threshold": 0.5,
    },
    "pa": {
        "vote_threshold": 0.51,
        "vote_mode": "score_weighted",
        "merge_range_gap": 0,
        "score_gate_threshold": 0.5,
    },
    "affi": {
        "vote_threshold": 0.05,
        "vote_mode": "binary",
        "merge_range_gap": 100,
        "score_gate_threshold": 0.0,
        "affi_endpoint_shrink": 10,
    },
}

# ---------------------------------------------------------------------------
# MSL
# ---------------------------------------------------------------------------
MSL_CHAMPION_VARIANT_KEY = "msl_v2_uni_u1_w1600s400_post0660_per_metric"
MSL_CHAMPION_LABEL = (
    "MSL champion: MSL_v2 univar_v1 + var0-only + none-norm + win1600/stride400 "
    "+ cot-lite + precise-hit post + per-metric aggregation"
)
MSL_V2_CSV_PATH = "dataset/MSL_v2/MSL_Dataset_test_example.csv"
MSL_V2_NORMAL_CSV_PATH = "dataset/MSL_v2/MSL_Dataset_normal_example.csv"
MSL_WINDOW_SIZE = 1600
MSL_STRIDE = 400
MSL_FULL_WINDOWS = 181

MSL_CHAMPION_OVERRIDES: dict[str, Any] = {
    **_SHARED_FEWSHOT,
    "csv_path": MSL_V2_CSV_PATH,
    "normalize_fit_csv": MSL_V2_NORMAL_CSV_PATH,
    "window_size": MSL_WINDOW_SIZE,
    "stride": MSL_STRIDE,
    "down_len": 1,
    "normalize_values": "none",
    "feature_select_mode": "first_k",
    "feature_select_k": 1,
    "prompt_profile": "msl_univar_v1",
    "prompt_reasoning_mode": "cot-lite",
    "show_attack_points": False,
    "prompt_max_intervals_auto": False,
    "prompt_max_intervals": 4,
    "few_shot_pool_use_all_windows": True,
    "few_shot_feature_align": "similarity_mad_subset",
}

MSL_CHAMPION_PRECISE_HIT_POST: dict[str, Any] = {
    "contrast_zthr": 2.0,
    "min_level_shift": 0.25,
    "min_sharp": 0.01,
    "max_len": 800,
    "pad": 16,
    "min_len": 16,
    "max_ranges": 2,
}

MSL_CHAMPION_AGGREGATION_PER_METRIC: dict[str, dict[str, Any]] = {
    "point": {
        "endpoint_shrink": 0,
        "vote_threshold": 0.05,
        "vote_mode": "binary",
        "merge_range_gap": 10,
        "score_gate_threshold": 0.0,
    },
    "pa": {
        "endpoint_shrink": 40,
        "vote_threshold": 0.05,
        "vote_mode": "binary",
        "merge_range_gap": 0,
        "score_gate_threshold": 0.0,
    },
    "affi": {
        "endpoint_shrink": 0,
        "vote_threshold": 0.05,
        "vote_mode": "binary",
        "merge_range_gap": 10,
        "score_gate_threshold": 0.0,
    },
}

# ---------------------------------------------------------------------------
# SMD (inlined; no results/ dependency)
# ---------------------------------------------------------------------------
SMD_CHAMPION_VARIANT_KEY = 'smd_per_entity_phase_c_20260717'
SMD_CHAMPION_LABEL = (
    "SMD champion: per-entity VLM-AD (MAD + similarity/lite_dual ratio=0.1 + Qwen3-VL-30B-A3B); 1-1 mad8/w1600s400/smd_hint/cot-lite; 2-1 mad8/w1600s200/smd_hint/cot-lite; 3-2 mad16/w1000s500/smd_hint_recall/cot-lite-pass; 3-7 mad4/w1600s400/smd_hint/cot-lite; no precise-hit; Point/PA/Affi per-metric post; n_points-weighted aggregate"
)
SMD_LOCKED_STACK: dict[str, Any] = {
    'use_vision': True,
    'model': 'qwen/qwen3-vl-30b-a3b-instruct',
    'base_url': 'https://openrouter.ai/api/v1',
    'check_model': False,
    'normalize_values': 'none',
    'feature_select_mode': 'mad',
    'few_shot_enable': True,
    'few_shot_vision_mark_anomaly': False,
    'few_shot_strategy': 'diverse_segment',
    'few_shot_dual_normal_strategy': 'random_per_segment',
    'few_shot_ratio': 0.1,
    'few_shot_pool_use_all_windows': True,
    'few_shot_normal_pool_source': 'train',
    'exclude_anomaly_pool_from_eval': True,
    'few_shot_feature_align': 'similarity_mad_subset',
    'few_shot_per_window': 1,
    'show_attack_points': False,
    'prompt_max_intervals_auto': False,
    'down_len': 1,
    'seed': 42,
}
SMD_PER_ENTITY: dict[str, dict[str, Any]] = {
    'machine-1-1': {
        'overrides': {
            'feature_select_k': 8,
            'window_size': 1600,
            'stride': 400,
            'prompt_profile': 'smd_hint',
            'prompt_reasoning_mode': 'cot-lite',
            'prompt_min_interval_len': 20,
            'parse_min_interval_len': 8,
            'prompt_max_intervals': 3,
            'parse_merge_gap': 30,
            'csv_path': 'dataset/SMD/vlm/machine-1-1/SMD_Dataset_test_example.csv',
            'normalize_fit_csv': 'dataset/SMD/vlm/machine-1-1/SMD_Dataset_normal_example.csv',
        },
        'aggregation_per_metric': {
            'point': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.2,
                'merge_range_gap': 200,
                'vote_mode': 'score_weighted',
                'score_gate_threshold': 0.0,
                'min_pred_len': 0,
                'max_events': 5,
                'window_min_len': 0,
            },
            'pa': {
                'endpoint_shrink': 20,
                'vote_threshold': 0.2,
                'merge_range_gap': 200,
                'vote_mode': 'score_weighted',
                'score_gate_threshold': 0.0,
                'min_pred_len': 0,
                'max_events': 5,
                'window_min_len': 0,
            },
            'affi': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.2,
                'merge_range_gap': 200,
                'vote_mode': 'score_weighted',
                'score_gate_threshold': 0.0,
                'min_pred_len': 0,
                'max_events': 5,
                'window_min_len': 0,
            },
        },
    },
    'machine-2-1': {
        'overrides': {
            'feature_select_k': 8,
            'window_size': 1600,
            'stride': 200,
            'prompt_profile': 'smd_hint',
            'prompt_reasoning_mode': 'cot-lite',
            'prompt_min_interval_len': 20,
            'parse_min_interval_len': 8,
            'prompt_max_intervals': 3,
            'parse_merge_gap': 30,
            'csv_path': 'dataset/SMD/vlm/machine-2-1/SMD_Dataset_test_example.csv',
            'normalize_fit_csv': 'dataset/SMD/vlm/machine-2-1/SMD_Dataset_normal_example.csv',
        },
        'aggregation_per_metric': {
            'point': {
                'endpoint_shrink': 0,
                'vote_threshold': 0.2,
                'merge_range_gap': 0,
                'vote_mode': 'score_weighted',
                'score_gate_threshold': 0.0,
                'min_pred_len': 15,
                'max_events': 0,
                'window_min_len': 0,
            },
            'pa': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.2,
                'merge_range_gap': 0,
                'vote_mode': 'score_weighted',
                'score_gate_threshold': 0.0,
                'min_pred_len': 0,
                'max_events': 0,
                'window_min_len': 0,
            },
            'affi': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.1,
                'merge_range_gap': 200,
                'vote_mode': 'score_weighted',
                'score_gate_threshold': 0.0,
                'min_pred_len': 0,
                'max_events': 0,
                'window_min_len': 0,
            },
        },
    },
    'machine-3-2': {
        'overrides': {
            'feature_select_k': 16,
            'window_size': 1000,
            'stride': 500,
            'prompt_profile': 'smd_hint_recall',
            'prompt_reasoning_mode': 'cot-lite-pass',
            'prompt_min_interval_len': 5,
            'parse_min_interval_len': 3,
            'prompt_max_intervals': 4,
            'parse_merge_gap': 30,
            'csv_path': 'dataset/SMD/vlm/machine-3-2/SMD_Dataset_test_example.csv',
            'normalize_fit_csv': 'dataset/SMD/vlm/machine-3-2/SMD_Dataset_normal_example.csv',
        },
        'aggregation_per_metric': {
            'point': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.05,
                'merge_range_gap': 0,
                'vote_mode': 'binary',
                'score_gate_threshold': 0.0,
                'min_pred_len': 20,
                'max_events': 0,
                'window_min_len': 0,
            },
            'pa': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.05,
                'merge_range_gap': 0,
                'vote_mode': 'binary',
                'score_gate_threshold': 0.0,
                'min_pred_len': 20,
                'max_events': 0,
                'window_min_len': 0,
            },
            'affi': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.05,
                'merge_range_gap': 50,
                'vote_mode': 'binary',
                'score_gate_threshold': 0.0,
                'min_pred_len': 20,
                'max_events': 0,
                'window_min_len': 0,
            },
        },
    },
    'machine-3-7': {
        'overrides': {
            'feature_select_k': 4,
            'window_size': 1600,
            'stride': 400,
            'prompt_profile': 'smd_hint',
            'prompt_reasoning_mode': 'cot-lite',
            'prompt_min_interval_len': 20,
            'parse_min_interval_len': 8,
            'prompt_max_intervals': 3,
            'parse_merge_gap': 30,
            'csv_path': 'dataset/SMD/vlm/machine-3-7/SMD_Dataset_test_example.csv',
            'normalize_fit_csv': 'dataset/SMD/vlm/machine-3-7/SMD_Dataset_normal_example.csv',
        },
        'aggregation_per_metric': {
            'point': {
                'endpoint_shrink': 0,
                'vote_threshold': 0.3,
                'merge_range_gap': 200,
                'vote_mode': 'binary',
                'score_gate_threshold': 0.0,
                'min_pred_len': 0,
                'max_events': 0,
                'window_min_len': 0,
            },
            'pa': {
                'endpoint_shrink': 10,
                'vote_threshold': 0.3,
                'merge_range_gap': 200,
                'vote_mode': 'binary',
                'score_gate_threshold': 0.0,
                'min_pred_len': 5,
                'max_events': 0,
                'window_min_len': 0,
            },
            'affi': {
                'endpoint_shrink': 20,
                'vote_threshold': 0.2,
                'merge_range_gap': 0,
                'vote_mode': 'score_weighted',
                'score_gate_threshold': 0.0,
                'min_pred_len': 15,
                'max_events': 0,
                'window_min_len': 0,
            },
        },
    },
}

# ---------------------------------------------------------------------------
# PSM
# ---------------------------------------------------------------------------
PSM_CHAMPION_VARIANT_KEY = "psm_cover_bal_pass_mad16_w200s100_aops_ndiv"
PSM_CHAMPION_LABEL = (
    "PSM champion: cover_bal_pass + mad16 + win200/stride100 + cot-lite-pass "
    "+ few-shot aops/ndiv + per-metric aggregation"
)

PSM_CHAMPION_OVERRIDES: dict[str, Any] = {
    **_SHARED_FEWSHOT,
    "csv_path": PSM_TEST_CSV,
    "normalize_fit_csv": PSM_NORMAL_CSV,
    "window_size": 200,
    "stride": 100,
    "down_len": 1,
    "normalize_values": "none",
    "feature_select_mode": "mad",
    "feature_select_k": 16,
    "prompt_profile": "psm_hint_cover_bal",
    "prompt_reasoning_mode": "cot-lite-pass",
    "show_attack_points": False,
    "prompt_max_intervals_auto": False,
    "prompt_max_intervals": 2,
    "few_shot_strategy": "one_per_segment",
    "few_shot_dual_normal_strategy": "diverse_segment",
    "few_shot_pool_use_all_windows": True,
    "few_shot_feature_align": "similarity_mad_subset",
    "parse_merge_gap": 10,
    "parse_min_interval_len": 3,
    "prompt_min_interval_len": 5,
    "prompt_prefer_wide_intervals": False,
}

PSM_CHAMPION_AGGREGATION_PER_METRIC: dict[str, dict[str, Any]] = {
    "point": {
        "endpoint_shrink": 5,
        "vote_threshold": 0.05,
        "vote_mode": "binary",
        "merge_range_gap": 0,
        "score_gate_threshold": 0.85,
    },
    "pa": {
        "endpoint_shrink": 10,
        "vote_threshold": 0.51,
        "vote_mode": "binary",
        "merge_range_gap": 0,
        "score_gate_threshold": 0.0,
    },
    "affi": {
        "endpoint_shrink": 20,
        "vote_threshold": 0.05,
        "vote_mode": "binary",
        "merge_range_gap": 0,
        "score_gate_threshold": 0.3,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_AGG_KEYS = ("vote_threshold", "vote_mode", "merge_range_gap", "score_gate_threshold")
RUNS_ROOT = PROJECT_ROOT / "runs"


def composite_f1(point_f1: float | None, pa_f1: float | None, affi_f1: float | None) -> float | None:
    """Equal-weight mean of point / PA / Affiliation F1."""
    if point_f1 is None or pa_f1 is None or affi_f1 is None:
        return None
    return round((float(point_f1) + float(pa_f1) + float(affi_f1)) / 3.0, 6)

# Only keys that champion configs actually pass into triLAD.py.
# (api_key / output_dir are handled outside this allowlist.)
_RUN_VLM_AD_CLI_KEYS = frozenset(
    {
        "csv_path",
        "window_size",
        "stride",
        "model",
        "base_url",
        "max_windows",
        "use_vision",
        "show_attack_points",
        "normalize_values",
        "normalize_fit_csv",
        "down_len",
        "seed",
        "check_model",
        "concurrency",
        "few_shot_enable",
        "few_shot_ratio",
        "few_shot_per_window",
        "few_shot_strategy",
        "few_shot_dual_normal_strategy",
        "few_shot_pool_use_all_windows",
        "few_shot_normal_pool_source",
        "exclude_anomaly_pool_from_eval",
        "few_shot_vision_mark_anomaly",
        "few_shot_feature_align",
        "prompt_max_intervals",
        "prompt_max_intervals_auto",
        "prompt_reasoning_mode",
        "prompt_profile",
        "prompt_output_mode",
        "feature_select_mode",
        "feature_select_k",
        "parse_merge_gap",
        "parse_min_interval_len",
        "prompt_min_interval_len",
        "prompt_prefer_wide_intervals",
    }
)


def cfg_to_argv(cfg: dict) -> list[str]:
    argv: list[str] = []
    for key, val in cfg.items():
        if key not in _RUN_VLM_AD_CLI_KEYS or val is None:
            continue
        flag = f"--{key}"
        if key == "use_vision":
            if val:
                argv.append(flag)
            continue
        if isinstance(val, bool):
            argv.extend([flag, str(val)])
        else:
            argv.extend([flag, str(val)])
    return argv


def apply_overrides(exp: dict, overrides: dict) -> dict:
    out = deepcopy(exp)
    out["config"] = deepcopy(exp["config"])
    out["config"].update(overrides)
    return out


def _base_exp(*, normalize_values: str = "minmax", concurrency: int = 8) -> dict:
    return {
        "id": "08",
        "name": "cot_lite",
        "config": {
            "use_vision": True,
            "model": QWEN30B,
            "base_url": OPENROUTER_BASE,
            "check_model": False,
            "concurrency": concurrency,
            "normalize_values": normalize_values,
            "few_shot_enable": True,
            "few_shot_vision_mark_anomaly": False,
            "few_shot_sampler": "similarity",
            "few_shot_feature_align": "similarity_mad_subset",
            "few_shot_per_window": 1,
            "few_shot_normal_pool_source": "train",
            "exclude_anomaly_pool_from_eval": True,
        },
    }


def run_TriLAD(
    cfg: dict,
    out_dir: Path,
    logs_dir: Path,
    *,
    log_name: str = "run.log",
) -> dict[str, Any]:
    """Always start a fresh inference (never reuse prior outputs)."""
    if out_dir.exists():
        for name in (*WINDOW_PREDICTIONS_NAMES, "metrics.json", "run_status.json"):
            p = out_dir / name
            if p.exists():
                p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "output_dir": str(out_dir.relative_to(PROJECT_ROOT)),
        "config": cfg,
        "status": "pending",
    }
    status_path = out_dir / "run_status.json"

    cfg = materialize_prompt_max_intervals(deepcopy(cfg))
    if cfg.get("_materialized_prompt_max_intervals_from_auto") is not None:
        computed = cfg.pop("_materialized_prompt_max_intervals_from_auto")
        print(f"[INFO] Materialized prompt_max_intervals: auto -> {computed}", flush=True)

    (out_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cmd = [
        sys.executable,
        "triLAD.py",
        *cfg_to_argv(cfg),
        "--output_dir",
        str(out_dir),
    ]
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / log_name
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n[{datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT)
        logf.write(f"[{datetime.now().isoformat()}] EXIT={proc.returncode}\n")
    record["inference_exit_code"] = proc.returncode
    record["log_path"] = str(log_path.relative_to(PROJECT_ROOT))
    record["status"] = "ok" if proc.returncode == 0 else "failed_inference"
    status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# Offline score
# ---------------------------------------------------------------------------


def _agg_kwargs(agg: dict[str, Any]) -> dict[str, Any]:
    return {k: agg[k] for k in _AGG_KEYS if k in agg}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(label)
    keys = ("point_f1", "pa_f1", "affi_f1", "composite_f1", "avg_f1", "joint_pa_affi")
    parts = [f"{k}={metrics[k]:.6f}" for k in keys if isinstance(metrics.get(k), (int, float))]
    if parts:
        print("  " + "  ".join(parts))
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


# Prefers inference_outputs naming; falls back to pipeline run_dir convention.
WINDOW_PREDICTIONS_NAMES = (
    "window_predictions_pre_agg.jsonl",
    "window_predictions.jsonl",
)


def resolve_window_predictions(run_dir: Path) -> Path:
    for name in WINDOW_PREDICTIONS_NAMES:
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing window predictions under {run_dir} "
        f"(tried: {', '.join(WINDOW_PREDICTIONS_NAMES)})"
    )


def load_window_outputs(jsonl_path: Path) -> list[DetectionOutput]:
    outs: list[DetectionOutput] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            outs.append(
                DetectionOutput(
                    window_id=int(row["window_id"]),
                    start_idx=int(row["start_idx"]),
                    end_idx=int(row["end_idx"]),
                    anomaly_ranges_local=row.get("anomaly_ranges_local") or [],
                    anomaly_score=float(row.get("anomaly_score") or 0.0),
                    rationale=str(row.get("rationale", "")),
                    raw_response=str(row.get("raw_response", "")),
                    parsed_ok=bool(row.get("parsed_ok", False)),
                )
            )
    return outs


def build_eval_mask_from_run(run_dir: Path, gt_len: int) -> np.ndarray:
    """Exclude few-shot pool windows from point-wise eval (SWaT convention)."""
    metrics_path = run_dir / "metrics.json"
    mask = np.ones(gt_len, dtype=bool)
    try:
        jsonl = resolve_window_predictions(run_dir)
    except FileNotFoundError:
        return mask
    pool_ids: set[int] = set()
    if metrics_path.exists():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        pool_ids = {int(x) for x in (meta.get("few_shot", {}) or {}).get("pool_window_ids") or []}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        wid = int(row["window_id"])
        if pool_ids and wid in pool_ids:
            mask[int(row["start_idx"]) : int(row["end_idx"])] = False
    return mask


def _metric_bundle(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pt = pointwise_f1(gt, pred)
    pa = point_adjust_f1(gt, pred)
    af = affiliation_f1(gt, pred)
    return {
        "point_f1": pt["f1"],
        "point_precision": pt["precision"],
        "point_recall": pt["recall"],
        "pa_f1": pa["f1"],
        "pa_precision": pa["precision"],
        "pa_recall": pa["recall"],
        "affi_f1": af["f1"],
        "affi_precision": af["precision"],
        "affi_recall": af["recall"],
        "composite_f1": composite_f1(pt["f1"], pa["f1"], af["f1"]),
        "joint_pa_affi": round(pa["f1"] + af["f1"], 6),
    }


def _pred_with_post(outs: list[DetectionOutput], gt_len: int, post: dict[str, Any]) -> np.ndarray:
    sh = int(post.get("endpoint_shrink") or post.get("affi_endpoint_shrink") or 0)
    cur = apply_endpoint_shrink(outs, sh) if sh else outs
    return window_outputs_to_point_labels(cur, gt_len, **_agg_kwargs(post))


def _filter_events(pred: np.ndarray, *, min_pred_len: int = 0, max_events: int = 0) -> np.ndarray:
    if min_pred_len <= 0 and max_events <= 0:
        return pred
    segs: list[tuple[int, int]] = []
    n = len(pred)
    i = 0
    while i < n:
        if pred[i] == 0:
            i += 1
            continue
        j = i
        while j < n and pred[j] == 1:
            j += 1
        segs.append((i, j - 1))
        i = j
    keep = segs
    if min_pred_len > 0:
        keep = [(a, b) for a, b in keep if (b - a + 1) >= min_pred_len]
    if max_events > 0 and len(keep) > max_events:
        keep = sorted(keep, key=lambda t: -(t[1] - t[0] + 1))[:max_events]
    out = np.zeros(n, dtype=np.int32)
    for a, b in keep:
        out[a : b + 1] = 1
    return out


def _resolve_run_dir(dataset: str, run_dir: str | None, tag: str | None) -> Path:
    """Resolve a scoreable run directory. Requires --run_dir or --tag (no historical fallback)."""
    if run_dir:
        p = Path(run_dir)
        rd = p if p.is_absolute() else PROJECT_ROOT / p
    elif tag:
        rd = RUNS_ROOT / dataset / tag
    else:
        raise FileNotFoundError(
            f"score for {dataset} requires --run_dir <path> or --tag <tag> "
            f"(expected under runs/{dataset}/<tag>/)."
        )
    try:
        resolve_window_predictions(rd)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing window predictions under {rd}") from exc
    return rd


def _write_score(run_dir: Path, scored: dict[str, Any]) -> Path:
    out = run_dir / "champion_score.json"
    out.write_text(json.dumps(scored, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def score_swat(run_dir: Path) -> dict[str, Any]:
    rd = Path(run_dir)
    outs = load_window_outputs(resolve_window_predictions(rd))
    gt_path = rd / "gt_point_labels.npy"
    if gt_path.exists():
        gt = np.load(gt_path).astype(np.int32)
    else:
        csv_path = PROJECT_ROOT / SWAT_CHAMPION_OVERRIDES["csv_path"]
        gt = pd.read_csv(csv_path)["Attack"].to_numpy(dtype=np.int32)
    per: dict[str, Any] = {}
    for metric, agg in SWAT_CHAMPION_AGGREGATION_PER_METRIC.items():
        pred = _pred_with_post(outs, len(gt), agg)
        m = evaluate_metrics(gt, pred)
        if metric == "point":
            per["point_f1"] = m["point_wise_f1"]["f1"]
            per["point_precision"] = m["point_wise_f1"]["precision"]
            per["point_recall"] = m["point_wise_f1"]["recall"]
        elif metric == "pa":
            per["pa_f1"] = m["point_adjust_f1"]["f1"]
            per["pa_precision"] = m["point_adjust_f1"]["precision"]
            per["pa_recall"] = m["point_adjust_f1"]["recall"]
        else:
            per["affi_f1"] = m["affi_f1"]["f1"]
            per["affi_precision"] = m["affi_f1"]["precision"]
            per["affi_recall"] = m["affi_f1"]["recall"]
    per["composite_f1"] = composite_f1(per["point_f1"], per["pa_f1"], per["affi_f1"])
    per["joint_pa_affi"] = round(per["pa_f1"] + per["affi_f1"], 6)
    per["run_dir"] = _rel(rd)
    per["aggregation_per_metric"] = SWAT_CHAMPION_AGGREGATION_PER_METRIC
    _write_score(rd, per)
    return per


def _msl_post_cfg(d: dict[str, Any] | None = None) -> PreciseHitPostConfig:
    base = dict(MSL_CHAMPION_PRECISE_HIT_POST)
    if d:
        base.update(d)
    return PreciseHitPostConfig(**base)


def msl_load_refined_outputs(
    run_dir: Path,
    *,
    post: PreciseHitPostConfig | None = None,
) -> tuple[list[DetectionOutput], np.ndarray]:
    post = post or _msl_post_cfg()
    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    csv_path = str(PROJECT_ROOT / (cfg.get("csv_path") or MSL_V2_CSV_PATH))
    fit_csv = str(PROJECT_ROOT / (cfg.get("normalize_fit_csv") or MSL_V2_NORMAL_CSV_PATH))
    ws = int(cfg.get("window_size") or MSL_WINDOW_SIZE)
    st = int(cfg.get("stride") or MSL_STRIDE)
    preds = [
        json.loads(line)
        for line in resolve_window_predictions(run_dir).read_text().splitlines()
        if line.strip()
    ]
    wins = build_eval_windows(
        csv_path=csv_path,
        window_size=ws,
        stride=st,
        down_len=1,
        normalize_mode="none",
        normalize_fit_csv=fit_csv,
    )
    by = {w.start_idx: w for w in wins}
    gt = pd.read_csv(csv_path)["Attack"].to_numpy(dtype=np.int32)
    outs: list[DetectionOutput] = []
    for r in preds:
        w = by.get(r["start_idx"])
        if w is None:
            continue
        x = np.asarray(w.values, dtype=float)[:, 0]
        refined = refine_window_ranges(x, r.get("anomaly_ranges_local") or [], post)
        outs.append(
            DetectionOutput(
                window_id=int(r.get("window_id") or 0),
                start_idx=int(r["start_idx"]),
                end_idx=int(r.get("end_idx") or w.end_idx),
                anomaly_ranges_local=refined,
                anomaly_score=float(r.get("anomaly_score") or (0.85 if refined else 0.0)),
                rationale="",
                raw_response="",
                parsed_ok=True,
            )
        )
    return outs, gt


def score_msl(run_dir: Path) -> dict[str, Any]:
    """Precise-hit refine + frozen per-metric aggregation (champion recipe)."""
    rd = Path(run_dir)
    post_cfg = _msl_post_cfg()
    outs, gt = msl_load_refined_outputs(rd, post=post_cfg)
    per: dict[str, Any] = {}
    for metric, agg in MSL_CHAMPION_AGGREGATION_PER_METRIC.items():
        m = _metric_bundle(gt, _pred_with_post(outs, len(gt), agg))
        if metric == "point":
            per.update(
                {
                    "point_f1": m["point_f1"],
                    "point_precision": m["point_precision"],
                    "point_recall": m["point_recall"],
                }
            )
        elif metric == "pa":
            per.update(
                {
                    "pa_f1": m["pa_f1"],
                    "pa_precision": m["pa_precision"],
                    "pa_recall": m["pa_recall"],
                }
            )
        else:
            per.update(
                {
                    "affi_f1": m["affi_f1"],
                    "affi_precision": m["affi_precision"],
                    "affi_recall": m["affi_recall"],
                }
            )
    per["composite_f1"] = composite_f1(per["point_f1"], per["pa_f1"], per["affi_f1"])
    per["joint_pa_affi"] = round(per["pa_f1"] + per["affi_f1"], 6)
    per["run_dir"] = _rel(rd)
    per["n_windows"] = len(outs)
    per["precise_hit_post"] = post_cfg.__dict__
    per["aggregation_per_metric"] = MSL_CHAMPION_AGGREGATION_PER_METRIC
    _write_score(rd, per)
    return per


def _smd_load_outputs(
    run_dir: Path,
    *,
    csv_path: str,
    fit_csv: str,
    window_size: int,
    stride: int,
) -> tuple[list[DetectionOutput], np.ndarray]:
    preds = [
        json.loads(line)
        for line in resolve_window_predictions(run_dir).read_text().splitlines()
        if line.strip()
    ]
    wins = build_eval_windows(
        csv_path=csv_path,
        window_size=window_size,
        stride=stride,
        down_len=1,
        normalize_mode="none",
        normalize_fit_csv=fit_csv,
    )
    by = {w.start_idx: w for w in wins}
    gt = pd.read_csv(csv_path)["Attack"].to_numpy(dtype=np.int32)
    outs: list[DetectionOutput] = []
    for r in preds:
        w = by.get(r["start_idx"])
        if w is None:
            continue
        ranges = r.get("anomaly_ranges_local") or []
        outs.append(
            DetectionOutput(
                window_id=int(r.get("window_id") or 0),
                start_idx=int(r["start_idx"]),
                end_idx=int(r.get("end_idx") or w.end_idx),
                anomaly_ranges_local=ranges,
                anomaly_score=float(r.get("anomaly_score") or (0.85 if ranges else 0.0)),
                rationale="",
                raw_response="",
                parsed_ok=True,
            )
        )
    return outs, gt


def _smd_pred_with_post(outs: list[DetectionOutput], gt: np.ndarray, post: dict) -> np.ndarray:
    pred = _pred_with_post(outs, len(gt), post)
    return _filter_events(
        pred,
        min_pred_len=int(post.get("min_pred_len") or 0),
        max_events=int(post.get("max_events") or 0),
    )


def _weighted_mean(vals: list[float], weights: list[float]) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vals, dtype=float)
    return float(np.sum(v * w) / max(1e-12, np.sum(w)))


def _resolve_smd_entity_dirs(tag: str) -> dict[str, Path]:
    dirs = {e: RUNS_ROOT / "smd" / tag / e for e in SMD_ENTITIES}
    missing = []
    for e, p in dirs.items():
        try:
            resolve_window_predictions(p)
        except FileNotFoundError:
            missing.append(e)
    if missing:
        raise FileNotFoundError(f"SMD score missing entities for tag={tag}: {missing}")
    return dirs


def score_smd(*, tag: str | None = None, entity_dirs: dict[str, Path] | None = None) -> dict[str, Any]:
    if entity_dirs is None:
        if not tag:
            raise FileNotFoundError("SMD score requires --tag <tag> (dirs under runs/smd/<tag>/).")
        dirs = _resolve_smd_entity_dirs(tag)
    else:
        dirs = entity_dirs
    per: dict[str, Any] = {}
    for entity, pe in SMD_PER_ENTITY.items():
        run_dir = dirs[entity]
        ov = pe["overrides"]
        outs, gt = _smd_load_outputs(
            run_dir,
            csv_path=str(PROJECT_ROOT / ov["csv_path"]),
            fit_csv=str(PROJECT_ROOT / ov["normalize_fit_csv"]),
            window_size=int(ov["window_size"]),
            stride=int(ov["stride"]),
        )
        posts = pe["aggregation_per_metric"]
        m_pt = _metric_bundle(gt, _smd_pred_with_post(outs, gt, posts["point"]))
        m_pa = _metric_bundle(gt, _smd_pred_with_post(outs, gt, posts["pa"]))
        m_af = _metric_bundle(gt, _smd_pred_with_post(outs, gt, posts["affi"]))
        row = {
            "point_f1": m_pt["point_f1"],
            "pa_f1": m_pa["pa_f1"],
            "affi_f1": m_af["affi_f1"],
            "avg_f1": (m_pt["point_f1"] + m_pa["pa_f1"] + m_af["affi_f1"]) / 3.0,
            "n_points": int(len(gt)),
            "run_dir": _rel(run_dir),
        }
        per[entity] = row
        _write_score(run_dir, {**row, "aggregation_per_metric": posts})
    machines = list(per.keys())
    weights = [float(per[e]["n_points"]) for e in machines]
    point = _weighted_mean([per[e]["point_f1"] for e in machines], weights)
    pa = _weighted_mean([per[e]["pa_f1"] for e in machines], weights)
    affi = _weighted_mean([per[e]["affi_f1"] for e in machines], weights)
    avg = (point + pa + affi) / 3.0
    scored = {
        "aggregate": {
            "point_f1": point,
            "pa_f1": pa,
            "affi_f1": affi,
            "avg_f1": avg,
            "composite_f1": avg,
            "weights_n_points": weights,
        },
        "per_entity": per,
        "tag": tag,
    }
    summary_dir = RUNS_ROOT / "smd" / (tag or "_manual")
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "champion_score.json").write_text(
        json.dumps(scored, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return scored


def score_psm(run_dir: Path) -> dict[str, Any]:
    rd = Path(run_dir)
    outs = load_window_outputs(resolve_window_predictions(rd))
    gt_path = rd / "gt_point_labels.npy"
    if gt_path.exists():
        gt = np.load(gt_path).astype(np.int32)
    else:
        gt = pd.read_csv(PROJECT_ROOT / PSM_TEST_CSV)["Attack"].to_numpy(dtype=np.int32)
    posts = PSM_CHAMPION_AGGREGATION_PER_METRIC
    m_pt = _metric_bundle(gt, _pred_with_post(outs, len(gt), posts["point"]))
    m_pa = _metric_bundle(gt, _pred_with_post(outs, len(gt), posts["pa"]))
    m_af = _metric_bundle(gt, _pred_with_post(outs, len(gt), posts["affi"]))
    avg = (m_pt["point_f1"] + m_pa["pa_f1"] + m_af["affi_f1"]) / 3.0
    per = {
        "run_dir": _rel(rd),
        "n_windows": len(outs),
        "point_f1": m_pt["point_f1"],
        "point_precision": m_pt["point_precision"],
        "point_recall": m_pt["point_recall"],
        "pa_f1": m_pa["pa_f1"],
        "pa_precision": m_pa["pa_precision"],
        "pa_recall": m_pa["pa_recall"],
        "affi_f1": m_af["affi_f1"],
        "affi_precision": m_af["affi_precision"],
        "affi_recall": m_af["affi_recall"],
        "composite_f1": avg,
        "avg_f1": avg,
        "aggregation_per_metric": posts,
    }
    _write_score(rd, per)
    return per


def _score_after_run(dataset: str, rec: dict[str, Any]) -> dict[str, Any] | None:
    if rec.get("status") != "ok":
        return None
    if dataset == "smd":
        tag = rec.get("tag")
        entity_dirs = {
            e: PROJECT_ROOT / row["output_dir"]
            if not Path(row["output_dir"]).is_absolute()
            else Path(row["output_dir"])
            for e, row in (rec.get("per_entity") or {}).items()
            if row.get("status") == "ok" and row.get("output_dir")
        }
        if len(entity_dirs) != len(SMD_ENTITIES):
            print(f"[WARN] SMD offline score skipped: incomplete entities={list(entity_dirs)}", flush=True)
            return None
        return score_smd(tag=tag, entity_dirs=entity_dirs)
    out = rec.get("output_dir")
    if not out:
        return None
    rd = Path(out) if Path(out).is_absolute() else PROJECT_ROOT / out
    return {"swat": score_swat, "msl": score_msl, "psm": score_psm}[dataset](rd)


def _show_block(variant: str, label: str, payload: dict[str, Any]) -> None:
    print(f"variant={variant}")
    print(label)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def show_swat() -> None:
    _show_block(
        SWAT_CHAMPION_VARIANT_KEY,
        SWAT_CHAMPION_LABEL,
        {
            "overrides": SWAT_CHAMPION_OVERRIDES,
            "aggregation_per_metric": SWAT_CHAMPION_AGGREGATION_PER_METRIC,
            "full_windows": SWAT_FULL_WINDOWS,
        },
    )


def show_msl() -> None:
    _show_block(
        MSL_CHAMPION_VARIANT_KEY,
        MSL_CHAMPION_LABEL,
        {
            "overrides": MSL_CHAMPION_OVERRIDES,
            "precise_hit_post": MSL_CHAMPION_PRECISE_HIT_POST,
            "aggregation_per_metric": MSL_CHAMPION_AGGREGATION_PER_METRIC,
            "full_windows": MSL_FULL_WINDOWS,
        },
    )


def show_smd() -> None:
    entities = {
        e: {"overrides": pe["overrides"], "aggregation_per_metric": pe["aggregation_per_metric"]}
        for e, pe in SMD_PER_ENTITY.items()
    }
    _show_block(
        SMD_CHAMPION_VARIANT_KEY,
        SMD_CHAMPION_LABEL,
        {"locked_stack": SMD_LOCKED_STACK, "per_entity": entities},
    )


def show_psm() -> None:
    _show_block(
        PSM_CHAMPION_VARIANT_KEY,
        PSM_CHAMPION_LABEL,
        {
            "overrides": PSM_CHAMPION_OVERRIDES,
            "aggregation_per_metric": PSM_CHAMPION_AGGREGATION_PER_METRIC,
            "full_windows": PSM_FULL_WINDOWS,
        },
    )


def run_swat(*, concurrency: int, dry_run: bool, tag: str) -> dict:
    ov = deepcopy(SWAT_CHAMPION_OVERRIDES)
    base = apply_overrides(_base_exp(normalize_values="minmax", concurrency=concurrency), ov)
    cfg = deepcopy(base["config"])
    cfg["concurrency"] = concurrency
    cfg["max_windows"] = 0
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_ROOT / "swat" / tag
    logs = RUNS_ROOT / "logs" / "swat" / tag
    print(f"[RUN] {SWAT_CHAMPION_VARIANT_KEY}: {SWAT_CHAMPION_LABEL}", flush=True)
    if dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return {"status": "dry_run", "config": cfg, "output_dir": str(out_dir.relative_to(PROJECT_ROOT))}
    return run_TriLAD(cfg, out_dir, logs, log_name="swat.log")


def run_msl(*, concurrency: int, dry_run: bool, tag: str) -> dict:
    ov = deepcopy(MSL_CHAMPION_OVERRIDES)
    base = apply_overrides(_base_exp(normalize_values="none", concurrency=concurrency), ov)
    cfg = deepcopy(base["config"])
    cfg["concurrency"] = concurrency
    cfg["max_windows"] = 0
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_ROOT / "msl" / tag
    logs = RUNS_ROOT / "logs" / "msl" / tag
    print(f"[RUN] {MSL_CHAMPION_VARIANT_KEY}: {MSL_CHAMPION_LABEL}", flush=True)
    if dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return {"status": "dry_run", "config": cfg, "output_dir": str(out_dir.relative_to(PROJECT_ROOT))}
    return run_TriLAD(cfg, out_dir, logs, log_name="msl.log")


def _build_smd_entity_cfg(entity: str, concurrency: int) -> dict[str, Any]:
    pe = SMD_PER_ENTITY[entity]
    cfg = {**SMD_LOCKED_STACK, **pe["overrides"]}
    cfg["concurrency"] = int(concurrency)
    cfg["max_windows"] = 0
    for k in ("csv_path", "normalize_fit_csv"):
        p = Path(cfg[k])
        if not p.is_absolute():
            cfg[k] = str(PROJECT_ROOT / p)
    return cfg


def _run_smd_entity(entity: str, *, tag: str, concurrency: int, logs_dir: Path) -> dict[str, Any]:
    cfg = _build_smd_entity_cfg(entity, concurrency)
    out_dir = RUNS_ROOT / "smd" / tag / entity
    return {"entity": entity, **run_TriLAD(cfg, out_dir, logs_dir, log_name=f"{entity}.log")}


def run_smd(*, concurrency: int, dry_run: bool, tag: str) -> dict[str, Any]:
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    entities = list(SMD_PER_ENTITY.keys())
    logs_dir = RUNS_ROOT / "logs" / "smd" / tag
    print(f"[RUN] SMD parallel entities={len(entities)} concurrency/job={concurrency} tag={tag}", flush=True)
    if dry_run:
        for e in entities:
            print(f"--- {e} ---")
            print(json.dumps(_build_smd_entity_cfg(e, concurrency), ensure_ascii=False, indent=2))
        return {"status": "dry_run"}
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(entities)) as pool:
        futs = {
            pool.submit(_run_smd_entity, e, tag=tag, concurrency=concurrency, logs_dir=logs_dir): e
            for e in entities
        }
        for fut in as_completed(futs):
            row = fut.result()
            results[row["entity"]] = row
            print(f"[DONE] {row['entity']} status={row.get('status')} dir={row.get('output_dir')}", flush=True)
    return {"tag": tag, "per_entity": results}


def run_psm(*, concurrency: int, dry_run: bool, tag: str) -> dict:
    ov = deepcopy(PSM_CHAMPION_OVERRIDES)
    base = apply_overrides(_base_exp(normalize_values="none", concurrency=concurrency), ov)
    cfg = deepcopy(base["config"])
    cfg["concurrency"] = concurrency
    cfg["max_windows"] = 0
    for k in ("csv_path", "normalize_fit_csv"):
        p = Path(cfg[k])
        if not p.is_absolute():
            cfg[k] = str(PROJECT_ROOT / p)
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_ROOT / "psm" / tag
    logs = RUNS_ROOT / "logs" / "psm" / tag
    print(f"[RUN] {PSM_CHAMPION_VARIANT_KEY}: {PSM_CHAMPION_LABEL}", flush=True)
    if dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return {"status": "dry_run", "config": cfg, "output_dir": str(out_dir.relative_to(PROJECT_ROOT))}
    return run_TriLAD(cfg, out_dir, logs, log_name="psm.log")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM-AD champion reproduction (SWaT / MSL / SMD / PSM)")
    p.add_argument("dataset", choices=["swat", "msl", "smd", "psm"])
    p.add_argument(
        "action",
        choices=["show", "run", "score"],
        help="show=print config; run=fresh API inference + offline score; score=offline only",
    )
    p.add_argument("--concurrency", type=int, default=8, help="API concurrency (default: 8 for all datasets)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--tag",
        default="",
        help="Run tag: run writes runs/<dataset>/<tag>/; score resolves the same path",
    )
    p.add_argument(
        "--run_dir",
        default="",
        help="Existing run dir for score (SWaT/MSL/PSM); alternative to --tag",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.action == "show":
        {"swat": show_swat, "msl": show_msl, "smd": show_smd, "psm": show_psm}[args.dataset]()
        return

    if args.action == "score":
        if args.dataset == "smd":
            if not args.tag:
                raise SystemExit("smd score requires --tag <tag> (dirs under runs/smd/<tag>/)")
            scored = score_smd(tag=args.tag)
            a = scored["aggregate"]
            print(
                f"[SMD score] P={a['point_f1']:.4f} PA={a['pa_f1']:.4f} "
                f"Affi={a['affi_f1']:.4f} avg={a['avg_f1']:.4f}",
                flush=True,
            )
            for e, pe in scored["per_entity"].items():
                print(
                    f"  {e}: P={pe['point_f1']:.4f} PA={pe['pa_f1']:.4f} "
                    f"Affi={pe['affi_f1']:.4f} avg={pe['avg_f1']:.4f} dir={pe['run_dir']}",
                    flush=True,
                )
            return
        if not args.run_dir and not args.tag:
            raise SystemExit(
                f"{args.dataset} score requires --run_dir <path> or --tag <tag> "
                f"(runs/{args.dataset}/<tag>/)"
            )
        rd = _resolve_run_dir(args.dataset, args.run_dir or None, args.tag or None)
        scored = {"swat": score_swat, "msl": score_msl, "psm": score_psm}[args.dataset](rd)
        _print_metrics(f"[{args.dataset.upper()} score] run_dir={scored.get('run_dir')}", scored)
        return

    kwargs = dict(concurrency=args.concurrency, dry_run=args.dry_run, tag=args.tag)
    if args.dataset == "swat":
        rec = run_swat(**kwargs)
    elif args.dataset == "msl":
        rec = run_msl(**kwargs)
    elif args.dataset == "psm":
        rec = run_psm(**kwargs)
    else:
        rec = run_smd(**kwargs)
    print(
        f"[DONE] {json.dumps({k: rec.get(k) for k in ('status', 'output_dir', 'tag') if k in rec}, ensure_ascii=False)}",
        flush=True,
    )
    if not args.dry_run:
        scored = _score_after_run(args.dataset, rec)
        if scored is None:
            return
        if args.dataset == "smd":
            a = scored["aggregate"]
            _print_metrics("[SMD offline score]", a)
        else:
            _print_metrics(f"[{args.dataset.upper()} offline score]", scored)


if __name__ == "__main__":
    main()
