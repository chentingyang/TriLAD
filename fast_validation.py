#!/usr/bin/env python3
"""Fast offline validation on frozen champion inference_outputs.

Reads window_predictions_pre_agg.jsonl + gt_point_labels.npy, applies frozen champion
per-metric aggregation, writes pred_point_labels.npy and prints Point / PA / Affi
precision, recall, F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import _bootstrap_trilad as _bootstrap

_bootstrap.bootstrap()

from champion import (  # noqa: E402
    SMD_ENTITIES,
    SMD_PER_ENTITY,
    SWAT_CHAMPION_AGGREGATION_PER_METRIC,
    load_window_outputs,
    msl_load_refined_outputs,
    resolve_window_predictions,
    _msl_post_cfg,
    _pred_with_post,
    _smd_load_outputs,
    _smd_pred_with_post,
    _weighted_mean,
)
from utils.metrics import affiliation_f1, point_adjust_f1, pointwise_f1  # noqa: E402

DEFAULT_OUTPUTS = PROJECT_ROOT / "inference_outputs"


def _prf(gt: np.ndarray, pred: np.ndarray, metric: str) -> dict[str, float]:
    if metric == "point":
        row = pointwise_f1(gt, pred)
    elif metric == "pa":
        row = point_adjust_f1(gt, pred)
    else:
        row = affiliation_f1(gt, pred)
    return {"precision": float(row["precision"]), "recall": float(row["recall"]), "f1": float(row["f1"])}


def _metrics_per_kind(gt: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {kind: _prf(gt, preds[kind], kind) for kind in ("point", "pa", "affi")}


def _avg_f1(metrics: dict[str, dict[str, float]]) -> float:
    return (metrics["point"]["f1"] + metrics["pa"]["f1"] + metrics["affi"]["f1"]) / 3.0


def _save_pred(run_dir: Path, pred: np.ndarray, *, write: bool) -> None:
    if not write:
        return
    np.save(run_dir / "pred_point_labels.npy", pred.astype(np.int32))


def validate_swat(run_dir: Path, *, write_preds: bool) -> dict[str, Any]:
    outs = load_window_outputs(resolve_window_predictions(run_dir))
    gt = np.load(run_dir / "gt_point_labels.npy").astype(np.int32)
    posts = SWAT_CHAMPION_AGGREGATION_PER_METRIC
    preds = {k: _pred_with_post(outs, len(gt), posts[k]) for k in ("point", "pa", "affi")}
    metrics = _metrics_per_kind(gt, preds)
    _save_pred(run_dir, preds["point"], write=write_preds)
    avg_f1 = _avg_f1(metrics)
    return {
        "dataset": "SWaT",
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "n_points": int(len(gt)),
        "n_windows": len(outs),
        "metrics": metrics,
        "avg_f1": avg_f1,
    }


def validate_psm(run_dir: Path, *, write_preds: bool) -> dict[str, Any]:
    from champion import PSM_CHAMPION_AGGREGATION_PER_METRIC  # noqa: WPS433

    outs = load_window_outputs(resolve_window_predictions(run_dir))
    gt = np.load(run_dir / "gt_point_labels.npy").astype(np.int32)
    posts = PSM_CHAMPION_AGGREGATION_PER_METRIC
    preds = {k: _pred_with_post(outs, len(gt), posts[k]) for k in ("point", "pa", "affi")}
    metrics = _metrics_per_kind(gt, preds)
    _save_pred(run_dir, preds["point"], write=write_preds)
    avg_f1 = _avg_f1(metrics)
    return {
        "dataset": "PSM",
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "n_points": int(len(gt)),
        "n_windows": len(outs),
        "metrics": metrics,
        "avg_f1": avg_f1,
    }


def validate_msl(run_dir: Path, *, write_preds: bool) -> dict[str, Any]:
    from champion import MSL_CHAMPION_AGGREGATION_PER_METRIC  # noqa: WPS433

    outs, gt = msl_load_refined_outputs(run_dir, post=_msl_post_cfg())
    posts = MSL_CHAMPION_AGGREGATION_PER_METRIC
    preds = {k: _pred_with_post(outs, len(gt), posts[k]) for k in ("point", "pa", "affi")}
    metrics = _metrics_per_kind(gt, preds)
    _save_pred(run_dir, preds["point"], write=write_preds)
    avg_f1 = _avg_f1(metrics)
    return {
        "dataset": "MSL",
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "n_points": int(len(gt)),
        "n_windows": len(outs),
        "metrics": metrics,
        "avg_f1": avg_f1,
    }


def validate_smd_entity(entity: str, run_dir: Path, *, write_preds: bool) -> dict[str, Any]:
    pe = SMD_PER_ENTITY[entity]
    ov = pe["overrides"]
    outs, gt = _smd_load_outputs(
        run_dir,
        csv_path=str(PROJECT_ROOT / ov["csv_path"]),
        fit_csv=str(PROJECT_ROOT / ov["normalize_fit_csv"]),
        window_size=int(ov["window_size"]),
        stride=int(ov["stride"]),
    )
    posts = pe["aggregation_per_metric"]
    preds = {k: _smd_pred_with_post(outs, gt, posts[k]) for k in ("point", "pa", "affi")}
    metrics = _metrics_per_kind(gt, preds)
    _save_pred(run_dir, preds["point"], write=write_preds)
    avg_f1 = _avg_f1(metrics)
    return {
        "dataset": "SMD",
        "entity": entity,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "n_points": int(len(gt)),
        "n_windows": len(outs),
        "metrics": metrics,
        "avg_f1": avg_f1,
    }


def validate_smd(smd_root: Path, *, write_preds: bool) -> dict[str, Any]:
    per_entity: dict[str, Any] = {}
    for entity in SMD_ENTITIES:
        run_dir = smd_root / entity
        per_entity[entity] = validate_smd_entity(entity, run_dir, write_preds=write_preds)
    weights = [float(per_entity[e]["n_points"]) for e in SMD_ENTITIES]
    aggregate_metrics: dict[str, dict[str, float]] = {}
    for kind in ("point", "pa", "affi"):
        aggregate_metrics[kind] = {
            "precision": _weighted_mean([per_entity[e]["metrics"][kind]["precision"] for e in SMD_ENTITIES], weights),
            "recall": _weighted_mean([per_entity[e]["metrics"][kind]["recall"] for e in SMD_ENTITIES], weights),
            "f1": _weighted_mean([per_entity[e]["metrics"][kind]["f1"] for e in SMD_ENTITIES], weights),
        }
    avg_f1 = (aggregate_metrics["point"]["f1"] + aggregate_metrics["pa"]["f1"] + aggregate_metrics["affi"]["f1"]) / 3.0
    return {
        "dataset": "SMD",
        "run_dir": str(smd_root.relative_to(PROJECT_ROOT)),
        "per_entity": per_entity,
        "metrics": aggregate_metrics,
        "avg_f1": avg_f1,
        "weights_n_points": weights,
    }


def _print_row(label: str, metrics: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'metric':<8} {'precision':>10} {'recall':>10} {'f1':>10}")
    for name in ("point", "pa", "affi"):
        m = metrics[name]
        print(f"{name:<8} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1']:10.4f}")
    avg = (metrics["point"]["f1"] + metrics["pa"]["f1"] + metrics["affi"]["f1"]) / 3.0
    print(f"avg_f1: {avg:.4f}")


def run_all(outputs_root: Path, *, write_preds: bool, datasets: set[str] | None) -> dict[str, Any]:
    want = datasets or {"swat", "psm", "msl", "smd"}
    results: list[dict[str, Any]] = []

    if "swat" in want and (outputs_root / "swat").is_dir():
        try:
            resolve_window_predictions(outputs_root / "swat")
        except FileNotFoundError:
            pass
        else:
            row = validate_swat(outputs_root / "swat", write_preds=write_preds)
            results.append(row)
            _print_row("SWaT", row["metrics"])

    if "psm" in want and (outputs_root / "psm").is_dir():
        try:
            resolve_window_predictions(outputs_root / "psm")
        except FileNotFoundError:
            pass
        else:
            row = validate_psm(outputs_root / "psm", write_preds=write_preds)
            results.append(row)
            _print_row("PSM", row["metrics"])

    if "msl" in want and (outputs_root / "msl").is_dir():
        try:
            resolve_window_predictions(outputs_root / "msl")
        except FileNotFoundError:
            pass
        else:
            row = validate_msl(outputs_root / "msl", write_preds=write_preds)
            results.append(row)
            _print_row("MSL", row["metrics"])

    if "smd" in want and (outputs_root / "smd").is_dir():
        row = validate_smd(outputs_root / "smd", write_preds=write_preds)
        results.append(row)
        _print_row("SMD (n_points weighted)", row["metrics"])
        for entity in SMD_ENTITIES:
            _print_row(f"SMD {entity}", row["per_entity"][entity]["metrics"])

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "outputs_root": str(outputs_root.relative_to(PROJECT_ROOT)),
        "write_preds": write_preds,
        "results": results,
    }
    out_json = outputs_root / "validation_results.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[WRITE] {out_json.relative_to(PROJECT_ROOT)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast offline champion validation on inference_outputs")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS,
        help="Root with swat/psm/msl/smd subdirs (default: inference_outputs)",
    )
    parser.add_argument(
        "--dataset",
        choices=["swat", "psm", "msl", "smd"],
        nargs="*",
        help="Validate only selected dataset(s)",
    )
    parser.add_argument(
        "--no-write-preds",
        action="store_true",
        help="Skip writing pred_point_labels.npy",
    )
    args = parser.parse_args()
    outputs_root = args.outputs_dir if args.outputs_dir.is_absolute() else PROJECT_ROOT / args.outputs_dir
    if not outputs_root.is_dir():
        raise SystemExit(f"Missing outputs dir: {outputs_root}")
    ds = set(args.dataset) if args.dataset else None
    run_all(outputs_root, write_preds=not args.no_write_preds, datasets=ds)


if __name__ == "__main__":
    main()
