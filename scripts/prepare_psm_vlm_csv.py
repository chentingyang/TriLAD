#!/usr/bin/env python3
"""Export PSM train/test CSV under dataset/PSM → TriLAD wide CSV (+ minmax)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT_ROOT / "dataset" / "PSM"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "PSM" / "vlm"


def _feat_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("feature_")]
    if cols:
        return cols
    return [
        c
        for c in df.columns
        if c.lower() not in {"timestamp_(min)", "timestamp", "time", "label"}
    ]


def _write_csv(values: np.ndarray, labels: np.ndarray, path: Path) -> None:
    n_feat = values.shape[1]
    cols = [f"var_{i}" for i in range(n_feat)]
    df = pd.DataFrame(values.astype(np.float32), columns=cols)
    df.insert(0, "Timestamp", np.arange(len(df), dtype=np.int64))
    df["Attack"] = labels.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_psm(src: Path, out_dir: Path) -> dict:
    train_df = pd.read_csv(src / "train.csv")
    test_df = pd.read_csv(src / "test.csv")
    label_df = pd.read_csv(src / "test_label.csv")
    feat = _feat_cols(train_df)
    train = np.nan_to_num(train_df[feat].to_numpy(dtype=np.float32), nan=0.0)
    test = np.nan_to_num(test_df[feat].to_numpy(dtype=np.float32), nan=0.0)
    labels = np.nan_to_num(label_df["label"].to_numpy(dtype=np.float32), nan=0.0).reshape(-1)
    if len(labels) != len(test):
        n = min(len(labels), len(test))
        labels = labels[:n]
        test = test[:n]
    labels = labels.astype(np.int32)

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(train, np.zeros(len(train), dtype=np.int32), out_dir / "PSM_Dataset_normal.csv")
    _write_csv(test, labels, out_dir / "PSM_Dataset_test.csv")
    attack = np.vstack([train, test])
    attack_lab = np.concatenate([np.zeros(len(train), dtype=np.int32), labels])
    _write_csv(attack, attack_lab, out_dir / "PSM_Dataset_attack.csv")

    segs = int((np.diff(np.concatenate([[0], (labels > 0).astype(int), [0]])) == 1).sum())
    meta = {
        "dataset": "PSM",
        "n_features": int(train.shape[1]),
        "train_points": int(len(train)),
        "test_points": int(len(test)),
        "label_rate": float(labels.mean()),
        "n_anom_segments": segs,
        "value_range": [float(test.min()), float(test.max())],
        "normalize": "minmax_fit_train",
        "note": "CSV values minmax-scaled on train; TriLAD input schema Timestamp+var_*+Attack",
        "test_csv": str(out_dir / "PSM_Dataset_test.csv"),
        "normal_csv": str(out_dir / "PSM_Dataset_normal.csv"),
        "attack_csv": str(out_dir / "PSM_Dataset_attack.csv"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare PSM wide CSV for TriLAD")
    p.add_argument("--src", default=str(DEFAULT_SRC), help="Dir with train.csv/test.csv/test_label.csv")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = p.parse_args()
    meta = export_psm(Path(args.src), Path(args.out_dir))
    print(
        f"PSM VLM CSV: train={meta['train_points']} test={meta['test_points']} "
        f"C={meta['n_features']} rate={meta['label_rate']:.4f} segs={meta['n_anom_segments']}"
    )
    print(f"Wrote under {args.out_dir}")


if __name__ == "__main__":
    main()
