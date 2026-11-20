#!/usr/bin/env python3
"""Export SMD entity npy under dataset/SMD → TriLAD wide CSV (+ minmax)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT_ROOT / "dataset" / "SMD"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "SMD" / "vlm"
DEFAULT_MACHINES = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7"]


def _write_csv(values: np.ndarray, labels: np.ndarray, path: Path) -> None:
    n_feat = values.shape[1]
    cols = [f"var_{i}" for i in range(n_feat)]
    df = pd.DataFrame(values.astype(np.float32), columns=cols)
    df.insert(0, "Timestamp", np.arange(len(df), dtype=np.int64))
    df["Attack"] = labels.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_entity(src_root: Path, entity: str, out_root: Path) -> dict:
    train = np.nan_to_num(np.load(src_root / f"{entity}_train.npy"), nan=0.0).astype(np.float32)
    test = np.nan_to_num(np.load(src_root / f"{entity}_test.npy"), nan=0.0).astype(np.float32)
    labels = np.load(src_root / f"{entity}_labels.npy")
    if labels.ndim == 2:
        labels = (labels > 0).any(axis=1)
    labels = labels.reshape(-1).astype(np.int32)
    if len(labels) != len(test):
        raise ValueError(f"{entity}: label/test length mismatch")

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)

    ent_dir = out_root / entity
    ent_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(train, np.zeros(len(train), dtype=np.int32), ent_dir / "SMD_Dataset_normal.csv")
    _write_csv(test, labels, ent_dir / "SMD_Dataset_test.csv")
    attack = np.vstack([train, test])
    attack_lab = np.concatenate([np.zeros(len(train), dtype=np.int32), labels])
    _write_csv(attack, attack_lab, ent_dir / "SMD_Dataset_attack.csv")

    segs = int((np.diff(np.concatenate([[0], (labels > 0).astype(int), [0]])) == 1).sum())
    meta = {
        "entity": entity,
        "n_features": int(train.shape[1]),
        "train_points": int(len(train)),
        "test_points": int(len(test)),
        "label_rate": float(labels.mean()),
        "n_anom_segments": segs,
        "value_range": [float(test.min()), float(test.max())],
        "normalize": "minmax_fit_train",
        "note": "CSV values minmax-scaled on train; TriLAD input schema Timestamp+var_*+Attack",
        "test_csv": str(ent_dir / "SMD_Dataset_test.csv"),
        "normal_csv": str(ent_dir / "SMD_Dataset_normal.csv"),
        "attack_csv": str(ent_dir / "SMD_Dataset_attack.csv"),
    }
    (ent_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare SMD wide CSV for TriLAD")
    p.add_argument(
        "--src_root",
        default=str(DEFAULT_SRC),
        help="Dir with <entity>_train.npy / _test.npy / _labels.npy",
    )
    p.add_argument("--out_root", default=str(DEFAULT_OUT))
    p.add_argument("--machines", default=",".join(DEFAULT_MACHINES))
    args = p.parse_args()
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    out = Path(args.out_root)
    metas = [export_entity(Path(args.src_root), m, out) for m in machines]
    summary = {"machines": machines, "entities": metas}
    (out / "meta.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for m in metas:
        print(
            f"{m['entity']}: train={m['train_points']} test={m['test_points']} "
            f"rate={m['label_rate']:.4f} segs={m['n_anom_segments']} "
            f"range=[{m['value_range'][0]:.3f},{m['value_range'][1]:.3f}]"
        )
    print(f"Wrote VLM CSVs under {out}")


if __name__ == "__main__":
    main()
