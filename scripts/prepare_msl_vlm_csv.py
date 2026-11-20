#!/usr/bin/env python3
"""Convert MSL npy/pkl under dataset/MSL_v2 into TriLAD wide CSV (+ minmax)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from prepare_msl_data import prepare_msl_npy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "dataset" / "MSL_v2"


def _load_arrays(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npy_ok = all(
        (data_root / name).exists()
        for name in ("MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy")
    )
    if not npy_ok:
        prepare_msl_npy(data_root, data_root)
    train = np.nan_to_num(np.load(data_root / "MSL_train.npy"), nan=0.0).astype(np.float32)
    test = np.nan_to_num(np.load(data_root / "MSL_test.npy"), nan=0.0).astype(np.float32)
    labels = (
        np.nan_to_num(np.load(data_root / "MSL_test_label.npy"), nan=0.0)
        .astype(np.int32)
        .reshape(-1)
    )
    return train, test, labels


def _minmax_fit_transform(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train).astype(np.float32)
    test_s = scaler.transform(test).astype(np.float32)
    return train_s, test_s


def prepare_msl_vlm_csv(
    data_root: str | Path,
    out_root: str | Path | None = None,
) -> dict[str, str | int | float]:
    data_root = Path(data_root)
    out_root = Path(out_root or data_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train, test, test_label = _load_arrays(data_root)
    train, test = _minmax_fit_transform(train, test)
    n_feat = train.shape[1]
    feature_cols = [f"var_{i}" for i in range(n_feat)]

    attack_labels = np.concatenate([np.zeros(len(train), dtype=np.int32), test_label])
    attack_wide = np.vstack([train, test])
    attack_df = pd.DataFrame(attack_wide, columns=feature_cols)
    attack_df.insert(0, "Timestamp", np.arange(len(attack_df), dtype=np.int64))
    attack_df["Attack"] = attack_labels.astype(np.float32)

    normal_df = pd.DataFrame(train, columns=feature_cols)
    normal_df.insert(0, "Timestamp", np.arange(len(normal_df), dtype=np.int64))
    normal_df["Attack"] = 0.0

    attack_path = out_root / "MSL_Dataset_attack.csv"
    normal_path = out_root / "MSL_Dataset_normal.csv"
    test_path = out_root / "MSL_Dataset_test.csv"
    attack_df.to_csv(attack_path, index=False)
    normal_df.to_csv(normal_path, index=False)

    test_df = pd.DataFrame(test, columns=feature_cols)
    test_df.insert(0, "Timestamp", np.arange(len(test_df), dtype=np.int64))
    test_df["Attack"] = test_label.astype(np.float32)
    test_df.to_csv(test_path, index=False)

    test_windows = max(0, (len(test) - 400) // 400 + 1) if len(test) >= 400 else 0

    meta = {
        "n_features": int(n_feat),
        "train_points": int(len(train)),
        "test_points": int(len(test)),
        "attack_points": int(len(attack_df)),
        "label_rate_test": float(np.mean(test_label > 0)),
        "normalize": "minmax_fit_train",
        "attack_csv": str(attack_path),
        "normal_csv": str(normal_path),
        "test_csv": str(test_path),
        "n_windows_win400_stride400_test_only": int(test_windows),
    }
    (out_root / "MSL_vlm_meta.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in meta.items()) + "\n",
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MSL wide CSV for TriLAD")
    parser.add_argument("--data_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out_root", default="", help="defaults to data_root")
    args = parser.parse_args()
    meta = prepare_msl_vlm_csv(args.data_root, args.out_root or args.data_root)
    print("Prepared MSL VLM CSV:", meta)


if __name__ == "__main__":
    main()
