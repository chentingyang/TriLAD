#!/usr/bin/env python3
"""Convert MSL pkl files under dataset/ into npy arrays used by prepare_msl_vlm_csv."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "dataset" / "MSL_v2"


def prepare_msl_npy(
    data_root: str | Path,
    out_root: str | Path | None = None,
) -> dict[str, int | float]:
    data_root = Path(data_root)
    out_root = Path(out_root or data_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train = np.asarray(pickle.load(open(data_root / "MSL_train.pkl", "rb")), dtype=np.float32)
    test = np.asarray(pickle.load(open(data_root / "MSL_test.pkl", "rb")), dtype=np.float32)
    labels = np.asarray(
        pickle.load(open(data_root / "MSL_test_label.pkl", "rb")), dtype=np.int32
    ).reshape(-1)

    train = np.nan_to_num(train, nan=0.0)
    test = np.nan_to_num(test, nan=0.0)
    labels = np.nan_to_num(labels, nan=0.0).astype(np.int32)

    if train.shape[1] != test.shape[1]:
        raise ValueError(f"train/test feature mismatch: {train.shape} vs {test.shape}")
    if len(labels) != len(test):
        raise ValueError(f"label/test length mismatch: {len(labels)} vs {len(test)}")

    np.save(out_root / "MSL_train.npy", train)
    np.save(out_root / "MSL_test.npy", test)
    np.save(out_root / "MSL_test_label.npy", labels)

    meta = {
        "n_features": int(train.shape[1]),
        "train_points": int(len(train)),
        "test_points": int(len(test)),
        "label_rate": float(np.mean(labels > 0)),
    }
    (out_root / "meta.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in meta.items()) + "\n",
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="MSL pkl → npy under dataset/MSL_v2")
    parser.add_argument("--data_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out_root", default="", help="defaults to data_root")
    args = parser.parse_args()
    meta = prepare_msl_npy(args.data_root, args.out_root or args.data_root)
    print("Prepared MSL npy data:", meta)


if __name__ == "__main__":
    main()
