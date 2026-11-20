# inference_outputs

Frozen **raw inference** artifacts.

Each directory contains:
- `window_predictions_pre_agg.jsonl` — per-window model raw output (**before** window→point aggregation)
- `gt_point_labels.npy` — point-level ground truth
- `pred_point_labels.npy` — optional point-level prediction (**after** aggregation; written by `fast_validation.py`)

Offline re-score: `python3 fast_validation.py`.
