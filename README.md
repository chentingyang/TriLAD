# TriLAD

This is the source code of "TriLAD: Multimodal LLM with Tri-Layer Retrieval for Multivariate Time Series Anomaly Detection".

## Run

Inference entry: [`triLAD.py`](triLAD.py).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/ahstat/affiliation-metrics-py.git"
cp .env.example .env   # set OPENROUTER_API_KEY
```

```bash
python triLAD.py \
  --csv_path dataset/SWaT/SWaT_Dataset_attack_example.csv \
  --normalize_fit_csv dataset/SWaT/SWaT_Dataset_normal_example.csv \
  --normalize_values minmax \
  --output_dir runs/adhoc
```

`--csv_path` is the test/attack series. `--normalize_fit_csv` is the training/normal series: it fits the scaler when normalization is on, and it is the default source of normal few-shot windows. The default model is `qwen/qwen3-vl-30b-a3b-instruct`. Anomaly-pool windows are kept out of inference and out of the metrics. Results are written under `--output_dir`.

## Verifiable outputs

[`inference_outputs/`](inference_outputs) holds frozen inference outputs that can be re-scored without calling the model.

```
inference_outputs/
├── swat/
├── msl/
├── psm/
├── smd/machine-1-1/
├── smd/machine-2-1/
├── smd/machine-3-2/
├── smd/machine-3-7/
└── validation_results.json
```

Each dataset directory contains:

- `window_predictions_pre_agg.jsonl` — per-window model output before aggregation
- `gt_point_labels.npy` — point-level ground truth
- `pred_point_labels.npy` — point-level prediction after aggregation

Re-score all four datasets:

```bash
python fast_validation.py
```

The command prints Point / PA / Affiliation precision, recall, and F1, and writes `inference_outputs/validation_results.json`.

## Champion configs

Champion settings for SWaT, MSL, SMD, and PSM live in [`scripts/champion.py`](scripts/champion.py). Print a config without calling the API:

```bash
python scripts/champion.py swat show
python scripts/champion.py msl show
python scripts/champion.py smd show
python scripts/champion.py psm show
```

| Dataset | Config symbols |
|---------|----------------|
| SWaT | `SWAT_CHAMPION_OVERRIDES`, `SWAT_CHAMPION_AGGREGATION_PER_METRIC` |
| MSL | `MSL_CHAMPION_OVERRIDES`, `MSL_CHAMPION_PRECISE_HIT_POST`, `MSL_CHAMPION_AGGREGATION_PER_METRIC` |
| SMD | `SMD_LOCKED_STACK`, `SMD_PER_ENTITY` |
| PSM | `PSM_CHAMPION_OVERRIDES`, `PSM_CHAMPION_AGGREGATION_PER_METRIC` |

`show` prints those blocks. `run` starts a fresh API inference through `triLAD.py` and then scores offline. `score` re-scores an existing run and does not call the API.

```bash
python scripts/champion.py swat run --concurrency 8
python scripts/champion.py msl score --tag <tag>
python scripts/champion.py smd score --tag <tag>
```

For the full version of each dataset, contact tingyangchen@hust.edu.cn.
