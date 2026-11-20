from __future__ import annotations

import base64
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from TriLAD.feature_selection import compute_grid_shape
from TriLAD.schemas import WindowSample


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace("/", "").replace("_", "").replace(" ", "")


def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"attack", "normalattack", "timestamp"}
    return [c for c in df.columns if normalize_column_name(c) not in excluded]


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    attack_col = None
    for col in df.columns:
        if normalize_column_name(col) in {"attack", "normalattack"}:
            attack_col = col
            break
    if attack_col is None:
        return np.zeros(len(df), dtype=np.int32)
    raw = df[attack_col].astype(str).str.strip().str.lower()
    mapped = raw.map(
        {
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0,
            "attack": 1,
            "normal": 0,
            "true": 1,
            "false": 0,
        }
    )
    numeric = pd.to_numeric(raw, errors="coerce")
    labels = mapped.where(mapped.notna(), (numeric >= 1).astype(np.int32))
    return labels.fillna(0).astype(np.int32).to_numpy()


def load_csv(
    csv_path: str | Path,
    normalize_mode: str = "none",
    normalize_fit_csv: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    feats = pick_feature_columns(df)
    feat_df = df[feats].apply(pd.to_numeric, errors="coerce")
    feat_df = (
        feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    )
    values = feat_df.to_numpy(dtype=np.float32)
    labels = extract_labels(df)
    mode = normalize_mode.strip().lower()
    if mode not in {"none", "minmax", "standard"}:
        raise ValueError("normalize_mode must be one of: none, minmax, standard")
    if mode in {"minmax", "standard"}:
        scaler = MinMaxScaler() if mode == "minmax" else StandardScaler()
        if normalize_fit_csv is not None:
            fit_df = pd.read_csv(Path(normalize_fit_csv))
            if fit_df.shape[1] == 1:
                fit_df = pd.read_csv(Path(normalize_fit_csv), sep=";")
            fit_feat_df = fit_df[feats].apply(pd.to_numeric, errors="coerce")
            fit_feat_df = (
                fit_feat_df.replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            scaler.fit(fit_feat_df.to_numpy(dtype=np.float32))
        else:
            scaler.fit(values)
        values = scaler.transform(values).astype(np.float32)
    return values, labels, feats


def downsample(np_data: np.ndarray, down_len: int, is_label: bool = False) -> np.ndarray:
    if down_len <= 1:
        return np_data
    down_time_len = len(np_data) // down_len
    if down_time_len <= 0:
        raise ValueError(
            f"down_len={down_len} is too large for sequence length={len(np_data)}"
        )
    if is_label:
        d_data = np_data[: down_time_len * down_len].reshape(-1, down_len)
        return np.round(np.max(d_data, axis=1))
    _, col_num = np_data.shape
    data_t = np_data.transpose()
    d_data = data_t[:, : down_time_len * down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)
    return d_data.transpose()


def make_windows(
    values: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    window_size: int = 200,
    stride: int = 100,
) -> list[WindowSample]:
    out: list[WindowSample] = []
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    wid = 0
    for start in range(0, max(0, len(values) - window_size + 1), stride):
        end = start + window_size
        out.append(
            WindowSample(
                window_id=wid,
                start_idx=start,
                end_idx=end,
                values=values[start:end].tolist(),
                feature_names=feature_names,
                gt_labels=labels[start:end].astype(int).tolist(),
            )
        )
        wid += 1
    return out


def _attack_interval_spans(attack_window: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(attack_window)
    while i < n:
        if attack_window[i] == 1:
            j = i + 1
            while j < n and attack_window[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _plot_sample_on_axes_block(
    sample: WindowSample,
    axes_block: np.ndarray,
    plot_indices: list[int],
    *,
    y_margin_ratio: float,
    show_attack_points: bool,
    attack_shade_color: str,
    attack_shade_alpha: float,
    subplot_titles: bool,
) -> None:
    """Fill one (sub_rows, cols) axes block with MAD-selected feature panels."""
    window_data = np.asarray(sample.values, dtype=float)
    labels = (
        np.asarray(sample.gt_labels, dtype=np.int32) if show_attack_points else None
    )
    sub_rows, cols = axes_block.shape
    for panel_idx in range(sub_rows * cols):
        r, c = divmod(panel_idx, cols)
        ax = axes_block[r, c]
        if panel_idx >= len(plot_indices):
            ax.axis("off")
            continue
        feat_idx = plot_indices[panel_idx]
        y = window_data[:, feat_idx]
        if labels is not None:
            for x0, x1 in _attack_interval_spans(labels):
                ax.axvspan(
                    x0,
                    x1,
                    facecolor=attack_shade_color,
                    alpha=attack_shade_alpha,
                    lw=0,
                    zorder=0,
                )
        ax.plot(y, color="steelblue", linewidth=1.2, zorder=2)
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        if np.isclose(y_min, y_max):
            delta = 1.0 if np.isclose(y_min, 0.0) else abs(y_min) * 0.1
            ax.set_ylim(y_min - delta, y_max + delta)
        else:
            pad = (y_max - y_min) * y_margin_ratio
            ax.set_ylim(y_min - pad, y_max + pad)
        if subplot_titles:
            title = sample.feature_names[feat_idx]
            if len(title) > 14:
                title = title[:12] + ".."
            ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])


def window_to_image_b64_swat_style(
    sample: WindowSample,
    dpi: int = 120,
    y_margin_ratio: float = 0.15,
    show_attack_points: bool = False,
    attack_shade_color: str = "red",
    attack_shade_alpha: float = 0.22,
    feature_indices: list[int] | None = None,
    subplot_titles: bool = True,
    fixed_grid: tuple[int, int] | None = None,
    figsize_per_panel: float = 2.0,
) -> str:
    if y_margin_ratio < 0:
        raise ValueError("y_margin_ratio must be >= 0")
    window_data = np.asarray(sample.values, dtype=float)
    plot_indices = (
        list(range(window_data.shape[1]))
        if feature_indices is None
        else list(feature_indices)
    )
    num_panels = len(plot_indices)
    if fixed_grid is None:
        rows, cols = compute_grid_shape(num_panels)
    else:
        rows, cols = fixed_grid
    fig_w = max(6.0, cols * figsize_per_panel)
    fig_h = max(6.0, rows * figsize_per_panel)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.atleast_2d(axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.subplots_adjust(hspace=0.12, wspace=0.1)
    _plot_sample_on_axes_block(
        sample,
        axes,
        plot_indices,
        y_margin_ratio=y_margin_ratio,
        show_attack_points=show_attack_points,
        attack_shade_color=attack_shade_color,
        attack_shade_alpha=attack_shade_alpha,
        subplot_titles=subplot_titles,
    )
    for panel_idx in range(rows * cols):
        if panel_idx >= num_panels:
            r, c = divmod(panel_idx, cols)
            axes[r, c].axis("off")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
