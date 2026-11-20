from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from utils.data_pipeline import window_to_image_b64_swat_style
from TriLAD.feature_selection import (
    FeatureSelectionResult,
    select_features,
    subset_window_values,
)
from TriLAD.schemas import WindowSample


@dataclass
class VisionEncoderConfig:
    feature_select_mode: str = "none"
    feature_select_k: int = 16
    feature_select_seed: int = 0
    use_fixed_7x7_grid: bool = False
    subplot_titles: bool = True
    dpi: int = 120
    y_margin_ratio: float = 0.15
    show_attack_points: bool = False
    attack_shade_color: str = "red"
    attack_shade_alpha: float = 0.22
    baseline_mean: np.ndarray | None = None
    baseline_std: np.ndarray | None = None


def encode_window_image_b64(
    sample: WindowSample,
    config: VisionEncoderConfig,
    *,
    force_feature_indices: list[int] | None = None,
    force_show_attack_points: bool | None = None,
) -> tuple[str, list[int]]:
    """
    Render one window to base64 PNG.

    When force_feature_indices is set, skip per-window MAD and plot exactly those columns
    (used for few-shot / target vision feature alignment).
    """
    if force_feature_indices is not None:
        indices = list(force_feature_indices)
    else:
        window_data = np.asarray(sample.values, dtype=np.float64)
        sel = select_features(
            window_data,
            method=config.feature_select_mode,
            k=config.feature_select_k,
            seed=config.feature_select_seed + sample.window_id,
            baseline_mean=config.baseline_mean,
            baseline_std=config.baseline_std,
            feature_names=list(sample.feature_names),
        )
        indices = sel.indices

    fixed_grid = (7, 7) if config.use_fixed_7x7_grid else None
    show_attack = (
        config.show_attack_points
        if force_show_attack_points is None
        else force_show_attack_points
    )
    b64 = window_to_image_b64_swat_style(
        sample,
        dpi=config.dpi,
        y_margin_ratio=config.y_margin_ratio,
        show_attack_points=show_attack,
        attack_shade_color=config.attack_shade_color,
        attack_shade_alpha=config.attack_shade_alpha,
        feature_indices=indices,
        subplot_titles=config.subplot_titles,
        fixed_grid=fixed_grid,
    )
    return b64, indices


def build_vision_image_encoder(
    config: VisionEncoderConfig,
) -> tuple[Callable[..., str], Callable[[WindowSample], FeatureSelectionResult]]:
    """Returns (encode_image_b64, select_for_window). encode_image_b64 accepts force_feature_indices=."""

    def select_for_window(sample: WindowSample) -> FeatureSelectionResult:
        window_data = np.asarray(sample.values, dtype=np.float64)
        return select_features(
            window_data,
            method=config.feature_select_mode,
            k=config.feature_select_k,
            seed=config.feature_select_seed + sample.window_id,
            baseline_mean=config.baseline_mean,
            baseline_std=config.baseline_std,
            feature_names=list(sample.feature_names),
        )

    def encode_image_b64(
        sample: WindowSample,
        *,
        force_feature_indices: list[int] | None = None,
        force_show_attack_points: bool | None = None,
    ) -> str:
        b64, _ = encode_window_image_b64(
            sample,
            config,
            force_feature_indices=force_feature_indices,
            force_show_attack_points=force_show_attack_points,
        )
        return b64

    return encode_image_b64, select_for_window


def apply_feature_selection_to_sample(
    sample: WindowSample,
    selection: FeatureSelectionResult,
) -> WindowSample:
    sub_values, sub_names = subset_window_values(
        sample.values,
        sample.feature_names,
        selection.indices,
    )
    return WindowSample(
        window_id=sample.window_id,
        start_idx=sample.start_idx,
        end_idx=sample.end_idx,
        values=sub_values,
        feature_names=sub_names,
        gt_labels=sample.gt_labels,
        image_path=sample.image_path,
    )
