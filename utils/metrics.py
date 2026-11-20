from __future__ import annotations

import numpy as np

try:
    from affiliation.generics import convert_vector_to_events
    from affiliation.metrics import pr_from_events

    _HUET_AVAILABLE = True
except ImportError:
    _HUET_AVAILABLE = False


def pointwise_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def point_adjust_predict(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_arr = y_true.astype(int)
    pred_arr = y_pred.astype(int).copy()
    adjusted = pred_arr.copy()
    anomaly_state = False
    n = len(pred_arr)
    for i in range(n):
        if true_arr[i] == 1 and pred_arr[i] == 1 and not anomaly_state:
            anomaly_state = True
            j = i
            while j >= 0 and true_arr[j] == 1:
                adjusted[j] = 1
                j -= 1
        elif true_arr[i] == 0:
            anomaly_state = False
        if anomaly_state:
            adjusted[i] = 1
    return adjusted.astype(np.int32)


def point_adjust_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return pointwise_f1(y_true.astype(int), point_adjust_predict(y_true, y_pred))


def to_intervals(binary: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 0
    n = len(binary)
    while i < n:
        if binary[i] == 1:
            j = i + 1
            while j < n and binary[j] == 1:
                j += 1
            intervals.append((i, j - 1))
            i = j
        else:
            i += 1
    return intervals


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    inter = max(0, right - left + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def event_iou_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Legacy event-level max-IoU metric (formerly named affiliation_f1)."""
    gt_events = to_intervals(y_true.astype(int))
    pr_events = to_intervals(y_pred.astype(int))
    if len(gt_events) == 0 and len(pr_events) == 0:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    if len(gt_events) == 0 or len(pr_events) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    p_scores = [max(_interval_iou(p, g) for g in gt_events) for p in pr_events]
    r_scores = [max(_interval_iou(g, p) for p in pr_events) for g in gt_events]
    precision = float(np.mean(p_scores)) if p_scores else 0.0
    recall = float(np.mean(r_scores)) if r_scores else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def affiliation_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Official Huet et al. (KDD 2022) affiliation precision/recall/F1.
    Requires package: affiliation-metrics (pip install affiliation-metrics).
    """
    if not _HUET_AVAILABLE:
        raise ImportError(
            "affiliation-metrics is required for official Affiliation-F1. "
            "Install with: pip install affiliation-metrics"
        )
    gt = y_true.astype(int)
    pr = y_pred.astype(int)
    if np.count_nonzero(gt) == 0 and np.count_nonzero(pr) == 0:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    if np.count_nonzero(gt) == 0 or np.count_nonzero(pr) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    events_gt = convert_vector_to_events(gt)
    events_pred = convert_vector_to_events(pr)
    trange = (0, len(gt))
    aff = pr_from_events(events_pred, events_gt, trange)
    precision = float(aff["precision"])
    recall = float(aff["recall"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "point_wise_f1": pointwise_f1(y_true, y_pred),
        "point_adjust_f1": point_adjust_f1(y_true, y_pred),
        "affi_f1": affiliation_f1(y_true, y_pred),
        "event_iou_f1": event_iou_f1(y_true, y_pred),
    }
