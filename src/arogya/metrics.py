"""Diagnostic equity metrics: per-group performance and fairness scores."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as sp_stats

from arogya.models import GroupMetrics, MetricName


# ---------------------------------------------------------------------------
# Low-level metric computations
# ---------------------------------------------------------------------------

def _safe_divide(num: float, den: float) -> float | None:
    """Return *num / den* or ``None`` when *den* is zero."""
    return float(num / den) if den > 0 else None


def sensitivity(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float | None:
    """True-positive rate (recall)."""
    pos = y_true == 1
    if pos.sum() == 0:
        return None
    return float((y_pred_bin[pos] == 1).sum() / pos.sum())


def specificity(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float | None:
    """True-negative rate."""
    neg = y_true == 0
    if neg.sum() == 0:
        return None
    return float((y_pred_bin[neg] == 0).sum() / neg.sum())


def ppv(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float | None:
    """Positive predictive value (precision)."""
    pred_pos = y_pred_bin == 1
    if pred_pos.sum() == 0:
        return None
    return float(y_true[pred_pos].sum() / pred_pos.sum())


def npv(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float | None:
    """Negative predictive value."""
    pred_neg = y_pred_bin == 0
    if pred_neg.sum() == 0:
        return None
    return float((y_true[pred_neg] == 0).sum() / pred_neg.sum())


def f1_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float | None:
    """F1 score (harmonic mean of precision and recall)."""
    sens = sensitivity(y_true, y_pred_bin)
    prec = ppv(y_true, y_pred_bin)
    if sens is None or prec is None or (sens + prec) == 0:
        return None
    return float(2 * sens * prec / (sens + prec))


def accuracy(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float | None:
    """Classification accuracy."""
    if len(y_true) == 0:
        return None
    return float((y_true == y_pred_bin).sum() / len(y_true))


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Area under the ROC curve via the trapezoidal rule.

    Returns ``None`` when fewer than two classes are present.
    """
    if len(np.unique(y_true)) < 2:
        return None
    # Sort by descending predicted probability
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    tpr_prev, fpr_prev = 0.0, 0.0
    auc_val = 0.0
    tp, fp = 0.0, 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc_val += 0.5 * (tpr + tpr_prev) * (fpr - fpr_prev)
        tpr_prev, fpr_prev = tpr, fpr
    return float(np.clip(auc_val, 0.0, 1.0))


def calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float | None:
    """Expected calibration error (ECE) using equal-width bins."""
    if len(y_true) == 0:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_pred = float(y_prob[mask].mean())
        avg_true = float(y_true[mask].mean())
        ece += mask.sum() / len(y_true) * abs(avg_pred - avg_true)
    return float(ece)


# ---------------------------------------------------------------------------
# Per-group evaluation
# ---------------------------------------------------------------------------

def compute_group_metrics(
    group_name: str,
    axis: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> GroupMetrics:
    """Compute all metrics for a single demographic group."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred_bin = (y_prob >= threshold).astype(int)

    n = len(y_true)
    prev = float(y_true.mean()) if n > 0 else 0.0

    return GroupMetrics(
        group_name=group_name,
        axis=axis,
        n_samples=n,
        prevalence=prev,
        sensitivity=sensitivity(y_true, y_pred_bin),
        specificity=specificity(y_true, y_pred_bin),
        ppv=ppv(y_true, y_pred_bin),
        npv=npv(y_true, y_pred_bin),
        auc=auc_roc(y_true, y_prob),
        calibration_error=calibration_error(y_true, y_prob),
        f1=f1_score(y_true, y_pred_bin),
        accuracy=accuracy(y_true, y_pred_bin),
    )


# ---------------------------------------------------------------------------
# Equity scores
# ---------------------------------------------------------------------------

def worst_best_ratio(
    groups: Sequence[GroupMetrics],
    metric: MetricName = MetricName.AUC,
) -> float | None:
    """Ratio of worst-group to best-group performance for *metric*.

    Returns a value in [0, 1] where 1.0 means perfect equity.
    Returns ``None`` if no valid values exist.
    """
    values = [g.metric(metric) for g in groups if g.metric(metric) is not None]
    if not values:
        return None
    best = max(values)
    worst = min(values)
    if best == 0:
        return None
    return float(worst / best)


def auc_gap(groups: Sequence[GroupMetrics]) -> float | None:
    """Absolute gap between best-group and worst-group AUC."""
    aucs = [g.auc for g in groups if g.auc is not None]
    if not aucs:
        return None
    return float(max(aucs) - min(aucs))


def diagnostic_equity_score(
    groups: Sequence[GroupMetrics],
    weights: dict[MetricName, float] | None = None,
) -> float:
    """Composite Diagnostic Equity Score in [0, 1].

    Aggregates worst/best ratios across multiple metrics with optional
    *weights*.  A score of 1.0 indicates perfect parity; lower values
    signal increasing disparity.
    """
    if weights is None:
        weights = {
            MetricName.AUC: 0.35,
            MetricName.SENSITIVITY: 0.25,
            MetricName.SPECIFICITY: 0.20,
            MetricName.CALIBRATION_ERROR: 0.20,
        }

    total_weight = 0.0
    weighted_sum = 0.0

    for metric, w in weights.items():
        if metric == MetricName.CALIBRATION_ERROR:
            # For calibration error, equity means all groups are equally
            # (well-)calibrated.  We invert so that *lower* error = *higher*
            # contribution to equity.
            ces = [g.calibration_error for g in groups if g.calibration_error is not None]
            if ces:
                worst_ce = max(ces)
                ratio = 1.0 - worst_ce  # 0 error -> 1.0
                weighted_sum += w * max(ratio, 0.0)
                total_weight += w
        else:
            ratio = worst_best_ratio(groups, metric)
            if ratio is not None:
                weighted_sum += w * ratio
                total_weight += w

    if total_weight == 0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))
