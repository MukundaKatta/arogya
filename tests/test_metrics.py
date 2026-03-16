"""Tests for arogya.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from arogya.metrics import (
    accuracy,
    auc_gap,
    auc_roc,
    calibration_error,
    compute_group_metrics,
    diagnostic_equity_score,
    f1_score,
    npv,
    ppv,
    sensitivity,
    specificity,
    worst_best_ratio,
)
from arogya.models import GroupMetrics, MetricName


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.05])
    return y_true, y_prob


@pytest.fixture()
def biased_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Predictions that are somewhat biased (imperfect)."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.4, 0.8, 0.1, 0.6, 0.2, 0.7, 0.3])
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Individual metric tests
# ---------------------------------------------------------------------------

class TestSensitivity:
    def test_perfect(self, perfect_predictions: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = perfect_predictions
        y_pred_bin = (y_prob >= 0.5).astype(int)
        assert sensitivity(y_true, y_pred_bin) == 1.0

    def test_no_positives(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 1])
        assert sensitivity(y_true, y_pred) is None

    def test_partial(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        assert sensitivity(y_true, y_pred) == pytest.approx(0.5)


class TestSpecificity:
    def test_perfect(self, perfect_predictions: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = perfect_predictions
        y_pred_bin = (y_prob >= 0.5).astype(int)
        assert specificity(y_true, y_pred_bin) == 1.0

    def test_no_negatives(self) -> None:
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 0, 1])
        assert specificity(y_true, y_pred) is None


class TestPPV:
    def test_perfect(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert ppv(y_true, y_pred) == 1.0

    def test_imperfect(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0])
        assert ppv(y_true, y_pred) == pytest.approx(2 / 3)


class TestNPV:
    def test_perfect(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert npv(y_true, y_pred) == 1.0


class TestF1:
    def test_perfect(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert f1_score(y_true, y_pred) == 1.0

    def test_zero(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        # sensitivity=0, ppv=0 => f1 = None (0/0)
        result = f1_score(y_true, y_pred)
        assert result is None or result == 0.0


class TestAccuracy:
    def test_perfect(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        assert accuracy(y_true, y_pred) == 1.0

    def test_empty(self) -> None:
        assert accuracy(np.array([]), np.array([])) is None


class TestAUC:
    def test_perfect(self, perfect_predictions: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = perfect_predictions
        auc = auc_roc(y_true, y_prob)
        assert auc is not None
        assert auc == pytest.approx(1.0)

    def test_random(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=1000)
        y_prob = rng.rand(1000)
        auc = auc_roc(y_true, y_prob)
        assert auc is not None
        assert 0.4 < auc < 0.6  # roughly 0.5 for random

    def test_single_class(self) -> None:
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7])
        assert auc_roc(y_true, y_prob) is None


class TestCalibrationError:
    def test_perfectly_calibrated(self) -> None:
        # Assign each sample a probability equal to its label
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = calibration_error(y_true, y_prob, n_bins=5)
        assert ece is not None
        assert ece == pytest.approx(0.0, abs=0.05)

    def test_empty(self) -> None:
        assert calibration_error(np.array([]), np.array([])) is None


# ---------------------------------------------------------------------------
# Composite metrics
# ---------------------------------------------------------------------------

class TestComputeGroupMetrics:
    def test_returns_group_metrics(self, perfect_predictions: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_prob = perfect_predictions
        gm = compute_group_metrics("test_group", "race", y_true, y_prob)
        assert isinstance(gm, GroupMetrics)
        assert gm.group_name == "test_group"
        assert gm.axis == "race"
        assert gm.n_samples == 6
        assert gm.sensitivity == 1.0
        assert gm.specificity == 1.0
        assert gm.auc is not None
        assert gm.auc == pytest.approx(1.0)


class TestWorstBestRatio:
    def test_equal_groups(self) -> None:
        groups = [
            GroupMetrics(group_name="a", axis="race", n_samples=50, prevalence=0.5, auc=0.85),
            GroupMetrics(group_name="b", axis="race", n_samples=50, prevalence=0.5, auc=0.85),
        ]
        ratio = worst_best_ratio(groups, MetricName.AUC)
        assert ratio == pytest.approx(1.0)

    def test_disparate_groups(self) -> None:
        groups = [
            GroupMetrics(group_name="a", axis="race", n_samples=50, prevalence=0.5, auc=0.90),
            GroupMetrics(group_name="b", axis="race", n_samples=50, prevalence=0.5, auc=0.60),
        ]
        ratio = worst_best_ratio(groups, MetricName.AUC)
        assert ratio is not None
        assert ratio == pytest.approx(0.60 / 0.90)


class TestAUCGap:
    def test_no_gap(self) -> None:
        groups = [
            GroupMetrics(group_name="a", axis="sex", n_samples=50, prevalence=0.5, auc=0.80),
            GroupMetrics(group_name="b", axis="sex", n_samples=50, prevalence=0.5, auc=0.80),
        ]
        assert auc_gap(groups) == pytest.approx(0.0)

    def test_with_gap(self) -> None:
        groups = [
            GroupMetrics(group_name="a", axis="sex", n_samples=50, prevalence=0.5, auc=0.90),
            GroupMetrics(group_name="b", axis="sex", n_samples=50, prevalence=0.5, auc=0.70),
        ]
        assert auc_gap(groups) == pytest.approx(0.20)


class TestDiagnosticEquityScore:
    def test_perfect_equity(self) -> None:
        groups = [
            GroupMetrics(
                group_name="a", axis="race", n_samples=100, prevalence=0.5,
                auc=0.90, sensitivity=0.85, specificity=0.85, calibration_error=0.02,
            ),
            GroupMetrics(
                group_name="b", axis="race", n_samples=100, prevalence=0.5,
                auc=0.90, sensitivity=0.85, specificity=0.85, calibration_error=0.02,
            ),
        ]
        score = diagnostic_equity_score(groups)
        assert score == pytest.approx(1.0, abs=0.02)

    def test_imperfect_equity(self) -> None:
        groups = [
            GroupMetrics(
                group_name="a", axis="race", n_samples=100, prevalence=0.5,
                auc=0.95, sensitivity=0.90, specificity=0.90, calibration_error=0.01,
            ),
            GroupMetrics(
                group_name="b", axis="race", n_samples=100, prevalence=0.5,
                auc=0.65, sensitivity=0.55, specificity=0.60, calibration_error=0.15,
            ),
        ]
        score = diagnostic_equity_score(groups)
        assert 0.0 < score < 0.90
