"""Tests for arogya.auditor.BiasAuditor."""

from __future__ import annotations

import numpy as np
import pytest

from arogya.auditor import BiasAuditor
from arogya.demographics import DemographicAxis, DemographicSpec
from arogya.domains import ClinicalDomain
from arogya.models import AuditResult, PredictionRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(
    n: int = 200,
    seed: int = 42,
    bias_group: str = "group_b",
    bias_amount: float = 0.15,
) -> list[PredictionRecord]:
    """Generate synthetic prediction records with controllable bias.

    ``bias_group`` receives predictions shifted by ``-bias_amount`` so
    that the model performs worse on that group.
    """
    rng = np.random.RandomState(seed)
    records: list[PredictionRecord] = []
    groups = ["group_a", "group_b"]
    sexes = ["male", "female"]

    for i in range(n):
        y_true = rng.randint(0, 2)
        group = groups[i % 2]
        sex = sexes[i % 2]

        # Base prediction with some noise
        if y_true == 1:
            y_pred = min(0.7 + rng.normal(0, 0.1), 1.0)
        else:
            y_pred = max(0.3 + rng.normal(0, 0.1), 0.0)

        # Introduce bias for the target group
        if group == bias_group:
            y_pred = np.clip(y_pred - bias_amount, 0.0, 1.0)

        records.append(
            PredictionRecord(
                sample_id=str(i),
                y_true=y_true,
                y_pred=float(y_pred),
                demographics={"race": group, "sex": sex},
            )
        )
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBiasAuditorPredictions:
    def test_basic_audit(self) -> None:
        records = _make_records(n=200, bias_amount=0.15)
        spec = DemographicSpec(axes=[DemographicAxis.RACE])
        auditor = BiasAuditor(
            demographic_spec=spec,
            model_name="test_model",
            min_group_size=10,
        )
        result = auditor.audit_predictions(records)

        assert isinstance(result, AuditResult)
        assert result.model_name == "test_model"
        assert result.total_samples == 200
        assert len(result.axes) == 1
        assert result.axes[0].axis == "race"
        assert result.axes[0].n_groups == 2
        assert 0.0 <= result.overall_equity_score <= 1.0

    def test_detects_bias(self) -> None:
        """The biased group should be flagged as the worst group."""
        records = _make_records(n=400, bias_amount=0.25, seed=7)
        spec = DemographicSpec(axes=[DemographicAxis.RACE])
        auditor = BiasAuditor(demographic_spec=spec, min_group_size=10)
        result = auditor.audit_predictions(records)

        race_axis = result.axes[0]
        assert race_axis.worst_group == "group_b"
        assert race_axis.equity_score < 0.95  # significant disparity

    def test_equity_high_when_no_bias(self) -> None:
        records = _make_records(n=400, bias_amount=0.0, seed=99)
        spec = DemographicSpec(axes=[DemographicAxis.RACE])
        auditor = BiasAuditor(demographic_spec=spec, min_group_size=10)
        result = auditor.audit_predictions(records)

        assert result.overall_equity_score > 0.85

    def test_multiple_axes(self) -> None:
        records = _make_records(n=400)
        spec = DemographicSpec(axes=[DemographicAxis.RACE, DemographicAxis.SEX])
        auditor = BiasAuditor(demographic_spec=spec, min_group_size=10)
        result = auditor.audit_predictions(records)

        axis_names = {a.axis for a in result.axes}
        assert "race" in axis_names
        assert "sex" in axis_names

    def test_mitigations_generated(self) -> None:
        records = _make_records(n=400, bias_amount=0.25)
        spec = DemographicSpec(axes=[DemographicAxis.RACE])
        auditor = BiasAuditor(demographic_spec=spec, min_group_size=10)
        result = auditor.audit_predictions(records)

        # If equity is poor enough, mitigations should be recommended
        if result.overall_equity_score < 0.90:
            assert len(result.mitigations) > 0

    def test_min_group_size_skips_small_groups(self) -> None:
        """Groups below min_group_size should be excluded."""
        records = _make_records(n=50)
        spec = DemographicSpec(axes=[DemographicAxis.RACE])
        auditor = BiasAuditor(
            demographic_spec=spec,
            min_group_size=100,  # higher than any single group
        )
        result = auditor.audit_predictions(records)
        # No axis should have computed groups
        assert len(result.axes) == 0


class TestBiasAuditorArrays:
    def test_from_arrays(self) -> None:
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        y_prob = np.clip(y_true * 0.6 + rng.normal(0.2, 0.1, n), 0, 1)
        demographics = {
            "race": np.array(["group_a" if i % 2 == 0 else "group_b" for i in range(n)]),
        }
        spec = DemographicSpec(axes=[DemographicAxis.RACE])
        auditor = BiasAuditor(demographic_spec=spec, min_group_size=10)
        result = auditor.audit_from_arrays(y_true, y_prob, demographics)

        assert isinstance(result, AuditResult)
        assert result.total_samples == n

    def test_with_model_fn(self) -> None:
        rng = np.random.RandomState(42)
        n = 100
        X = rng.rand(n, 5)
        y_true = rng.randint(0, 2, size=n)
        demographics = {
            "sex": np.array(["male" if i % 2 == 0 else "female" for i in range(n)]),
        }

        def dummy_model(x: np.ndarray) -> np.ndarray:
            return np.clip(x[:, 0] * 0.8 + 0.1, 0, 1)

        spec = DemographicSpec(axes=[DemographicAxis.SEX])
        auditor = BiasAuditor(demographic_spec=spec, min_group_size=10)
        result = auditor.audit_with_model(dummy_model, X, y_true, demographics)

        assert isinstance(result, AuditResult)
        assert result.total_samples == n


class TestBiasAuditorDomain:
    def test_domain_annotation(self) -> None:
        records = _make_records(n=200, bias_amount=0.2)
        spec = DemographicSpec(axes=[DemographicAxis.RACE])
        auditor = BiasAuditor(
            demographic_spec=spec,
            clinical_domain=ClinicalDomain.DERMATOLOGY,
            min_group_size=10,
        )
        result = auditor.audit_predictions(records)
        assert result.clinical_domain == "dermatology"
