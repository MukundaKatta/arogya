"""Pydantic data models for Arogya audit pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MetricName(str, Enum):
    """Supported diagnostic performance metrics."""

    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    PPV = "ppv"
    NPV = "npv"
    AUC = "auc"
    CALIBRATION_ERROR = "calibration_error"
    F1 = "f1"
    ACCURACY = "accuracy"


class MitigationStrategy(str, Enum):
    """Bias mitigation strategies."""

    RESAMPLING = "resampling"
    REWEIGHTING = "reweighting"
    AUGMENTATION = "augmentation"
    THRESHOLD_TUNING = "threshold_tuning"
    CALIBRATION = "calibration"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    TERMINAL = "terminal"
    JSON = "json"
    HTML = "html"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class PredictionRecord(BaseModel):
    """A single prediction with demographic annotations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    y_true: int = Field(..., ge=0, le=1, description="Ground-truth binary label")
    y_pred: float = Field(
        ..., ge=0.0, le=1.0, description="Model predicted probability"
    )
    demographics: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of demographic axis name to group label",
    )
    clinical_domain: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroupMetrics(BaseModel):
    """Performance metrics for a single demographic group."""

    group_name: str
    axis: str
    n_samples: int = Field(..., ge=0)
    prevalence: float = Field(..., ge=0.0, le=1.0)
    sensitivity: float | None = None
    specificity: float | None = None
    ppv: float | None = None
    npv: float | None = None
    auc: float | None = None
    calibration_error: float | None = None
    f1: float | None = None
    accuracy: float | None = None

    def metric(self, name: MetricName) -> float | None:
        """Retrieve a metric value by enum name."""
        return getattr(self, name.value, None)


class AxisEquitySummary(BaseModel):
    """Equity summary for a single demographic axis."""

    axis: str
    n_groups: int
    groups: list[GroupMetrics]
    best_group: str
    worst_group: str
    best_auc: float | None = None
    worst_auc: float | None = None
    auc_gap: float | None = None
    equity_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="1.0 = perfect equity, 0.0 = maximum disparity",
    )


class MitigationRecommendation(BaseModel):
    """A single mitigation recommendation."""

    strategy: MitigationStrategy
    axis: str
    affected_groups: list[str]
    priority: int = Field(..., ge=1, le=5, description="1 = highest priority")
    description: str
    expected_improvement: str


class AuditResult(BaseModel):
    """Complete result of a bias audit."""

    model_name: str = "unnamed_model"
    clinical_domain: str | None = None
    total_samples: int
    threshold: float = 0.5
    axes: list[AxisEquitySummary]
    overall_equity_score: float = Field(..., ge=0.0, le=1.0)
    mitigations: list[MitigationRecommendation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("overall_equity_score", mode="before")
    @classmethod
    def _clamp_equity(cls, v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))
