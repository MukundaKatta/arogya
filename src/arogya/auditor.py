"""BiasAuditor — core engine that runs a model across demographic groups."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np

from arogya.demographics import DemographicAxis, DemographicSpec, axis_from_string
from arogya.domains import ClinicalDomain
from arogya.metrics import (
    auc_gap,
    compute_group_metrics,
    diagnostic_equity_score,
    worst_best_ratio,
)
from arogya.mitigations import BiasMitigator
from arogya.models import (
    AuditResult,
    AxisEquitySummary,
    GroupMetrics,
    MetricName,
    PredictionRecord,
)

# Type alias for a model prediction function:
#   f(features) -> array of probabilities in [0, 1]
PredictionFn = Callable[[np.ndarray], np.ndarray]


class BiasAuditor:
    """Evaluate a diagnostic model for bias across demographic groups.

    Parameters
    ----------
    threshold:
        Decision threshold applied to predicted probabilities.
    demographic_spec:
        Which axes/groups to audit.  Defaults to all 12 axes with their
        standard groups.
    clinical_domain:
        Optional clinical domain for domain-specific risk annotation.
    model_name:
        Human-readable model name used in reports.
    min_group_size:
        Minimum number of samples in a group before metrics are computed.
        Groups smaller than this are skipped with a warning.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        demographic_spec: DemographicSpec | None = None,
        clinical_domain: ClinicalDomain | None = None,
        model_name: str = "unnamed_model",
        min_group_size: int = 20,
    ) -> None:
        self.threshold = threshold
        self.spec = demographic_spec or DemographicSpec()
        self.clinical_domain = clinical_domain
        self.model_name = model_name
        self.min_group_size = min_group_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_predictions(
        self,
        records: Sequence[PredictionRecord],
    ) -> AuditResult:
        """Run the audit on a sequence of ``PredictionRecord`` objects.

        Returns a fully populated ``AuditResult``.
        """
        # Partition records by (axis, group)
        partitions: dict[str, dict[str, list[PredictionRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in records:
            for axis_str, group_label in rec.demographics.items():
                partitions[axis_str][group_label].append(rec)

        axis_summaries: list[AxisEquitySummary] = []
        all_group_metrics: list[GroupMetrics] = []

        for axis in self.spec.axes:
            axis_key = axis.value
            group_map = partitions.get(axis_key, {})
            if not group_map:
                continue

            group_metrics_list: list[GroupMetrics] = []
            for group_label, group_records in group_map.items():
                if len(group_records) < self.min_group_size:
                    continue
                y_true = np.array([r.y_true for r in group_records])
                y_prob = np.array([r.y_pred for r in group_records])
                gm = compute_group_metrics(
                    group_name=group_label,
                    axis=axis_key,
                    y_true=y_true,
                    y_prob=y_prob,
                    threshold=self.threshold,
                )
                group_metrics_list.append(gm)

            if not group_metrics_list:
                continue

            # Identify best / worst by AUC (fall back to sensitivity)
            def _sort_key(gm: GroupMetrics) -> float:
                return gm.auc if gm.auc is not None else (gm.sensitivity or 0.0)

            sorted_groups = sorted(group_metrics_list, key=_sort_key)
            worst = sorted_groups[0]
            best = sorted_groups[-1]

            gap = auc_gap(group_metrics_list)
            eq = diagnostic_equity_score(group_metrics_list)

            summary = AxisEquitySummary(
                axis=axis_key,
                n_groups=len(group_metrics_list),
                groups=group_metrics_list,
                best_group=best.group_name,
                worst_group=worst.group_name,
                best_auc=best.auc,
                worst_auc=worst.auc,
                auc_gap=gap,
                equity_score=eq,
            )
            axis_summaries.append(summary)
            all_group_metrics.extend(group_metrics_list)

        overall_eq = (
            diagnostic_equity_score(all_group_metrics) if all_group_metrics else 0.0
        )

        # Generate mitigations
        mitigator = BiasMitigator(clinical_domain=self.clinical_domain)
        mitigations = mitigator.recommend(axis_summaries)

        return AuditResult(
            model_name=self.model_name,
            clinical_domain=self.clinical_domain.value if self.clinical_domain else None,
            total_samples=len(records),
            threshold=self.threshold,
            axes=axis_summaries,
            overall_equity_score=overall_eq,
            mitigations=mitigations,
        )

    def audit_from_arrays(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        demographics: dict[str, np.ndarray],
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> AuditResult:
        """Convenience wrapper that builds ``PredictionRecord`` objects from
        parallel numpy arrays.

        Parameters
        ----------
        y_true:
            Binary ground-truth labels, shape ``(n,)``.
        y_prob:
            Predicted probabilities, shape ``(n,)``.
        demographics:
            Mapping of axis name to an array of group labels, each of
            shape ``(n,)``.
        sample_ids:
            Optional per-sample identifiers.
        """
        n = len(y_true)
        if sample_ids is None:
            sample_ids = [str(i) for i in range(n)]

        records: list[PredictionRecord] = []
        for i in range(n):
            demo = {axis: str(vals[i]) for axis, vals in demographics.items()}
            records.append(
                PredictionRecord(
                    sample_id=sample_ids[i],
                    y_true=int(y_true[i]),
                    y_pred=float(y_prob[i]),
                    demographics=demo,
                    clinical_domain=(
                        self.clinical_domain.value if self.clinical_domain else None
                    ),
                )
            )
        return self.audit_predictions(records)

    def audit_with_model(
        self,
        predict_fn: PredictionFn,
        X: np.ndarray,
        y_true: np.ndarray,
        demographics: dict[str, np.ndarray],
        *,
        sample_ids: Sequence[str] | None = None,
    ) -> AuditResult:
        """Run the model prediction function, then audit the results.

        Parameters
        ----------
        predict_fn:
            Callable that accepts a feature matrix ``X`` of shape
            ``(n, d)`` and returns predicted probabilities of shape
            ``(n,)``.
        X:
            Feature matrix.
        y_true:
            Ground-truth binary labels.
        demographics:
            Per-axis group-label arrays.
        sample_ids:
            Optional per-sample identifiers.
        """
        y_prob = predict_fn(X)
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        return self.audit_from_arrays(
            y_true=y_true,
            y_prob=y_prob,
            demographics=demographics,
            sample_ids=sample_ids,
        )
