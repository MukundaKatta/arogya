"""BiasMitigator — automatic mitigation recommendations."""

from __future__ import annotations

from typing import Sequence

from arogya.domains import ClinicalDomain, risks_for_axis
from arogya.models import (
    AxisEquitySummary,
    MitigationRecommendation,
    MitigationStrategy,
)


# Equity-score thresholds for priority assignment
_PRIORITY_THRESHOLDS: list[tuple[float, int]] = [
    (0.60, 1),  # critical
    (0.70, 2),  # high
    (0.80, 3),  # medium
    (0.90, 4),  # low
    (1.01, 5),  # informational
]


def _priority_for_score(equity_score: float) -> int:
    for threshold, priority in _PRIORITY_THRESHOLDS:
        if equity_score < threshold:
            return priority
    return 5


class BiasMitigator:
    """Generate actionable mitigation recommendations from audit results.

    Parameters
    ----------
    clinical_domain:
        When set, domain-specific risk context is folded into
        recommendations.
    equity_threshold:
        Axes with an equity score below this value trigger
        recommendations.
    """

    def __init__(
        self,
        clinical_domain: ClinicalDomain | None = None,
        equity_threshold: float = 0.90,
    ) -> None:
        self.clinical_domain = clinical_domain
        self.equity_threshold = equity_threshold

    def recommend(
        self,
        axis_summaries: Sequence[AxisEquitySummary],
    ) -> list[MitigationRecommendation]:
        """Return a prioritised list of mitigation recommendations."""
        recommendations: list[MitigationRecommendation] = []

        for summary in axis_summaries:
            if summary.equity_score >= self.equity_threshold:
                continue

            priority = _priority_for_score(summary.equity_score)
            affected = self._underperforming_groups(summary)

            # 1. Resampling
            recommendations.append(
                MitigationRecommendation(
                    strategy=MitigationStrategy.RESAMPLING,
                    axis=summary.axis,
                    affected_groups=affected,
                    priority=priority,
                    description=(
                        f"Oversample under-represented groups on the "
                        f"'{summary.axis}' axis to balance training data.  "
                        f"Target groups: {', '.join(affected)}."
                    ),
                    expected_improvement=(
                        "Reduce AUC gap by equalising label distribution "
                        "across groups."
                    ),
                )
            )

            # 2. Re-weighting
            recommendations.append(
                MitigationRecommendation(
                    strategy=MitigationStrategy.REWEIGHTING,
                    axis=summary.axis,
                    affected_groups=affected,
                    priority=min(priority + 1, 5),
                    description=(
                        f"Apply inverse-frequency or inverse-performance "
                        f"sample weights for groups on the '{summary.axis}' "
                        f"axis to penalise errors on disadvantaged groups."
                    ),
                    expected_improvement=(
                        "Improve worst-group sensitivity without requiring "
                        "additional data collection."
                    ),
                )
            )

            # 3. Augmentation
            recommendations.append(
                MitigationRecommendation(
                    strategy=MitigationStrategy.AUGMENTATION,
                    axis=summary.axis,
                    affected_groups=affected,
                    priority=min(priority + 1, 5),
                    description=(
                        f"Apply targeted data augmentation (rotation, colour "
                        f"jitter, synthetic generation) for under-performing "
                        f"groups on the '{summary.axis}' axis."
                    ),
                    expected_improvement=(
                        "Increase effective training-set diversity for "
                        "disadvantaged groups."
                    ),
                )
            )

            # 4. Threshold tuning
            if summary.auc_gap is not None and summary.auc_gap > 0.05:
                recommendations.append(
                    MitigationRecommendation(
                        strategy=MitigationStrategy.THRESHOLD_TUNING,
                        axis=summary.axis,
                        affected_groups=affected,
                        priority=priority,
                        description=(
                            f"Tune per-group decision thresholds on the "
                            f"'{summary.axis}' axis to equalise sensitivity "
                            f"or specificity across groups."
                        ),
                        expected_improvement=(
                            "Equalise operating-point performance without "
                            "retraining the model."
                        ),
                    )
                )

            # 5. Post-hoc calibration
            recommendations.append(
                MitigationRecommendation(
                    strategy=MitigationStrategy.CALIBRATION,
                    axis=summary.axis,
                    affected_groups=affected,
                    priority=min(priority + 2, 5),
                    description=(
                        f"Apply group-aware Platt scaling or isotonic "
                        f"regression to improve calibration on the "
                        f"'{summary.axis}' axis."
                    ),
                    expected_improvement=(
                        "Reduce calibration error disparity across groups."
                    ),
                )
            )

            # Domain-specific context
            domain_risks = risks_for_axis(summary.axis)
            for risk in domain_risks:
                if (
                    self.clinical_domain is not None
                    and risk.domain != self.clinical_domain
                ):
                    continue
                recommendations.append(
                    MitigationRecommendation(
                        strategy=MitigationStrategy.AUGMENTATION,
                        axis=summary.axis,
                        affected_groups=affected,
                        priority=max(priority - 1, 1),
                        description=(
                            f"[{risk.domain.value}] {risk.description} "
                            f"Consider domain-specific data collection and "
                            f"validation for affected groups."
                        ),
                        expected_improvement=(
                            f"Address known {risk.risk_level}-risk disparity "
                            f"in {risk.domain.value}."
                        ),
                    )
                )

        # Sort by priority (ascending = most critical first)
        recommendations.sort(key=lambda r: r.priority)
        return recommendations

    @staticmethod
    def _underperforming_groups(summary: AxisEquitySummary) -> list[str]:
        """Identify groups whose AUC is below the axis median."""
        aucs = [(g.group_name, g.auc) for g in summary.groups if g.auc is not None]
        if not aucs:
            return [summary.worst_group]
        median_auc = sorted(a for _, a in aucs)[len(aucs) // 2]
        return [name for name, auc in aucs if auc <= median_auc]
