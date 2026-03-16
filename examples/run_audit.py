#!/usr/bin/env python3
"""Example: run a bias audit on synthetic diagnostic data.

Usage::

    python examples/run_audit.py
"""

from __future__ import annotations

import numpy as np

from arogya.auditor import BiasAuditor
from arogya.demographics import DemographicAxis, DemographicSpec
from arogya.domains import ClinicalDomain
from arogya.models import ReportFormat
from arogya.report import DiagnosticEquityReport


def main() -> None:
    rng = np.random.RandomState(2024)
    n = 1_000

    # --- Simulate ground-truth and predictions ---
    y_true = rng.randint(0, 2, size=n)

    # Base model: decent but not great
    noise = rng.normal(0, 0.12, size=n)
    y_prob = np.clip(y_true * 0.55 + 0.2 + noise, 0, 1)

    # --- Assign demographic attributes ---
    races = np.array(
        rng.choice(
            ["white", "black", "hispanic", "asian"],
            size=n,
            p=[0.50, 0.20, 0.20, 0.10],
        )
    )
    sexes = np.array(rng.choice(["male", "female"], size=n))
    skin_tones = np.array(
        rng.choice(
            [
                "fitzpatrick_I", "fitzpatrick_II", "fitzpatrick_III",
                "fitzpatrick_IV", "fitzpatrick_V", "fitzpatrick_VI",
            ],
            size=n,
        )
    )

    # --- Inject bias: reduce prediction quality for darker skin tones ---
    for tone in ("fitzpatrick_V", "fitzpatrick_VI"):
        mask = skin_tones == tone
        y_prob[mask] = np.clip(y_prob[mask] - 0.12, 0, 1)

    # --- Also inject a smaller racial bias ---
    y_prob[races == "black"] = np.clip(y_prob[races == "black"] - 0.06, 0, 1)

    demographics = {
        "race": races,
        "sex": sexes,
        "skin_tone": skin_tones,
    }

    # --- Configure auditor ---
    spec = DemographicSpec(
        axes=[
            DemographicAxis.RACE,
            DemographicAxis.SEX,
            DemographicAxis.SKIN_TONE,
        ]
    )
    auditor = BiasAuditor(
        threshold=0.5,
        demographic_spec=spec,
        clinical_domain=ClinicalDomain.DERMATOLOGY,
        model_name="SkinLesionClassifier-v2",
        min_group_size=20,
    )

    # --- Run audit ---
    result = auditor.audit_from_arrays(y_true, y_prob, demographics)

    # --- Render reports ---
    report = DiagnosticEquityReport(result)

    # Terminal output (Rich)
    print("=" * 72)
    report.render(ReportFormat.TERMINAL)

    # JSON export
    json_str = report.render(ReportFormat.JSON, output_path="audit_result.json")
    print("\nJSON audit saved to audit_result.json")

    # HTML export
    report.render(ReportFormat.HTML, output_path="equity_report.html")
    print("HTML report saved to equity_report.html")


if __name__ == "__main__":
    main()
