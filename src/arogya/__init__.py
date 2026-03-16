"""Arogya — AI Diagnostic Bias Detector Across Demographics."""

__version__ = "0.1.0"

from arogya.auditor import BiasAuditor
from arogya.demographics import DemographicAxis, DemographicGroup
from arogya.domains import ClinicalDomain
from arogya.metrics import diagnostic_equity_score
from arogya.mitigations import BiasMitigator
from arogya.report import DiagnosticEquityReport

__all__ = [
    "BiasAuditor",
    "BiasMitigator",
    "ClinicalDomain",
    "DemographicAxis",
    "DemographicGroup",
    "DiagnosticEquityReport",
    "diagnostic_equity_score",
]
