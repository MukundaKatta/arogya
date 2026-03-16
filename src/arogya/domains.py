"""Six clinical domains supported by Arogya bias auditing."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClinicalDomain(str, Enum):
    """Supported clinical domains (6 total)."""

    DERMATOLOGY = "dermatology"
    RADIOLOGY = "radiology"
    CARDIOLOGY = "cardiology"
    OPHTHALMOLOGY = "ophthalmology"
    PATHOLOGY = "pathology"
    DENTAL = "dental"


# ---------------------------------------------------------------------------
# Domain-specific bias risk profiles
# ---------------------------------------------------------------------------

class BiasRiskFactor(BaseModel):
    """A known bias risk factor for a domain-axis combination."""

    domain: ClinicalDomain
    axis: str
    risk_level: str = Field(..., pattern="^(low|medium|high|critical)$")
    description: str


# Pre-defined risk factors based on known disparities in medical AI literature.
DOMAIN_BIAS_RISKS: list[BiasRiskFactor] = [
    # Dermatology
    BiasRiskFactor(
        domain=ClinicalDomain.DERMATOLOGY,
        axis="skin_tone",
        risk_level="critical",
        description=(
            "Skin-lesion classifiers trained predominantly on lighter skin "
            "tones exhibit significantly lower sensitivity on Fitzpatrick V–VI."
        ),
    ),
    BiasRiskFactor(
        domain=ClinicalDomain.DERMATOLOGY,
        axis="race",
        risk_level="high",
        description=(
            "Diagnostic accuracy for melanoma and other conditions varies "
            "across racial groups due to training-data imbalance."
        ),
    ),
    # Radiology
    BiasRiskFactor(
        domain=ClinicalDomain.RADIOLOGY,
        axis="sex",
        risk_level="medium",
        description=(
            "Chest X-ray models may encode sex-specific anatomical priors "
            "that bias disease detection (e.g., cardiac silhouette norms)."
        ),
    ),
    BiasRiskFactor(
        domain=ClinicalDomain.RADIOLOGY,
        axis="age_group",
        risk_level="high",
        description=(
            "Pediatric and elderly imaging differs structurally; models "
            "trained on adult-centric datasets generalise poorly."
        ),
    ),
    # Cardiology
    BiasRiskFactor(
        domain=ClinicalDomain.CARDIOLOGY,
        axis="sex",
        risk_level="high",
        description=(
            "Women present atypical cardiac symptoms; ML models often "
            "underdiagnose cardiac events in female patients."
        ),
    ),
    BiasRiskFactor(
        domain=ClinicalDomain.CARDIOLOGY,
        axis="race",
        risk_level="high",
        description=(
            "Race-based corrections in cardiac risk scores can propagate "
            "systemic disparities into algorithmic predictions."
        ),
    ),
    # Ophthalmology
    BiasRiskFactor(
        domain=ClinicalDomain.OPHTHALMOLOGY,
        axis="race",
        risk_level="high",
        description=(
            "Diabetic retinopathy screening models show variable performance "
            "across racial groups linked to fundus pigmentation."
        ),
    ),
    BiasRiskFactor(
        domain=ClinicalDomain.OPHTHALMOLOGY,
        axis="age_group",
        risk_level="medium",
        description=(
            "Age-related macular degeneration models may under-represent "
            "younger-onset cases in training data."
        ),
    ),
    # Pathology
    BiasRiskFactor(
        domain=ClinicalDomain.PATHOLOGY,
        axis="geography",
        risk_level="medium",
        description=(
            "Histopathology slide preparation varies by institution and "
            "region, creating distributional shift for geographically "
            "under-represented groups."
        ),
    ),
    BiasRiskFactor(
        domain=ClinicalDomain.PATHOLOGY,
        axis="sex",
        risk_level="medium",
        description=(
            "Sex-linked cancer subtypes may be under-sampled, affecting "
            "model calibration for sex-specific conditions."
        ),
    ),
    # Dental
    BiasRiskFactor(
        domain=ClinicalDomain.DENTAL,
        axis="age_group",
        risk_level="medium",
        description=(
            "Dental X-ray AI may perform differently on deciduous vs. "
            "permanent dentition in pediatric populations."
        ),
    ),
    BiasRiskFactor(
        domain=ClinicalDomain.DENTAL,
        axis="insurance_status",
        risk_level="medium",
        description=(
            "Uninsured populations may have later-stage presentations, "
            "shifting the pre-test probability distribution."
        ),
    ),
]


def risks_for_domain(domain: ClinicalDomain) -> list[BiasRiskFactor]:
    """Return known bias risk factors for *domain*."""
    return [r for r in DOMAIN_BIAS_RISKS if r.domain == domain]


def risks_for_axis(axis: str) -> list[BiasRiskFactor]:
    """Return known bias risk factors across all domains for *axis*."""
    return [r for r in DOMAIN_BIAS_RISKS if r.axis == axis]


def domain_from_string(name: str) -> ClinicalDomain:
    """Resolve a string to a ``ClinicalDomain`` enum member."""
    name_lower = name.lower()
    for member in ClinicalDomain:
        if member.name.lower() == name_lower or member.value == name_lower:
            return member
    valid = ", ".join(m.value for m in ClinicalDomain)
    raise ValueError(f"Unknown clinical domain {name!r}. Valid domains: {valid}")
