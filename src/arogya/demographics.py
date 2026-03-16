"""Twelve demographic axes and group definitions for diagnostic bias auditing."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from pydantic import BaseModel, Field


class DemographicAxis(str, Enum):
    """Supported demographic axes (12 total)."""

    RACE = "race"
    SEX = "sex"
    AGE_GROUP = "age_group"
    BMI_CATEGORY = "bmi_category"
    SKIN_TONE = "skin_tone"
    INSURANCE_STATUS = "insurance_status"
    LANGUAGE = "language"
    INCOME_BRACKET = "income_bracket"
    DISABILITY_STATUS = "disability_status"
    GEOGRAPHY = "geography"
    EDUCATION_LEVEL = "education_level"
    PREGNANCY_STATUS = "pregnancy_status"


# ---------------------------------------------------------------------------
# Default group labels for each axis
# ---------------------------------------------------------------------------

DEFAULT_GROUPS: dict[DemographicAxis, list[str]] = {
    DemographicAxis.RACE: [
        "white", "black", "hispanic", "asian", "native_american",
        "pacific_islander", "multiracial", "other",
    ],
    DemographicAxis.SEX: ["male", "female", "intersex"],
    DemographicAxis.AGE_GROUP: [
        "pediatric_0_17", "young_adult_18_34", "adult_35_54",
        "older_adult_55_74", "elderly_75_plus",
    ],
    DemographicAxis.BMI_CATEGORY: [
        "underweight", "normal", "overweight", "obese_class_1",
        "obese_class_2", "obese_class_3",
    ],
    DemographicAxis.SKIN_TONE: [
        "fitzpatrick_I", "fitzpatrick_II", "fitzpatrick_III",
        "fitzpatrick_IV", "fitzpatrick_V", "fitzpatrick_VI",
    ],
    DemographicAxis.INSURANCE_STATUS: [
        "private", "medicare", "medicaid", "uninsured", "tricare", "other",
    ],
    DemographicAxis.LANGUAGE: [
        "english", "spanish", "mandarin", "hindi", "arabic",
        "french", "other",
    ],
    DemographicAxis.INCOME_BRACKET: [
        "low", "lower_middle", "middle", "upper_middle", "high",
    ],
    DemographicAxis.DISABILITY_STATUS: [
        "none", "physical", "cognitive", "sensory", "multiple",
    ],
    DemographicAxis.GEOGRAPHY: ["urban", "suburban", "rural", "frontier"],
    DemographicAxis.EDUCATION_LEVEL: [
        "less_than_high_school", "high_school", "some_college",
        "bachelors", "graduate",
    ],
    DemographicAxis.PREGNANCY_STATUS: [
        "not_pregnant", "pregnant_first_trimester",
        "pregnant_second_trimester", "pregnant_third_trimester",
        "postpartum",
    ],
}


class DemographicGroup(BaseModel):
    """A single demographic group within an axis."""

    axis: DemographicAxis
    label: str
    description: str = ""

    def __hash__(self) -> int:
        return hash((self.axis, self.label))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DemographicGroup):
            return NotImplemented
        return self.axis == other.axis and self.label == other.label

    @property
    def key(self) -> str:
        """Short key string ``axis:label``."""
        return f"{self.axis.value}:{self.label}"


class DemographicSpec(BaseModel):
    """Specification of which axes and groups to audit."""

    axes: list[DemographicAxis] = Field(
        default_factory=lambda: list(DemographicAxis),
    )
    custom_groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Optional per-axis override of group labels.",
    )

    def groups_for(self, axis: DemographicAxis) -> list[str]:
        """Return the group labels for *axis*, honouring custom overrides."""
        if axis.value in self.custom_groups:
            return self.custom_groups[axis.value]
        return DEFAULT_GROUPS.get(axis, [])

    def all_groups(self) -> list[DemographicGroup]:
        """Expand the spec into a flat list of ``DemographicGroup`` objects."""
        result: list[DemographicGroup] = []
        for axis in self.axes:
            for label in self.groups_for(axis):
                result.append(DemographicGroup(axis=axis, label=label))
        return result


def axis_from_string(name: str) -> DemographicAxis:
    """Resolve a string to a ``DemographicAxis`` enum member.

    Accepts both the enum value (``"race"``) and the enum name
    (``"RACE"``), case-insensitively.
    """
    name_upper = name.upper()
    for member in DemographicAxis:
        if member.name == name_upper or member.value == name.lower():
            return member
    valid = ", ".join(m.value for m in DemographicAxis)
    raise ValueError(f"Unknown demographic axis {name!r}. Valid axes: {valid}")
