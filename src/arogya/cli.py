"""Click CLI for Arogya: ``arogya audit`` and ``arogya report``."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np

from arogya.auditor import BiasAuditor
from arogya.demographics import DemographicSpec, axis_from_string, DemographicAxis
from arogya.domains import ClinicalDomain, domain_from_string
from arogya.models import AuditResult, PredictionRecord, ReportFormat
from arogya.report import DiagnosticEquityReport


@click.group()
@click.version_option(package_name="arogya")
def cli() -> None:
    """Arogya -- AI Diagnostic Bias Detector Across Demographics."""


# -----------------------------------------------------------------------
# arogya audit
# -----------------------------------------------------------------------

@cli.command()
@click.option(
    "--data", "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to a CSV or JSON file with prediction records.",
)
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    default=None,
    help="Path to a pickled model (sklearn-compatible with predict_proba).",
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=0.5,
    show_default=True,
    help="Decision threshold for binarising predictions.",
)
@click.option(
    "--domain",
    type=click.Choice([d.value for d in ClinicalDomain], case_sensitive=False),
    default=None,
    help="Clinical domain for domain-specific risk annotation.",
)
@click.option(
    "--model-name",
    type=str,
    default="unnamed_model",
    show_default=True,
    help="Human-readable model name for reports.",
)
@click.option(
    "--min-group-size",
    type=int,
    default=20,
    show_default=True,
    help="Minimum group size before metrics are computed.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Write JSON audit result to this path.",
)
@click.option(
    "--axes",
    type=str,
    default=None,
    help="Comma-separated list of axes to audit (default: all).",
)
def audit(
    data: str,
    model: str | None,
    threshold: float,
    domain: str | None,
    model_name: str,
    min_group_size: int,
    output: str | None,
    axes: str | None,
) -> None:
    """Run a bias audit on a dataset of predictions."""
    data_path = Path(data)

    # Parse axes
    spec = DemographicSpec()
    if axes:
        spec = DemographicSpec(
            axes=[axis_from_string(a.strip()) for a in axes.split(",")]
        )

    clinical_domain = domain_from_string(domain) if domain else None

    auditor = BiasAuditor(
        threshold=threshold,
        demographic_spec=spec,
        clinical_domain=clinical_domain,
        model_name=model_name,
        min_group_size=min_group_size,
    )

    records = _load_records(data_path)

    # If a model is provided, run predictions first
    if model:
        predict_fn = _load_model(Path(model))
        # Re-predict using the model -- expect records to have 'features' in metadata
        for rec in records:
            features = rec.metadata.get("features")
            if features is not None:
                x = np.array(features, dtype=float).reshape(1, -1)
                rec.y_pred = float(predict_fn(x).ravel()[0])

    result = auditor.audit_predictions(records)

    # Render to terminal
    report = DiagnosticEquityReport(result)
    report.render(ReportFormat.TERMINAL)

    # Optionally save JSON
    if output:
        json_str = report.render(ReportFormat.JSON, output_path=output)
        click.echo(f"\nAudit result written to {output}")


# -----------------------------------------------------------------------
# arogya report
# -----------------------------------------------------------------------

@cli.command()
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to a JSON audit-result file.",
)
@click.option(
    "--format", "-f",
    "fmt",
    type=click.Choice(["terminal", "json", "html"], case_sensitive=False),
    default="terminal",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="File path for JSON/HTML output.",
)
def report(input_path: str, fmt: str, output: str | None) -> None:
    """Render a previously generated audit result."""
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = AuditResult.model_validate(raw)
    report_obj = DiagnosticEquityReport(result)
    report_format = ReportFormat(fmt.lower())

    rendered = report_obj.render(report_format, output_path=output)
    if rendered and not output:
        click.echo(rendered)
    elif output:
        click.echo(f"Report written to {output}")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_records(path: Path) -> list[PredictionRecord]:
    """Load prediction records from CSV or JSON."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [PredictionRecord.model_validate(r) for r in raw]
        raise click.ClickException("JSON file must contain a list of records.")

    if suffix == ".csv":
        import csv

        records: list[PredictionRecord] = []
        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                # Identify demographic columns (prefixed with "demo_")
                demographics: dict[str, str] = {}
                for col, val in row.items():
                    if col.startswith("demo_"):
                        demographics[col.removeprefix("demo_")] = val

                records.append(
                    PredictionRecord(
                        sample_id=row.get("sample_id", str(i)),
                        y_true=int(row["y_true"]),
                        y_pred=float(row["y_pred"]),
                        demographics=demographics,
                        clinical_domain=row.get("clinical_domain"),
                    )
                )
        return records

    raise click.ClickException(f"Unsupported file format: {suffix}")


def _load_model(path: Path) -> Any:
    """Load a pickled model and return its predict_proba (or predict)."""
    with path.open("rb") as fh:
        model = pickle.load(fh)  # noqa: S301

    if hasattr(model, "predict_proba"):
        return lambda x: model.predict_proba(x)[:, 1]
    if callable(model):
        return model
    raise click.ClickException(
        "Model must have a predict_proba method or be callable."
    )
