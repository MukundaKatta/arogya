"""DiagnosticEquityReport — Rich terminal, JSON, and HTML report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arogya.models import AuditResult, AxisEquitySummary, ReportFormat


class DiagnosticEquityReport:
    """Generate and export diagnostic equity reports.

    Parameters
    ----------
    result:
        The ``AuditResult`` to render.
    """

    def __init__(self, result: AuditResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        fmt: ReportFormat = ReportFormat.TERMINAL,
        output_path: str | Path | None = None,
    ) -> str | None:
        """Render the report in the requested format.

        Returns the rendered string for JSON/HTML, or ``None`` for
        terminal output (printed directly).
        """
        if fmt == ReportFormat.TERMINAL:
            self._render_terminal()
            return None
        elif fmt == ReportFormat.JSON:
            text = self._render_json()
        elif fmt == ReportFormat.HTML:
            text = self._render_html()
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        if output_path is not None:
            Path(output_path).write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------------
    # Terminal (Rich)
    # ------------------------------------------------------------------

    def _render_terminal(self) -> None:
        console = Console()
        r = self.result

        # Header
        eq_color = self._equity_color(r.overall_equity_score)
        header = Text()
        header.append("Arogya Diagnostic Equity Report\n", style="bold cyan")
        header.append(f"Model: {r.model_name}\n")
        if r.clinical_domain:
            header.append(f"Domain: {r.clinical_domain}\n")
        header.append(f"Samples: {r.total_samples:,}  |  Threshold: {r.threshold}\n")
        header.append("Overall Equity Score: ", style="bold")
        header.append(f"{r.overall_equity_score:.3f}", style=f"bold {eq_color}")
        console.print(Panel(header, title="Summary", border_style="cyan"))

        # Per-axis tables
        for axis_summary in r.axes:
            self._print_axis_table(console, axis_summary)

        # Mitigations
        if r.mitigations:
            mit_table = Table(
                title="Mitigation Recommendations",
                show_lines=True,
                border_style="yellow",
            )
            mit_table.add_column("P", justify="center", width=3)
            mit_table.add_column("Strategy", min_width=14)
            mit_table.add_column("Axis", min_width=10)
            mit_table.add_column("Description", ratio=3)
            for m in r.mitigations:
                p_style = "bold red" if m.priority <= 2 else "yellow"
                mit_table.add_row(
                    Text(str(m.priority), style=p_style),
                    m.strategy.value,
                    m.axis,
                    m.description,
                )
            console.print(mit_table)

    def _print_axis_table(
        self, console: Console, summary: AxisEquitySummary
    ) -> None:
        eq_color = self._equity_color(summary.equity_score)
        title = (
            f"Axis: {summary.axis}  |  Equity: "
            f"[{eq_color}]{summary.equity_score:.3f}[/{eq_color}]  |  "
            f"AUC gap: {summary.auc_gap:.4f}" if summary.auc_gap is not None
            else f"Axis: {summary.axis}  |  Equity: "
                 f"[{eq_color}]{summary.equity_score:.3f}[/{eq_color}]"
        )
        table = Table(title=title, show_lines=False, border_style="blue")
        table.add_column("Group", min_width=16)
        table.add_column("N", justify="right", min_width=6)
        table.add_column("Prev", justify="right", min_width=6)
        table.add_column("Sens", justify="right", min_width=6)
        table.add_column("Spec", justify="right", min_width=6)
        table.add_column("PPV", justify="right", min_width=6)
        table.add_column("AUC", justify="right", min_width=6)
        table.add_column("ECE", justify="right", min_width=6)

        for g in summary.groups:
            style = "bold red" if g.group_name == summary.worst_group else ""
            table.add_row(
                Text(g.group_name, style=style),
                str(g.n_samples),
                self._fmt(g.prevalence),
                self._fmt(g.sensitivity),
                self._fmt(g.specificity),
                self._fmt(g.ppv),
                self._fmt(g.auc),
                self._fmt(g.calibration_error),
            )

        console.print(table)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _render_json(self) -> str:
        payload = self._to_dict()
        return json.dumps(payload, indent=2, default=str)

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _render_html(self) -> str:
        r = self.result
        lines: list[str] = [
            "<!DOCTYPE html>",
            "<html lang='en'><head><meta charset='utf-8'>",
            "<title>Arogya Equity Report</title>",
            "<style>",
            "body{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a}",
            "h1{color:#0e7490}",
            "table{border-collapse:collapse;width:100%;margin:1rem 0}",
            "th,td{border:1px solid #d1d5db;padding:.5rem .75rem;text-align:right}",
            "th{background:#f3f4f6;text-align:center}",
            "td:first-child,th:first-child{text-align:left}",
            ".worst{color:#dc2626;font-weight:bold}",
            ".score{font-size:1.5rem;font-weight:bold}",
            ".score.good{color:#16a34a}.score.warn{color:#ca8a04}.score.bad{color:#dc2626}",
            ".card{border:1px solid #e5e7eb;border-radius:.5rem;padding:1rem;margin:1rem 0}",
            "</style></head><body>",
            f"<h1>Arogya Diagnostic Equity Report</h1>",
            f"<div class='card'>",
            f"<p><strong>Model:</strong> {r.model_name}</p>",
        ]
        if r.clinical_domain:
            lines.append(f"<p><strong>Domain:</strong> {r.clinical_domain}</p>")
        score_cls = self._equity_css_class(r.overall_equity_score)
        lines += [
            f"<p><strong>Samples:</strong> {r.total_samples:,} &nbsp;|&nbsp; "
            f"<strong>Threshold:</strong> {r.threshold}</p>",
            f"<p>Overall Equity Score: "
            f"<span class='score {score_cls}'>{r.overall_equity_score:.3f}</span></p>",
            "</div>",
        ]

        for axis_summary in r.axes:
            lines += self._html_axis_table(axis_summary)

        if r.mitigations:
            lines.append("<h2>Mitigation Recommendations</h2>")
            lines.append("<table><tr><th>P</th><th>Strategy</th><th>Axis</th>"
                         "<th>Description</th></tr>")
            for m in r.mitigations:
                cls = "worst" if m.priority <= 2 else ""
                lines.append(
                    f"<tr><td class='{cls}'>{m.priority}</td>"
                    f"<td>{m.strategy.value}</td><td>{m.axis}</td>"
                    f"<td style='text-align:left'>{m.description}</td></tr>"
                )
            lines.append("</table>")

        lines += [
            f"<footer><p>Generated by Arogya v0.1.0 at "
            f"{datetime.now(timezone.utc).isoformat()}</p></footer>",
            "</body></html>",
        ]
        return "\n".join(lines)

    def _html_axis_table(self, s: AxisEquitySummary) -> list[str]:
        score_cls = self._equity_css_class(s.equity_score)
        gap_str = f" | AUC gap: {s.auc_gap:.4f}" if s.auc_gap is not None else ""
        lines = [
            f"<h3>Axis: {s.axis} &mdash; Equity: "
            f"<span class='score {score_cls}'>{s.equity_score:.3f}</span>"
            f"{gap_str}</h3>",
            "<table><tr><th>Group</th><th>N</th><th>Prev</th><th>Sens</th>"
            "<th>Spec</th><th>PPV</th><th>AUC</th><th>ECE</th></tr>",
        ]
        for g in s.groups:
            cls = " class='worst'" if g.group_name == s.worst_group else ""
            lines.append(
                f"<tr{cls}><td>{g.group_name}</td><td>{g.n_samples}</td>"
                f"<td>{self._fmt(g.prevalence)}</td>"
                f"<td>{self._fmt(g.sensitivity)}</td>"
                f"<td>{self._fmt(g.specificity)}</td>"
                f"<td>{self._fmt(g.ppv)}</td>"
                f"<td>{self._fmt(g.auc)}</td>"
                f"<td>{self._fmt(g.calibration_error)}</td></tr>"
            )
        lines.append("</table>")
        return lines

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_dict(self) -> dict[str, Any]:
        r = self.result
        return {
            "model_name": r.model_name,
            "clinical_domain": r.clinical_domain,
            "total_samples": r.total_samples,
            "threshold": r.threshold,
            "overall_equity_score": r.overall_equity_score,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "axes": [
                {
                    "axis": a.axis,
                    "equity_score": a.equity_score,
                    "auc_gap": a.auc_gap,
                    "best_group": a.best_group,
                    "worst_group": a.worst_group,
                    "groups": [g.model_dump() for g in a.groups],
                }
                for a in r.axes
            ],
            "mitigations": [m.model_dump() for m in r.mitigations],
        }

    @staticmethod
    def _fmt(v: float | None, decimals: int = 3) -> str:
        return f"{v:.{decimals}f}" if v is not None else "—"

    @staticmethod
    def _equity_color(score: float) -> str:
        if score >= 0.90:
            return "green"
        elif score >= 0.75:
            return "yellow"
        return "red"

    @staticmethod
    def _equity_css_class(score: float) -> str:
        if score >= 0.90:
            return "good"
        elif score >= 0.75:
            return "warn"
        return "bad"
