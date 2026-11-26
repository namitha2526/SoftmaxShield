"""Structured reporters for JSON and human-readable terminal output."""

from __future__ import annotations

import json
from typing import Any, Dict

from softmaxshield.core.models import ShieldReport


class JSONReporter:
    """Formats verification report into standardized JSON string."""

    @staticmethod
    def format(report: ShieldReport, indent: int = 2) -> str:
        """Serialize ShieldReport to pretty JSON string."""
        return json.dumps(report.to_dict(), indent=indent)


class ConsoleReporter:
    """Formats verification report into clean terminal output with severity indicators."""

    @staticmethod
    def format(report: ShieldReport, verbose: bool = False) -> str:
        """Format ShieldReport for terminal display."""
        lines = []
        status_text = "PASSED" if report.passed else "FAILED"
        status_symbol = "✔" if report.passed else "✖"
        banner = f"SoftmaxShield Verification [{status_symbol} {status_text}] - {report.file_name}"
        lines.append("=" * len(banner))
        lines.append(banner)
        lines.append("=" * len(banner))
        lines.append(
            f"Total Issues Detected: {report.total_issues} | Time: {report.execution_time_ms:.2f} ms"
        )
        lines.append("-" * len(banner))

        if not report.issues:
            lines.append("No verification issues detected. Code is statically verified.")
            return "\n".join(lines)

        for idx, issue in enumerate(report.issues, 1):
            severity_str = f"[{issue.severity.value}]"
            lines.append(
                f"\n#{idx} {severity_str} {issue.issue_type.value} at line {issue.line}, col {issue.col}"
            )
            lines.append(f"   Identifier : {issue.detected_identifier}")
            lines.append(f"   Status     : {issue.verification_status.value}")
            lines.append(f"   Explanation: {issue.explanation}")
            if issue.suggested_correction:
                lines.append(f"   Suggestion : {issue.suggested_correction}")
            if issue.confidence_score is not None:
                lines.append(f"   Confidence : {issue.confidence_score:.4f}")

        if verbose and report.metadata:
            lines.append("\n" + "-" * len(banner))
            lines.append("Metadata:")
            for k, v in report.metadata.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

