"""Unit tests for JSON/Console reporters and CLI execution."""

import json
from pathlib import Path
import pytest
from softmaxshield.cli import main
from softmaxshield.core.models import (
    IssueCategory,
    Severity,
    ShieldIssue,
    ShieldReport,
    VerificationStatus,
)
from softmaxshield.reporter.formatter import ConsoleReporter, JSONReporter


def test_json_reporter_schema():
    """Verify JSON reporter outputs all required schema fields."""
    issue = ShieldIssue(
        file_name="app.py",
        line=10,
        col=4,
        detected_identifier="reqeusts",
        issue_type=IssueCategory.INVALID_PACKAGE,
        severity=Severity.CRITICAL,
        verification_status=VerificationStatus.VERIFIED_INVALID,
        explanation="Package does not exist.",
        suggested_correction="requests",
        confidence_score=0.45,
    )
    report = ShieldReport(
        file_name="app.py",
        passed=False,
        total_issues=1,
        issues=[issue],
        execution_time_ms=12.5,
    )

    json_str = JSONReporter.format(report)
    parsed = json.loads(json_str)

    assert parsed["file_name"] == "app.py"
    assert parsed["passed"] is False
    assert parsed["total_issues"] == 1
    assert len(parsed["issues"]) == 1

    issue_dict = parsed["issues"][0]
    required_fields = [
        "file_name",
        "line",
        "col",
        "detected_identifier",
        "issue_type",
        "severity",
        "verification_status",
        "explanation",
        "suggested_correction",
    ]
    for rf in required_fields:
        assert rf in issue_dict, f"Missing required field {rf}"
    assert issue_dict["issue_type"] == "INVALID_PACKAGE"
    assert issue_dict["severity"] == "CRITICAL"


def test_console_reporter_format():
    """Verify human-readable console reporter output."""
    report = ShieldReport(
        file_name="valid.py",
        passed=True,
        total_issues=0,
        issues=[],
        execution_time_ms=5.0,
    )
    output = ConsoleReporter.format(report)
    assert "PASSED" in output
    assert "valid.py" in output
    assert "No verification issues detected" in output


def test_cli_check_valid_file(tmp_path, capsys):
    """Verify CLI exit code 0 on valid Python code."""
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("import math\nprint(math.sqrt(9))")

    exit_code = main(["check", str(valid_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PASSED" in captured.out


def test_cli_check_invalid_file(tmp_path, capsys):
    """Verify CLI exit code 1 on code with verification errors."""
    invalid_file = tmp_path / "invalid.py"
    invalid_file.write_text("import math\nprint(math.fast_sqrt(9))")

    exit_code = main(["check", str(invalid_file)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "FAILED" in captured.out
    assert "INVALID_API" in captured.out


def test_cli_check_json_output(tmp_path, capsys):
    """Verify CLI --json outputs parseable JSON report."""
    test_file = tmp_path / "test_json.py"
    test_file.write_text("import os\nprint(os.getenv('HOME'))")

    exit_code = main(["check", str(test_file), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)
    assert data["passed"] is True
    assert data["file_name"] == str(test_file)


def test_cli_missing_file(capsys):
    """Verify CLI exit code 2 when target file does not exist."""
    exit_code = main(["check", "nonexistent_file_xyz.py"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not found" in captured.err


def test_cli_auto_correct_flag(tmp_path, capsys):
    """Verify CLI --auto-correct repairs the file on disk."""
    fixable_file = tmp_path / "fixable.py"
    fixable_file.write_text("import math\nx = math.fast_sqrt(25)")

    exit_code = main(["check", str(fixable_file), "--auto-correct"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Successfully auto-corrected" in captured.out
    # Check that file content was updated
    updated_content = fixable_file.read_text()
    assert "math.sqrt" in updated_content
    assert "fast_sqrt" not in updated_content

