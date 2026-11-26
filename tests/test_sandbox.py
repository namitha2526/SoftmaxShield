"""Unit tests verifying the isolated execution sandbox runner."""

import pytest
from softmaxshield.evaluation.sandbox_runner import SandboxRunner


def test_sandbox_successful_execution():
    """Verify clean execution of valid Python code and assertions."""
    runner = SandboxRunner(timeout_seconds=2.0)
    result = runner.execute_snippet(
        code="a = 10\nb = 20\nresult = a + b",
        assertion_code="assert result == 30",
    )
    assert result.success is True
    assert result.timed_out is False
    assert result.assertion_passed is True
    assert result.return_code == 0


def test_sandbox_assertion_failure():
    """Verify accurate capture of assertion failures."""
    runner = SandboxRunner(timeout_seconds=2.0)
    result = runner.execute_snippet(
        code="val = 42",
        assertion_code="assert val == 99",
    )
    assert result.success is False
    assert result.assertion_passed is False
    assert result.error_type == "AssertionError"


def test_sandbox_syntax_error():
    """Verify accurate capture of syntax errors."""
    runner = SandboxRunner(timeout_seconds=2.0)
    result = runner.execute_snippet(
        code="def broken_func(:\n    pass",
    )
    assert result.success is False
    assert result.error_type == "SyntaxError"


def test_sandbox_pattern_assertion():
    """Verify string pattern assertions against code without executing live I/O."""
    runner = SandboxRunner(timeout_seconds=2.0)
    code = "import requests\nresp = requests.get('https://example.com', timeout=5)"
    result = runner.execute_snippet(
        code=code,
        assertion_code="assert 'requests.get' in code and 'timeout=5' in code",
    )
    assert result.success is True
    assert result.assertion_passed is True

