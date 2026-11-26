"""Unit tests for API, signature, attribute, and keyword argument verification."""

import functools
import inspect
import pytest
from softmaxshield.analyzer.ast_visitor import parse_source_code
from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import (
    CallReference,
    IssueCategory,
    ModuleType,
    Severity,
    VerificationStatus,
)
from softmaxshield.verifier.api_verifier import APIVerifier


def test_verify_valid_api_call():
    """Verify standard library function call with valid arguments."""
    code = "import math\nmath.sqrt(16)"
    ast_res = parse_source_code(code, "test.py")
    verifier = APIVerifier()

    issues = verifier.verify_call(
        call=ast_res.calls[0],
        file_name="test.py",
        ast_result=ast_res,
        module_type=ModuleType.STDLIB,
    )
    assert len(issues) == 0


def test_verify_invalid_api_attribute():
    """Verify that a nonexistent attribute on a valid module is flagged as INVALID_API with suggestion."""
    code = "import math\nmath.fast_sqrt(16)"
    ast_res = parse_source_code(code, "test.py")
    verifier = APIVerifier()

    issues = verifier.verify_call(
        call=ast_res.calls[0],
        file_name="test.py",
        ast_result=ast_res,
        module_type=ModuleType.STDLIB,
    )
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == IssueCategory.INVALID_API
    assert issue.severity == Severity.HIGH
    assert issue.suggested_correction == "math.sqrt"
    assert "does not exist on" in issue.explanation


def test_verify_invalid_keyword_argument():
    """Verify that an unsupported keyword argument is flagged as INVALID_ARGUMENT."""
    code = "import os\nos.getenv('PATH', invalid_kwarg='hello')"
    ast_res = parse_source_code(code, "test.py")
    verifier = APIVerifier()

    issues = verifier.verify_call(
        call=ast_res.calls[0],
        file_name="test.py",
        ast_result=ast_res,
        module_type=ModuleType.STDLIB,
    )
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == IssueCategory.INVALID_ARGUMENT
    assert issue.severity == Severity.HIGH
    assert "invalid_kwarg" in issue.explanation


def test_verify_positional_only_argument_violation():
    """Verify that passing a positional-only argument as a keyword is flagged as INVALID_ARGUMENT."""
    code = "import math\nmath.sqrt(x=16)"
    ast_res = parse_source_code(code, "test.py")
    verifier = APIVerifier()

    issues = verifier.verify_call(
        call=ast_res.calls[0],
        file_name="test.py",
        ast_result=ast_res,
        module_type=ModuleType.STDLIB,
    )
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == IssueCategory.INVALID_ARGUMENT
    assert "positional-only" in issue.explanation


def test_verify_kwargs_support_in_signature():
    """Verify that functions accepting **kwargs allow arbitrary kwargs without false positives."""
    code = "import requests\nrequests.get('https://example.com', timeout=5, custom_header_flag='extra')"
    ast_res = parse_source_code(code, "test.py")
    verifier = APIVerifier()

    issues = verifier.verify_call(
        call=ast_res.calls[0],
        file_name="test.py",
        ast_result=ast_res,
        module_type=ModuleType.THIRD_PARTY,
    )
    # requests.get takes **kwargs, so custom_header_flag is allowed
    assert len(issues) == 0


def test_inconclusive_uninstalled_module_not_flagged_invalid():
    """Verify that uninstalled third-party modules report INCONCLUSIVE instead of false INVALID_API."""
    code = "import some_uninstalled_service\nsome_uninstalled_service.process(timeout=30)"
    ast_res = parse_source_code(code, "test.py")
    # Non-strict mode
    verifier = APIVerifier(ShieldConfig(strict_mode=False))

    issues = verifier.verify_call(
        call=ast_res.calls[0],
        file_name="test.py",
        ast_result=ast_res,
        module_type=ModuleType.THIRD_PARTY,
    )
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == IssueCategory.UNVERIFIED_REFERENCE
    assert issue.verification_status == VerificationStatus.INCONCLUSIVE
    assert issue.severity == Severity.INFO


def test_decorator_unwrapping():
    """Verify inspect.unwrap properly unwraps decorated functions for signature analysis."""
    def sample_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @sample_decorator
    def target_function(name: str, *, count: int = 1):
        return f"{name} {count}"

    unwrapped = inspect.unwrap(target_function)
    sig = inspect.signature(unwrapped)
    assert "name" in sig.parameters
    assert "count" in sig.parameters
