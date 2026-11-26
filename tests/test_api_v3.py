"""Unit tests for the expanded FastAPI endpoints (/correct, /evaluate)."""

import pytest
from softmaxshield.api.service import (
    CorrectRequest,
    CorrectResponse,
    EvaluateRequest,
    EvaluateResponse,
    correct_code_snippet,
    evaluate_benchmark_endpoint,
)


def test_api_correct_endpoint_clean_code():
    """Verify that /correct endpoint passes clean valid code without modifications."""
    req = CorrectRequest(
        code="import math\nval = math.sqrt(16.0)\n",
        file_name="clean.py",
    )
    resp: CorrectResponse = correct_code_snippet(req)
    assert resp.success is True
    assert resp.rounds_attempted == 0
    assert "math.sqrt" in resp.final_code


def test_api_correct_endpoint_with_issues():
    """Verify that /correct endpoint triggers self-correction loop when issues are found."""
    req = CorrectRequest(
        code="import fast_numpy as fnp\nres = fnp.sum([1, 2, 3])",
        file_name="halluc.py",
        strict=True,
    )
    resp: CorrectResponse = correct_code_snippet(req)
    assert resp.rounds_attempted >= 1
    assert isinstance(resp.execution_time_ms, float)


def test_api_evaluate_endpoint():
    """Verify that /evaluate endpoint successfully executes benchmark slices."""
    req = EvaluateRequest(
        max_tasks=2,
        configs=["baseline_llm", "full_softmaxshield"],
    )
    resp: EvaluateResponse = evaluate_benchmark_endpoint(req)
    assert resp.status == "completed"
    assert resp.total_tasks == 2
    assert len(resp.configurations) == 2
    assert "Baseline Llm" in resp.markdown_table
    assert "Full Softmaxshield" in resp.markdown_table

