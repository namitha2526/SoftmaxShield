"""Unit tests for the Self-Correction Engine."""

import unittest.mock as mock
import pytest
from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import (
    IssueCategory,
    Severity,
    ShieldIssue,
    VerificationStatus,
)
from softmaxshield.corrector.client import MockLLMClient
from softmaxshield.corrector.engine import SelfCorrectionEngine
from softmaxshield.pipeline import ShieldPipeline


def test_build_prompt_includes_details():
    """Verify prompt formatting includes line numbers, categories, and suggested corrections."""
    engine = SelfCorrectionEngine()
    issues = [
        ShieldIssue(
            file_name="example.py",
            line=2,
            col=5,
            detected_identifier="math.fast_sqrt",
            issue_type=IssueCategory.INVALID_API,
            severity=Severity.HIGH,
            verification_status=VerificationStatus.VERIFIED_INVALID,
            explanation="Attribute 'fast_sqrt' does not exist.",
            suggested_correction="math.sqrt",
        )
    ]
    prompt = engine.build_prompt("import math\nmath.fast_sqrt(9)", issues)
    assert "Line 2" in prompt
    assert "INVALID_API" in prompt
    assert "math.fast_sqrt" in prompt
    assert "math.sqrt" in prompt
    assert "Modify ONLY the flagged lines" in prompt


def test_single_round_successful_correction():
    """Verify code with an invalid API is corrected on the first round."""
    broken_code = "import math\nx = math.fast_sqrt(16)"
    mock_client = MockLLMClient()
    pipeline = ShieldPipeline()
    engine = SelfCorrectionEngine(pipeline=pipeline, llm_client=mock_client)

    result = engine.run(broken_code, file_name="math_test.py")

    assert result.success is True
    assert result.rounds_attempted == 1
    assert "fast_sqrt" not in result.final_code
    assert "math.sqrt" in result.final_code
    assert result.original_code == broken_code
    assert len(result.attempts) == 1
    assert "--- round_1_before.py" in result.attempts[0].diff


def test_multi_round_successful_correction():
    """Verify iterative correction fixing issues step by step across multiple rounds."""
    broken_code = "import math\nx = math.fast_sqrt(16)\ny = math.fast_sqrt(25)"
    # Client that only fixes 1 issue per round
    mock_client = MockLLMClient(partial_correction=True)
    config = ShieldConfig(max_correction_rounds=3)
    pipeline = ShieldPipeline(config)
    engine = SelfCorrectionEngine(pipeline=pipeline, llm_client=mock_client, config=config)

    result = engine.run(broken_code, file_name="multi_test.py")

    assert result.success is True
    assert result.rounds_attempted == 2
    assert "math.fast_sqrt" not in result.final_code
    assert result.final_code.count("math.sqrt") == 2
    assert len(result.attempts) == 2


def test_correction_stops_at_max_rounds():
    """Verify correction stops and returns failure when limit of 3 rounds is reached."""
    broken_code = "import math\nx = math.fast_sqrt(16)"
    # Client that refuses to change anything
    mock_client = MockLLMClient(should_fail=True)
    config = ShieldConfig(max_correction_rounds=3)
    pipeline = ShieldPipeline(config)
    engine = SelfCorrectionEngine(pipeline=pipeline, llm_client=mock_client, config=config)

    result = engine.run(broken_code, file_name="fail_test.py")

    assert result.success is False
    assert result.rounds_attempted == 3
    assert len(result.attempts) == 3
    assert mock_client.call_count == 3


def test_already_valid_code_skips_correction():
    """Verify that valid code does not trigger LLM calls."""
    valid_code = "import math\nx = math.sqrt(16)"
    mock_client = MockLLMClient()
    pipeline = ShieldPipeline()
    engine = SelfCorrectionEngine(pipeline=pipeline, llm_client=mock_client)

    result = engine.run(valid_code, file_name="valid.py")

    assert result.success is True
    assert result.rounds_attempted == 0
    assert mock_client.call_count == 0
    assert len(result.attempts) == 0

