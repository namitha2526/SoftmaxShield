"""Unit tests for Confidence Analysis, stable softmax, and token logprob scoring."""

import math
import pytest
from softmaxshield.confidence.analyzer import ConfidenceAnalyzer, stable_softmax
from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import IssueCategory, Severity, TokenLogprob, VerificationStatus


def test_stable_softmax_basic():
    """Verify standard softmax behavior and sum to 1.0."""
    logits = [1.0, 2.0, 3.0]
    probs = stable_softmax(logits)
    assert len(probs) == 3
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
    # Highest logit must produce highest probability
    assert probs[2] > probs[1] > probs[0]


def test_stable_softmax_extreme_values():
    """Verify numerical stability with large logits preventing overflow."""
    extreme_logits = [1000.0, 1001.0, 1002.0]
    probs = stable_softmax(extreme_logits)
    assert len(probs) == 3
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
    assert not any(math.isnan(p) for p in probs)

    empty_probs = stable_softmax([])
    assert empty_probs == []

    single_probs = stable_softmax([42.0])
    assert single_probs == [1.0]


def test_calculate_token_probability():
    """Verify logprob conversion to probability."""
    analyzer = ConfidenceAnalyzer()
    # logprob 0.0 -> probability 1.0
    assert math.isclose(analyzer.calculate_token_probability(0.0), 1.0)
    # logprob -0.69315 -> probability ~0.5
    assert math.isclose(analyzer.calculate_token_probability(-0.69315), 0.5, rel_tol=1e-3)
    # large negative logprob -> ~0.0
    assert analyzer.calculate_token_probability(-100.0) == 0.0


def test_compute_identifier_confidence_multi_token():
    """Verify confidence calculation across tokens for an identifier."""
    tokens = [
        TokenLogprob(token="import", logprob=-0.01),
        TokenLogprob(token=" ", logprob=-0.01),
        TokenLogprob(token="req", logprob=-2.3),    # p ~ 0.100
        TokenLogprob(token="eusts", logprob=-0.05), # p ~ 0.951
    ]
    analyzer = ConfidenceAnalyzer()
    score = analyzer.compute_identifier_confidence("reqeusts", tokens)
    assert score is not None
    # Conservative minimum token probability
    assert math.isclose(score, math.exp(-2.3), rel_tol=1e-3)


def test_assess_identifier_flags_low_confidence():
    """Verify low-confidence identifier triggers LOW_CONFIDENCE issue."""
    tokens = [
        TokenLogprob(token="reqeusts", logprob=-1.6),  # p ~ 0.201 < threshold 0.85
    ]
    config = ShieldConfig(confidence_threshold=0.85)
    analyzer = ConfidenceAnalyzer(config)

    issue = analyzer.assess_identifier(
        identifier="reqeusts",
        file_name="test.py",
        line=1,
        col=7,
        token_logprobs=tokens,
    )
    assert issue is not None
    assert issue.issue_type == IssueCategory.LOW_CONFIDENCE
    assert issue.severity == Severity.WARNING
    assert issue.verification_status == VerificationStatus.SUSPICIOUS
    assert issue.confidence_score is not None
    assert issue.confidence_score < 0.85
    assert "distinct from factual API validity" in issue.explanation


def test_graceful_fallback_without_token_logprobs():
    """Verify analyzer disables gracefully when no token logprobs are available."""
    analyzer = ConfidenceAnalyzer()
    score = analyzer.compute_identifier_confidence("requests", None)
    assert score is None

    issue = analyzer.assess_identifier("requests", "test.py", 1, 0, None)
    assert issue is None

