"""Confidence analysis engine with numerically stable softmax and token logprob association."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import (
    IssueCategory,
    Severity,
    ShieldIssue,
    TokenLogprob,
    VerificationStatus,
)


def stable_softmax(logits: Sequence[float]) -> List[float]:
    """Compute numerically stable softmax probabilities over a sequence of real-valued logits.

    Uses the shift invariance property: softmax(z) = softmax(z - max(z)).
    """
    if not logits:
        return []
    if len(logits) == 1:
        return [1.0]

    max_val = max(logits)
    # Subtract max to prevent overflow in exp
    exponents = [math.exp(val - max_val) for val in logits]
    total = sum(exponents)
    if total == 0.0 or math.isnan(total):
        # Uniform fallback in degenerate condition
        n = len(logits)
        return [1.0 / n] * n

    return [exp_val / total for exp_val in exponents]


class ConfidenceAnalyzer:
    """Analyzes token log probabilities from LLM outputs to measure identifier generation confidence."""

    def __init__(self, config: Optional[ShieldConfig] = None) -> None:
        self.config = config or ShieldConfig()

    def calculate_token_probability(self, logprob: float) -> float:
        """Convert a single log probability to a bounded probability in [0.0, 1.0]."""
        if math.isnan(logprob) or logprob < -50.0:
            return 0.0
        try:
            return max(0.0, min(1.0, math.exp(logprob)))
        except OverflowError:
            return 1.0 if logprob > 0 else 0.0

    def compute_identifier_confidence(
        self,
        identifier: str,
        token_logprobs: Optional[List[Union[TokenLogprob, Dict[str, Any]]]],
    ) -> Optional[float]:
        """Associate token logprobs with a code identifier and calculate the confidence score.

        If the model does not provide token probabilities, returns None gracefully.
        """
        if not token_logprobs or not identifier:
            return None

        # Normalize token items to TokenLogprob objects
        normalized_tokens: List[TokenLogprob] = []
        for item in token_logprobs:
            if isinstance(item, TokenLogprob):
                normalized_tokens.append(item)
            elif isinstance(item, dict):
                normalized_tokens.append(
                    TokenLogprob(
                        token=str(item.get("token", "")),
                        logprob=float(item.get("logprob", 0.0)),
                        top_logprobs=item.get("top_logprobs"),
                    )
                )

        target = identifier.strip()
        matched_probs: List[float] = []

        # Find tokens that form or overlap with the target identifier
        full_text = "".join(t.token for t in normalized_tokens)
        if target not in full_text:
            # Fallback: check individual sub-tokens
            sub_parts = [p for p in target.replace(".", " ").replace("_", " ").split() if p]
            for t in normalized_tokens:
                clean_t = t.token.strip()
                if clean_t and any(part in clean_t or clean_t in part for part in sub_parts):
                    prob = self.calculate_token_probability(t.logprob)
                    matched_probs.append(prob)
        else:
            # Reconstruct spans to match exact token window
            current_pos = 0
            start_idx = full_text.find(target)
            end_idx = start_idx + len(target)

            for t in normalized_tokens:
                tok_len = len(t.token)
                tok_start = current_pos
                tok_end = current_pos + tok_len
                current_pos = tok_end

                if max(start_idx, tok_start) < min(end_idx, tok_end):
                    prob = self.calculate_token_probability(t.logprob)
                    matched_probs.append(prob)

        if not matched_probs:
            return None

        # Conservative confidence metric: minimum token probability across the identifier
        return min(matched_probs)

    def assess_identifier(
        self,
        identifier: str,
        file_name: str,
        line: int,
        col: int,
        token_logprobs: Optional[List[Union[TokenLogprob, Dict[str, Any]]]],
    ) -> Optional[ShieldIssue]:
        """Assess an identifier and emit a LOW_CONFIDENCE issue if score is below threshold.

        Distinguishes model confidence from factual API validity.
        """
        score = self.compute_identifier_confidence(identifier, token_logprobs)
        if score is None:
            # Layer disabled gracefully for models without token logprobs
            return None

        if score < self.config.confidence_threshold:
            return ShieldIssue(
                file_name=file_name,
                line=line,
                col=col,
                detected_identifier=identifier,
                issue_type=IssueCategory.LOW_CONFIDENCE,
                severity=Severity.WARNING,
                verification_status=VerificationStatus.SUSPICIOUS,
                explanation=(
                    f"Identifier '{identifier}' generated with low LLM confidence ({score:.3f} < "
                    f"{self.config.confidence_threshold:.2f}). Note: Model generation confidence is distinct from factual API validity "
                    f"and must be verified independently."
                ),
                suggested_correction=None,
                confidence_score=score,
            )

        return None
