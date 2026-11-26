"""LLM Client protocol and mock implementations for the Self-Correction Engine."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Protocol

from softmaxshield.core.models import ShieldIssue
from softmaxshield.corrector.semantic_analyzer import SemanticAnalyzer, SemanticProposal

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients used in self-correction."""

    def generate_correction(
        self,
        prompt: str,
        current_code: str,
        issues: List[ShieldIssue],
    ) -> str:
        """Generate corrected Python code addressing the specified issues."""
        ...


class MockLLMClient:
    """Semantic-aware deterministic mock LLM client for testing and offline execution.

    Uses SemanticAnalyzer to understand operation intent, retrieve signatures from RAG,
    generate semantically authentic replacements (not mere lexical string similarities),
    score confidence, and reject low-confidence/hallucinated fixes.
    """

    def __init__(
        self,
        should_fail: bool = False,
        partial_correction: bool = False,
        semantic_analyzer: Optional[SemanticAnalyzer] = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.should_fail = should_fail
        self.partial_correction = partial_correction
        self.confidence_threshold = confidence_threshold
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer(confidence_threshold=confidence_threshold)
        self.call_count = 0
        self.last_proposals: List[SemanticProposal] = []

    def generate_correction(
        self,
        prompt: str,
        current_code: str,
        issues: List[ShieldIssue],
    ) -> str:
        """Apply semantically authentic corrections line by line."""
        self.call_count += 1
        self.last_proposals = []
        if self.should_fail:
            # Return unchanged code to simulate LLM failing to fix
            return current_code

        lines = current_code.splitlines(keepends=True)
        applied = 0

        for issue in issues:
            if self.partial_correction and applied >= 1:
                break

            line_idx = issue.line - 1
            if 0 <= line_idx < len(lines):
                orig_line = lines[line_idx]

                # Semantic-aware proposal generation
                proposal = self.semantic_analyzer.propose_correction(
                    issue=issue,
                    line_text=orig_line,
                    full_code=current_code,
                    prompt_context=prompt,
                )
                self.last_proposals.append(proposal)

                # Validation & Confidence Threshold Gate:
                # Reject corrections that are low-confidence or semantically unclear
                if proposal.requires_manual_fix or proposal.confidence < self.confidence_threshold:
                    logger.info(
                        "Correction skipped for '%s' (conf: %.2f < %.2f, manual: %s): %s",
                        issue.detected_identifier,
                        proposal.confidence,
                        self.confidence_threshold,
                        proposal.requires_manual_fix,
                        proposal.reasoning,
                    )
                    continue

                # Apply proposal to code line
                modified_line = self.semantic_analyzer.apply_proposal_to_line(orig_line, proposal)
                if modified_line != orig_line:
                    lines[line_idx] = modified_line
                    applied += 1

        new_code = "".join(lines)

        # Syntactic Validation: Ensure the result parses cleanly as Python AST
        if not self.semantic_analyzer.validate_code_syntax(new_code):
            logger.error("Syntactic validation failed on corrected code. Reverting to original.")
            return current_code

        return new_code

