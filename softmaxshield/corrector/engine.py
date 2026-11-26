"""Self-correction feedback engine orchestrating multi-round LLM code repair."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import CorrectionAttempt, ShieldIssue, ShieldReport
from softmaxshield.corrector.client import LLMClientProtocol, MockLLMClient
from softmaxshield.corrector.semantic_analyzer import SemanticAnalyzer
from softmaxshield.pipeline import ShieldPipeline

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Outcome of the self-correction engine."""

    original_code: str
    final_code: str
    success: bool
    rounds_attempted: int
    attempts: List[CorrectionAttempt] = field(default_factory=list)
    final_report: Optional[ShieldReport] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    semantic_reasonings: List[str] = field(default_factory=list)


class SelfCorrectionEngine:
    """Manages iterative feedback loop with an LLM to correct flagged code errors."""

    def __init__(
        self,
        pipeline: Optional[ShieldPipeline] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        config: Optional[ShieldConfig] = None,
        rag_engine: Optional[Any] = None,
    ) -> None:
        self.config = config or ShieldConfig()
        self.pipeline = pipeline or ShieldPipeline(self.config)
        self.rag_engine = rag_engine

        conf_thresh = getattr(self.config, "correction_confidence_threshold", 0.75)
        if llm_client is None:
            analyzer = SemanticAnalyzer(rag_engine=self.rag_engine, confidence_threshold=conf_thresh)
            self.llm_client = MockLLMClient(
                semantic_analyzer=analyzer,
                confidence_threshold=conf_thresh,
            )
        else:
            self.llm_client = llm_client

    def build_prompt(self, current_code: str, issues: List[ShieldIssue]) -> str:
        """Construct structured prompt instructing the model to repair flagged lines only."""
        issues_text_lines = []
        for idx, issue in enumerate(issues, 1):
            suggestion_part = (
                f" Suggested correction: '{issue.suggested_correction}'."
                if issue.suggested_correction
                else ""
            )
            issues_text_lines.append(
                f"- Issue {idx} [Line {issue.line}, Col {issue.col}]: "
                f"Category: {issue.issue_type.value} | Identifier: `{issue.detected_identifier}`\n"
                f"  Explanation: {issue.explanation}{suggestion_part}"
            )

        issues_summary = "\n".join(issues_text_lines)

        return (
            "You are an automated Python security and static verification code repair agent.\n"
            "Static verification detected hallucinated or invalid identifiers in the code below.\n\n"
            "=== VERIFICATION FINDINGS ===\n"
            f"{issues_summary}\n\n"
            "=== STRICT CONSTRAINTS ===\n"
            "1. Modify ONLY the flagged lines to resolve the identified issues.\n"
            "2. Preserve all other lines, comments, and logic unchanged.\n"
            "3. Do NOT invent new unverified packages or APIs.\n"
            "4. Return ONLY the complete corrected Python source code.\n"
            "5. Do NOT include conversational explanations or markdown backticks.\n\n"
            "=== CURRENT CODE ===\n"
            f"{current_code}\n"
        )

    def clean_llm_response(self, response_text: str) -> str:
        """Strip markdown fences or conversational preambles from model response."""
        text = response_text.strip()
        # Remove markdown code fences if present (e.g. ```python ... ```)
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text

    def run(
        self,
        source_code: str,
        file_name: str = "<input>",
    ) -> CorrectionResult:
        """Execute the self-correction loop up to max_rounds."""
        original_code = source_code
        current_code = source_code
        attempts: List[CorrectionAttempt] = []

        # Initial verification
        report = self.pipeline.verify_code(current_code, file_name=file_name)
        if report.passed:
            return CorrectionResult(
                original_code=original_code,
                final_code=current_code,
                success=True,
                rounds_attempted=0,
                attempts=[],
                final_report=report,
            )

        max_rounds = self.config.max_correction_rounds

        for round_num in range(1, max_rounds + 1):
            prompt = self.build_prompt(current_code, report.issues)
            raw_corrected = self.llm_client.generate_correction(
                prompt=prompt,
                current_code=current_code,
                issues=report.issues,
            )
            corrected_code = self.clean_llm_response(raw_corrected)

            # Compute diff
            diff = "".join(
                difflib.unified_diff(
                    current_code.splitlines(keepends=True),
                    corrected_code.splitlines(keepends=True),
                    fromfile=f"round_{round_num}_before.py",
                    tofile=f"round_{round_num}_after.py",
                )
            )

            # Re-verify new code
            new_report = self.pipeline.verify_code(corrected_code, file_name=file_name)
            is_resolved = new_report.passed

            attempt = CorrectionAttempt(
                round_number=round_num,
                code_before=current_code,
                code_after=corrected_code,
                issues_found=report.issues,
                resolved=is_resolved,
                diff=diff,
            )
            attempts.append(attempt)

            current_code = corrected_code
            report = new_report

            if is_resolved:
                logger.info("Verification passed on round %d!", round_num)
                confidence_map: Dict[str, float] = {}
                reasonings: List[str] = []
                if hasattr(self.llm_client, "last_proposals"):
                    for p in getattr(self.llm_client, "last_proposals", []):
                        confidence_map[p.original_target] = p.confidence
                        if p.reasoning:
                            reasonings.append(f"{p.original_target}: {p.reasoning}")

                return CorrectionResult(
                    original_code=original_code,
                    final_code=current_code,
                    success=True,
                    rounds_attempted=round_num,
                    attempts=attempts,
                    final_report=new_report,
                    confidence_scores=confidence_map,
                    semantic_reasonings=reasonings,
                )

        # Reached round limit without passing completely
        logger.warning("Reached maximum correction rounds (%d) without full pass.", max_rounds)
        confidence_map: Dict[str, float] = {}
        reasonings: List[str] = []
        if hasattr(self.llm_client, "last_proposals"):
            for p in getattr(self.llm_client, "last_proposals", []):
                confidence_map[p.original_target] = p.confidence
                if p.reasoning:
                    reasonings.append(f"{p.original_target}: {p.reasoning}")

        return CorrectionResult(
            original_code=original_code,
            final_code=current_code,
            success=False,
            rounds_attempted=max_rounds,
            attempts=attempts,
            final_report=report,
            confidence_scores=confidence_map,
            semantic_reasonings=reasonings,
        )

