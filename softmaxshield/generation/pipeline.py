"""End-to-end grounded generation pipeline combining RAG, Prompt Construction, LLM Generation, and SoftmaxShield Verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from softmaxshield.core.config import ShieldConfig
from softmaxshield.core.models import ShieldReport
from softmaxshield.corrector.client import LLMClientProtocol, MockLLMClient
from softmaxshield.corrector.engine import CorrectionResult, SelfCorrectionEngine
from softmaxshield.generation.docs_extractor import DocChunk, LibraryDocsCatalog
from softmaxshield.generation.prompt_templates import PromptStrategy
from softmaxshield.generation.rag_engine import RAGGenerationEngine
from softmaxshield.pipeline import ShieldPipeline


@dataclass
class GroundedGenerationResult:
    """Result of end-to-end grounded generation and verification."""

    user_query: str
    strategy: PromptStrategy
    prompt_used: str
    retrieved_chunks: List[DocChunk]
    generated_code: str
    verification_report: ShieldReport
    correction_result: Optional[CorrectionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "user_query": self.user_query,
            "strategy": self.strategy.value,
            "prompt_used": self.prompt_used,
            "retrieved_apis": [c.api_name for c in self.retrieved_chunks],
            "generated_code": self.generated_code,
            "passed": self.verification_report.passed,
            "total_issues": self.verification_report.total_issues,
            "issues": [i.to_dict() for i in self.verification_report.issues],
            "auto_corrected": self.correction_result.success if self.correction_result else False,
        }


class GroundedGenerationPipeline:
    """Orchestrates: User Prompt -> RAG Retrieval -> Prompt Construction -> LLM Generation -> SoftmaxShield Verification."""

    def __init__(
        self,
        rag_engine: Optional[RAGGenerationEngine] = None,
        verification_pipeline: Optional[ShieldPipeline] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        config: Optional[ShieldConfig] = None,
    ) -> None:
        self.config = config or ShieldConfig()
        self.rag_engine = rag_engine or RAGGenerationEngine()
        self.verifier = verification_pipeline or ShieldPipeline(self.config)
        self.llm_client = llm_client or MockLLMClient()
        self.corrector = SelfCorrectionEngine(
            pipeline=self.verifier,
            llm_client=self.llm_client,
            config=self.config,
            rag_engine=self.rag_engine,
        )

    def generate_and_verify(
        self,
        user_query: str,
        strategy: PromptStrategy = PromptStrategy.CHAIN_OF_VERIFICATION,
        allowed_libraries: Optional[List[str]] = None,
        version_filter: Optional[Union[str, Dict[str, str]]] = None,
        auto_correct: bool = False,
    ) -> GroundedGenerationResult:
        """Execute end-to-end generation and verification workflow."""
        # 1. RAG Retrieval & Prompt Construction
        rag_result = self.rag_engine.build_grounded_prompt(
            user_query=user_query,
            strategy=strategy,
            top_k=4,
            allowed_libraries=allowed_libraries,
            version_filter=version_filter,
        )

        # 2. LLM Generation
        raw_code = self.llm_client.generate_correction(
            prompt=rag_result.prompt,
            current_code=user_query,
            issues=[],
        )
        cleaned_code = self.corrector.clean_llm_response(raw_code)

        # 3. SoftmaxShield Verification
        initial_report = self.verifier.verify_code(cleaned_code, file_name="generated_code.py")

        # 4. Self-Correction (if requested and issues exist)
        correction_result = None
        final_code = cleaned_code
        final_report = initial_report

        if auto_correct and not initial_report.passed:
            correction_result = self.corrector.run(cleaned_code, file_name="generated_code.py")
            final_code = correction_result.final_code
            final_report = correction_result.final_report or self.verifier.verify_code(final_code)

        return GroundedGenerationResult(
            user_query=user_query,
            strategy=strategy,
            prompt_used=rag_result.prompt,
            retrieved_chunks=rag_result.retrieved_chunks,
            generated_code=final_code,
            verification_report=final_report,
            correction_result=correction_result,
        )

