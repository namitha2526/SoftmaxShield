"""FastAPI REST API Service for SoftmaxShield Generation & Verification."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from softmaxshield.core.config import ShieldConfig
from softmaxshield.generation.docs_extractor import DocChunk, get_ten_libraries_docs
from softmaxshield.generation.pipeline import GroundedGenerationPipeline
from softmaxshield.generation.prompt_templates import PromptStrategy
from softmaxshield.generation.rag_engine import RAGGenerationEngine
from softmaxshield.pipeline import ShieldPipeline

app = FastAPI(
    title="SoftmaxShield API Service",
    description="REST API service for RAG-grounded code generation and static verification of LLM Python code.",
    version="0.1.0",
)

# Global shared pipeline instances
_pipeline: Optional[GroundedGenerationPipeline] = None
_verifier: Optional[ShieldPipeline] = None
_rag_engine: Optional[RAGGenerationEngine] = None


def get_grounded_pipeline() -> GroundedGenerationPipeline:
    """Lazy initialize and return singleton GroundedGenerationPipeline."""
    global _pipeline, _verifier, _rag_engine
    if _pipeline is None:
        config = ShieldConfig()
        _rag_engine = RAGGenerationEngine()
        _verifier = ShieldPipeline(config)
        _pipeline = GroundedGenerationPipeline(
            rag_engine=_rag_engine,
            verification_pipeline=_verifier,
            config=config,
        )
    return _pipeline


# --- Pydantic Schemas ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Developer natural language coding instruction")
    strategy: PromptStrategy = Field(
        default=PromptStrategy.CHAIN_OF_VERIFICATION,
        description="Prompt defence strategy: CONSTRAINED, FEW_SHOT, or CHAIN_OF_VERIFICATION",
    )
    libraries: Optional[List[str]] = Field(
        default=None,
        description="Optional list of permitted libraries (e.g. ['fastapi', 'pydantic'])",
    )
    version_filter: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional library version constraints, e.g. {'pandas': '2.2'}",
    )
    auto_correct: bool = Field(
        default=True,
        description="Whether to run the self-correction engine if static verification finds issues",
    )


class GenerateResponse(BaseModel):
    user_query: str
    strategy: str
    retrieved_apis: List[str]
    generated_code: str
    passed: bool
    total_issues: int
    issues: List[Dict[str, Any]]
    auto_corrected: bool


class VerifyRequest(BaseModel):
    code: str = Field(..., description="Python source code to statically verify")
    file_name: str = Field(default="<api_input>.py", description="Virtual file name for reporting")
    strict: bool = Field(default=False, description="Enable strict verification mode")


class VerifyResponse(BaseModel):
    file_name: str
    passed: bool
    total_issues: int
    execution_time_ms: float
    issues: List[Dict[str, Any]]


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=4, ge=1, le=20)
    libraries: Optional[List[str]] = None
    version: Optional[str] = None


class ChunkItem(BaseModel):
    library: str
    version: str
    api_name: str
    signature: str
    docstring: str
    source_url: str
    parameters: List[str]
    return_type: str


class RetrieveResponse(BaseModel):
    query: str
    total_retrieved: int
    chunks: List[ChunkItem]


class UpdateDocsRequest(BaseModel):
    library: str
    version: str
    chunks: List[ChunkItem]


class CorrectRequest(BaseModel):
    code: str = Field(..., description="Python source code to statically verify and repair")
    file_name: str = Field(default="<correction_input>.py", description="Virtual file name")
    strict: bool = Field(default=False, description="Enable strict mode")


class CorrectionAttemptModel(BaseModel):
    round_number: int
    resolved: bool
    diff: str
    issues_count: int


class CorrectResponse(BaseModel):
    original_code: str
    final_code: str
    success: bool
    rounds_attempted: int
    diff: str
    attempts: List[CorrectionAttemptModel]
    execution_time_ms: float


class EvaluateRequest(BaseModel):
    configs: Optional[List[str]] = Field(
        default=None,
        description="Configurations to evaluate: baseline_llm, prompt_engineering, rag_only, fine_tuned_model, guardrails_only, full_softmaxshield",
    )
    max_tasks: Optional[int] = Field(
        default=None,
        description="Optional limit on number of benchmark tasks to evaluate",
    )


class EvaluateResponse(BaseModel):
    status: str
    total_tasks: int
    configurations: List[Dict[str, Any]]
    markdown_table: str


# --- API Endpoints ---

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Service health and catalog status check."""
    pipeline = get_grounded_pipeline()
    total_chunks = len(pipeline.rag_engine.catalog.chunks)
    return {
        "status": "healthy",
        "service": "SoftmaxShield",
        "version": "0.1.0",
        "catalog_chunks_indexed": total_chunks,
        "timestamp": time.time(),
    }


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["Generation"])
def generate_and_verify_code(req: GenerateRequest) -> GenerateResponse:
    """Execute RAG retrieval -> prompt construction -> generation -> SoftmaxShield verification."""
    pipeline = get_grounded_pipeline()
    result = pipeline.generate_and_verify(
        user_query=req.prompt,
        strategy=req.strategy,
        allowed_libraries=req.libraries,
        version_filter=req.version_filter,
        auto_correct=req.auto_correct,
    )
    res_dict = result.to_dict()
    return GenerateResponse(**res_dict)


@app.post("/api/v1/verify", response_model=VerifyResponse, tags=["Verification"])
def verify_code_snippet(req: VerifyRequest) -> VerifyResponse:
    """Directly verify Python code using the multi-layer static verification engine."""
    pipeline = get_grounded_pipeline()
    pipeline.verifier.config.strict_mode = req.strict
    report = pipeline.verifier.verify_code(req.code, file_name=req.file_name)
    return VerifyResponse(
        file_name=report.file_name,
        passed=report.passed,
        total_issues=report.total_issues,
        execution_time_ms=report.execution_time_ms,
        issues=[i.to_dict() for i in report.issues],
    )


@app.post("/api/v1/retrieve", response_model=RetrieveResponse, tags=["RAG"])
def retrieve_docs(req: RetrieveRequest) -> RetrieveResponse:
    """Query the vector database for relevant documentation chunks."""
    pipeline = get_grounded_pipeline()
    chunks = pipeline.rag_engine.retrieve_signatures(
        query=req.query,
        top_k=req.top_k,
        library_filter=req.libraries,
        version_filter=req.version,
    )
    chunk_items = [
        ChunkItem(
            library=c.library,
            version=c.version,
            api_name=c.api_name,
            signature=c.signature,
            docstring=c.docstring,
            source_url=c.source_url,
            parameters=c.parameters,
            return_type=c.return_type,
        )
        for c in chunks
    ]
    return RetrieveResponse(
        query=req.query,
        total_retrieved=len(chunk_items),
        chunks=chunk_items,
    )


@app.post("/api/v1/docs/update", tags=["RAG"])
def update_library_docs(req: UpdateDocsRequest) -> Dict[str, Any]:
    """Register or update documentation chunks for a library version."""
    pipeline = get_grounded_pipeline()
    new_doc_chunks = [
        DocChunk(
            library=c.library,
            version=c.version,
            api_name=c.api_name,
            symbol_type="function",
            signature=c.signature,
            docstring=c.docstring,
            source_url=c.source_url,
            parameters=c.parameters,
            return_type=c.return_type,
        )
        for c in req.chunks
    ]
    total = pipeline.rag_engine.update_library_documentation(
        library=req.library,
        version=req.version,
        new_chunks=new_doc_chunks,
    )
    return {
        "status": "success",
        "library": req.library,
        "version": req.version,
        "total_catalog_chunks": total,
    }


@app.post("/api/v1/correct", response_model=CorrectResponse, tags=["Correction"])
def correct_code_snippet(req: CorrectRequest) -> CorrectResponse:
    """Run the self-correction feedback loop to iteratively repair detected errors."""
    import difflib

    pipeline = get_grounded_pipeline()
    pipeline.verifier.config.strict_mode = req.strict

    t0 = time.perf_counter()
    result = pipeline.corrector.run(req.code, file_name=req.file_name)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    attempt_models = [
        CorrectionAttemptModel(
            round_number=att.round_number,
            resolved=att.resolved,
            diff=att.diff,
            issues_count=len(att.issues_found),
        )
        for att in result.attempts
    ]

    diff_lines = list(
        difflib.unified_diff(
            result.original_code.splitlines(keepends=True),
            result.final_code.splitlines(keepends=True),
            fromfile="original.py",
            tofile="corrected.py",
        )
    )
    full_diff = "".join(diff_lines)

    return CorrectResponse(
        original_code=result.original_code,
        final_code=result.final_code,
        success=result.success,
        rounds_attempted=result.rounds_attempted,
        diff=full_diff,
        attempts=attempt_models,
        execution_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/v1/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate_benchmark_endpoint(req: EvaluateRequest) -> EvaluateResponse:
    """Run or retrieve comparative evaluation across architectural configurations."""
    from softmaxshield.evaluation.evaluator import BenchmarkEvaluator, format_markdown_table

    evaluator = BenchmarkEvaluator()
    task_subset = evaluator.tasks[: req.max_tasks] if req.max_tasks else evaluator.tasks
    target_configs = req.configs or [
        "baseline_llm",
        "prompt_engineering",
        "rag_only",
        "fine_tuned_model",
        "guardrails_only",
        "full_softmaxshield",
    ]

    summaries = []
    for cfg in target_configs:
        summary, _ = evaluator.evaluate_configuration(cfg, task_subset=task_subset)
        summaries.append(summary)

    table_md = format_markdown_table(summaries)
    return EvaluateResponse(
        status="completed",
        total_tasks=len(task_subset),
        configurations=[s.to_dict() for s in summaries],
        markdown_table=table_md,
    )


