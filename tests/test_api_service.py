"""Unit tests for SoftmaxShield FastAPI REST API service endpoints and handlers."""

import pytest
from softmaxshield.api.service import (
    ChunkItem,
    GenerateRequest,
    RetrieveRequest,
    UpdateDocsRequest,
    VerifyRequest,
    generate_and_verify_code,
    health_check,
    retrieve_docs,
    update_library_docs,
    verify_code_snippet,
)
from softmaxshield.generation.prompt_templates import PromptStrategy


def test_health_endpoint():
    """Verify healthcheck endpoint reports healthy status and chunk statistics."""
    data = health_check()
    assert data["status"] == "healthy"
    assert data["service"] == "SoftmaxShield"
    assert data["catalog_chunks_indexed"] >= 20


def test_retrieve_endpoint():
    """Verify documentation retrieval returns relevant API references."""
    req = RetrieveRequest(
        query="HTTP GET request with timeout",
        top_k=3,
        libraries=["requests"],
    )
    res = retrieve_docs(req)
    assert res.total_retrieved >= 1
    found_symbols = [c.api_name for c in res.chunks]
    assert "requests.get" in found_symbols


def test_retrieve_endpoint_metadata_filtering():
    """Verify version and package filtering in retrieval endpoint."""
    req = RetrieveRequest(
        query="dataframe operations",
        top_k=5,
        libraries=["pandas"],
        version="2.2",
    )
    res = retrieve_docs(req)
    for c in res.chunks:
        assert c.library == "pandas"
        assert c.version.startswith("2.2")


def test_verify_endpoint_valid_code():
    """Verify /api/v1/verify on clean standard library code."""
    req = VerifyRequest(
        code="import math\nx = math.sqrt(25)",
        file_name="clean.py",
        strict=False,
    )
    res = verify_code_snippet(req)
    assert res.passed is True
    assert res.total_issues == 0


def test_verify_endpoint_hallucinated_code():
    """Verify /api/v1/verify detects hallucinated API functions."""
    req = VerifyRequest(
        code="import math\nx = math.fast_sqrt(25)",
        file_name="hallucinated.py",
        strict=False,
    )
    res = verify_code_snippet(req)
    assert res.passed is False
    assert res.total_issues >= 1
    assert res.issues[0]["issue_type"] == "INVALID_API"


def test_generate_and_verify_pipeline_endpoint():
    """Verify full end-to-end generation and verification pipeline."""
    req = GenerateRequest(
        prompt="Write a function using requests.get to ping health endpoint",
        strategy=PromptStrategy.CHAIN_OF_VERIFICATION,
        libraries=["requests"],
        auto_correct=False,
    )
    res = generate_and_verify_code(req)
    assert res.user_query == req.prompt
    assert len(res.retrieved_apis) >= 1
    assert "requests.get" in res.retrieved_apis


def test_docs_update_endpoint():
    """Verify registering or updating documentation for a new library version."""
    req = UpdateDocsRequest(
        library="custom_lib",
        version="1.0.0",
        chunks=[
            ChunkItem(
                library="custom_lib",
                version="1.0.0",
                api_name="custom_lib.execute_pipeline",
                signature="custom_lib.execute_pipeline(steps: list)",
                docstring="Executes a pipeline of steps.",
                source_url="https://custom.org/docs",
                parameters=["steps"],
                return_type="bool",
            )
        ],
    )
    res = update_library_docs(req)
    assert res["status"] == "success"
    assert res["library"] == "custom_lib"
    assert res["version"] == "1.0.0"
