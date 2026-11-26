"""Unit tests for generation-side defences: RAG, Vector Store, Templates, and Fine-Tuning."""

import json
from pathlib import Path
import pytest
from softmaxshield.generation.docs_extractor import (
    DocChunk,
    LibraryDocsCatalog,
    get_ten_libraries_docs,
)
from softmaxshield.generation.prompt_templates import (
    PromptStrategy,
    build_chain_of_verification_prompt,
    build_constrained_prompt,
    build_few_shot_prompt,
)
from softmaxshield.generation.rag_engine import RAGGenerationEngine
from softmaxshield.generation.vector_store import FAISSVectorStore, InMemoryVectorStore
from softmaxshield.scripts.continued_pretraining import prepare_raw_docs_corpus
from softmaxshield.scripts.generate_finetuning_data import generate_dataset


def test_docs_extractor_ten_libraries():
    """Verify that catalog contains chunks across all 10 target libraries."""
    catalog = get_ten_libraries_docs()
    expected_libs = {
        "numpy", "pandas", "requests", "flask", "fastapi",
        "scikit-learn", "torch", "matplotlib", "sqlalchemy", "pydantic"
    }
    found_libs = {c.library.lower() for c in catalog.chunks}
    for lib in expected_libs:
        assert lib in found_libs, f"Missing library {lib} in documentation catalog"

    # Verify chunk properties
    first_chunk = catalog.chunks[0]
    assert first_chunk.signature is not None
    assert first_chunk.version is not None
    assert "Library:" in first_chunk.to_embedding_text()
    assert "[" in first_chunk.to_prompt_context()


def test_in_memory_vector_store():
    """Verify in-memory vector store indexing and keyword-boosted retrieval."""
    store = InMemoryVectorStore()
    chunk1 = DocChunk(
        library="requests",
        version="2.31.0",
        api_name="requests.get",
        symbol_type="function",
        signature="requests.get(url, params=None)",
        docstring="Send HTTP GET request",
        parameters=["url", "params"],
        return_type="requests.Response",
    )
    chunk2 = DocChunk(
        library="pandas",
        version="2.2.2",
        api_name="pandas.DataFrame.merge",
        symbol_type="method",
        signature="pandas.DataFrame.merge(self, right)",
        docstring="Merge two DataFrames",
        parameters=["right"],
        return_type="pandas.DataFrame",
    )
    store.add_documents([chunk1, chunk2])

    results = store.search("HTTP GET request", top_k=1)
    assert len(results) == 1
    assert results[0][0].api_name == "requests.get"

    pandas_results = store.search("merge DataFrames", top_k=1)
    assert len(pandas_results) == 1
    assert pandas_results[0][0].api_name == "pandas.DataFrame.merge"


def test_rag_generation_engine_prompts():
    """Verify RAG engine detection, retrieval, and prompt synthesis."""
    engine = RAGGenerationEngine(use_faiss=False)

    # Test library detection
    libs = engine.detect_libraries("Create a FastAPI endpoint with Pydantic validation")
    assert "fastapi" in libs
    assert "pydantic" in libs

    # Test Constrained Prompt Strategy
    constrained_res = engine.build_grounded_prompt(
        "Make a GET request using requests",
        strategy=PromptStrategy.CONSTRAINED,
    )
    assert "STRICT GENERATION CONSTRAINTS (CLOSED-WORLD INSTRUCTION)" in constrained_res.prompt
    assert "requests" in constrained_res.prompt

    # Test Few-Shot Prompt Strategy
    few_shot_res = engine.build_grounded_prompt(
        "Make a GET request using requests",
        strategy=PromptStrategy.FEW_SHOT,
    )
    assert "EXAMPLE 1:" in few_shot_res.prompt
    assert "TARGET TASK:" in few_shot_res.prompt

    # Test Chain-of-Verification Strategy
    cove_res = engine.build_grounded_prompt(
        "Make a GET request using requests",
        strategy=PromptStrategy.CHAIN_OF_VERIFICATION,
    )
    assert "PHASE 1: DRAFT IMPLEMENTATION PLAN" in cove_res.prompt
    assert "PHASE 2: IDENTIFIER INVENTORY & CHECKLIST" in cove_res.prompt
    assert "PHASE 3: STATIC API VERIFICATION AUDIT" in cove_res.prompt
    assert "PHASE 4: FINAL VERIFIED PYTHON CODE" in cove_res.prompt


def test_fine_tuning_dataset_generation(tmp_path):
    """Verify dataset generation outputs valid JSONL with >= 50 test records and negative examples."""
    dataset_file = tmp_path / "test_dataset.jsonl"
    count = generate_dataset(str(dataset_file), target_count=60)
    assert count >= 60

    lines = dataset_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == count

    negatives = [json.loads(line) for line in lines if json.loads(line).get("negative_example") is not None]
    assert len(negatives) > 0, "Dataset must include negative refusal examples"

    neg_sample = negatives[0]
    assert neg_sample["expected_behavior"] in ("refuse_hallucination", "refuse_nonexistent_package", "warn_unverified_api")
    assert "prompt" in neg_sample
    assert "correct_code" in neg_sample


def test_prepare_raw_docs_corpus(tmp_path):
    """Verify continued pre-training corpus preparation creates readable text file."""
    corpus_path = tmp_path / "raw_corpus.txt"
    created_path = prepare_raw_docs_corpus(str(corpus_path))
    assert Path(created_path).is_file()
    content = Path(created_path).read_text(encoding="utf-8")
    assert "# Library:" in content
    assert "def " in content

