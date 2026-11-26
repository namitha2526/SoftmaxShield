"""Generation-side defences for SoftmaxShield: RAG, Vector Store, and Prompt Templates."""

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
    format_prompt_with_retrieved_docs,
)
from softmaxshield.generation.rag_engine import RAGGenerationEngine
from softmaxshield.generation.vector_store import (
    FAISSVectorStore,
    InMemoryVectorStore,
    get_default_vector_store,
)

__all__ = [
    "DocChunk",
    "LibraryDocsCatalog",
    "get_ten_libraries_docs",
    "FAISSVectorStore",
    "InMemoryVectorStore",
    "get_default_vector_store",
    "RAGGenerationEngine",
    "PromptStrategy",
    "build_constrained_prompt",
    "build_few_shot_prompt",
    "build_chain_of_verification_prompt",
    "format_prompt_with_retrieved_docs",
]

