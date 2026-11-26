"""RAG Generation Engine integrating documentation retrieval with generation-side prompts and metadata filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
from softmaxshield.generation.vector_store import FAISSVectorStore, InMemoryVectorStore


@dataclass
class RAGPromptResult:
    """Output from the RAG generation engine containing prompt and retrieved evidence."""

    prompt: str
    retrieved_chunks: List[DocChunk]
    strategy: PromptStrategy
    detected_libraries: List[str]


class RAGGenerationEngine:
    """Retrieval-Augmented Generation engine for grounding LLM code generation."""

    def __init__(
        self,
        catalog: Optional[LibraryDocsCatalog] = None,
        use_faiss: bool = False,
    ) -> None:
        self.catalog = catalog or get_ten_libraries_docs()
        self.vector_store = FAISSVectorStore() if use_faiss else InMemoryVectorStore()
        # Index all documentation chunks
        self.vector_store.add_documents(self.catalog.chunks)

    def detect_libraries(self, query: str) -> List[str]:
        """Detect which of the 10 libraries are mentioned in or relevant to the user query."""
        known_libs = [
            "numpy", "pandas", "requests", "flask", "fastapi",
            "scikit-learn", "torch", "matplotlib", "sqlalchemy", "pydantic"
        ]
        alias_map = {
            "np": "numpy",
            "pd": "pandas",
            "pytorch": "torch",
            "sklearn": "scikit-learn",
            "plt": "matplotlib",
        }
        detected = []
        q_lower = query.lower()

        for lib in known_libs:
            if lib in q_lower:
                detected.append(lib)

        for alias, lib in alias_map.items():
            if alias in q_lower and lib not in detected:
                detected.append(lib)

        return detected

    def retrieve_signatures(
        self,
        query: str,
        top_k: int = 4,
        library_filter: Optional[List[str]] = None,
        version_filter: Optional[Union[str, Dict[str, str]]] = None,
    ) -> List[DocChunk]:
        """Retrieve the top-k most relevant function and class signatures with metadata filters."""
        search_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            library_filter=library_filter,
            version_filter=version_filter,
        )
        return [chunk for chunk, score in search_results]

    def update_library_documentation(
        self,
        library: str,
        version: str,
        new_chunks: List[DocChunk],
    ) -> int:
        """Update the underlying catalog and re-index the vector store with new library version documentation."""
        total = self.catalog.update_library_docs(library, version, new_chunks)
        # Rebuild vector store index
        self.vector_store = InMemoryVectorStore()
        self.vector_store.add_documents(self.catalog.chunks)
        return total

    def build_grounded_prompt(
        self,
        user_query: str,
        strategy: PromptStrategy = PromptStrategy.CHAIN_OF_VERIFICATION,
        top_k: int = 4,
        allowed_libraries: Optional[List[str]] = None,
        version_filter: Optional[Union[str, Dict[str, str]]] = None,
    ) -> RAGPromptResult:
        """Retrieve relevant signatures and inject them into the selected prompt template."""
        detected = self.detect_libraries(user_query)
        lib_filter = allowed_libraries or detected or None

        retrieved = self.retrieve_signatures(
            query=user_query,
            top_k=top_k,
            library_filter=lib_filter,
            version_filter=version_filter,
        )

        if strategy == PromptStrategy.CONSTRAINED:
            effective_libs = allowed_libraries or (detected if detected else ["standard library"])
            prompt = build_constrained_prompt(
                user_query=user_query,
                allowed_libraries=effective_libs,
                retrieved_chunks=retrieved,
            )
        elif strategy == PromptStrategy.FEW_SHOT:
            prompt = build_few_shot_prompt(
                user_query=user_query,
                retrieved_chunks=retrieved,
            )
        elif strategy == PromptStrategy.CHAIN_OF_VERIFICATION:
            prompt = build_chain_of_verification_prompt(
                user_query=user_query,
                retrieved_chunks=retrieved,
            )
        else:
            raise ValueError(f"Unknown PromptStrategy: {strategy}")

        return RAGPromptResult(
            prompt=prompt,
            retrieved_chunks=retrieved,
            strategy=strategy,
            detected_libraries=detected,
        )
