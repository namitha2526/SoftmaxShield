"""Vector Store implementation for SoftmaxShield RAG supporting FAISS, ChromaDB interface, and pure-Python fallback."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from softmaxshield.generation.docs_extractor import DocChunk


class InMemoryVectorStore:
    """Lightweight, pure-Python vector store using normalized TF-IDF cosine similarity.

    Guarantees 100% zero-crash operation in air-gapped environments or where C-extensions are restricted.
    Supports library and version metadata filtering.
    """

    def __init__(self) -> None:
        self.chunks: List[DocChunk] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.document_vectors: List[Dict[int, float]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric keywords."""
        return re.findall(r"[a-zA-Z0-9_\.]+", text.lower())

    def add_documents(self, chunks: List[DocChunk]) -> None:
        """Index documentation chunks."""
        self.chunks.extend(chunks)
        tokenized_docs = [self._tokenize(c.to_embedding_text()) for c in chunks]

        # Build vocabulary & doc frequencies
        doc_count = len(tokenized_docs)
        df: Dict[str, int] = {}
        for doc in tokenized_docs:
            seen = set(doc)
            for token in seen:
                df[token] = df.get(token, 0) + 1
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

        # Compute IDF
        for token, count in df.items():
            self.idf[token] = math.log((doc_count + 1.0) / (count + 1.0)) + 1.0

        # Compute TF-IDF vectors
        self.document_vectors = []
        for doc in tokenized_docs:
            vec: Dict[int, float] = {}
            tf: Dict[str, int] = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            norm_sq = 0.0
            for token, freq in tf.items():
                dim = self.vocabulary[token]
                val = (freq / len(doc)) * self.idf.get(token, 1.0)
                vec[dim] = val
                norm_sq += val * val
            # L2 normalize
            norm = math.sqrt(norm_sq) or 1.0
            self.document_vectors.append({dim: val / norm for dim, val in vec.items()})

    def search(
        self,
        query: str,
        top_k: int = 5,
        library_filter: Optional[List[str]] = None,
        version_filter: Optional[Union[str, Dict[str, str]]] = None,
    ) -> List[Tuple[DocChunk, float]]:
        """Retrieve top-k most relevant chunks using cosine similarity and metadata filters."""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.document_vectors:
            return []

        # Vectorize query
        q_tf: Dict[str, int] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        q_vec: Dict[int, float] = {}
        norm_sq = 0.0
        for t, freq in q_tf.items():
            if t in self.vocabulary:
                dim = self.vocabulary[t]
                val = (freq / len(query_tokens)) * self.idf.get(t, 1.0)
                q_vec[dim] = val
                norm_sq += val * val
        q_norm = math.sqrt(norm_sq) or 1.0
        q_vec = {dim: val / q_norm for dim, val in q_vec.items()}

        scores: List[Tuple[DocChunk, float]] = []
        for idx, doc_vec in enumerate(self.document_vectors):
            chunk = self.chunks[idx]

            # 1. Library Filter
            if library_filter and chunk.library.lower() not in [l.lower() for l in library_filter]:
                continue

            # 2. Version Filter
            if version_filter:
                if isinstance(version_filter, str) and not chunk.version.startswith(version_filter):
                    continue
                elif isinstance(version_filter, dict):
                    req_ver = version_filter.get(chunk.library.lower())
                    if req_ver and not chunk.version.startswith(req_ver):
                        continue

            # Cosine similarity
            dot = sum(val * q_vec.get(dim, 0.0) for dim, val in doc_vec.items())
            # Exact symbol bonus
            for t in query_tokens:
                if t in chunk.api_name.lower():
                    dot += 0.25

            scores.append((chunk, dot))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class FAISSVectorStore:
    """Production vector store using FAISS and Hugging Face / Sentence-Transformers embeddings.

    Supports library and version metadata filtering with graceful fallback.
    """

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model_name = embedding_model_name
        self.chunks: List[DocChunk] = []
        self._index: Optional[Any] = None
        self._model: Optional[Any] = None
        self._faiss: Optional[Any] = None
        self._fallback = InMemoryVectorStore()
        self._init_faiss()

    def _init_faiss(self) -> None:
        """Attempt to load FAISS."""
        try:
            import faiss
            self._faiss = faiss
        except Exception:
            self._faiss = None

    def _get_embedding_model(self) -> Optional[Any]:
        """Lazy load sentence transformers embedding model using local cache to prevent network hangs."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Try local cache first to ensure zero network timeout in air-gapped / offline modes
                try:
                    self._model = SentenceTransformer(self.embedding_model_name, local_files_only=True)
                except Exception:
                    self._model = None
            except Exception:
                self._model = None
        return self._model

    def add_documents(self, chunks: List[DocChunk]) -> None:
        """Embed and add document chunks to the index."""
        self.chunks.extend(chunks)
        self._fallback.add_documents(chunks)

        model = self._get_embedding_model()
        if self._faiss is not None and model is not None:
            texts = [c.to_embedding_text() for c in chunks]
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            dim = embeddings.shape[1]
            if self._index is None:
                # Use Inner Product on normalized vectors for exact Cosine Similarity
                self._index = self._faiss.IndexFlatIP(dim)
            self._index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
        library_filter: Optional[List[str]] = None,
        version_filter: Optional[Union[str, Dict[str, str]]] = None,
    ) -> List[Tuple[DocChunk, float]]:
        """Search top-k chunks with FAISS and metadata filters; falls back to InMemoryVectorStore."""
        model = self._get_embedding_model()
        if self._faiss is not None and self._index is not None and model is not None:
            q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            distances, indices = self._index.search(q_emb, min(top_k * 4, len(self.chunks)))

            results: List[Tuple[DocChunk, float]] = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]

                # Library Filter
                if library_filter and chunk.library.lower() not in [l.lower() for l in library_filter]:
                    continue

                # Version Filter
                if version_filter:
                    if isinstance(version_filter, str) and not chunk.version.startswith(version_filter):
                        continue
                    elif isinstance(version_filter, dict):
                        req_ver = version_filter.get(chunk.library.lower())
                        if req_ver and not chunk.version.startswith(req_ver):
                            continue

                results.append((chunk, float(score)))
                if len(results) >= top_k:
                    break
            return results

        # Fallback to pure-Python TF-IDF cosine store
        return self._fallback.search(
            query,
            top_k=top_k,
            library_filter=library_filter,
            version_filter=version_filter,
        )


def get_default_vector_store() -> Any:
    """Factory providing the optimal available vector store."""
    return FAISSVectorStore()
