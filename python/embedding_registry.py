"""Embedding backend registry — env-var-driven dispatch + BGE-M3 placeholder.

Per Sprint 2 Phase 7 Commit 6 / handoff Section 5.6. Wraps the existing
EmbeddingProvider in embed_provider.py with two additions:

1. `get_embedding_backend(name=None, allow_missing=False)`: top-level factory
   that resolves backend choice from arg → env `NEURAL_MEMORY_EMBED_BACKEND`
   → 'auto'. Returns the underlying backend object (with .embed() method).

2. `BgeM3Backend`: optional BGE-M3 hybrid (dense + sparse + multi-vector)
   adapter. Lazy imports FlagEmbedding; raises BackendUnavailable when the
   model library isn't installed. Use allow_missing=True to get None instead.

Existing MiniLM/TF-IDF/Hash fallbacks continue to work unchanged. No mandatory
heavy model dependency added to default install.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class BackendUnavailable(RuntimeError):
    """Raised when a requested embedding backend cannot be loaded."""


class BgeM3Backend:
    """BGE-M3 hybrid retrieval backbone (dense + sparse + multi-vector).

    Optional dependency: requires `FlagEmbedding` package to be installed.
    Without it, instantiation raises BackendUnavailable.

    Surface matches the existing backend protocol — `.embed(text) -> list[float]`
    plus optional `.embed_sparse(text)` and `.embed_multi(text)` for hybrid use.
    """

    dim = 1024  # BGE-M3 dense vector dimension

    def __init__(self) -> None:
        # Per Reviewer #1: ImportError-only catch was insufficient.
        # BGEM3FlagModel construction can raise OSError (missing weights),
        # RuntimeError (CUDA mismatch), or HuggingFace network errors.
        # All of these should signal "backend unavailable" — not propagate
        # as uncaught exceptions through allow_missing=True callers.
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError as exc:
            raise BackendUnavailable(
                "BGE-M3 requires FlagEmbedding: pip install FlagEmbedding"
            ) from exc
        try:
            self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        except Exception as exc:
            raise BackendUnavailable(
                f"BGE-M3 model load failed (network? CUDA? weights?): {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        result = self._model.encode([text], return_dense=True,
                                    return_sparse=False, return_colbert_vecs=False)
        return result["dense_vecs"][0].tolist()

    def embed_sparse(self, text: str) -> dict[str, float]:
        """Sparse lexical token weights — returns {token: weight} dict.

        Hindsight-style sparse retrieval channel; can be persisted in
        memories.metadata_json for query-time scoring without re-embedding.
        """
        result = self._model.encode([text], return_dense=False,
                                    return_sparse=True, return_colbert_vecs=False)
        return result["lexical_weights"][0]


def get_embedding_backend(
    name: Optional[str] = None,
    *,
    allow_missing: bool = False,
) -> Optional[Any]:
    """Resolve and return an embedding backend.

    Resolution order:
        1. `name` argument (if provided)
        2. `NEURAL_MEMORY_EMBED_BACKEND` env var
        3. 'auto'

    Recognized names:
        - 'auto'    : same as EmbeddingProvider's auto-detect
        - 'default' : alias for 'auto' (used by tests)
        - 'sentence-transformers' : MiniLM via sentence-transformers
        - 'tfidf'   : TF-IDF + SVD
        - 'hash'    : deterministic hash backend (zero deps)
        - 'bge-m3'  : BGE-M3 hybrid (requires FlagEmbedding)

    Args:
        name: explicit backend name; takes priority over env var.
        allow_missing: if True, return None when backend is unavailable
            instead of raising BackendUnavailable.

    Returns:
        Backend instance with .embed() method, or None if allow_missing=True
        and backend is unavailable.
    """
    if name is None:
        name = os.environ.get("NEURAL_MEMORY_EMBED_BACKEND", "auto")

    if name in ("auto", "default"):
        from embed_provider import EmbeddingProvider
        provider = EmbeddingProvider(backend="auto")
        return provider.backend

    if name == "bge-m3":
        try:
            return BgeM3Backend()
        except BackendUnavailable:
            if allow_missing:
                return None
            raise

    # Other named backends delegate to EmbeddingProvider's existing dispatch.
    from embed_provider import EmbeddingProvider
    try:
        provider = EmbeddingProvider(backend=name)
        return provider.backend
    except Exception as exc:
        if allow_missing:
            return None
        raise BackendUnavailable(f"backend {name!r} unavailable: {exc}") from exc
