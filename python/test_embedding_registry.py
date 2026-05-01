"""Acceptance tests for Sprint 2 Phase 7 Commit 6 — embedding backend registry.

Per addendum lines 338-374. Five contracts:

  1. Default embedding backend still loads + .embed() returns a vector
  2. get_embedding_backend(name='default') returns a working backend
  3. get_embedding_backend('bge-m3', allow_missing=True) returns None or
     a backend with .embed (graceful when FlagEmbedding not installed)
  4. get_embedding_backend('hash') returns the deterministic hash backend
  5. NEURAL_MEMORY_EMBED_BACKEND env var dispatch works

Stdlib unittest. Run:
    python3 python/test_embedding_registry.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_registry import (  # noqa: E402
    BackendUnavailable,
    BgeM3Backend,
    get_embedding_backend,
)


class EmbeddingRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = os.environ.pop("NEURAL_MEMORY_EMBED_BACKEND", None)

    def tearDown(self) -> None:
        if self._saved_env is not None:
            os.environ["NEURAL_MEMORY_EMBED_BACKEND"] = self._saved_env
        else:
            os.environ.pop("NEURAL_MEMORY_EMBED_BACKEND", None)

    def test_default_backend_loads_and_embeds(self) -> None:
        backend = get_embedding_backend(name="default")
        self.assertIsNotNone(backend)
        vec = backend.embed("test memory")
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)

    def test_auto_backend_loads_and_embeds(self) -> None:
        backend = get_embedding_backend(name="auto")
        self.assertIsNotNone(backend)
        vec = backend.embed("hello world")
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)

    def test_hash_backend_is_deterministic(self) -> None:
        backend = get_embedding_backend(name="hash")
        self.assertIsNotNone(backend)
        v1 = backend.embed("repeatable text")
        v2 = backend.embed("repeatable text")
        self.assertEqual(v1, v2)

    def test_bge_m3_is_optional(self) -> None:
        backend = get_embedding_backend(name="bge-m3", allow_missing=True)
        # Either None (FlagEmbedding not installed) or a working backend
        if backend is not None:
            self.assertTrue(hasattr(backend, "embed"))

    def test_bge_m3_raises_without_allow_missing(self) -> None:
        # If FlagEmbedding is installed, this won't raise — skip gracefully
        try:
            backend = BgeM3Backend()
            # Backend loaded; the test is only meaningful when missing
            self.skipTest("FlagEmbedding installed; cannot test missing path")
        except BackendUnavailable:
            with self.assertRaises(BackendUnavailable):
                get_embedding_backend(name="bge-m3", allow_missing=False)

    def test_env_var_dispatches_backend(self) -> None:
        os.environ["NEURAL_MEMORY_EMBED_BACKEND"] = "hash"
        backend = get_embedding_backend()  # no name arg → reads env
        self.assertEqual(backend.__class__.__name__, "HashBackend")

    def test_explicit_name_overrides_env_var(self) -> None:
        """Explicit name MUST win over env var. Per Reviewer #2 finding,
        previous version of this test didn't actually assert override
        behavior — only that something returned."""
        os.environ["NEURAL_MEMORY_EMBED_BACKEND"] = "hash"
        # Get the env-only resolution first (should be HashBackend)
        env_only = get_embedding_backend()
        self.assertEqual(env_only.__class__.__name__, "HashBackend",
                         "env var dispatch baseline broken")
        # Now explicit override — should NOT be HashBackend (auto/default
        # resolves via EmbeddingProvider's auto-detect, which prefers
        # sentence-transformers > TF-IDF > Hash)
        explicit = get_embedding_backend(name="default")
        self.assertNotEqual(
            explicit.__class__.__name__, "HashBackend",
            "explicit name='default' did not override NEURAL_MEMORY_EMBED_BACKEND=hash; "
            f"got {explicit.__class__.__name__}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
