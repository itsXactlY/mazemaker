"""GPU-accelerated vector recall for mazemaker.

Loads pre-computed embeddings onto GPU for sub-millisecond cosine similarity search.
Much faster than Python loop or C++ Hopfield network.

Usage:
    engine = GpuRecallEngine()
    engine.load()
    results = engine.recall("query text", k=5)
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Per-DB isolation: the production GpuRecallEngine cache lives in
# ~/.mazemaker/engine/gpu_cache/ and is hardcoded for the operator's main
# DB. When a tool / harness opens a *different* DB (per-conv benchmarks,
# ephemeral test stores, etc), the embeddings cached here do NOT match
# the IDs in the open DB — recall_batch returns IDs from the production
# corpus, which leak into REM bridge candidates as -1 (or stray IDs from
# another DB) and produce silently-broken connection_history rows.
# Override with MAZEMAKER_GPU_CACHE_DIR to point at an isolated cache
# (or an empty dir to force fallback to per-query SQLite recall).
_CACHE_DIR = Path(os.environ.get(
    "MAZEMAKER_GPU_CACHE_DIR",
    str(Path.home() / ".mazemaker" / "engine" / "gpu_cache"),
))
_EMBEDDINGS_PATH = _CACHE_DIR / "embeddings.npy"
# Metadata moved from pickle → JSON. pickle.load on operator-writable
# files in ~/.mazemaker/engine/gpu_cache/ is an RCE primitive; the dict
# is just {ids: int[], labels: str[], contents: str[]}, all JSON-trivial.
_METADATA_PATH = _CACHE_DIR / "metadata.json"


def _derive_per_db_cache_dir(db_path) -> Path:
    """Derive a stable, DB-fingerprinted cache subdir from db_path.

    Production has historically used a single global cache for ~/.mazemaker/engine/memory.db.
    Per-DB cache eliminates cross-DB contamination when tools/harnesses
    open a DB different from the production one (per-conv benchmarks,
    customer pods, multi-tenant test fixtures, etc.).

    Returns the global default when db_path is None, falsy, or matches the
    historical production DB — preserves backwards compatibility.
    """
    import hashlib
    if not db_path:
        return _CACHE_DIR
    s = str(Path(db_path).resolve())
    # Preserve the production global cache for the canonical operator DB.
    if s == str(Path.home() / ".mazemaker" / "memory.db") or \
       s == str(Path.home() / ".mazemaker" / "engine" / "memory.db"):
        return _CACHE_DIR
    digest = hashlib.sha256(s.encode()).hexdigest()[:12]
    return _CACHE_DIR.parent / "gpu_cache_per_db" / digest


class GpuRecallEngine:
    """GPU-accelerated cosine similarity search over mazemaker embeddings."""

    def __init__(self, db_path=None):
        self._device = None
        self._emb_tensor = None  # (N, dim) float32 on GPU
        # Cached row-normalised view of _emb_tensor. Built lazily the first
        # time recall() encounters a non-unit query (so we don't pay the
        # normalisation cost on already-normalised models like e5-large).
        self._emb_tensor_normed = None
        self._ids = []
        self._labels = []
        self._contents = []
        self._dim = 1024
        self._loaded = False
        # Per-DB cache isolation — picks the global cache for the production
        # DB, an isolated sha256-derived subdir for any other DB. Use the
        # property `cache_dir` to read; downstream code (build_gpu_cache)
        # should use this rather than the module-level _CACHE_DIR.
        self._cache_dir: Path = _derive_per_db_cache_dir(db_path)
        self._emb_path: Path = self._cache_dir / "embeddings.npy"
        self._meta_path: Path = self._cache_dir / "metadata.json"

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def load(self, embed_fn=None, embed_batch_fn=None) -> bool:
        """Load embeddings onto GPU.

        Args:
            embed_fn: Optional callable(text) -> list[float] for query embedding.
                      If None, uses sentence-transformers.
            embed_batch_fn: Optional callable(list[str]) -> list[list[float]]
                            for batched query embedding. Used by recall_batch
                            to encode N queries in a single backend call,
                            collapsing the per-query embed-server IPC overhead.

        Returns:
            True if loaded successfully.
        """
        import torch

        # Select device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # Load cached embeddings — per-DB cache (falls back to global for
        # the production DB).
        if not self._emb_path.exists():
            return False

        emb_array = np.load(str(self._emb_path))
        self._dim = emb_array.shape[1]

        # If metadata.json is missing (e.g. a legacy cache still holds
        # metadata.pkl) refuse to load. Callers rebuild the cache via
        # build_gpu_cache.build(); pickle is never loaded.
        if not self._meta_path.exists():
            return False
        with open(str(self._meta_path), "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._ids = meta["ids"]
        self._labels = meta["labels"]
        self._contents = meta["contents"]

        # Move to GPU
        self._emb_tensor = torch.tensor(emb_array, device=self._device, dtype=torch.float32)
        # Invalidate any prior normalised cache — a reload (e.g. after the
        # source DB grew) means the row layout changed.
        self._emb_tensor_normed = None

        # Store embed function
        self._embed_fn = embed_fn
        self._embed_batch_fn = embed_batch_fn

        self._loaded = True
        return True

    def load_from_store(self, store, embed_fn=None, embed_batch_fn=None) -> bool:
        """Populate the GPU tensor directly from a Mazemaker store.

        Use this when the on-disk gpu_cache is absent or untrusted —
        the canonical example is the PostgreSQL backend, which keeps
        embeddings in a pgvector column rather than in a SQLite blob,
        so build_gpu_cache.py (a SQLite-only tool) can't help. Without
        this path, MM_DB_BACKEND=postgres always left self._gpu = None
        and NREM/REM's GPU-fast think_ids / recall_batch silently
        degraded to per-call CPU/numpy.

        Both SQLiteStore and PostgresStore expose get_all() with the
        embedding column already decoded to list[float], so the loader
        is fully backend-agnostic.

        Returns True on success, False if the store is empty or torch
        is unavailable.
        """
        try:
            import torch
        except ImportError:
            return False

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        rows = store.get_all()
        if not rows:
            return False

        ids: list[int] = []
        labels: list[str] = []
        contents: list[str] = []
        vecs: list[list[float]] = []
        for r in rows:
            emb = r.get("embedding")
            if not emb:
                continue
            ids.append(int(r["id"]))
            labels.append(r.get("label") or "")
            contents.append(r.get("content") or "")
            vecs.append(emb)

        if not ids:
            return False

        arr = np.asarray(vecs, dtype=np.float32)
        self._dim = int(arr.shape[1])
        self._emb_tensor = torch.tensor(arr, device=self._device, dtype=torch.float32)
        self._emb_tensor_normed = None
        self._ids = ids
        self._labels = labels
        self._contents = contents
        self._embed_fn = embed_fn
        self._embed_batch_fn = embed_batch_fn
        self._loaded = True
        return True

    def add_one(self, mem_id: int, label: str, content: str, embedding) -> None:
        """Append one new memory to the in-GPU tensor. Called from
        Mazemaker.remember() so newly-stored memories are searchable on
        GPU immediately (no cache rebuild needed).

        Embedding can be list[float], np.ndarray, or torch.Tensor.
        """
        if not self._loaded:
            return
        import torch
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, device=self._device, dtype=torch.float32)
        else:
            embedding = embedding.to(self._device, dtype=torch.float32)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        self._emb_tensor = torch.cat([self._emb_tensor, embedding], dim=0)
        # Invalidate normalised cache; recall() will recompute on next miss.
        self._emb_tensor_normed = None
        self._ids.append(int(mem_id))
        self._labels.append(label or "")
        self._contents.append(content or "")

    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Search memories by semantic similarity.

        Args:
            query: Search query text.
            k: Number of results.

        Returns:
            List of {id, label, content, similarity} dicts.
        """
        if not self._loaded:
            return []

        import torch

        # Embed query
        if self._embed_fn is None:
            raise RuntimeError("No embed function configured")

        query_vec = self._embed_fn(query)
        # Dim guard: a query produced by the active backend (e.g. 1024-d
        # FastEmbed) against a GPU cache that was populated by a different
        # backend (e.g. 384-d MiniLM) would crash inside torch.matmul. The
        # outer try/except in memory_client catches the crash, but doing
        # the check up-front skips the exception machinery and yields a
        # clean fast path through the CPU/HNSW fallbacks instead.
        if len(query_vec) != self._dim:
            return []
        q = torch.tensor(query_vec, device=self._device, dtype=torch.float32)

        # Always row-normalise the stored tensor + the query before the
        # matmul. The DB can carry mixed-magnitude rows (some written by
        # an older engine that didn't pass normalize_embeddings=True; some
        # from bulk imports that did). Without normalising both sides we
        # get raw projection scores in arbitrary range — REM's filter of
        # `0.3 < sim < 0.95` then drops every candidate. Cost is one-time
        # per process: norms cached in self._emb_tensor_normed.
        if self._emb_tensor_normed is None:
            norms = torch.norm(self._emb_tensor, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)
            self._emb_tensor_normed = self._emb_tensor / norms
        emb_normed = self._emb_tensor_normed
        q_mag = torch.norm(q)
        if q_mag > 1e-12:
            q = q / q_mag

        # Cosine similarity (dot product of normalized vectors) — always
        # in [-1, 1] now, so REM's similarity-window filter behaves.
        sims = torch.matmul(emb_normed, q)

        # Top-k
        top_k = torch.topk(sims, min(k, len(self._ids)))

        results = []
        for idx, sim in zip(top_k.indices.cpu().numpy(), top_k.values.cpu().numpy()):
            results.append({
                "id": self._ids[idx],
                "label": self._labels[idx],
                "content": self._contents[idx],
                "similarity": float(sim),
            })

        return results

    def recall_batch(self, queries: "list[str]", k: int = 5) -> "list[list[dict]]":
        """Batched semantic recall — encodes all N queries in a single
        embed_batch call and runs ONE matmul (B, dim) × (N_corpus, dim).T
        followed by topk along the batch axis. Bypasses N rounds of
        embed-server IPC + N CPU-Python result-build loops.

        Returns a list of result lists, indexed parallel to `queries`.
        Falls back to a sequential loop when the batch embed function
        wasn't supplied to load() — the public API stays the same.
        """
        if not self._loaded or not queries:
            return [[] for _ in queries]

        import torch

        if self._embed_batch_fn is None:
            return [self.recall(q, k=k) for q in queries]

        try:
            vecs = self._embed_batch_fn(list(queries))
        except Exception:
            return [self.recall(q, k=k) for q in queries]
        if not vecs or len(vecs) != len(queries):
            return [self.recall(q, k=k) for q in queries]

        Q = torch.tensor(vecs, device=self._device, dtype=torch.float32)
        if Q.ndim != 2 or Q.shape[1] != self._dim:
            return [[] for _ in queries]

        # Lazy build / reuse normalised corpus tensor (same as recall())
        if self._emb_tensor_normed is None:
            norms = torch.norm(self._emb_tensor, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)
            self._emb_tensor_normed = self._emb_tensor / norms

        # Row-normalise queries
        q_norms = torch.norm(Q, dim=1, keepdim=True)
        q_norms = torch.clamp(q_norms, min=1e-12)
        Q = Q / q_norms

        # Batch cosine: (B, dim) @ (dim, N) → (B, N)
        sims = torch.matmul(Q, self._emb_tensor_normed.T)

        eff_k = min(k, sims.shape[1])
        top_k = torch.topk(sims, eff_k, dim=1)
        idx_cpu = top_k.indices.cpu().numpy()
        val_cpu = top_k.values.cpu().numpy()

        out: "list[list[dict]]" = []
        for b in range(idx_cpu.shape[0]):
            row = []
            for j in range(eff_k):
                idx = int(idx_cpu[b, j])
                row.append({
                    "id": self._ids[idx],
                    "label": self._labels[idx],
                    "content": self._contents[idx],
                    "similarity": float(val_cpu[b, j]),
                })
            out.append(row)
        return out

    def stats(self) -> dict:
        """Return engine stats."""
        return {
            "loaded": self._loaded,
            "device": str(self._device) if self._device else None,
            "memories": len(self._ids),
            "dim": self._dim,
            "vram_mb": self._emb_tensor.element_size() * self._emb_tensor.nelement() / 1024 / 1024 if self._emb_tensor is not None else 0,
        }

    def shutdown(self):
        """Free GPU memory."""
        if self._emb_tensor is not None:
            del self._emb_tensor
            self._emb_tensor = None
        if self._emb_tensor_normed is not None:
            del self._emb_tensor_normed
            self._emb_tensor_normed = None
        if self._device and self._device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
        self._loaded = False
