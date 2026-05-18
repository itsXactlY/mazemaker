#!/usr/bin/env python3
"""
colbert_helper.py — ColBERT-style late-interaction token extractor for Mazemaker.

This is the storage-and-rerank channel that powers the 2nd-stage
ColBERT-style late-interaction rerank in `memory_client.recall`.
BGE-M3 natively emits per-token contextual embeddings as a hybrid
dense+sparse+colbert model; we already pay the forward-pass cost via
the shared embed-server. The only thing missing was storing the
token-level outputs and scoring with them at recall.

Design constraints (set by the deploy environment):

  * The shared embed-server (UNIX socket) currently serves ONLY
    {dense embedding, recall} — it does NOT expose token-level
    embeddings. We are explicitly NOT to restart it (it's holding a
    CUDA model that other pods are using).
  * Therefore this module loads its OWN copy of BGE-M3 in-process,
    on-demand, when colbert is enabled. ~1.4 GB extra VRAM. Singleton
    inside the process so a single Mazemaker instance only loads once.
  * Storage shape: top-N tokens by L2 norm, fp16, packed `(N, dim)`
    little-endian = N*dim*2 bytes per memory. Defaults: N=32, dim=1024
    → 64 KB/memory. 230k memories ≈ 14.7 GB extra disk on the BLOB column.
    OPT-IN ONLY via MM_COLBERT_ENABLED=1.

Public API:

  * `colbert_available()` — cheap probe; True if torch+model load works
  * `encode_tokens(text, top_k=32)` → np.ndarray (top_k, dim) fp16
  * `pack_tokens(arr)` → bytes (for SQLite BLOB / Postgres BYTEA)
  * `unpack_tokens(blob, dim=1024)` → np.ndarray (?, dim) fp16
  * `score_late_interaction(q_tokens, doc_tokens_list)` →
        list[float] in [0, 1], len == len(doc_tokens_list).
        Batched on GPU when available, falls back to numpy.

Top-32 selection rule: filter out [CLS]/[SEP]/[PAD] (and any token whose
attention_mask is 0), sort the surviving rows by L2 norm desc, keep
top-K. If <K real tokens, repeat (cyclic pad) — does not bias
max-similarity since duplicates don't add new max candidates.
"""
from __future__ import annotations

import logging
import os
import struct
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 32
DEFAULT_DIM = 1024  # BGE-M3
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

def _candidate_model_dirs() -> list[Path]:
    """Search path for the BGE-M3 snapshot.

    Operators can prepend additional dirs via MM_MODEL_DIRS (colon-
    separated). Without this, a custom EMBED_MODEL deployed to an
    unusual cache dir was invisible — colbert_helper would refuse to
    load and the rerank channel silently disabled itself.
    """
    extras: list[Path] = []
    raw = os.environ.get("MM_MODEL_DIRS", "")
    if raw:
        for p in raw.split(":"):
            p = p.strip()
            if p:
                extras.append(Path(os.path.expanduser(p)))
    return extras + [
        Path.home() / ".mazemaker" / "engine" / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]


_MODEL_DIR_CANDIDATES = _candidate_model_dirs()


def _resolve_snapshot(model_name: str) -> Optional[str]:
    safe = model_name.replace("/", "--")
    for base in _MODEL_DIR_CANDIDATES:
        cache_base = base / f"models--{safe}"
        refs_main = cache_base / "refs" / "main"
        if refs_main.exists():
            try:
                snap = cache_base / "snapshots" / refs_main.read_text().strip()
                if snap.exists() and (snap / "config.json").exists():
                    return str(snap)
            except Exception:
                pass
        snaps = cache_base / "snapshots"
        if snaps.exists():
            for s in sorted(snaps.iterdir(), reverse=True):
                if (s / "config.json").exists():
                    return str(s)
    return None


_state_lock = threading.Lock()
_state: dict = {
    "tokenizer": None,
    "model": None,
    "device": None,
    "dim": DEFAULT_DIM,
    "loaded": False,
    "failed": False,
    "fail_reason": None,
}


def _load_once() -> bool:
    """Lazy-load tokenizer + transformer. Returns True on success."""
    if _state["loaded"]:
        return True
    if _state["failed"]:
        return False
    with _state_lock:
        if _state["loaded"]:
            return True
        if _state["failed"]:
            return False
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModel
        except Exception as exc:
            _state["failed"] = True
            _state["fail_reason"] = f"transformers/torch import failed: {exc}"
            logger.warning("colbert: %s", _state["fail_reason"])
            return False

        snap = _resolve_snapshot(MODEL_NAME)
        if snap is None:
            _state["failed"] = True
            _state["fail_reason"] = (
                f"colbert: BGE-M3 snapshot not found locally for '{MODEL_NAME}'. "
                "Ensure the embed-server's model dir is populated."
            )
            # Pass fail_reason as an argument, not as the format string.
            # The previous logger.warning(_state["fail_reason"]) treated
            # the message as a printf-style template — if the underlying
            # exception text contained a stray %s it would crash the
            # logger or produce a garbled second message.
            logger.warning("%s", _state["fail_reason"])
            return False

        try:
            import torch as _t
            tok = AutoTokenizer.from_pretrained(snap)
            mdl = AutoModel.from_pretrained(snap)
            dev = os.environ.get("MM_COLBERT_DEVICE")
            if not dev:
                if _t.cuda.is_available():
                    try:
                        free_mb = _t.cuda.mem_get_info(0)[0] / 1024 ** 2
                        dev = "cuda" if free_mb > 1800 else "cpu"
                    except Exception:
                        dev = "cuda"
                else:
                    dev = "cpu"
            mdl = mdl.to(dev).eval()
            _state["tokenizer"] = tok
            _state["model"] = mdl
            _state["device"] = dev
            _state["dim"] = int(getattr(mdl.config, "hidden_size", DEFAULT_DIM))
            _state["loaded"] = True
            logger.info(
                "colbert: BGE-M3 token-emitter ARMED on %s (dim=%d, snap=%s)",
                dev, _state["dim"], snap,
            )
            return True
        except Exception as exc:
            _state["failed"] = True
            _state["fail_reason"] = f"colbert: model load failed: {exc}"
            logger.warning(_state["fail_reason"])
            return False


def colbert_available() -> bool:
    return _load_once()


def colbert_dim() -> int:
    return int(_state.get("dim") or DEFAULT_DIM)


def colbert_status() -> dict:
    return {
        "loaded": bool(_state["loaded"]),
        "failed": bool(_state["failed"]),
        "device": _state.get("device"),
        "dim": _state.get("dim"),
        "fail_reason": _state.get("fail_reason"),
    }


def _select_top_tokens(
    hidden: "np.ndarray",
    mask: "np.ndarray",
    special_token_positions: "np.ndarray",
    top_k: int,
) -> "np.ndarray":
    """Pick the `top_k` non-special, non-pad tokens with the largest L2 norm.

    `hidden` is (T, D). `mask` is (T,) ints/bools where 1 = real token.
    `special_token_positions` is a boolean (T,) marking [CLS]/[SEP].
    Returns (top_k, D) fp32; cyclic-pads when fewer real tokens.
    """
    keep = (mask.astype(bool)) & (~special_token_positions.astype(bool))
    if not keep.any():
        return np.zeros((top_k, hidden.shape[1]), dtype=np.float32)
    real = hidden[keep]
    norms = np.linalg.norm(real, axis=1)
    order = np.argsort(-norms)
    real = real[order]
    if real.shape[0] >= top_k:
        return real[:top_k].astype(np.float32, copy=False)
    reps = (top_k + real.shape[0] - 1) // real.shape[0]
    out = np.tile(real, (reps, 1))[:top_k]
    return out.astype(np.float32, copy=False)


def encode_tokens(text: str, top_k: int = DEFAULT_TOP_K, max_length: int = 512) -> Optional["np.ndarray"]:
    """Run BGE-M3 once and return the top-K token embeddings as fp16.

    Returns None if the model isn't loadable (caller should treat as
    "skip the colbert channel for this memory"). Output shape: (top_k, dim).
    Embeddings are L2-normalised per token row so cosine == dot-product
    downstream.
    """
    if not text or not text.strip():
        return None
    if not _load_once():
        return None
    import torch
    tok = _state["tokenizer"]
    mdl = _state["model"]
    dev = _state["device"]
    enc = tok(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    input_ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    special = enc["special_tokens_mask"].to(dev)
    with torch.no_grad():
        out = mdl(input_ids=input_ids, attention_mask=attn)
        hidden = out.last_hidden_state[0]
        norms = hidden.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        hidden = hidden / norms
        hidden_np = hidden.detach().to("cpu", dtype=torch.float32).numpy()
    mask_np = attn[0].detach().cpu().numpy()
    special_np = special[0].detach().cpu().numpy()
    top = _select_top_tokens(hidden_np, mask_np, special_np, top_k)
    return top.astype(np.float16)


def encode_tokens_batch(
    texts: "list[str]",
    top_k: int = DEFAULT_TOP_K,
    max_length: int = 512,
    batch_size: int = 32,
) -> "list[Optional[np.ndarray]]":
    """Batched variant — used by the migration script."""
    out: list[Optional[np.ndarray]] = [None] * len(texts)
    if not _load_once():
        return out
    import torch
    tok = _state["tokenizer"]
    mdl = _state["model"]
    dev = _state["device"]
    pending_idx: list[int] = []
    pending_text: list[str] = []
    for i, t in enumerate(texts):
        if t and t.strip():
            pending_idx.append(i)
            pending_text.append(t)
    if not pending_text:
        return out
    for start in range(0, len(pending_text), batch_size):
        chunk_idx = pending_idx[start:start + batch_size]
        chunk_text = pending_text[start:start + batch_size]
        enc = tok(
            chunk_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        input_ids = enc["input_ids"].to(dev)
        attn = enc["attention_mask"].to(dev)
        special = enc["special_tokens_mask"].to(dev)
        with torch.no_grad():
            o = mdl(input_ids=input_ids, attention_mask=attn)
            hidden = o.last_hidden_state
            norms = hidden.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            hidden = hidden / norms
            hidden_np = hidden.detach().to("cpu", dtype=torch.float32).numpy()
        mask_np = attn.detach().cpu().numpy()
        special_np = special.detach().cpu().numpy()
        for b, gi in enumerate(chunk_idx):
            top = _select_top_tokens(hidden_np[b], mask_np[b], special_np[b], top_k)
            out[gi] = top.astype(np.float16)
    return out


_HEADER_MAGIC = b"CB1"


def pack_tokens(arr: "np.ndarray") -> bytes:
    """Pack a (K, D) fp16 array as bytes.

    Layout: [3-byte magic 'CB1'][1-byte version=0x01][2-byte K][2-byte D]
            [K*D*2 bytes fp16 little-endian row-major].
    """
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16)
    if arr.ndim != 2:
        raise ValueError(f"pack_tokens: expected 2D, got {arr.shape}")
    k, d = arr.shape
    header = _HEADER_MAGIC + bytes([0x01]) + struct.pack("<HH", k, d)
    body = arr.astype("<f2", copy=False).tobytes()
    return header + body


def unpack_tokens(blob: Optional[bytes], expected_dim: int = DEFAULT_DIM) -> Optional["np.ndarray"]:
    if not blob or len(blob) < 8:
        return None
    if blob[:3] != _HEADER_MAGIC:
        if len(blob) % (expected_dim * 2) != 0:
            return None
        return np.frombuffer(blob, dtype="<f2").reshape(-1, expected_dim)
    version = blob[3]
    if version != 0x01:
        return None
    k, d = struct.unpack("<HH", blob[4:8])
    expected_len = k * d * 2
    if len(blob) - 8 != expected_len:
        return None
    return np.frombuffer(blob[8:8 + expected_len], dtype="<f2").reshape(k, d)


def score_late_interaction(
    q_tokens: "np.ndarray",
    doc_tokens_list: "list[Optional[np.ndarray]]",
) -> "list[float]":
    """ColBERT max-sim scoring: per query token take MAX over doc tokens,
    sum over Q, divide by Q. None doc entries → 0.0."""
    if q_tokens is None or len(doc_tokens_list) == 0:
        return [0.0] * len(doc_tokens_list)
    qf = np.ascontiguousarray(q_tokens, dtype=np.float32)
    Q, D = qf.shape

    try:
        import torch
        dev = _state.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cuda" and len(doc_tokens_list) >= 4:
            valid_mask = [d is not None for d in doc_tokens_list]
            valid_docs = [d for d in doc_tokens_list if d is not None]
            if not valid_docs:
                return [0.0] * len(doc_tokens_list)
            max_k = max(d.shape[0] for d in valid_docs)
            B = len(valid_docs)
            doc_tensor = np.zeros((B, max_k, D), dtype=np.float32)
            for i, d in enumerate(valid_docs):
                doc_tensor[i, :d.shape[0], :] = d.astype(np.float32, copy=False)
            qt = torch.from_numpy(qf).to(dev)
            dt = torch.from_numpy(doc_tensor).to(dev)
            qb = qt.unsqueeze(0).expand(B, Q, D)
            sim = torch.bmm(qb, dt.transpose(1, 2))
            max_per_q, _ = sim.max(dim=2)
            score = max_per_q.sum(dim=1) / float(Q)
            arr = score.detach().to("cpu").numpy()
            out: list[float] = []
            j = 0
            for ok in valid_mask:
                if ok:
                    out.append(float(arr[j]))
                    j += 1
                else:
                    out.append(0.0)
            return out
    except Exception:
        pass

    out = []
    for d in doc_tokens_list:
        if d is None:
            out.append(0.0)
            continue
        df = np.ascontiguousarray(d, dtype=np.float32)
        sim = qf @ df.T
        max_per_q = sim.max(axis=1)
        out.append(float(max_per_q.sum() / Q))
    return out


__all__ = [
    "DEFAULT_TOP_K",
    "colbert_available",
    "colbert_dim",
    "colbert_status",
    "encode_tokens",
    "encode_tokens_batch",
    "pack_tokens",
    "unpack_tokens",
    "score_late_interaction",
]
