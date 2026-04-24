"""Configuration for Neural Memory plugin.

Defaults:
  DB path:               ~/.neural_memory/memory.db
  Embedding backend:     auto (sentence-transformers > tfidf > hash)
  Consolidation interval: 300s
  Max episodic memories:  50000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = str(Path.home() / ".neural_memory" / "memory.db")
DEFAULT_EMBEDDING_BACKEND = "auto"          # auto | sentence-transformers | tfidf | hash
DEFAULT_CONSOLIDATION_INTERVAL = 0          # 0 = disabled
DEFAULT_MAX_EPISODIC = 0                    # 0 = unlimited
DEFAULT_SIMILARITY_THRESHOLD = 0.15         # auto-connect threshold
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_PREFETCH_LIMIT = 5

# Advanced retrieval knobs. Defaults preserve old behavior unless config opts in.
DEFAULT_RETRIEVAL_MODE = "semantic"         # semantic | hybrid | advanced | skynet
DEFAULT_RETRIEVAL_CANDIDATES = 64
DEFAULT_USE_HNSW = "auto"                  # auto | true | false
DEFAULT_LAZY_GRAPH = False
DEFAULT_THINK_ENGINE = "bfs"               # bfs | ppr
DEFAULT_RERANK = False
DEFAULT_RRF_K = 60
DEFAULT_SALIENCE_DECAY_K = 0.03
DEFAULT_PPR_ALPHA = 0.15
DEFAULT_PPR_ITERS = 20
DEFAULT_PPR_HOPS = 2

# Session write policy: durable extraction by default; raw archives are opt-in.
DEFAULT_SESSION_EXTRACT_FACTS = True
DEFAULT_SESSION_FACT_LIMIT = 5
DEFAULT_STORE_RAW_TURNS = False
DEFAULT_ARCHIVE_RAW_TURNS = False


def get_config() -> Dict[str, Any]:
    """Read neural memory config from ~/.hermes/config.yaml.

    Falls back to defaults if the file doesn't exist or the section is missing.
    """
    config = {
        "db_path": DEFAULT_DB_PATH,
        "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
        "consolidation_interval": DEFAULT_CONSOLIDATION_INTERVAL,
        "max_episodic": DEFAULT_MAX_EPISODIC,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "search_limit": DEFAULT_SEARCH_LIMIT,
        "prefetch_limit": DEFAULT_PREFETCH_LIMIT,
        "retrieval_mode": DEFAULT_RETRIEVAL_MODE,
        "retrieval_candidates": DEFAULT_RETRIEVAL_CANDIDATES,
        "use_hnsw": DEFAULT_USE_HNSW,
        "lazy_graph": DEFAULT_LAZY_GRAPH,
        "think_engine": DEFAULT_THINK_ENGINE,
        "rerank": DEFAULT_RERANK,
        "rrf_k": DEFAULT_RRF_K,
        "salience_decay_k": DEFAULT_SALIENCE_DECAY_K,
        "ppr_alpha": DEFAULT_PPR_ALPHA,
        "ppr_iters": DEFAULT_PPR_ITERS,
        "ppr_hops": DEFAULT_PPR_HOPS,
        "session_extract_facts": DEFAULT_SESSION_EXTRACT_FACTS,
        "session_fact_limit": DEFAULT_SESSION_FACT_LIMIT,
        "store_raw_turns": DEFAULT_STORE_RAW_TURNS,
        "archive_raw_turns": DEFAULT_ARCHIVE_RAW_TURNS,
    }

    try:
        from hermes_cli.config import load_config
        hermes_cfg = load_config()
        neural_cfg = hermes_cfg.get("memory", {}).get("neural", {}) or {}
        if isinstance(neural_cfg, dict):
            config.update({k: v for k, v in neural_cfg.items() if v is not None})
    except Exception:
        pass

    # Expand $HERMES_HOME and ~ in db_path
    db_path = config.get("db_path", DEFAULT_DB_PATH)
    if isinstance(db_path, str):
        try:
            from hermes_constants import get_hermes_home
            db_path = db_path.replace("$HERMES_HOME", str(get_hermes_home()))
            db_path = db_path.replace("${HERMES_HOME}", str(get_hermes_home()))
        except Exception:
            pass
        db_path = os.path.expanduser(db_path)
        config["db_path"] = db_path

    return config
