"""mcp_schemas.py — Mazemaker MCP tool schemas + dispatch tables.

This is the single source of truth for the engine's MCP tool surface.
Both the in-process Hermes plugin (`python/__init__.py`) and the
customer-pod adapter (`mazemaker-v2-stack/backend/client/pod/mazemaker/tools.py`)
import from this module so there is exactly one schema definition for
every Mazemaker MCP tool.

The module is import-clean — no Hermes dependency, no engine side-
effects, no I/O.  Safe to load from any context.
"""

from __future__ import annotations

MAZEMAKER_REMEMBER_SCHEMA = {
    "name": "mazemaker_remember",
    "description": (
        "STORE a fact, preference, decision, or piece of context that the user "
        "will expect you to remember in future turns or sessions. Call this WHENEVER "
        "the user states a durable fact about themselves, their setup, their "
        "preferences, or makes a decision. Examples: 'I prefer fish shell' → call "
        "this; 'we decided to use FastEmbed' → call this; 'the bug was at line 870' "
        "→ call this. Auto-embeds and auto-connects to similar memories. Storing "
        "is cheap — err on the side of more, with a stable label."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The DURABLE fact to remember. One sentence ideally. NOT the "
                    "whole conversation, NOT your reasoning — just the fact the "
                    "user wants you to recall later."
                ),
            },
            "label": {
                "type": "string",
                "description": (
                    "Stable topic slug like 'pref:shell', 'decision:embedding', "
                    "'bug:dream-engine', 'fact:user-name'. Reusing labels lets "
                    "future writes update or fuse with the existing memory."
                ),
            },
        },
        "required": ["content"],
    },
}

MAZEMAKER_RECALL_SCHEMA = {
    "name": "mazemaker_recall",
    "description": (
        "SEARCH the persistent mazemaker. Call this AT THE START of every turn "
        "where the user references prior context, asks 'do you remember…', mentions "
        "their preferences/setup/files/projects, or where you would otherwise be "
        "tempted to answer from your parametric weights. Returns top-k memories "
        "ranked by semantic+graph relevance — these are facts you've stored across "
        "sessions, more reliable than your training data for anything user-specific. "
        "Cheap to call; missing a recall when one was warranted is the #1 way to "
        "look stupid."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search phrase. Use the user's key entities + topic. "
                    "E.g. 'shell preference', 'fastembed model choice', "
                    "'dream engine bug'. Concrete is better than abstract."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 5).",
            },
        },
        "required": ["query"],
    },
}

MAZEMAKER_THINK_SCHEMA = {
    "name": "mazemaker_think",
    "description": (
        "EXPLORE adjacent memories via spreading activation from a known memory id. "
        "After a recall hit on memory N, call this to surface the cluster around N — "
        "it's how you find context the user didn't explicitly query for. Use when "
        "an answer needs background, when you want to remember the broader picture, "
        "or when the user asks 'what else…'/'related…'. Returns activated memories "
        "ranked by graph proximity * decay."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "Starting memory ID (typically from a recent neural_recall hit).",
            },
            "depth": {
                "type": "integer",
                "description": "Hop depth — 2-3 for tight clusters, 5+ for broader exploration. Default 3.",
            },
        },
        "required": ["memory_id"],
    },
}

MAZEMAKER_GRAPH_SCHEMA = {
    "name": "mazemaker_graph",
    "description": (
        "META view of the memory store: total count, connection count, top edges. "
        "Call when the user asks 'what do you remember about me?', 'how much memory "
        "do you have?', 'show me the structure'. Cheap aggregate — don't use as a "
        "substitute for neural_recall on specific topics."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

MAZEMAKER_RECALL_MULTI_SCHEMA = {
    "name": "mazemaker_recall_multi",
    "description": (
        "MULTI-ANGLE recall: run two-or-more query phrasings in parallel "
        "and fuse the results via Reciprocal Rank Fusion. Call this when "
        "you suspect one phrasing might miss the right memory because the "
        "user phrased the stored fact differently. Example: gold memory "
        "says 'I love espresso', user later asks 'what coffee do I "
        "prefer' — single-shot recall on either phrasing alone may miss "
        "the other; multi-angle with both phrasings catches both. "
        "Two-to-five angles is the sweet spot; more is rarely worth the "
        "tokens. Falls back to plain recall on engines without the "
        "multi-recall feature."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "angles": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Two or more query phrasings to recall in parallel. "
                    "Vary entity wording, time framing, and explicit-vs-"
                    "implicit framing across angles."
                ),
            },
            "k": {
                "type": "integer",
                "description": "Max fused results to return (default 5).",
            },
            "fuse": {
                "type": "boolean",
                "description": (
                    "Default true (RRF-fused flat list). Set false to "
                    "receive raw per-angle results as a list of lists."
                ),
            },
        },
        "required": ["angles"],
    },
}

MAZEMAKER_RECALL_ADVANCED_SCHEMA = {
    "name": "mazemaker_recall_advanced",
    "description": (
        "ADVANCED recall — same query/answer shape as mazemaker_recall, "
        "but exposes per-call channel tuning: ColBERT and DAE weights, "
        "temporal weight, MMR lambda for diversity, score-percentile "
        "filter, and retrieval mode. Use only when the default calibration "
        "of mazemaker_recall is missing a result you know exists. The "
        "engine's default channel weights are bench-tuned for the median "
        "corpus; overriding them is a debugging tool, not the hot path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search phrase.",
            },
            "k": {
                "type": "integer",
                "description": "Max results to return (default 5).",
            },
            "mode": {
                "type": "string",
                "enum": ["semantic", "hybrid", "advanced", "skynet", "lean", "trim"],
                "description": (
                    "Retrieval mode. Default inherits the engine setting; "
                    "use 'lean' for fast prose recall, 'skynet' for the "
                    "full multi-channel fusion."
                ),
            },
            "colbert_weight": {
                "type": "number",
                "description": (
                    "Per-call ColBERT@1.5 rerank weight. 0 = off, >1 = "
                    "emphasised. Bench champion is 2.5–3.0."
                ),
            },
            "dae_weight": {
                "type": "number",
                "description": (
                    "Per-call Dream-Augmented Embeddings channel weight. "
                    "0 = off, 1 = baseline."
                ),
            },
            "temporal_weight": {
                "type": "number",
                "description": (
                    "Recency-decay weight in the relevance formula. "
                    "Default 0.2; higher emphasises freshly-stored memories."
                ),
            },
            "score_percentile": {
                "type": "number",
                "description": (
                    "Drop the bottom X fraction of candidates by rank "
                    "before truncating to k. Range [0, 1)."
                ),
            },
            "mmr_lambda": {
                "type": "number",
                "description": (
                    "Maximal-marginal-relevance diversity weight. "
                    "0 = off (default), higher = more diversity."
                ),
            },
        },
        "required": ["query"],
    },
}

MAZEMAKER_AFE_FACTS_SCHEMA = {
    "name": "mazemaker_afe_facts",
    "description": (
        "Query AFE-extracted atomic facts in the corpus.  Filters by "
        "label prefix ('afe', 'afe:auto', '::api2::C' for rebake "
        "namespaces) and/or by source memory_id when the operator "
        "wants to inspect a single session's facts.  Pro feature."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_id": {"type": "integer"},
            "label_prefix": {"type": "string"},
            "limit": {"type": "integer"},
        },
    },
}

MAZEMAKER_SYNTH_LINEAGE_SCHEMA = {
    "name": "mazemaker_synth_lineage",
    "description": (
        "Given a Stage S synthesis memory_id, return the contributing "
        "AFE atomic facts (the cluster that fed the LLM-distilled "
        "rephrasing) + edge weights.  Pro feature."
    ),
    "parameters": {
        "type": "object",
        "properties": {"memory_id": {"type": "integer"}},
        "required": ["memory_id"],
    },
}

MAZEMAKER_DIAGNOSE_SCHEMA = {
    "name": "mazemaker_diagnose",
    "description": (
        "Typed-miss diagnostic — identify queries where the live "
        "corpus failed to surface a high-similarity match within "
        "top-K.  Returns per-question-type misses + samples.  Use "
        "when the operator wants to know 'where does my pod struggle'.  "
        "Pro feature."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question_type": {
                "type": "string",
                "enum": ["preference", "temporal", "factual",
                         "general", "ssa", "ssu", "ms", "tr", "ku"],
            },
            "top_k_threshold": {"type": "integer"},
            "sample_size": {"type": "integer"},
        },
    },
}

MAZEMAKER_REBAKE_SCHEMA = {
    "name": "mazemaker_rebake",
    "description": (
        "Query-conditional Stage C rebake on specified gold sessions.  "
        "Productized version of benchmarks/targeted_rebake/rebake.py: "
        "runs the AFE Stage C extractor with a per-call prompt against "
        "the specified missed-query sessions, inserts new atomic facts "
        "at the specified salience.  Pro + managed-pod only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "missed_query_ids": {
                "type": "array", "items": {"type": "integer"},
            },
            "extraction_prompt": {"type": "string"},
            "model": {"type": "string"},
            "salience": {"type": "number"},
        },
        "required": ["missed_query_ids", "extraction_prompt"],
    },
}

MAZEMAKER_ABLATE_SCHEMA = {
    "name": "mazemaker_ablate",
    "description": (
        "Channel-ablation matrix: run each query under each channel-"
        "isolated configuration so the operator can see which channel "
        "is load-bearing on the live corpus.  Channels: semantic, fts, "
        "colbert, dae, ppr, intent, temporal, salience, canonical.  "
        "Pro feature."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channels": {"type": "array", "items": {"type": "string"}},
            "queries": {"type": "array", "items": {"type": "string"}},
            "k": {"type": "integer"},
        },
        "required": ["channels", "queries"],
    },
}

MAZEMAKER_DREAM_STATS_SCHEMA = {
    "name": "mazemaker_dream_stats",
    "description": (
        "Aggregate dream-engine stats — total sessions, strengthened "
        "edges, weakened edges, bridges found, insights crystallised. "
        "Optional `phase` arg filters the response to one of the 7 "
        "phases when the backend returns a phase-keyed structure."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": ["nrem", "supersedes", "rem", "insight",
                         "afe", "synthesis", "dae", "all"],
                "description": "Optional phase filter (default 'all').",
            },
        },
    },
}

MAZEMAKER_QUOTA_SCHEMA = {
    "name": "mazemaker_quota",
    "description": (
        "Live quota state for the running pod's license — tier, "
        "remaining-calls-today, remaining-this-month, grace-until "
        "timestamp.  When the pod's meter is wired (Lite/Pro pods), "
        "also returns a per-tool `by_tool` breakdown of today's call "
        "counts so customers can see which tool is eating their "
        "quota."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

MAZEMAKER_HEALTH_SCHEMA = {
    "name": "mazemaker_health",
    "description": (
        "Corpus + AFE + dream + DAE single-pull health.  Returns a "
        "nested status blob: memory + connection counts, last-dream-"
        "cycle stats, AFE fact coverage, DAE vector freshness.  Pro-"
        "only sections return 'n/a' on Community.  All tiers."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

MAZEMAKER_SUPERSEDES_LOG_SCHEMA = {
    "name": "mazemaker_supersedes_log",
    "description": (
        "Recent SUPERSEDES detections from the memory_revisions table "
        "in chronological-DESC order: (old_id, new_id, cosine, "
        "numeric_overlap, ts).  All tiers — conflict supersession is "
        "a Community feature per the public tier matrix."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "since": {"type": "number",
                      "description": "Unix-timestamp cutoff (optional)."},
            "limit": {"type": "integer"},
        },
    },
}

MAZEMAKER_DREAM_CONFIG_SCHEMA = {
    "name": "mazemaker_dream_config",
    "description": (
        "GET or SET the dream-engine cadence + threshold knobs.  Default "
        "action 'get' returns the current values for idle_threshold "
        "(seconds since last activity before a cycle triggers), "
        "memory_threshold (new-memories count that triggers a cycle), "
        "max_memories (NREM batch size), max_isolated (REM batch size), "
        "and dae_recompute_every (NREM cycles between DAE recomputes). "
        "Action 'set' applies any of those as keyword overrides on the "
        "running DreamEngine; takes effect on the next loop tick."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set"],
                "description": "get (default) or set.",
            },
            "idle_threshold": {"type": "integer"},
            "memory_threshold": {"type": "integer"},
            "max_memories": {"type": "integer"},
            "max_isolated": {"type": "integer"},
            "dae_recompute_every": {"type": "integer"},
        },
    },
}

MAZEMAKER_DREAM_CONTROL_SCHEMA = {
    "name": "mazemaker_dream_control",
    "description": (
        "Control the autonomous dream daemon: pause (stop the loop), "
        "resume (start it again), or status (report running/idle + "
        "cycle count).  Cycles already in progress are not interrupted "
        "by pause.  Pro feature in the operator's intended tier model; "
        "engine-side check is delegated to the daemon's existing gates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["pause", "resume", "status"],
                "description": "pause / resume / status (default 'status').",
            },
        },
    },
}

MAZEMAKER_CLASSIFY_INTENT_SCHEMA = {
    "name": "mazemaker_classify_intent",
    "description": (
        "CLASSIFY a query's intent without running a recall. Returns one of "
        "{preference, temporal, factual, general}. Cheap regex — no LLM, "
        "no embedding. Useful as a routing decision: preference-intent "
        "queries benefit from mazemaker_recall_multi with rephrased "
        "angles; factual-intent queries are usually one-shot; temporal-"
        "intent queries benefit from a higher temporal_weight in "
        "mazemaker_recall_advanced."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to classify.",
            },
        },
        "required": ["query"],
    },
}

# Per-phase dream-trigger tools — fission of mazemaker_dream's phase
# enum into a flat tool surface. Each tool takes no params and
# dispatches to Memory.dream(phase=<X>) via the engine. Pro-only
# phases (afe / synthesis / dae) return a structured skipped payload
# on community licenses.
_DREAM_PHASE_TOOL_DESCRIPTIONS = {
    "mazemaker_dream_nrem": (
        "Run NREM only: replay + strengthen active edges + prune "
        "inactive ones. Three-slice sampling (recent / random / "
        "low-salience). Ships in Community."
    ),
    "mazemaker_dream_supersedes": (
        "Run SUPERSEDES phase only: cross-session conflict detection "
        "via cosine similarity + numeric-token overlap. Marks the "
        "older memory '[SUPERSEDED]' and records the revision. "
        "Ships in Community."
    ),
    "mazemaker_dream_rem": (
        "Run REM phase only: bridge-discovery between isolated "
        "memories via batched semantic recall. Returns bridges found "
        "+ isolated-memory queue depth. Ships in Community."
    ),
    "mazemaker_dream_insight": (
        "Run Insight phase only: Louvain community detection on the "
        "connection graph + cluster-summary memories with anchor "
        "samples. Ships in Community."
    ),
    "mazemaker_dream_afe": (
        "Run AFE phase only: Atomic Fact Extraction Stage A (markdown) "
        "→ Stage B (NER) → Stage C (LLM user-state). Bulk-writes "
        "atomic facts with 'afe:auto' label. Pro feature; community "
        "calls return a structured skipped payload."
    ),
    "mazemaker_dream_synthesize": (
        "Run Stage S synthesis phase only: LLM-distilled "
        "crystallisation of cross-source patterns from AFE-candidate "
        "clusters. ~10% yield by design. Pro feature."
    ),
    "mazemaker_dream_dae": (
        "Run DAE phase only: recompute Dream-Augmented Embeddings — "
        "the second embedding vector weighted toward graph neighbours. "
        "Full-corpus pass. Pro feature."
    ),
}
DREAM_PHASE_TOOL_ALIAS = {
    "mazemaker_dream_nrem":       "nrem",
    "mazemaker_dream_supersedes": "supersedes",
    "mazemaker_dream_rem":        "rem",
    "mazemaker_dream_insight":    "insight",
    "mazemaker_dream_afe":        "afe",
    "mazemaker_dream_synthesize": "synthesis",
    "mazemaker_dream_dae":        "dae",
}

_DREAM_PHASE_SCHEMAS = [
    {
        "name": tool_name,
        "description": _DREAM_PHASE_TOOL_DESCRIPTIONS[tool_name],
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    for tool_name in DREAM_PHASE_TOOL_ALIAS
]


ALL_TOOL_SCHEMAS = [
    MAZEMAKER_REMEMBER_SCHEMA,
    MAZEMAKER_RECALL_SCHEMA,
    MAZEMAKER_RECALL_MULTI_SCHEMA,
    MAZEMAKER_RECALL_ADVANCED_SCHEMA,
    MAZEMAKER_CLASSIFY_INTENT_SCHEMA,
    MAZEMAKER_DREAM_CONFIG_SCHEMA,
    MAZEMAKER_DREAM_CONTROL_SCHEMA,
    MAZEMAKER_DREAM_STATS_SCHEMA,
    MAZEMAKER_QUOTA_SCHEMA,
    MAZEMAKER_AFE_FACTS_SCHEMA,
    MAZEMAKER_SYNTH_LINEAGE_SCHEMA,
    MAZEMAKER_DIAGNOSE_SCHEMA,
    MAZEMAKER_REBAKE_SCHEMA,
    MAZEMAKER_ABLATE_SCHEMA,
    MAZEMAKER_HEALTH_SCHEMA,
    MAZEMAKER_SUPERSEDES_LOG_SCHEMA,
    MAZEMAKER_THINK_SCHEMA,
    MAZEMAKER_GRAPH_SCHEMA,
    *_DREAM_PHASE_SCHEMAS,
]
