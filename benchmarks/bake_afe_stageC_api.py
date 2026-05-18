#!/usr/bin/env python3
"""bake_afe_stageC_api.py — Stage C extraction via OpenAI API.

Replaces the local `ollama run qwen2.5:3b` subprocess path with parallel
HTTP calls to OpenAI. Motivation: the qwen2.5:3b Stage C bake produced
2,706 user-side facts that lifted ssp R@5 0.2333→0.3667 on godbench, but
the remaining ~19/30 ssp golds aren't in top-5 — likely because the small
local model missed subtle preferences or phrased them in a way that didn't
match the bench query semantics. A stronger extractor *might* recover
some of those.

Different label namespace `::api::C` keeps the new facts side-by-side
with the existing `::afe::C` qwen facts; comparison benches can ablate
each independently.

Implementation:
- Parallel API calls (asyncio gather, configurable concurrency).
- Same JSON-array prompt shape as `_stage_c_llm` in afe.py — but
  cleaner output because the OpenAI streaming wrapper doesn't leak
  ANSI cursor codes into stdout.
- Embeds each new fact via the shared embed.sock socket so the
  pipeline stays consistent (no separate embedder load).
- Bulk-writes via PostgresStore.remember_batch.

USAGE
    OPENAI_API_KEY=$(cat ~/.benchkey) python benchmarks/bake_afe_stageC_api.py \\
        --variant oracle --model gpt-5-nano-2025-08-07 \\
        --max-sources 0 --concurrency 16

    # smoke test
    python benchmarks/bake_afe_stageC_api.py --variant oracle \\
        --model gpt-5-nano-2025-08-07 --max-sources 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

CACHE_DB = "mm10m_bench"

# Same atomic-fact gate as afe.py:_looks_atomic so the API output
# follows the same shape constraints downstream.
_NUMERIC_TOKEN_RE = re.compile(
    r"\b\d+\b|\$\d|\b\d+\s?(?:USD|EUR|GBP)\b|\b\d{4}[\-/]\d{1,2}[\-/]\d{1,2}\b"
)


def _looks_atomic(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 4 or len(s) > 280:
        return False
    sl = s.lower()
    if sl.startswith(("user ", "the user ")):
        return True
    if not _NUMERIC_TOKEN_RE.search(s) and not re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", s):
        return False
    return True


PROMPT = (
    "Extract specific atomic facts about the USER from the conversation "
    "passage below. Target two kinds:\n"
    "  (1) User-side statements: preferences, opinions, decisions, "
    "intentions, personal details (job, location, possessions, "
    "relationships, habits, plans). Phrase as 'user prefers X', "
    "'user owns Y', 'user plans to Z', 'user dislikes W', etc.\n"
    "  (2) Concrete entity facts mentioned by/about the user: "
    "specific numbers, dollar amounts, named entities, dates, "
    "addresses, version identifiers.\n\n"
    "Each fact must be a short atomic string under 200 characters. "
    "Capture subtle/implicit preferences too: if the user mentions a "
    "scenario or constraint, infer the underlying preference. "
    "Skip generic advice the assistant gave. Skip facts about the "
    "assistant. Output ONLY a JSON array of objects "
    "`[{{\"text\": \"...\"}}, ...]`, nothing else.\n\n"
    "PASSAGE:\n{passage}\n\nFACTS:"
)


async def _extract_one(
    client, sem, model: str, content: str, sid: str, truncate_at: int
) -> list[str]:
    truncated = content[:truncate_at]
    prompt = PROMPT.format(passage=truncated)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                # Don't pass response_format on nano — older SDK paths reject it.
                temperature=0.0 if "gpt-5" not in model else 1.0,
            )
        except Exception as e:
            print(f"  [{sid}] API error: {e}", flush=True)
            return []
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        return []
    # Strip code fences if any
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for f in parsed:
        if isinstance(f, str):
            t = f.strip()
        elif isinstance(f, dict):
            # Accept any plausible key
            t = None
            for key in ("text", "fact", "Fact", "statement", "value", "atom"):
                v = f.get(key)
                if isinstance(v, str) and v.strip():
                    t = v
                    break
            if t is None:
                continue
            t = t.strip()
        else:
            continue
        if _looks_atomic(t):
            out.append(t[:280])
    return out


async def main_async(args) -> int:
    schema = f"longmemeval_{args.variant}_bgem3_1024"
    os.environ["MM_DB_BACKEND"] = "postgres"
    os.environ["MM_POSTGRES_DB"] = CACHE_DB
    os.environ["MM_POSTGRES_SCHEMA"] = schema

    print(f"[api-bake] DB={CACHE_DB} schema={schema} model={args.model} "
          f"concurrency={args.concurrency}", flush=True)

    from memory_client import Mazemaker
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="semantic",
        rerank=False,
    )

    # Pull eligible source sessions: not chunked, not AFE-derived.
    # Idempotency: skip sessions whose sid already has an api::C fact.
    with nm.store._cursor() as (_conn, cur):
        cur.execute(
            "SELECT DISTINCT split_part(label, '::', 1) "
            "FROM memories WHERE label LIKE '%::api::C%'"
        )
        done_sids = {r[0] for r in cur.fetchall()}
    with nm.store._cursor() as (_conn, cur):
        cur.execute(
            "SELECT id, label, content FROM memories "
            "WHERE label LIKE %s "
            "  AND label NOT LIKE %s "
            "  AND label NOT LIKE %s "
            "  AND length(content) >= 500",
            ("session:%", "%::chunk::%", "%::afe::%"),
        )
        rows = [r for r in cur.fetchall() if r[1] not in done_sids]
    print(f"[api-bake] eligible after dedup: {len(rows)} sessions"
          f" (already-done: {len(done_sids)})", flush=True)
    if args.max_sources and args.max_sources > 0:
        rows = rows[: args.max_sources]
    print(f"[api-bake] {len(rows):,} sessions to process", flush=True)
    if not rows:
        return 0

    # Get an embedder via Mazemaker's shared socket
    embedder = nm.embedder

    # Read API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_file = Path.home() / ".benchkey"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        print("[api-bake] FATAL: no OPENAI_API_KEY and no ~/.benchkey")
        return 2

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.time()
    # Each row: (id, label, content). Label = "session:answer_X".
    # We need sid = label[len("session:"):]
    tasks = []
    sids = []
    src_ids = []
    contents_keep: list[str] = []
    for src_id, label, content in rows:
        sid = label[len("session:"):] if label.startswith("session:") else label
        sids.append(sid)
        src_ids.append(int(src_id))
        contents_keep.append(content)
        tasks.append(
            _extract_one(client, sem, args.model, content, sid, args.truncate_at)
        )

    print(f"[api-bake] firing {len(tasks)} extractions...", flush=True)
    all_facts = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    total_facts = sum(len(f) for f in all_facts)
    print(f"[api-bake] extracted {total_facts} facts in {elapsed:.1f}s "
          f"({total_facts/max(1,elapsed):.1f} facts/s)", flush=True)

    # Embed + bulk-write
    if total_facts == 0:
        print("[api-bake] nothing to write")
        return 0

    fact_payload: list[tuple[str, str]] = []  # (content, label)
    fact_source_ids: list[int] = []
    for sid, src_id, facts in zip(sids, src_ids, all_facts):
        for idx, txt in enumerate(facts):
            label = f"session:{sid}::api::C{idx}"
            fact_payload.append((txt, label))
            fact_source_ids.append(src_id)

    contents_only = [c for c, _ in fact_payload]
    print(f"[api-bake] embedding {len(contents_only)} facts...", flush=True)
    embeddings = embedder.embed_batch(contents_only)

    print(f"[api-bake] writing to PG...", flush=True)
    rows_for_insert = []
    for (txt, label), emb in zip(fact_payload, embeddings):
        rows_for_insert.append({"content": txt, "label": label, "embedding": emb})
    # Use the engine's remember_batch
    new_ids = nm.store.remember_batch(rows_for_insert)
    print(f"[api-bake] wrote {len(new_ids)} memories", flush=True)

    # Add canonical_of-like edge from each new fact to its source session
    pairs = []
    for src_id, new_id in zip(fact_source_ids, new_ids):
        pairs.append((int(new_id), int(src_id), 1.0))
    if pairs:
        nm.store.add_connections_batch(pairs, edge_type="derived_from")
        print(f"[api-bake] linked {len(pairs)} fact→source edges", flush=True)

    print(f"\n[api-bake] DONE  {len(new_ids)} memories in {time.time()-t0:.1f}s",
          flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="oracle", choices=["s", "m", "oracle"])
    p.add_argument("--model", default="gpt-5-nano-2025-08-07")
    p.add_argument("--max-sources", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--truncate-at", type=int, default=4000,
                   help="Truncate session content to N chars before API call")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
