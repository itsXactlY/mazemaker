#!/usr/bin/env python3
"""Targeted Stage C rebake — extract user-side facts for the specific
gold sessions that the engine missed on a benchmark, insert them at
salience=2.0 with a distinct label namespace.

This is the formation-side lever that lifted inception_bench-oracle 500q R@5
from 0.7404 (retrieval-tuning ceiling) to 0.8043 (above the 0.80
stretch target). See ``decision_iter79_target_exceeded`` in auto-memory
for the full case study.

Methodology:

    1. Find queries whose gold session id isn't in top-5:
        python find_typed_misses.py <type> <misses.json>

    2. Run this script with a question-type-appropriate prompt:
        python rebake.py --misses <misses.json> --type <ssp|ssu|tr> \\
                         --namespace <api2|ssu|tr> \\
                         [--max-content 8000] [--concurrency 8] \\
                         [--temperature 1.0] [--salience 2.0]

    3. Re-run the benchmark with the same retrieval stack.

Each round typically lifts the targeted type by 6-22 top-5 hits while
trading 2-6 hits across other types via dilution. Stay under 6 facts
per session per round to avoid the dilution crossover.

Total cost: ~$0.01-0.02 per round at gpt-5-nano pricing.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))


# ───────────────────────────────────────────────────────────────────
# Prompt templates per question-type
# ───────────────────────────────────────────────────────────────────

PROMPT_SSP = """The user later asks: "{question}"

From the conversation passage below, extract every fact that describes
what the user owns, has, prefers, plans, budgets for, dislikes, or has
decided about their LIFE/POSSESSIONS/SITUATION/HABITS related to this
question's topic. Phrase each as a direct "user [verb] X" statement.

Focus on the user's perspective — not assistant suggestions.

Examples of good preference facts:
- "user owns a Sony A7R IV camera"
- "user's smartphone is an iPhone 14"
- "user prefers tropical destinations"
- "user has dietary restrictions: vegetarian, no dairy"
- "user works from home and commutes 2 days per week"
- "user's bedroom is 12x14 feet with a window facing east"
- "user enjoys jazz, blues, and classical music"

Each fact under 200 chars, atomic. Output ONLY a JSON array of strings.

PASSAGE:
{passage}

USER FACTS (JSON):"""

PROMPT_SSU = """A user has had a conversation. Later they ask: "{question}"

Read the conversation below and extract every USER-STATE fact relevant
to that question. These are concrete facts the user has DISCLOSED about
their life, possessions, history, schedule, identity, decisions, etc.
Pay close attention to specific numbers, names, dates, places, brands.

Phrase each fact as a direct "user [verb] X" statement with the concrete
value inline:
- "user owns a Sony A7R IV camera"
- "user's daily commute is 45 minutes each way"
- "user graduated with a Master's in Computer Science from MIT"
- "user works as a Software Engineer at Stripe"
- "user's dog is a Golden Retriever named Max"
- "user paid $1,200 for a designer handbag"
- "user has 23 playlists on Spotify"
- "user went to Tokyo for 2 weeks in November 2024"

Each fact under 200 chars, atomic, with concrete values where possible.
Output ONLY a JSON array of strings.

CONVERSATION:
{passage}

USER-STATE FACTS:"""

PROMPT_TR = """A user has had a conversation. Later they ask: "{question}"

Read the conversation below and extract every TIME-ANCHORED FACT about
the user: events with dates, durations, time-of-day, recurrences,
schedules, sequences. The user STATED these in the conversation.

For each time-anchored fact, include the concrete date/time/duration
inline. Examples:
- "user attended a bird watching workshop on 2024-03-15"
- "user has been bird watching since 2022"
- "user wakes up at 6:30 AM on Tuesdays and Thursdays"
- "user ordered her best friend's birthday gift 3 days before the party"
- "user bought a smoker on 2024-08-12"
- "user moved to the United States when they were 12 years old"
- "user spent 2 weeks in Japan during November 2024"

Also include any anchor events that other temporal questions might
need: "user's birthday party was on 2024-06-20", "user finished
reading 'Book' on date X".

Each fact under 200 chars, atomic, with concrete time/date values.
Output ONLY a JSON array of strings.

CONVERSATION:
{passage}

USER TIME-ANCHORED FACTS:"""

PROMPT_MS = """A user has had a conversation. Later they ask: "{question}"

Multi-session counting questions like "How many X did I do in the past Y?"
need every individual occurrence crystallised as a separately-retrievable
fact, with a concrete date so the downstream LLM can filter by time window.

Read the conversation below and extract every COUNTABLE EVENT the user
mentioned, with concrete dates/quantities inline. Examples:

- "user baked chocolate chip cookies on 2024-03-15"
- "user acquired 2 succulents at the garden center on 2024-03-10"
- "user serviced their road bike on 2024-03-20"
- "user visited the Met museum on 2024-02-14"
- "user attended a spinning class on 2024-03-08"
- "user tried Vietnamese cuisine for the first time on 2024-02-25"

Include the COUNT inline when the user states a number ("acquired 2",
"baked 3 batches"). Each event is its own atomic fact. Don't summarise
across sessions — emit one fact per concrete occurrence.

Each fact under 200 chars, atomic, with concrete dates and counts.
Output ONLY a JSON array of strings.

CONVERSATION:
{passage}

USER EVENT FACTS:"""


PROMPT_KU = """A user has had a conversation. Later they ask: "{question}"

Knowledge-update questions test whether the engine retrieves the LATEST
value of something the user has changed over time. The agent needs facts
that explicitly mark UPDATES with timestamps so the most recent state
wins the recall.

Read the conversation below and extract every UPDATE EVENT — anywhere the
user changes, replaces, switches, or modifies a previously-stated value.
Mark the update inline with a date. Examples:

- "user updated their phone number to 555-0199 on 2024-03-15"
- "user switched from Python 3.10 to 3.12 on 2024-02-28"
- "user replaced their old GPU with an RTX 5090 on 2024-01-10"
- "user changed their commute from car to bike on 2024-03-01"
- "user moved from Berlin to Vienna on 2024-02-14"
- "user revised the project deadline to 2024-04-30 on 2024-03-12"

Use verbs that mark mutation: updated, switched, replaced, changed,
moved, revised, upgraded, downgraded. Each fact must be the user's
NEW state with an anchor date. Each fact under 200 chars, atomic.
Output ONLY a JSON array of strings.

CONVERSATION:
{passage}

USER UPDATE FACTS:"""


PROMPTS = {"ssp": PROMPT_SSP, "ssu": PROMPT_SSU, "tr": PROMPT_TR, "ms": PROMPT_MS, "ku": PROMPT_KU}


async def _extract_one(client, sem, model: str, prompt_template: str,
                        sid: str, content: str, qtext: str,
                        max_content: int, temperature: float) -> list[str]:
    truncated = content[:max_content]
    prompt = prompt_template.format(question=qtext, passage=truncated)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        except Exception as e:
            print(f"  [{sid}] API error: {e}", flush=True)
            return []
    raw = (resp.choices[0].message.content or "").strip()
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
            t = None
            for key in ("text", "fact", "statement"):
                v = f.get(key)
                if isinstance(v, str) and v.strip():
                    t = v
                    break
            if not t:
                continue
            t = t.strip()
        else:
            continue
        if 4 <= len(t) <= 280:
            out.append(t[:280])
    return out


async def main_async(args) -> int:
    os.environ.setdefault("MM_DB_BACKEND", "postgres")
    os.environ.setdefault("MM_POSTGRES_DB", "mm10m_bench")
    os.environ.setdefault("MM_POSTGRES_SCHEMA", "longmemeval_oracle_bgem3_1024")

    misses_all = json.load(open(args.misses))
    if args.skip_abs:
        misses = [m for m in misses_all if not m["qid"].endswith("_abs")]
        skipped = len(misses_all) - len(misses)
        print(f"[rebake] {len(misses)} misses ({skipped} _abs skipped)", flush=True)
    else:
        misses = misses_all
        print(f"[rebake] {len(misses)} misses", flush=True)

    from memory_client import Mazemaker
    nm = Mazemaker(
        db_path="/dev/null",
        embedding_backend="auto",
        lazy_graph=True,
        retrieval_mode="semantic",
        rerank=False,
    )

    import psycopg
    pg_db = os.environ["MM_POSTGRES_DB"]
    pg_schema = os.environ["MM_POSTGRES_SCHEMA"]
    pg_user = os.environ.get("MM_POSTGRES_USER", "mazemaker")
    pg_pass = os.environ.get("MM_POSTGRES_PASSWORD", "")
    pg_host = os.environ.get("MM_POSTGRES_HOST", "localhost")

    with psycopg.connect(dbname=pg_db, user=pg_user, password=pg_pass, host=pg_host) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {pg_schema}")
            tasks = []
            for m in misses:
                for sid in m["gold_sids"]:
                    cur.execute("SELECT id, content FROM memories WHERE label=%s",
                                (f"session:{sid}",))
                    r = cur.fetchone()
                    if r:
                        tasks.append({
                            "qid": m["qid"], "qtext": m["question"],
                            "sid": sid, "src_id": r[0], "content": r[1],
                        })
    print(f"[rebake] {len(tasks)} sessions to extract from", flush=True)
    if not tasks:
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_file = Path.home() / ".benchkey"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        print("[rebake] FATAL: no OPENAI_API_KEY and no ~/.benchkey", file=sys.stderr)
        return 2

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)
    prompt_template = PROMPTS[args.type]

    print(f"[rebake] firing {len(tasks)} extractions (model={args.model}, "
          f"concurrency={args.concurrency})...", flush=True)
    coros = [
        _extract_one(client, sem, args.model, prompt_template,
                     t["sid"], t["content"], t["qtext"],
                     args.max_content, args.temperature)
        for t in tasks
    ]
    facts_per = await asyncio.gather(*coros)
    total = sum(len(f) for f in facts_per)
    print(f"[rebake] extracted {total} facts", flush=True)

    if args.sample > 0:
        print(f"\n[rebake] Sample of first {args.sample} sessions:", flush=True)
        for t, fs in zip(tasks[:args.sample], facts_per[:args.sample]):
            print(f"\n  [{t['sid']}] q={t['qtext'][:70]}")
            for f in fs[:5]:
                print(f"    → {f}")

    if total == 0:
        print("[rebake] nothing to write", flush=True)
        return 0

    if args.dry_run:
        print("\n[rebake] --dry-run set, not writing to DB", flush=True)
        return 0

    payload = []
    src_ids_aligned = []
    for t, facts in zip(tasks, facts_per):
        for idx, txt in enumerate(facts):
            payload.append((txt, f"session:{t['sid']}::{args.namespace}::C{idx}"))
            src_ids_aligned.append(t["src_id"])

    contents = [c for c, _ in payload]
    print(f"\n[rebake] embedding {len(contents)} facts...", flush=True)
    embeddings = nm.embedder.embed_batch(contents)

    print(f"[rebake] writing to PG with salience={args.salience}...", flush=True)
    rows = [
        {"content": txt, "label": label, "embedding": emb, "salience": args.salience}
        for (txt, label), emb in zip(payload, embeddings)
    ]
    new_ids = nm.store.remember_batch(rows)
    print(f"[rebake] wrote {len(new_ids)} memories", flush=True)

    pairs = [(int(nid), int(sid), 1.0) for sid, nid in zip(src_ids_aligned, new_ids)]
    if pairs:
        nm.store.add_connections_batch(pairs, edge_type="derived_from")
        print(f"[rebake] linked {len(pairs)} fact→source derived_from edges", flush=True)

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--misses", required=True, type=str,
                   help="JSON file produced by find_typed_misses.py")
    p.add_argument("--type", required=True, choices=("ssp", "ssu", "tr", "ms", "ku"),
                   help="Question-type bucket (selects the extraction prompt)")
    p.add_argument("--namespace", required=True, type=str,
                   help="Label namespace, e.g. 'api2' produces "
                        "session:<sid>::<namespace>::C<idx>")
    p.add_argument("--model", default="gpt-5-nano-2025-08-07")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-content", type=int, default=8000,
                   help="Truncate session content to N chars (default 8000)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--salience", type=float, default=2.0)
    p.add_argument("--skip-abs", action="store_true", default=True,
                   help="Skip _abs (abstention) variants — they have no gold fact")
    p.add_argument("--sample", type=int, default=3,
                   help="Print sample extractions from first N sessions (0=off)")
    p.add_argument("--dry-run", action="store_true",
                   help="Extract but don't write to DB")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
