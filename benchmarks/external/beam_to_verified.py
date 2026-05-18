#!/usr/bin/env python3
"""Convert BEAM-10M probing_questions.json (per conv) to the
mm_10m_eval-compatible verified_questions.json format so we can
reuse the existing openrouter_runner + deterministic scorer.

BEAM abilities (10):
  abstention, contradiction_resolution, event_ordering,
  information_extraction, instruction_following, knowledge_update,
  multi_session_reasoning, preference_following, summarization,
  temporal_reasoning

For each BEAM question we synthesise:
  qid             = f"{ability}_{i}"
  ability         = BEAM ability name (kept verbatim)
  question        = BEAM question text
  gold            = BEAM ideal_response (used by deterministic scorer
                    as a fallback when rubric doesn't yield a single
                    canonical value)
  rubric          = list of rubric items (BEAM sentence-level)
  accepted        = automatically extracted key tokens from each
                    rubric item (used by deterministic scorer)
  beam_metadata   = preserve difficulty, plan_reference, etc.

Usage:
  python beam_to_verified.py --beam-base /tmp/BEAM/chats/10M --out-dir /tmp/beam_verified
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Stop tokens that don't help downstream matching
STOP = {
    "the", "a", "an", "and", "of", "for", "to", "in", "is", "are",
    "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "with", "as", "i", "my", "me", "we",
    "our", "us", "this", "that", "these", "those", "his", "her",
    "their", "your", "you", "based", "provided", "chat", "there",
    "no", "information", "related", "about", "regarding", "around",
    "any", "specific", "explicit", "mentions", "states", "lists",
    "reports", "names", "describes", "details", "either", "or",
    "include", "includes", "contains", "but", "not", "from",
    "between", "during", "while", "until", "before", "after",
    "what", "when", "where", "which", "who", "how", "why", "by",
    "at", "on", "into", "out", "up", "down", "off", "over", "under",
}


META_PREFIX_PATTERNS = [
    r"^llm\s+response\s+should\s+(?:state|mention|include|describe|note|list|identify|capture|provide|address|acknowledge|cite|reference)\s*[:\-]?\s*",
    r"^response\s+should\s+(?:state|mention|include|describe|note|list|identify|capture|provide|address|acknowledge|cite|reference)\s*[:\-]?\s*",
    r"^the\s+(?:answer|response)\s+should\s+(?:state|mention|include|describe|note|list|identify|capture|provide|address|acknowledge|cite|reference)\s*[:\-]?\s*",
    r"^(?:state|mention|include|describe|note|list|identify|capture|provide|address|acknowledge|cite|reference|reports|names)\s+(?:that|how|the|a|an|whether|whether or not)?\s*",
    r"^based\s+on\s+the\s+(?:provided\s+)?chat,?\s+there\s+is\s+no\s+information\s+(?:about|on|regarding|related\s+to)\s*",
    r"^either\s+(?:omits|mentions|states|reports|notes)\s+(?:or\s+(?:omits|mentions|states|reports|notes))?\s+",
    r"^(?:correctly\s+)?(?:identifies|reports|states|mentions|notes)\s+(?:that|how|the|a|an)?\s*",
]
META_PREFIX_RE = re.compile("|".join(META_PREFIX_PATTERNS), re.IGNORECASE)


def strip_meta_prefix(rubric_item: str) -> str:
    """Strip meta-language prefixes like 'LLM response should state:' so
    only the substantive content remains for matching."""
    s = rubric_item.strip()
    # Apply repeatedly in case of nested prefixes
    for _ in range(3):
        new = META_PREFIX_RE.sub("", s, count=1).strip()
        if new == s:
            break
        s = new
    return s


def extract_key_tokens(rubric_item: str, max_tokens: int = 5) -> list[str]:
    """Pull out distinctive content tokens AFTER stripping rubric meta-
    language. Accepts:
      - alpha tokens ≥4 chars, non-stopword
      - numeric tokens (any length): 17, 88, 3.5, 99.9, 2024 ...
      - percent expressions: 88%, 17%
      - alpha+digit combos: gpt5, v1.2, etc.
    """
    cleaned = strip_meta_prefix(rubric_item)
    # Preserve percent signs by replacing with " percent " before
    # alpha-only stripping
    pct_marker = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1pct", cleaned)
    text = re.sub(r"[^a-z0-9 ]+", " ", pct_marker.lower())
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    seen = set()
    for t in text.split():
        if t in seen or t in STOP:
            continue
        # Accept: digit-only ≥1 char, alpha-only ≥4 chars, alphanumeric ≥3 chars
        if t.isdigit():
            pass
        elif t.isalpha() and len(t) >= 4:
            pass
        elif len(t) >= 3 and any(c.isdigit() for c in t):
            pass
        else:
            continue
        seen.add(t)
        tokens.append(t)
        if len(tokens) >= max_tokens:
            break
    return tokens


def convert_conv(conv: int, beam_base: Path, out_dir: Path) -> int:
    p = beam_base / str(conv) / "probing_questions" / "probing_questions.json"
    data = json.load(p.open())
    questions = []
    for ability, qs in data.items():
        for i, q in enumerate(qs):
            rubric = q.get("rubric", [])
            accepted_tokens = []
            for ri in rubric:
                accepted_tokens.append(extract_key_tokens(ri))
            # Build a flat accepted list for the deterministic scorer
            flat_accepted = []
            for toks in accepted_tokens:
                if toks:
                    flat_accepted.append(" ".join(toks))
            qid = f"{ability}_{i}"
            questions.append({
                "qid": qid,
                "ability": ability,
                "q_idx": i,
                "question": q["question"],
                "gold": q.get("ideal_response", ""),
                "rubric": rubric,
                "rubric_key_tokens": accepted_tokens,
                "accepted": flat_accepted,
                "difficulty": q.get("difficulty"),
                "beam_metadata": {
                    "abstention_type": q.get("abstention_type"),
                    "why_unanswerable": q.get("why_unanswerable"),
                    "plan_reference": q.get("plan_reference"),
                },
            })
    out = out_dir / f"conv-{conv}" / "verified_questions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"questions": questions, "source": "BEAM-10M"}, indent=2))
    return len(questions)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--beam-base", type=Path, default=Path("/tmp/BEAM/chats/10M"))
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/beam_verified"))
    p.add_argument("--convs", default="1,2,3,4,5,6,7,8,9,10")
    args = p.parse_args()
    convs = [int(c) for c in args.convs.split(",")]
    total = 0
    for c in convs:
        n = convert_conv(c, args.beam_base, args.out_dir)
        total += n
        print(f"  conv-{c}: {n} questions → {args.out_dir}/conv-{c}/verified_questions.json")
    print(f"\nTOTAL: {total} questions converted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
