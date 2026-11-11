#!/usr/bin/env python3
"""Translate README.md (or any markdown) to 12 languages via local LibreTranslate.

v2 (line-batched):
- Fenced code blocks (``` ... ```) are kept VERBATIM.
- Text is translated in batches of LINES (BATCH=50), so the API returns a
  1:1 line mapping and markdown structure can be post-processed:
    * horizontal rules (---) and table separator rows stay verbatim
    * blockquote lines keep their leading '>' (re-attached if the MT dropped it)
    * table rows keep their leading '|'
"""
import json
import re
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8089"
BATCH = 50
REQ_TIMEOUT = 300

HR_RE = re.compile(r"^\s*(?:-{3,}|_{3,})\s*$")


def split_segments(text: str):
    """List of (kind, text): kind='fence' (verbatim) or 'text'."""
    segments, cur, in_fence, kind = [], [], False, "text"

    def flush():
        nonlocal cur
        if cur:
            s = "\n".join(cur).strip("\n")
            if s.strip():
                segments.append((kind, s))
            cur = []

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            if in_fence:
                cur.append(line)
                flush()
                in_fence, kind = False, "text"
            else:
                flush()
                cur, in_fence, kind = [line], True, "fence"
        else:
            cur.append(line)
    flush()
    return segments


def is_verbatim(line: str) -> bool:
    if not line.strip():
        return True
    if HR_RE.match(line):
        return True
    # table separator rows: only spaces/colons/pipes/dashes, with at least one dash
    if set(line.strip()) <= set(" :|-") and "-" in line:
        return True
    return False


def translate_batch(lines: list, lang: str, retries=3):
    body = json.dumps({"q": lines, "source": "en", "target": lang}).encode()
    req = urllib.request.Request(
        BASE + "/translate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
                d = json.loads(r.read().decode())
            t = d.get("translatedText")
            if not t or len(t) != len(lines):
                raise ValueError(f"bad response: {d}")
            return t
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            if attempt == retries:
                raise
            print(f"  retry {attempt} ({lang}): {e}", flush=True)
            time.sleep(5)


def translate_segment(seg: str, lang: str):
    """Translate one text segment, line-batched, structure-aware."""
    lines = seg.split("\n")
    out, batch = [], []

    def flush_batch():
        if not batch:
            return
        tr = translate_batch(list(batch), lang)
        for src, t in zip(batch, tr):
            t = t.rstrip("\n")
            if src.startswith(">") and not t.startswith(">"):
                t = "> " + t  # re-attach blockquote marker
            if src.lstrip().startswith("|") and not t.lstrip().startswith("|"):
                t = "|" + t
            out.append(t)
        batch.clear()

    for line in lines:
        if is_verbatim(line):
            flush_batch()
            out.append(line)
        else:
            batch.append(line)
    flush_batch()
    return "\n".join(out)


def translate_text(text: str, lang: str):
    parts = []
    for kind, seg in split_segments(text):
        parts.append(seg if kind == "fence" else translate_segment(seg, lang))
    return "\n\n".join(p.strip("\n") for p in parts) + "\n"


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "/home/alca/projects/mazemaker/README.md"
    lang = sys.argv[2] if len(sys.argv) > 2 else "de"
    t0 = time.time()
    out = translate_text(open(src, encoding="utf-8").read(), lang)
    print(f"{time.time()-t0:.1f}s, {len(out)} chars")
    print(out[:800])
