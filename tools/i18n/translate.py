#!/usr/bin/env python3
"""
i18n/translate.py — batch-translate a markdown file through a self-hosted
LibreTranslate server (argos backend).

Splits the source markdown into section chunks (so tables, lists and code
blocks stay intact), POSTs each chunk to /translate, reassembles the
translated document, and writes one output file per target language.

Usage:
  python3 tools/i18n/translate.py README.md \
      --server http://127.0.0.1:8089 \
      --out README.{lang}.md \
      --langs de,fr,es,pt,it,nl,pl,ru,ar,zh,hi,ja,tr

Notes:
  - LibreTranslate refuses chunks > max-text-length (default 4000 chars);
    we pre-split at '##' / '###' / paragraph boundaries so no chunk overflows.
  - code blocks are skipped by LibreTranslate itself when format=markdown,
    so install instructions / code fences survive verbatim in every language.
  - model downloads are lazy: the first request for a language pair pulls
    the opus-mt model into the server's argos cache (several hundred MB).
"""

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

MAX_CHUNK = 3900  # stay under the server's 4000-char default


def chunk_lines(lines: list[str], max_len: int) -> list[str]:
    """Group consecutive lines into chunks that start at a heading and
    stay under max_len chars. A chunk may start mid-document (first
    chunk starts at line 0)."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append(cur)
            cur, cur_len = [], 0

    for line in lines:
        if re.match(r"^#{1,3} ", line) and cur_len > 0:
            flush()
        # force-split oversized plain chunks at paragraph boundaries
        if cur_len + len(line) + 1 > max_len and line.strip() == "" and cur_len > max_len // 2:
            flush()
        cur.append(line)
        cur_len += len(line) + 1
    flush()

    # sub-split any chunk that still overflows (long tables, etc.)
    final: list[str] = []
    for ch in chunks:
        text = "\n".join(ch)
        if len(text) <= max_len:
            final.append(text)
            continue
        # split on blank lines, greedily
        paras, buf, blen = [], [], 0
        for ln in ch:
            if ln.strip() == "" and buf:
                paras.append("\n".join(buf))
                buf, blen = [], 0
            else:
                buf.append(ln)
                blen += len(ln) + 1
        if buf:
            paras.append("\n".join(buf))
        acc, alen = [], 0
        for p in paras:
            if acc and alen + len(p) + 2 > max_len:
                final.append("\n".join(acc))
                acc, alen = [], 0
            acc.append(p)
            alen += len(p) + 2
        if acc:
            final.append("\n".join(acc))
    return final


def translate_chunk(server: str, source: str, target: str, q: str,
                    retries: int = 4) -> str:
    url = f"{server}/translate"
    for attempt in range(retries):
        r = requests.post(
            url,
            json={"q": q, "source": source, "target": target, "format": "markdown"},
            timeout=600,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data[0]["translatedText"]
            return data["translatedText"]
        # 429 / 500 (model download in flight) -> backoff and retry
        if r.status_code in (429, 500, 502, 503):
            wait = 15 * (attempt + 1)
            print(f"    server {r.status_code} ({wait}s backoff)", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"translate failed: HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"translate failed after {retries} attempts")


def translate_file(server: str, src_path: Path, out_template: str,
                   langs: list[str], source_lang: str = "en",
                   keep_header_note: bool = True) -> None:
    text = src_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    chunks = chunk_lines(lines, MAX_CHUNK)
    print(f"source: {src_path}  ({len(text)} chars, {len(chunks)} chunks)")

    # sanity: /languages must include every target
    r = requests.get(f"{server}/languages", timeout=30)
    r.raise_for_status()
    known = {l["code"] for l in r.json()}
    unknown = [l for l in langs if l not in known]
    if unknown:
        print(f"WARNING: not supported by this server: {unknown}", file=sys.stderr)

    for lang in langs:
        if lang in unknown:
            continue
        print(f"\n[{lang}] translating {len(chunks)} chunks ...", flush=True)
        out_parts = []
        for i, ch in enumerate(chunks):
            t0 = time.time()
            tr = translate_chunk(server, source_lang, lang, ch)
            out_parts.append(tr)
            print(f"    chunk {i + 1}/{len(chunks)}  {len(ch)} -> {len(tr)} chars  "
                  f"({time.time() - t0:.1f}s)", flush=True)
        doc = "\n".join(out_parts)
        if keep_header_note:
            note = (f"<!-- Translated from {source_lang} via self-hosted "
                    f"LibreTranslate (argos/opus-mt) on {date.today()}. "
                    f"Source of truth: {src_path.name} -->\n\n")
            doc = note + doc
        out_path = Path(out_template.format(lang=lang))
        out_path.write_text(doc, encoding="utf-8")
        print(f"[{lang}] wrote {out_path}  ({len(doc)} chars)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_file", help="markdown file to translate (English source)")
    ap.add_argument("--server", default="http://127.0.0.1:8089",
                    help="LibreTranslate base URL")
    ap.add_argument("--out", default="README.{lang}.md",
                    help="output template; {lang} is replaced by the ISO code")
    ap.add_argument("--langs", required=True,
                    help="comma-separated ISO codes, e.g. de,fr,es,pt,it,nl,pl,ru,ar,zh,hi,ja,tr")
    ap.add_argument("--source-lang", default="en")
    args = ap.parse_args()

    try:
        requests.get(f"{args.server}/health", timeout=10).raise_for_status()
    except Exception as e:
        sys.exit(f"server not reachable at {args.server}: {e}")

    translate_file(
        args.server,
        Path(args.source_file),
        args.out,
        [l.strip() for l in args.langs.split(",") if l.strip()],
        args.source_lang,
    )


if __name__ == "__main__":
    main()
