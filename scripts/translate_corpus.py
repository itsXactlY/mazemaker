#!/usr/bin/env python3
"""Corpus translator: translate any set of .md files to 12 languages via local LibreTranslate.

Usage:
  python3 scripts/translate_corpus.py [--files a.md b.md ...] [--langs ar de ...]

Defaults: the public-facing docs of the mazemaker repo.
Output:   i18n/<relative-path>.<lang>.md   (i.e. i18n/docs/architecture.de.md)
Resume:   a destination file that already exists is skipped.
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translate_lib import translate_text

REPO = "~/projects/mazemaker"

PUBLIC_DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "NOTICE",
    "docs/architecture.md",
    "docs/configuration.md",
    "docs/dream-engine.md",
    "docs/benchmarks.md",
    "docs/inception-bench.md",
    "docs/mcp-tools.md",
    "docs/federation.md",
    "docs/production-lessons.md",
    "docs/changelog-beta.md",
    "docs/testing.md",
    "docs/podman-ops.md",
    "benchmarks/README.md",
    "benchmarks/INCEPTION_BENCH_GUIDE.md",
]

LANGS = ["ar", "de", "es", "fr", "ga", "hi", "it", "ja", "ko", "pt", "ru", "zh-Hans"]


def out_path(repo: str, src_rel: str, lang: str) -> str:
    base = os.path.splitext(src_rel)[0]
    return os.path.join(repo, "i18n", base + "." + lang + ".md")


def translate_file(src_rel: str, langs: list, repo: str, verbose: bool) -> int:
    src = os.path.join(repo, src_rel)
    text = open(src, encoding="utf-8").read()
    written = 0
    for lang in langs:
        dest = out_path(repo, src_rel, lang)
        if os.path.exists(dest):
            print(f"  skip {lang}: {dest} exists")
            continue
        t0 = time.time()
        out = translate_text(text, lang)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "w", encoding="utf-8").write(out)
        written += 1
        print(f"  WROTE {dest} ({len(out)} chars, {time.time()-t0:.0f}s)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = args.files or PUBLIC_DOCS
    langs = args.langs or LANGS
    t0 = time.time()
    for f in files:
        if not os.path.exists(os.path.join(REPO, f)):
            print(f"MISSING: {f}")
            continue
        print(f"== {f} ({os.path.getsize(os.path.join(REPO, f))} bytes)")
        translate_file(f, langs, REPO, args.verbose)
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
