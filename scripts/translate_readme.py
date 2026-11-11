#!/usr/bin/env python3
"""Translate /home/alca/projects/mazemaker/README.md into 12 languages.

Writes i18n/README.<lang>.md. Reusable driver — the engine is translate_lib.py.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translate_lib import translate_text

SRC = "/home/alca/projects/mazemaker/README.md"
OUTDIR = "/home/alca/projects/mazemaker/i18n"
LOG = os.path.join(OUTDIR, "translate.log")

LANGS = [
    "de", "fr", "es", "pt", "it", "nl", "pl", "ru",
    "uk", "ja", "zh", "ko",
]

text = open(SRC, encoding="utf-8").read()
os.makedirs(OUTDIR, exist_ok=True)

log = open(LOG, "a", encoding="utf-8")
t0 = time.time()
for lang in LANGS:
    try:
        out = translate_text(text, lang)
        dest = os.path.join(OUTDIR, f"README.{lang}.md")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(out)
        log.write(f"WROTE README.{lang}.md ({len(out)} chars)\n")
        print(f"done {lang}: {len(out)} chars", flush=True)
    except Exception as e:
        log.write(f"FAILED {lang}: {e}\n")
        print(f"FAILED {lang}: {e}", flush=True)

log.write(f"README pass finished in {time.time()-t0:.0f}s\n")
print(f"all done in {time.time()-t0:.0f}s")
