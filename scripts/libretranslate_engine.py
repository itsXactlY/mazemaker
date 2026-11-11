"""LibreTranslate engine — structural-aware Markdown translation.

Strategy: translate LINE BY LINE, and for structural lines (headings,
blockquotes, table rows, separators) strip the markers before sending to
the MT engine and re-attach them afterwards. This prevents MT from
mangling `#`, `>`, `|`, `---` (argos is bad at zh: `##` -> `∮`, `###` ->
`□`/`•`/`·`; ar leaves the em-dash title untranslated).

Segment splitting still protects code fences from MT.

Usage:
    python3 scripts/libretranslate_engine.py README.md i18n/README.fr.md --langs de fr es pt it nl pl ru uk ja zh ko
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_API = "http://127.0.0.1:8089"

# per-language manual overrides for lines MT mangles (keyed by (lang, line_number_1based))
OVERRIDES = {
    ("ar", 1): "Mazemaker — نظام تشغيل وكلاء الذكاء الاصطناعي",
}

RE_FENCE = re.compile(r"^(```|~~~)")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
RE_QUOTE = re.compile(r"^>\s?(.*)$")
RE_HR = re.compile(r"^[\s\t:|-]*-[\s\t:|-]*$")
RE_PIPE_LINE = re.compile(r"^\|.*\|\s*$")
RE_EMPTY = re.compile(r"^\s*$")


def translate_one(api: str, q: str, target: str, retries: int = 2, timeout: int = 120) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            body = json.dumps({"q": q, "source": "en", "target": target}).encode("utf-8")
            req = urllib.request.Request(
                f"{api}/translate", data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))["translatedText"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_err = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"translate failed for target={target}: {last_err}")


def translate_structural_line(api: str, line: str, target: str) -> str:
    """Translate one Markdown line, preserving its structure."""
    m = RE_HEADING.match(line)
    if m:
        hashes, text = m.groups()
        tr = translate_one(api, text, target)
        tr = tr.strip()
        return f"{hashes} {tr}"

    m = RE_QUOTE.match(line)
    if m:
        text = m.group(1)
        if not text.strip():
            return line
        tr = translate_one(api, text, target)
        tr = " ".join(tr.split())
        return f"> {tr}"

    if RE_PIPE_LINE.match(line):
        stripped = line.strip().strip("|")
        tr = translate_one(api, stripped, target)
        tr = tr.strip()
        if "|" not in tr:
            # re-apply pipe count from the source line
            n = line.count("|") - 1
            parts = [p.strip() for p in tr.split(" ")]
            cells = parts[:n] if len(parts) >= n else parts + [""] * (n - len(parts))
            return "| " + " | ".join(cells) + " |"
        return tr

    if RE_HR.match(line) or RE_EMPTY.match(line):
        return line

    # plain paragraph line: translate as-is
    return translate_one(api, line, target)


def split_code_fences(text: str) -> list[tuple[bool, list[str]]]:
    segments: list[tuple[bool, list[str]]] = []
    fence: list[str] | None = None
    for line in text.splitlines():
        if RE_FENCE.match(line):
            if fence is None:
                fence = [line]
            else:
                segments.append((True, fence))
                fence = None
                continue
            continue
        if fence is not None:
            fence.append(line)
            continue
        if not segments or not segments[-1][0]:
            segments.append((False, [line]))
        else:
            segments[-1][1].append(line)
    if fence is not None:
        segments.append((True, fence))
    return segments


def translate_document(api: str, text: str, target: str) -> str:
    out: list[str] = []
    for is_code, lines in split_code_fences(text):
        if is_code:
            out.extend(lines)
            continue
        for i, line in enumerate(lines, start=1):
            key = (target, i)
            if key in OVERRIDES:
                out.append(OVERRIDES[key])
                continue
            if RE_HR.match(line) or RE_EMPTY.match(line):
                out.append(line)
                continue
            out.append(translate_structural_line(api, line, target))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--langs", required=True, help="space-separated target codes")
    ap.add_argument("--api", default=DEFAULT_API)
    args = ap.parse_args()

    langs = args.langs.split()
    text = open(args.src, encoding="utf-8").read()

    t0 = time.time()
    for lang in langs:
        out = translate_document(args.api, text, lang)
        path = args.dst.replace(".md", f".{lang}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"  {lang:4s} -> {path} ({len(out.splitlines())} lines)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
