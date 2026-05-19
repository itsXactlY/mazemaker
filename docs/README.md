# Mazemaker Documentation

The full documentation for the Mazemaker persistent-cognition engine.
Nine docs, organized by what you're trying to accomplish.

> **First-time visitor?** Start with the project [`README.md`](../README.md)
> at the repo root for the one-page pitch. Come back here when you want
> the details.

---

## Reading paths

### "I want to install and use it"

1. [`../README.md`](../README.md) — pitch + install one-liner
2. [`configuration.md`](configuration.md) — the minimal config + the recommended config
3. [`mcp-tools.md`](mcp-tools.md) — the LLM-callable surface
4. [`testing.md`](testing.md) — verify your install with the smoke test

### "I want to understand how it works"

1. [`architecture.md`](architecture.md) — the six-layer cognition stack
2. [`dream-engine.md`](dream-engine.md) — NREM / REM / Insight / AFE / Stage S / DAE
3. [`benchmarks.md`](benchmarks.md) — what the engine produces and how it was audited
4. [`inception-bench.md`](inception-bench.md) — the deterministic benchmark methodology

### "I want to reproduce the numbers"

1. [`benchmarks.md`](benchmarks.md) — full reproduction recipe
2. [`changelog-beta.md`](changelog-beta.md) — the 1.6 GB claim-evidence bundle (ProtonDrive)
3. [`inception-bench.md`](inception-bench.md) — run the deterministic bench yourself

### "I want to run it in production"

1. [`podman-ops.md`](podman-ops.md) — start / stop / check / update TL;DR card
2. [`configuration.md`](configuration.md) — every knob, every env var
3. [`dream-engine.md`](dream-engine.md) — standalone daemon for big corpora
4. [`production-lessons.md`](production-lessons.md) — what bites at scale
5. [`mcp-tools.md`](mcp-tools.md) — the integration shapes

### "I want to contribute or audit"

1. [`architecture.md`](architecture.md) — internal map
2. [`testing.md`](testing.md) — test surfaces + file structure
3. [`production-lessons.md`](production-lessons.md) — patched-bug index + operator rules
4. [`benchmarks/audit/`](../benchmarks/audit/) in the repo — eight rounds of GPT-5.5 audit (verbatim)

---

## File index

| Doc                                          | What it covers                                                          | Length |
|----------------------------------------------|-------------------------------------------------------------------------|--------|
| [`architecture.md`](architecture.md)         | Six-layer cognition stack · embedding backends · retrieval pipeline · GPU recall · graph · schema | medium |
| [`configuration.md`](configuration.md)       | Every YAML knob · every env var · retrieval-mode cheat sheet · tier-gated features · tuning recipes | medium |
| [`dream-engine.md`](dream-engine.md)         | NREM / SUPERSEDES / REM / Insight / AFE / DAE · triggers · sampling · GPU acceleration · standalone daemon | long   |
| [`benchmarks.md`](benchmarks.md)             | Inception Bench · LongMemEval-oracle · LongMemEval-S · Comparison Bench · audit story · reproduction recipe | long   |
| [`inception-bench.md`](inception-bench.md)   | Why external rubrics were broken · the methodology · the 12 scenarios · what we deliberately don't do | medium |
| [`mcp-tools.md`](mcp-tools.md)               | Nine tools, input/output JSON, integration shapes, quick-starts        | medium |
| [`podman-ops.md`](podman-ops.md)             | TL;DR podman card — start / stop / check status / update / logs / common failures | short |
| [`testing.md`](testing.md)                   | Smoke test · full suite · clean VM verification · file structure       | short  |
| [`production-lessons.md`](production-lessons.md) | Operator rules · benchmark-driven defaults · bench-noise discipline · external-audit handling · patched-bug index | medium |
| [`changelog-beta.md`](changelog-beta.md)     | Official Beta release notes · the threshold · six layers · engineering deliverables · 1.6 GB bundle | long   |

---

## Conventions

- **Code blocks** with bash use `$` only when the input differs from
  output. Most blocks omit it.
- **Defaults** are stated alongside every knob.
- **Pro-only features** are flagged in tables.
- **Cross-links** between docs use relative paths.
- **External links** (mazemaker.online, GitHub, ProtonDrive) are
  absolute.

---

## What's not in this directory

- **`UPPERCASE_*.md` files** in this directory are internal strategy
  documents (BENCHMARK_DOMINANCE_PLAN, PROPRIETARY_STACK_INTERNAL,
  RESEARCH_FRONTIER, SOTA_LANDSCAPE, DAE_DESIGN). Leave them; they're
  not the user-facing docs.
- **Marketing pages** live on
  [mazemaker.online](https://mazemaker.online/) — the
  [research](https://mazemaker.online/research) page, the
  [comparison](https://mazemaker.online/comparison/) page, the
  [architecture](https://mazemaker.online/architecture) page, the
  [blog](https://mazemaker.online/blog/), and the three deep-dives
  ([architect](https://mazemaker.online/architect/) /
  [onboarding](https://mazemaker.online/onboarding/) /
  [topology](https://mazemaker.online/topology/)).
- **Operator/agent-facing guidance** is in [`../CLAUDE.md`](../CLAUDE.md)
  at the repo root.

---

## Going further

- **Repo:** [github.com/itsXactlY/mazemaker](https://github.com/itsXactlY/mazemaker)
- **Console:** [mazemaker.dev](https://mazemaker.dev)
- **Marketing:** [mazemaker.online](https://mazemaker.online)
- **Contact:** `info@mazemaker.dev` · `enterprise@mazemaker.dev` ·
  `privacy@mazemaker.dev`
