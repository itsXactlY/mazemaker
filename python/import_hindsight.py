#!/usr/bin/env python3
"""
import_hindsight.py - Import Hindsight Cloud data into Mazemaker

Connects to Hindsight Cloud API, exports all banks/memories/mental models,
and imports them into Mazemaker with proper embeddings.

Usage:
    python import_hindsight.py --api-key YOUR_KEY [--base-url https://api.hindsight.vectorize.io]
    python import_hindsight.py --api-key YOUR_KEY --bank-id specific_bank
    python import_hindsight.py --api-key YOUR_KEY --export-only  # export to JSON, no import

Environment:
    HINDSIGHT_API_KEY    - API key (overrides --api-key)
    HINDSIGHT_BASE_URL   - Base URL (default: https://api.hindsight.vectorize.io)
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Mazemaker imports
sys.path.insert(0, str(Path(__file__).parent))
from mazemaker import Memory

DB_PATH = Path.home() / ".neural_memory" / "memory.db"


# ---------------------------------------------------------------------------
# Hindsight API Client (stdlib-only, no SDK dependency)
# ---------------------------------------------------------------------------

class HindsightClient:
    """Minimal Hindsight Cloud API client using only stdlib."""

    def __init__(self, api_key: str, base_url: str = "https://api.hindsight.vectorize.io"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session_timeout = 30

    def _request(self, method: str, path: str, data: Optional[dict] = None,
                 params: Optional[dict] = None) -> Any:
        """Make an authenticated API request."""
        url = f"{self.base_url}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if qs:
                url += f"?{qs}"

        body = json.dumps(data).encode() if data else None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        req = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(req, timeout=self.session_timeout) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(
                f"Hindsight API {method} {path} failed: {e.code} {e.reason}\n{error_body}"
            ) from e

    def list_banks(self) -> List[Dict[str, Any]]:
        """List all memory banks."""
        return self._request("GET", "/api/v1/banks")

    def get_bank_stats(self, bank_id: str) -> Dict[str, Any]:
        """Get bank statistics."""
        return self._request("GET", f"/api/v1/banks/{bank_id}/stats")

    def list_memories(self, bank_id: str, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """List memories in a bank."""
        return self._request("GET", f"/api/v1/banks/{bank_id}/memory",
                            params={"limit": limit, "offset": offset})

    def recall(self, bank_id: str, query: str, top_k: int = 50) -> Dict[str, Any]:
        """Recall memories matching a query."""
        return self._request("POST", f"/api/v1/banks/{bank_id}/recall",
                            data={"query": query, "top_k": top_k})

    def list_mental_models(self, bank_id: str) -> List[Dict[str, Any]]:
        """List mental models in a bank."""
        return self._request("GET", f"/api/v1/banks/{bank_id}/mental-models")

    def list_documents(self, bank_id: str) -> List[Dict[str, Any]]:
        """List documents in a bank."""
        return self._request("GET", f"/api/v1/banks/{bank_id}/documents")

    def list_entities(self, bank_id: str) -> List[Dict[str, Any]]:
        """List entities in a bank."""
        return self._request("GET", f"/api/v1/banks/{bank_id}/entities")


# ---------------------------------------------------------------------------
# Export Logic
# ---------------------------------------------------------------------------

def export_bank(client: HindsightClient, bank_id: str, bank_name: str,
                output_dir: Path) -> Dict[str, Any]:
    """Export all data from a single bank to JSON files."""
    bank_dir = output_dir / f"{bank_id}_{bank_name}"
    bank_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Exporting bank: {bank_name} ({bank_id})")
    print(f"{'='*60}")

    result = {"bank_id": bank_id, "bank_name": bank_name, "counts": {}}

    # 1. Bank stats
    try:
        stats = client.get_bank_stats(bank_id)
        (bank_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"  Stats: {json.dumps(stats, indent=2)[:200]}...")
    except Exception as e:
        print(f"  Stats: failed ({e})")
        stats = {}

    # 2. Export memories (paginated)
    all_memories = []
    offset = 0
    limit = 100
    print(f"  Exporting memories...", flush=True)
    while True:
        try:
            resp = client.list_memories(bank_id, limit=limit, offset=offset)
            items = resp.get("items", resp if isinstance(resp, list) else [])
            if not items:
                break
            all_memories.extend(items)
            offset += len(items)
            print(f"    [{len(all_memories)}] memories fetched...", flush=True)
            if len(items) < limit:
                break
        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            break

    (bank_dir / "memories.json").write_text(json.dumps(all_memories, indent=2))
    result["counts"]["memories"] = len(all_memories)
    print(f"  Memories: {len(all_memories)}")

    # 3. Export mental models
    try:
        models = client.list_mental_models(bank_id)
        if isinstance(models, dict):
            models = models.get("items", [])
        (bank_dir / "mental_models.json").write_text(json.dumps(models, indent=2))
        result["counts"]["mental_models"] = len(models)
        print(f"  Mental models: {len(models)}")
    except Exception as e:
        print(f"  Mental models: failed ({e})")
        result["counts"]["mental_models"] = 0

    # 4. Export documents
    try:
        docs = client.list_documents(bank_id)
        if isinstance(docs, dict):
            docs = docs.get("items", [])
        (bank_dir / "documents.json").write_text(json.dumps(docs, indent=2))
        result["counts"]["documents"] = len(docs)
        print(f"  Documents: {len(docs)}")
    except Exception as e:
        print(f"  Documents: failed ({e})")
        result["counts"]["documents"] = 0

    # 5. Export entities
    try:
        entities = client.list_entities(bank_id)
        if isinstance(entities, dict):
            entities = entities.get("items", [])
        (bank_dir / "entities.json").write_text(json.dumps(entities, indent=2))
        result["counts"]["entities"] = len(entities)
        print(f"  Entities: {len(entities)}")
    except Exception as e:
        print(f"  Entities: failed ({e})")
        result["counts"]["entities"] = 0

    return result


def export_all(client: HindsightClient, output_dir: Path,
               bank_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Export all banks (or a single bank) to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if bank_id:
        # Export single bank
        try:
            stats = client.get_bank_stats(bank_id)
            name = stats.get("name", bank_id)
        except Exception:
            # Narrowed from bare `except:` so KeyboardInterrupt /
            # SystemExit propagate. API failures, network errors,
            # JSON decode errors are still caught and the bank id
            # itself is used as a fallback name.
            name = bank_id
        results.append(export_bank(client, bank_id, name, output_dir))
    else:
        # List and export all banks
        banks = client.list_banks()
        if isinstance(banks, dict):
            banks = banks.get("items", banks)
        print(f"\nFound {len(banks)} banks")
        for bank in banks:
            bid = bank.get("id", bank.get("bank_id", ""))
            bname = bank.get("name", bid)
            results.append(export_bank(client, bid, bname, output_dir))

    # Write summary
    summary = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "banks": results,
        "total_memories": sum(r["counts"].get("memories", 0) for r in results),
        "total_mental_models": sum(r["counts"].get("mental_models", 0) for r in results),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  Export complete: {summary['total_memories']} memories, "
          f"{summary['total_mental_models']} mental models")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# Import into Mazemaker
# ---------------------------------------------------------------------------

def import_bank(mem: Memory, bank_dir: Path, bank_id: str, bank_name: str,
                batch_size: int = 256) -> Dict[str, int]:
    """Import a single bank's data into Mazemaker."""
    import sqlite3

    print(f"\n{'='*60}")
    print(f"  Importing bank: {bank_name} ({bank_id})")
    print(f"{'='*60}")

    embedder = mem._embedder
    dim = embedder.dim
    counts = {"memories": 0, "mental_models": 0, "connections": 0}

    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return _import_bank_body(mem, bank_dir, bank_id, bank_name, batch_size, conn, counts, embedder, dim)
    finally:
        conn.close()


def _import_bank_body(mem, bank_dir, bank_id, bank_name, batch_size, conn, counts, embedder, dim):
    """Body of import_bank, called under conn try/finally for cleanup."""
    # 1. Import memories
    mem_file = bank_dir / "memories.json"
    if mem_file.exists():
        memories = json.loads(mem_file.read_text())
        print(f"  Importing {len(memories)} memories...")

        texts = []
        labels = []
        skipped = 0

        for m in memories:
            # Hindsight memory structure: content/text, id, tags, metadata
            content = (m.get("content") or m.get("text") or "").strip()
            if not content:
                skipped += 1
                continue

            mem_id = m.get("id", "")
            tags = m.get("tags", [])
            ts = m.get("timestamp", m.get("mentioned_at", ""))
            if isinstance(ts, dict):
                ts = ts.get("value", "")

            # Build label: hindsight:{bank}:{tags}:{id}
            tag_str = ",".join(tags[:3]) if tags else ""
            label = f"hindsight:{bank_name}:{tag_str}:{mem_id[:12]}" if mem_id else f"hindsight:{bank_name}:{len(labels)}"

            # Truncate very long content
            if len(content) > 8000:
                content = content[:8000] + "..."

            # Store metadata in content for context
            context_parts = []
            if ts:
                context_parts.append(f"[{ts}]")
            if tags:
                context_parts.append(f"tags: {', '.join(tags)}")
            if context_parts:
                content = " ".join(context_parts) + "\n" + content

            texts.append(content)
            labels.append(label)

        print(f"    {len(texts)} non-empty, {skipped} empty skipped")

        # Batch embed and insert
        t0 = time.time()
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            embeddings = embedder.embed_batch(batch_texts)

            rows = [
                (label, text, struct.pack(f'{dim}f', *emb))
                for label, text, emb in zip(batch_labels, batch_texts, embeddings)
            ]
            conn.executemany(
                "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)",
                rows
            )
            conn.commit()

            done = min(i + batch_size, total)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"    [{done}/{total}] {rate:.1f} msg/s, ETA: {eta:.0f}s", flush=True)

        counts["memories"] = total
        elapsed = time.time() - t0
        print(f"    Done: {total} memories in {elapsed:.1f}s ({total/elapsed:.1f} msg/s)")

    # 2. Import mental models
    models_file = bank_dir / "mental_models.json"
    if models_file.exists():
        models = json.loads(models_file.read_text())
        if models:
            print(f"  Importing {len(models)} mental models...")

            texts = []
            labels = []
            for m in models:
                name = m.get("name", "unnamed")
                content = m.get("content", "")
                source_query = m.get("source_query", "")
                tags = m.get("tags", [])

                if not content:
                    continue

                # Build rich text: name + source + content + tags
                parts = [f"[Mental Model: {name}]"]
                if source_query:
                    parts.append(f"Source query: {source_query}")
                parts.append(content)
                if tags:
                    parts.append(f"Tags: {', '.join(tags)}")

                text = "\n".join(parts)
                label = f"hindsight:model:{bank_name}:{name}"

                texts.append(text)
                labels.append(label)

            if texts:
                embeddings = embedder.embed_batch(texts)
                rows = [
                    (label, text, struct.pack(f'{dim}f', *emb))
                    for label, text, emb in zip(labels, texts, embeddings)
                ]
                conn.executemany(
                    "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)",
                    rows
                )
                conn.commit()
                counts["mental_models"] = len(texts)
                print(f"    Done: {len(texts)} mental models")

    # 3. Import documents as memories
    docs_file = bank_dir / "documents.json"
    if docs_file.exists():
        docs = json.loads(docs_file.read_text())
        if docs:
            print(f"  Importing {len(docs)} documents...")

            texts = []
            labels = []
            for d in docs:
                content = (d.get("content") or d.get("text") or "").strip()
                if not content:
                    continue
                doc_id = d.get("id", "")
                title = d.get("title", doc_id[:20])

                if len(content) > 8000:
                    content = content[:8000] + "..."

                text = f"[Document: {title}]\n{content}"
                label = f"hindsight:doc:{bank_name}:{doc_id[:12]}"

                texts.append(text)
                labels.append(label)

            if texts:
                embeddings = embedder.embed_batch(texts)
                rows = [
                    (label, text, struct.pack(f'{dim}f', *emb))
                    for label, text, emb in zip(labels, texts, embeddings)
                ]
                conn.executemany(
                    "INSERT INTO memories (label, content, embedding) VALUES (?, ?, ?)",
                    rows
                )
                conn.commit()
                counts["documents"] = len(texts)
                print(f"    Done: {len(texts)} documents")

    return counts


def import_all(mem: Memory, export_dir: Path, batch_size: int = 256) -> None:
    """Import all exported banks into Mazemaker."""
    summary_file = export_dir / "export_summary.json"
    if not summary_file.exists():
        print(f"ERROR: No export summary found at {summary_file}")
        print("Run with --export-only first, or point --export-dir to existing export")
        sys.exit(1)

    summary = json.loads(summary_file.read_text())
    print(f"\nImporting from export: {summary['exported_at']}")
    print(f"Banks: {len(summary['banks'])}")

    total_counts = {"memories": 0, "mental_models": 0, "documents": 0}

    for bank_info in summary["banks"]:
        bank_id = bank_info["bank_id"]
        bank_name = bank_info["bank_name"]
        bank_dir = export_dir / f"{bank_id}_{bank_name}"

        if not bank_dir.exists():
            print(f"  WARNING: Bank dir not found: {bank_dir}")
            continue

        counts = import_bank(mem, bank_dir, bank_id, bank_name, batch_size)
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

    print(f"\n{'='*60}")
    print(f"  Import complete!")
    for k, v in total_counts.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Connection Building
# ---------------------------------------------------------------------------

def build_connections(threshold: float = 0.15, sample_size: int = 5000):
    """Build graph connections between imported memories."""
    import math
    import sqlite3

    print(f"\n=== Building connections (threshold={threshold}) ===")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute("SELECT id, embedding FROM memories ORDER BY id").fetchall()
        total = len(rows)
        print(f"  {total} memories total")

        if total == 0:
            return

        dim = len(rows[0][1]) // 4
        all_ids = []
        all_embs = []
        for row in rows:
            all_ids.append(row[0])
            all_embs.append(list(struct.unpack(f'{dim}f', row[1])))

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na * nb > 0 else 0

        connections = []
        t0 = time.time()

        if total <= sample_size:
            for i in range(total):
                for j in range(i + 1, total):
                    sim = cosine(all_embs[i], all_embs[j])
                    if sim > threshold:
                        connections.append((all_ids[i], all_ids[j], sim))
                if (i + 1) % 500 == 0:
                    print(f"  [{i + 1}/{total}] {len(connections)} connections", flush=True)
        else:
            window = 200
            for i in range(total):
                start = max(0, i - window)
                end = min(total, i + window)
                for j in range(start, end):
                    if j <= i:
                        continue
                    sim = cosine(all_embs[i], all_embs[j])
                    if sim > threshold:
                        connections.append((all_ids[i], all_ids[j], sim))
                if (i + 1) % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    print(f"  [{i + 1}/{total}] {len(connections)} conns, {rate:.0f}/s", flush=True)

        print(f"  Inserting {len(connections)} connections...")
        conn.executemany(
            "INSERT OR IGNORE INTO connections (source_id, target_id, weight, edge_type) VALUES (?, ?, ?, 'similar')",
            connections
        )
        conn.commit()
        print(f"  Done: {len(connections)} connections in {time.time() - t0:.1f}s")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Import Hindsight Cloud data into Mazemaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all banks to JSON (no import)
  python import_hindsight.py --api-key KEY --export-only

  # Export and import everything
  python import_hindsight.py --api-key KEY

  # Import from existing export
  python import_hindsight.py --import-only --export-dir ~/hindsight_export

  # Export specific bank
  python import_hindsight.py --api-key KEY --bank-id abc123
        """
    )
    parser.add_argument("--api-key", default=os.environ.get("HINDSIGHT_API_KEY"),
                        help="Hindsight API key (or set HINDSIGHT_API_KEY)")
    parser.add_argument("--base-url", default=os.environ.get("HINDSIGHT_BASE_URL",
                        "https://api.hindsight.vectorize.io"),
                        help="Hindsight API base URL")
    parser.add_argument("--bank-id", help="Export only this specific bank")
    parser.add_argument("--export-dir", default=str(Path.home() / "hindsight_export"),
                        help="Directory for export files")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export to JSON, don't import")
    parser.add_argument("--import-only", action="store_true",
                        help="Only import from existing export dir")
    parser.add_argument("--no-connections", action="store_true",
                        help="Skip connection building after import")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Embedding batch size")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Connection similarity threshold")

    args = parser.parse_args()
    export_dir = Path(args.export_dir)

    if not args.import_only:
        if not args.api_key:
            print("ERROR: --api-key or HINDSIGHT_API_KEY required")
            sys.exit(1)

        client = HindsightClient(args.api_key, args.base_url)
        print(f"Connected to Hindsight: {args.base_url}")

        export_all(client, export_dir, args.bank_id)

        if args.export_only:
            print("\nExport complete. Skipping import (--export-only).")
            return

    if not args.export_only:
        mem = Memory()
        print("Mazemaker initialized")
        import_all(mem, export_dir, args.batch_size)

        if not args.no_connections:
            build_connections(args.threshold)

    print("\nDone! Run 'mazemaker_remember' or check ~/.neural_memory/memory.db")


if __name__ == "__main__":
    main()
