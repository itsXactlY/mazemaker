"""
JRWL MSSQL Store - Transparent proxy that routes MSSQL calls through JRWL.

Drop-in replacement for MSSQLStore that talks to JRWL Broker instead
of directly connecting to MSSQL. Used inside SmolVM when JRWL mode is enabled.

Usage:
    from jrwl import JRWLConfig
    from jrwl.store import JRWLMSSQLStore

    store = JRWLMSSQLStore()
    mid = store.store("test", "Hello", [0.1] * 384)
    store.close()
"""

import struct
from typing import Optional

from jrwl.client import JRWLClient
from jrwl.config import JRWLConfig


class JRWLMSSQLStore:
    """MSSQL-compatible store that routes through JRWL bus.

    Implements the same interface as mssql_store.MSSQLStore
    so it can be used as a drop-in replacement.
    """

    def __init__(self, config: JRWLConfig = None):
        self._config = config or JRWLConfig()
        self._client = JRWLClient(self._config)
        # Synchronous connect (run event loop for init)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context — store for later
            self._loop = loop
            self._async_mode = True
        except RuntimeError:
            # No running loop — create one for sync operations
            self._loop = asyncio.new_event_loop()
            self._async_mode = False
            self._loop.run_until_complete(self._client.connect())

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        if self._async_mode:
            # We're inside an event loop — can't use run_until_complete
            # This is a design limitation; use async methods instead
            raise RuntimeError(
                "JRWLMSSQLStore used in async context. Use async methods directly."
            )
        return self._loop.run_until_complete(coro)

    def store(self, label: str, content: str, embedding: list[float]) -> int:
        """Store a memory with binary embedding blob."""
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        # Use parameterized query with binary param
        affected = self._run_async(
            self._client.exec(
                "INSERT INTO memories (label, content, embedding, vector_dim) "
                "VALUES (?, ?, ?, ?)",
                [label, content, blob, len(embedding)],
            )
        )
        # Get the inserted ID
        rows = self._run_async(
            self._client.query("SELECT SCOPE_IDENTITY() AS new_id")
        )
        return int(rows[0]["new_id"]) if rows else 0

    def get(self, id_: int) -> Optional[dict]:
        """Get a single memory by ID."""
        rows = self._run_async(
            self._client.query(
                "SELECT id, label, content, embedding, vector_dim, salience, access_count "
                "FROM memories WHERE id = ?",
                [id_],
            )
        )
        if not rows:
            return None

        row = rows[0]
        blob = row.get("embedding")
        dim = row.get("vector_dim", 0)
        if blob and isinstance(blob, str):
            blob = bytes.fromhex(blob)
        embedding = list(struct.unpack(f'{dim}f', blob)) if blob and dim else []

        return {
            "id": row["id"],
            "label": row["label"],
            "content": row["content"],
            "embedding": embedding,
            "salience": row.get("salience", 1.0),
            "access_count": row.get("access_count", 0),
        }

    def get_all(self) -> list[dict]:
        """Get all memories."""
        rows = self._run_async(
            self._client.query(
                "SELECT id, label, content, embedding, vector_dim, salience, access_count "
                "FROM memories ORDER BY id"
            )
        )
        results = []
        for row in rows:
            blob = row.get("embedding")
            dim = row.get("vector_dim", 0)
            if blob and isinstance(blob, str):
                blob = bytes.fromhex(blob)
            embedding = list(struct.unpack(f'{dim}f', blob)) if blob and dim else []
            results.append({
                "id": row["id"],
                "label": row["label"],
                "content": row["content"],
                "embedding": embedding,
                "salience": row.get("salience", 1.0),
                "access_count": row.get("access_count", 0),
            })
        return results

    def touch(self, id_: int):
        """Update access timestamp and count."""
        self._run_async(
            self._client.exec(
                "UPDATE memories SET last_accessed = SYSUTCDATETIME(), "
                "access_count = access_count + 1 WHERE id = ?",
                [id_],
            )
        )

    def add_connection(self, source: int, target: int, weight: float, edge_type: str = "similar"):
        """Add or update a graph edge."""
        self._run_async(
            self._client.exec(
                "MERGE connections AS target "
                "USING (VALUES (?, ?, ?, ?)) AS source (source_id, target_id, weight, edge_type) "
                "ON target.source_id = source.source_id AND target.target_id = source.target_id "
                "WHEN MATCHED THEN "
                "    UPDATE SET weight = CASE WHEN source.weight > target.weight THEN source.weight ELSE target.weight END, "
                "               edge_type = source.edge_type "
                "WHEN NOT MATCHED THEN "
                "    INSERT (source_id, target_id, weight, edge_type) "
                "    VALUES (source.source_id, source.target_id, source.weight, source.edge_type);",
                [source, target, weight, edge_type],
            )
        )

    def get_connections(self, node_id: int) -> list[dict]:
        """Get all connections for a node."""
        rows = self._run_async(
            self._client.query(
                "SELECT source_id, target_id, weight, edge_type "
                "FROM connections WHERE source_id = ? OR target_id = ? "
                "ORDER BY weight DESC",
                [node_id, node_id],
            )
        )
        return [
            {"source": r["source_id"], "target": r["target_id"],
             "weight": r["weight"], "type": r["edge_type"]}
            for r in rows
        ]

    def stats(self) -> dict:
        """Get store statistics."""
        mc = self._run_async(self._client.query("SELECT COUNT(*) AS cnt FROM memories"))
        cc = self._run_async(self._client.query("SELECT COUNT(*) AS cnt FROM connections"))
        return {
            "memories": mc[0]["cnt"] if mc else 0,
            "connections": cc[0]["cnt"] if cc else 0,
        }

    def close(self):
        """Close the JRWL client connection."""
        if not self._async_mode:
            self._loop.run_until_complete(self._client.disconnect())
            self._loop.close()
