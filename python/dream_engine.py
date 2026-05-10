"""Dream Engine — autonomous background memory consolidation.

Implements three phases inspired by biological sleep:
  1. NREM — Replay & consolidation (strengthen, prune)
  2. REM  — Exploration & bridge discovery
  3. Insight — Abstraction & community detection

Runs as a background daemon during idle periods. Stores results
in the same SQLite DB as the main mazemaker, extended with
dream-specific tables.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema extensions for dream tables (SQLite)
# ---------------------------------------------------------------------------

_DREAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS dream_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,
    phase TEXT NOT NULL,
    memories_processed INTEGER DEFAULT 0,
    connections_strengthened INTEGER DEFAULT 0,
    connections_pruned INTEGER DEFAULT 0,
    bridges_found INTEGER DEFAULT 0,
    insights_created INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dream_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    insight_type TEXT NOT NULL,
    source_memory_id INTEGER,
    content TEXT,
    confidence REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS connection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    changed_at REAL NOT NULL,
    -- FK to dream_sessions(id) — populated by dream-engine writes; NULL for
    -- non-cycle writes (e.g. user-triggered remember() auto-connect). Lets
    -- /dream/cycles/{id}/diff resolve via index instead of timestamp window.
    dream_session_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_dream_insights_type
    ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session
    ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_nodes
    ON connection_history(source_id, target_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_changed_at
    ON connection_history(changed_at);
CREATE INDEX IF NOT EXISTS idx_conn_history_session
    ON connection_history(dream_session_id);
CREATE INDEX IF NOT EXISTS idx_dream_sessions_started_at
    ON dream_sessions(started_at);
"""


# ---------------------------------------------------------------------------
# Abstract Dream Backend
# ---------------------------------------------------------------------------

class DreamBackend:
    """Interface for dream storage backends (SQLite or Postgres)."""

    def start_session(self, phase: str) -> int:
        raise NotImplementedError

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def sample_for_dream(
        self,
        limit: int,
        recent_pct: float = 0.5,
        random_old_pct: float = 0.3,
        low_salience_pct: float = 0.2,
    ) -> List[Dict[str, Any]]:
        # Mixed sample for NREM replay so the cycle doesn't only re-touch the
        # newest <limit> memories and recycle the surface forever. Default mix:
        # 50% recent, 30% random across all memories, 20% lowest-salience.
        # Result is deduped and capped at limit.
        # Backward-compat fallback: subclasses that don't override get pure
        # recent — same as get_recent_memories.
        return self.get_recent_memories(limit)

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def sample_isolated_for_dream(
        self,
        limit: int,
        max_connections: int = 3,
        recent_pct: float = 0.5,
        random_old_pct: float = 0.3,
        low_salience_pct: float = 0.2,
    ) -> List[Dict[str, Any]]:
        # Same mix-philosophy as sample_for_dream but constrained to memories
        # with < max_connections edges. REM tries to bridge orphaned memories,
        # so picking only "recent isolated" means anything that got isolated
        # six months ago and was never revisited stays a permanent orphan.
        # Backward-compat fallback: subclasses get pure recent.
        return self.get_isolated_memories(max_connections, limit)

    def get_connections(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        raise NotImplementedError

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        raise NotImplementedError

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                      delta: float = 0.05,
                                      dream_session_id: Optional[int] = None) -> int:
        """Bulk strengthen. Returns count updated.
        When dream_session_id is given, every per-edge strengthen also writes
        a connection_history row tagged with that session for replay/diff."""
        count = 0
        for src, tgt in edges:
            self.strengthen_connection(src, tgt, delta)
            count += 1
        return count

    def batch_weaken_connections(self, threshold: float = 0.05,
                                  delta: float = 0.01,
                                  dream_session_id: Optional[int] = None) -> int:
        """Bulk weaken all connections above threshold. Returns count updated.
        When dream_session_id is given, the bulk-weaken is summarised with a
        single connection_history row (reason='nrem_bulk_weaken') so the diff
        endpoint can show the cycle's overall decay without exploding rows."""
        raise NotImplementedError

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> bool:
        """Insert a bridge edge. Returns True if newly inserted, False if skipped
        (e.g. edge already exists or self-loop). Backends must canonicalise
        source<target before INSERT so connection_history and the connections
        table stay aligned. Callers should only log_connection_change on True.
        """
        raise NotImplementedError

    def add_bridges_batch(
        self,
        bridges: "List[Tuple[int, int, float]]",
        dream_session_id: "Optional[int]" = None,
        reason: str = "rem_bridge",
    ) -> int:
        """Insert many bridges in ONE transaction. Returns count of newly
        inserted edges. Same semantics as add_bridge — self-loops dropped,
        existing edges skipped, source<target canonicalised — but bypasses
        the per-edge fsync of one commit per call. Default fallback path
        loops add_bridge for backends that don't override; SQLite has a
        proper bulk implementation."""
        if not bridges:
            return 0
        new_count = 0
        for s, t, w in bridges:
            if self.add_bridge(s, t, w):
                src, tgt = (s, t) if s < t else (t, s)
                self.log_connection_change(
                    src, tgt, 0.0, w, reason,
                    dream_session_id=dream_session_id,
                )
                new_count += 1
        return new_count

    def prune_weak(self, threshold: float = 0.05) -> int:
        raise NotImplementedError

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str,
                               dream_session_id: Optional[int] = None) -> None:
        raise NotImplementedError

    def get_memory_vectors(self, memory_ids: List[int]) -> Dict[int, List[float]]:
        """Return embeddings for memory IDs. Optional backend capability."""
        return {}

    def set_connection_weight(self, source_id: int, target_id: int,
                              weight: float, reason: str = "semantic_reweight") -> bool:
        raise NotImplementedError

    def add_typed_connection(self, source_id: int, target_id: int,
                             weight: float = 0.5,
                             edge_type: str = "similar") -> bool:
        if edge_type == "bridge":
            self.add_bridge(source_id, target_id, weight)
            return True
        raise NotImplementedError

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        raise NotImplementedError

    def add_insights_batch(self, items) -> int:
        """Bulk-insert insights in one transaction.

        ``items`` is an iterable of
        ``(session_id, insight_type, source_memory_id, content, confidence)``
        tuples. Returns the number of rows inserted. Default implementation
        loops the per-row method for backends without bulk support.
        """
        n = 0
        for sid, t, src, content, conf in items:
            self.add_insight(sid, t, src, content, conf)
            n += 1
        return n

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Delete history entries older than keep_days. Returns count deleted."""
        raise NotImplementedError

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Delete dream sessions older than keep_days. Returns count deleted."""
        raise NotImplementedError

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories."""
        raise NotImplementedError

    def get_dream_stats(self) -> Dict[str, Any]:
        raise NotImplementedError

    def recent_cluster_anchors(self, window_seconds: float) -> "Set[int]":
        """Source-memory ids already emitted as `cluster` insights inside
        the last ``window_seconds``. Used by `_phase_insight` to rotate
        away from re-emitting the same top-N largest communities every
        cycle. Default: empty set (no rotation, backwards-compat)."""
        return set()


# ---------------------------------------------------------------------------
# SQLite Dream Backend
# ---------------------------------------------------------------------------

class SQLiteDreamBackend(DreamBackend):
    """Dream backend using the existing mazemaker SQLite DB."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._persistent_conn = None
        self._ensure_tables()

    @property
    def conn(self):
        """Persistent connection for DreamWorker compatibility."""
        if self._persistent_conn is None:
            self._persistent_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        return self._persistent_conn

    def close(self):
        if self._persistent_conn:
            self._persistent_conn.close()
            self._persistent_conn = None

    def _ensure_tables(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_DREAM_SCHEMA)
            # Existing mazemaker DBs may predate typed/bi-temporal edges.
            has_connections = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='connections'"
            ).fetchone()
            if has_connections:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(connections)").fetchall()}
                if "edge_type" not in cols:
                    conn.execute("ALTER TABLE connections ADD COLUMN edge_type TEXT DEFAULT 'similar'")
                for col, sql in {
                    "event_time": "ALTER TABLE connections ADD COLUMN event_time REAL",
                    "ingestion_time": "ALTER TABLE connections ADD COLUMN ingestion_time REAL DEFAULT (unixepoch())",
                    "valid_from": "ALTER TABLE connections ADD COLUMN valid_from REAL",
                    "valid_to": "ALTER TABLE connections ADD COLUMN valid_to REAL",
                }.items():
                    if col not in cols:
                        try:
                            conn.execute(sql)
                        except sqlite3.OperationalError:
                            pass
                conn.execute("UPDATE connections SET edge_type = 'similar' WHERE edge_type IS NULL OR edge_type = ''")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_connections_edge_type_weight "
                    "ON connections(edge_type, weight)"
                )
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def start_session(self, phase: str) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO dream_sessions (started_at, phase) VALUES (?, ?)",
                (time.time(), phase)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        if session_id < 0:
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE dream_sessions SET "
                "finished_at = ?, memories_processed = ?, "
                "connections_strengthened = ?, connections_pruned = ?, "
                "bridges_found = ?, insights_created = ? WHERE id = ?",
                (
                    time.time(),
                    stats.get("processed", stats.get("explored", 0)),
                    stats.get("strengthened", 0),
                    stats.get("pruned", 0),
                    stats.get("bridges", 0),
                    stats.get("insights", 0),
                    session_id,
                )
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, content FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"id": r["id"], "content": r["content"] or ""} for r in rows]
        finally:
            conn.close()

    def sample_for_dream(
        self,
        limit: int,
        recent_pct: float = 0.5,
        random_old_pct: float = 0.3,
        low_salience_pct: float = 0.2,
    ) -> List[Dict[str, Any]]:
        recent_n = max(1, int(limit * recent_pct))
        random_n = max(0, int(limit * random_old_pct))
        low_n = max(0, limit - recent_n - random_n)

        seen: Set[int] = set()
        out: List[Dict[str, Any]] = []

        def _push(rows):
            for r in rows:
                rid = r["id"]
                if rid in seen:
                    continue
                seen.add(rid)
                out.append({"id": rid, "content": r["content"] or ""})
                if len(out) >= limit:
                    return True
            return False

        conn = self._connect()
        try:
            _push(conn.execute(
                "SELECT id, content FROM memories ORDER BY created_at DESC LIMIT ?",
                (recent_n,),
            ).fetchall())

            # Random across the whole table — this is what breaks the
            # recent-only surface trap. Over-fetch a touch to absorb dedup
            # collisions with the recent slice.
            if random_n > 0 and len(out) < limit:
                _push(conn.execute(
                    "SELECT id, content FROM memories ORDER BY RANDOM() LIMIT ?",
                    (random_n + max(0, recent_n // 4),),
                ).fetchall())

            # Lowest salience first, tie-break on stale last_accessed. These
            # are the candidates most likely to get pruned next NREM if they
            # don't get re-activated — give them a chance to be heard.
            if low_n > 0 and len(out) < limit:
                _push(conn.execute(
                    "SELECT id, content FROM memories "
                    "ORDER BY salience ASC, last_accessed ASC LIMIT ?",
                    (low_n + max(0, (recent_n + random_n) // 4),),
                ).fetchall())

            return out[:limit]
        finally:
            conn.close()

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT m.id, m.content,
                       (SELECT COUNT(*) FROM connections
                        WHERE source_id = m.id OR target_id = m.id) as cnt
                FROM memories m
                WHERE (SELECT COUNT(*) FROM connections
                       WHERE source_id = m.id OR target_id = m.id) < ?
                ORDER BY m.created_at DESC LIMIT ?
            """, (max_connections, limit)).fetchall()
            return [
                {"id": r["id"], "content": r["content"] or "", "connection_count": r["cnt"]}
                for r in rows
            ]
        finally:
            conn.close()

    def sample_isolated_for_dream(
        self,
        limit: int,
        max_connections: int = 3,
        recent_pct: float = 0.5,
        random_old_pct: float = 0.3,
        low_salience_pct: float = 0.2,
    ) -> List[Dict[str, Any]]:
        recent_n = max(1, int(limit * recent_pct))
        random_n = max(0, int(limit * random_old_pct))
        low_n = max(0, limit - recent_n - random_n)

        seen: Set[int] = set()
        out: List[Dict[str, Any]] = []

        def _push(rows):
            for r in rows:
                rid = r["id"]
                if rid in seen:
                    continue
                seen.add(rid)
                out.append({
                    "id": rid,
                    "content": r["content"] or "",
                    "connection_count": r["cnt"],
                })
                if len(out) >= limit:
                    return True
            return False

        # Subquery is identical across the three slices — only the ORDER BY
        # changes. Materialise the connection-count once via the correlated
        # subquery; SQLite is fine with it on tables of this size, and the
        # WHERE clause is the same across all three reads.
        base_select = (
            "SELECT m.id, m.content, "
            "(SELECT COUNT(*) FROM connections "
            " WHERE source_id = m.id OR target_id = m.id) AS cnt "
            "FROM memories m "
            "WHERE (SELECT COUNT(*) FROM connections "
            "       WHERE source_id = m.id OR target_id = m.id) < ? "
        )

        conn = self._connect()
        try:
            _push(conn.execute(
                base_select + "ORDER BY m.created_at DESC LIMIT ?",
                (max_connections, recent_n),
            ).fetchall())

            if random_n > 0 and len(out) < limit:
                _push(conn.execute(
                    base_select + "ORDER BY RANDOM() LIMIT ?",
                    (max_connections, random_n + max(0, recent_n // 4)),
                ).fetchall())

            if low_n > 0 and len(out) < limit:
                _push(conn.execute(
                    base_select + "ORDER BY m.salience ASC, m.last_accessed ASC LIMIT ?",
                    (max_connections, low_n + max(0, (recent_n + random_n) // 4)),
                ).fetchall())

            return out[:limit]
        finally:
            conn.close()

    def get_connections(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, target_id, weight, COALESCE(edge_type, 'similar') AS edge_type "
                "FROM connections WHERE weight >= 0.05"
            ).fetchall()
            return [
                {
                    "source_id": r["source_id"],
                    "target_id": r["target_id"],
                    "weight": r["weight"],
                    "edge_type": r["edge_type"],
                    "type": r["edge_type"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_memory_vectors(self, memory_ids: List[int]) -> Dict[int, List[float]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT id, embedding FROM memories WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
                tuple(memory_ids),
            ).fetchall()
            out: Dict[int, List[float]] = {}
            for r in rows:
                blob = r["embedding"]
                dim = len(blob) // 4 if blob else 0
                if dim:
                    out[r["id"]] = list(struct.unpack(f"{dim}f", blob))
            return out
        finally:
            conn.close()

    def set_connection_weight(self, source_id: int, target_id: int,
                              weight: float, reason: str = "semantic_reweight") -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, source_id, target_id, weight FROM connections "
                "WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
                (source_id, target_id, target_id, source_id),
            ).fetchone()
            if not row:
                return False
            new_weight = max(0.0, min(1.0, float(weight)))
            conn.execute("UPDATE connections SET weight = ? WHERE id = ?", (new_weight, row["id"]))
            conn.execute(
                "INSERT INTO connection_history "
                "(source_id, target_id, old_weight, new_weight, reason, changed_at, dream_session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (row["source_id"], row["target_id"], row["weight"], new_weight, reason, time.time()),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def add_typed_connection(self, source_id: int, target_id: int,
                             weight: float = 0.5,
                             edge_type: str = "similar") -> bool:
        if source_id == target_id:
            return False
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id, weight FROM connections WHERE source_id = ? AND target_id = ? AND COALESCE(edge_type, 'similar') = ?",
                (source_id, target_id, edge_type),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE connections SET weight = MAX(weight, ?) WHERE id = ?",
                    (weight, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at, event_time, ingestion_time, valid_from) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (source_id, target_id, weight, edge_type, time.time(), time.time(), time.time(), time.time()),
                )
            conn.commit()
            return True
        finally:
            conn.close()

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        """Bump an edge's weight by delta, capped at 1.0.

        Canonicalises (source < target) before the UPDATE so this works
        on the canonical rows produced by the iter-62 migration. Without
        the swap, a caller passing (max, min) would silently match no
        rows and the update would be a no-op — observed in practice
        when NREM activated_edges happened to be derived from a
        non-canonical source.
        """
        if source_id == target_id:
            return
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                      delta: float = 0.05,
                                      dream_session_id: Optional[int] = None) -> int:
        """Bulk strengthen via executemany. Returns count updated.

        Canonicalises every (src, tgt) pair before the UPDATE — same
        rationale as strengthen_connection above. NREM's activated_edges
        set already stores (min, max), so this is a no-op for the
        primary caller; the canonicalisation is here so any future
        caller that forgets to swap still hits the right rows.

        When dream_session_id is provided, also writes one
        connection_history row per edge tagged with that session — this
        is what makes NREM cycles visible in /dream/cycles/{id}/diff.
        """
        if not edges:
            return 0
        canon = []
        for src, tgt in edges:
            src, tgt = int(src), int(tgt)
            if src == tgt:
                continue
            if src > tgt:
                src, tgt = tgt, src
            canon.append((delta, src, tgt))
        if not canon:
            return 0
        conn = self._connect()
        try:
            conn.executemany(
                "UPDATE connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_id = ? AND target_id = ?",
                canon,
            )
            if dream_session_id is not None:
                # Snapshot the new weights so the history reflects post-update
                # values. We do a single SELECT per cycle to avoid N round-trips.
                pairs = [(s, t) for (_d, s, t) in canon]
                placeholders = ",".join("(?,?)" for _ in pairs)
                flat = [v for pair in pairs for v in pair]
                rows = conn.execute(
                    "SELECT source_id, target_id, weight FROM connections "
                    f"WHERE (source_id, target_id) IN ({placeholders})",
                    flat,
                ).fetchall()
                now = time.time()
                hist = [
                    (r["source_id"], r["target_id"],
                     max(0.0, r["weight"] - delta), r["weight"],
                     "nrem_strengthen", now, dream_session_id)
                    for r in rows
                ]
                conn.executemany(
                    "INSERT INTO connection_history "
                    "(source_id, target_id, old_weight, new_weight, reason, changed_at, dream_session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    hist,
                )
            conn.commit()
            return len(canon)
        finally:
            conn.close()

    def batch_weaken_connections(self, threshold: float = 0.05,
                                  delta: float = 0.01,
                                  dream_session_id: Optional[int] = None) -> int:
        """Bulk weaken all connections above threshold in one UPDATE.

        Writes ONE summary connection_history row per cycle when
        dream_session_id is given. We don't expand into per-edge rows here
        because batch_weaken can touch tens of thousands of edges per cycle
        — that would inflate connection_history without adding signal.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE weight > ?",
                (delta, threshold)
            )
            n = cursor.rowcount
            if n > 0 and dream_session_id is not None:
                conn.execute(
                    "INSERT INTO connection_history "
                    "(source_id, target_id, old_weight, new_weight, reason, changed_at, dream_session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (-1, -1, float(threshold), float(delta),
                     f"nrem_bulk_weaken:{n}", time.time(), dream_session_id),
                )
            conn.commit()
            return n
        finally:
            conn.close()

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> bool:
        """Insert a REM-discovered bridge edge. Returns True if newly inserted.

        Canonicalises source<target to match add_connection's invariant.
        Without this, the connections table held mixed-orientation rows:
        an edge added by remember()'s auto_connect path was always (min,max),
        but a bridge from REM could be (max,min). Any downstream code that
        assumed canonical form (\"WHERE source=? AND target=?\") would miss
        the bridge edge half the time.

        Returns False on self-loop or pre-existing edge so the caller can
        avoid writing a misleading connection_history row claiming the
        weight changed from 0.0.
        """
        if source_id == target_id:
            return False
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id FROM connections WHERE source_id = ? AND target_id = ?",
                (source_id, target_id),
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO connections (source_id, target_id, weight, edge_type, created_at) "
                "VALUES (?, ?, ?, 'bridge', ?)",
                (source_id, target_id, weight, time.time())
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def add_bridges_batch(
        self,
        bridges: List[Tuple[int, int, float]],
        dream_session_id: Optional[int] = None,
        reason: str = "rem_bridge",
    ) -> int:
        """Bulk-insert REM bridges in one transaction. Drops self-loops,
        canonicalises (source<target), de-duplicates within the batch and
        against the existing connections table, and INSERTs only the truly
        new edges via executemany. log_connection_change is bulk-inserted
        in the same transaction. One commit per cycle instead of one per
        bridge — typical REM phase: 6000 bridges goes from 12000 commits
        to 1."""
        if not bridges:
            return 0

        # Canonicalise + within-batch dedupe
        canon: List[Tuple[int, int, float]] = []
        seen: Set[Tuple[int, int]] = set()
        for s, t, w in bridges:
            if s == t:
                continue
            if s > t:
                s, t = t, s
            if (s, t) in seen:
                continue
            seen.add((s, t))
            canon.append((s, t, max(0.0, min(1.0, float(w)))))

        if not canon:
            return 0

        conn = self._connect()
        try:
            # Diff against existing edges via a temp staging table — handles
            # batches > 999 (SQLite's IN-clause parameter ceiling) and lets
            # us join directly against `connections` without chunking.
            conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _bridge_staging "
                "(source_id INTEGER, target_id INTEGER, weight REAL)"
            )
            conn.execute("DELETE FROM _bridge_staging")
            conn.executemany(
                "INSERT INTO _bridge_staging (source_id, target_id, weight) VALUES (?, ?, ?)",
                canon,
            )

            new_rows = conn.execute("""
                SELECT s.source_id, s.target_id, s.weight
                FROM _bridge_staging s
                WHERE NOT EXISTS (
                    SELECT 1 FROM connections c
                    WHERE c.source_id = s.source_id
                      AND c.target_id = s.target_id
                )
            """).fetchall()

            if not new_rows:
                conn.commit()
                return 0

            now = time.time()
            conn.executemany(
                "INSERT INTO connections "
                "(source_id, target_id, weight, edge_type, created_at) "
                "VALUES (?, ?, ?, 'bridge', ?)",
                [(int(r[0]), int(r[1]), float(r[2]), now) for r in new_rows],
            )
            if dream_session_id is not None:
                conn.executemany(
                    "INSERT INTO connection_history "
                    "(source_id, target_id, old_weight, new_weight, reason, "
                    "changed_at, dream_session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (int(r[0]), int(r[1]), 0.0, float(r[2]), reason, now, dream_session_id)
                        for r in new_rows
                    ],
                )
            conn.commit()
            return len(new_rows)
        finally:
            conn.close()

    def prune_weak(self, threshold: float = 0.05) -> int:
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM connections WHERE weight < ?",
                (threshold,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str,
                               dream_session_id: Optional[int] = None) -> None:
        # Defensive canonicalisation: every connections row is canonical
        # (source<target), so log rows must match or any join on
        # (source_id, target_id) loses half the history. Callers fixed in
        # iter-M already canonicalise; this guard makes the contract
        # function-level so a future caller can't silently regress.
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO connection_history "
                "(source_id, target_id, old_weight, new_weight, reason, changed_at, dream_session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (source_id, target_id, old_weight, new_weight, reason, time.time(), dream_session_id)
            )
            conn.commit()
        finally:
            conn.close()

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO dream_insights "
                "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, insight_type, source_memory_id, content, confidence, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def add_insights_batch(self, items) -> int:
        """Bulk-insert insights — one transaction, one fsync.

        Replaces N sequential ``add_insight`` calls (each its own commit)
        for the dream-cycle Insight phase, where 6000+ cluster + bridge
        insights would otherwise generate 6000 commits and stall the
        writer for tens of seconds. Returns the number of rows inserted.
        """
        rows = []
        now = time.time()
        for sid, ins_type, src, content, conf in items:
            rows.append((int(sid), str(ins_type), int(src), str(content),
                         float(conf), now))
        if not rows:
            return 0

        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO dream_insights "
                "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()

    def recent_cluster_anchors(self, window_seconds: float) -> Set[int]:
        cutoff = time.time() - float(window_seconds)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT source_memory_id FROM dream_insights "
                "WHERE insight_type = 'cluster' AND created_at >= ?",
                (cutoff,),
            ).fetchall()
            return {int(r[0]) for r in rows if r[0] is not None}
        finally:
            conn.close()

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Delete history entries older than keep_days."""
        conn = self._connect()
        try:
            cutoff = time.time() - (keep_days * 86400)
            count = conn.execute(
                "DELETE FROM connection_history WHERE changed_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Delete dream sessions older than keep_days and their associated insights.

        Uses correlated DELETE+subquery (no per-id parameter binding) so first-run
        cleanup of a large backlog can't trip SQLITE_MAX_VARIABLE_NUMBER. The
        previous implementation built `IN (?,?,?,...)` with one placeholder per
        old session — fine for the steady-state (3/cycle), broken if the user
        enabled cleanup after months of accumulation.
        """
        conn = self._connect()
        try:
            cutoff = time.time() - (keep_days * 86400)
            # Insights first — SQLite has no FK cascade in default schemas.
            conn.execute(
                "DELETE FROM dream_insights "
                "WHERE session_id IN (SELECT id FROM dream_sessions WHERE started_at < ?)",
                (cutoff,)
            )
            count = conn.execute(
                "DELETE FROM dream_sessions WHERE started_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories."""
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM connections "
                "WHERE source_id NOT IN (SELECT id FROM memories) "
                "OR target_id NOT IN (SELECT id FROM memories)"
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            s = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(memories_processed), 0), "
                "COALESCE(SUM(connections_strengthened), 0), "
                "COALESCE(SUM(connections_pruned), 0), "
                "COALESCE(SUM(bridges_found), 0), "
                "COALESCE(SUM(insights_created), 0) "
                "FROM dream_sessions"
            ).fetchone()

            insights = conn.execute(
                "SELECT insight_type, COUNT(*) FROM dream_insights GROUP BY insight_type"
            ).fetchall()

            return {
                "sessions": s[0] if s else 0,
                "total_processed": s[1] if s else 0,
                "total_strengthened": s[2] if s else 0,
                "total_pruned": s[3] if s else 0,
                "total_bridges": s[4] if s else 0,
                "total_insights": s[5] if s else 0,
                "insight_types": {r[0]: r[1] for r in insights},
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Dream Engine
# ---------------------------------------------------------------------------

class DreamEngine:
    """Autonomous background consolidation for mazemaker.

    Three phases:
      NREM  — Replay recent memories, strengthen active, prune dead
      REM   — Explore isolated memories, discover bridges via embedding
      Insight — Community detection, bridge identification, abstraction
    """

    def __init__(
        self,
        backend: DreamBackend,
        neural_memory: Optional[Any] = None,
        idle_threshold: float = 300.0,     # 5 min idle
        memory_threshold: int = 50,         # dream every N new memories
        max_memories_per_cycle: int = 2000,
        max_isolated_per_cycle: int = 800,
        sample_recent_pct: float = 0.5,
        sample_random_pct: float = 0.3,
        sample_low_salience_pct: float = 0.2,
    ):
        self._backend = backend
        self._memory = neural_memory        # Mazemaker instance for think/recall
        self._idle_threshold = idle_threshold
        self._memory_threshold = memory_threshold
        self._max_memories = max_memories_per_cycle
        self._max_isolated = max_isolated_per_cycle
        # NREM sampling mix — defaults give 50% recent, 30% random across
        # all memories, 20% lowest-salience. Recent-only would mean older
        # memories are never replayed and quietly decay below prune threshold.
        self._sample_recent_pct = sample_recent_pct
        self._sample_random_pct = sample_random_pct
        self._sample_low_salience_pct = sample_low_salience_pct

        self._stop_event = threading.Event()  # set = stop requested
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._memory_count_at_last_dream = 0
        self._dream_count = 0

    @classmethod
    def sqlite(cls, db_path: str, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with SQLite backend."""
        backend = SQLiteDreamBackend(db_path)
        return cls(backend, neural_memory, **kwargs)

    @classmethod
    def postgres(cls, dsn: Optional[str] = None,
                 neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with Postgres+pgvector backend."""
        from dream_postgres_store import DreamPostgresStore
        backend = DreamPostgresStore(dsn=dsn)
        return cls(backend, neural_memory, **kwargs)

    def start(self) -> None:
        """Start the background dream daemon."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._dream_loop, daemon=True, name="dream-engine"
        )
        self._thread.start()
        logger.info(
            "Dream engine started: idle=%ss, threshold=%d",
            self._idle_threshold, self._memory_threshold,
        )

    def stop(self) -> None:
        """Stop the dream daemon."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("Dream engine stopped after %d cycles", self._dream_count)

    def touch(self) -> None:
        """Signal activity — resets idle timer."""
        self._last_activity = time.time()

    def dream_now(self) -> Dict[str, Any]:
        """Force an immediate dream cycle. Returns stats."""
        return self._run_dream_cycle()

    # -- Main loop -----------------------------------------------------------

    def _dream_loop(self) -> None:
        """Background daemon: dream when idle or threshold reached."""
        while not self._stop_event.is_set():
            try:
                # Wait 30s but wake immediately if stop() is called
                if self._stop_event.wait(timeout=30.0):
                    break

                idle = time.time() - self._last_activity
                try:
                    stats = self._memory.stats() if self._memory else {"memories": 0}
                    total = stats.get("memories", 0)
                except Exception:
                    total = 0
                new_since_last = total - self._memory_count_at_last_dream

                should_dream = (
                    idle >= self._idle_threshold
                    or new_since_last >= self._memory_threshold
                )

                if should_dream:
                    logger.info(
                        "Dream cycle triggered: idle=%.0fs, new=%d",
                        idle, new_since_last,
                    )
                    self._run_dream_cycle()
                    # Reset idle timer so the next cycle doesn't re-trigger
                    # on the very next 30s wake.  Without this, every poll
                    # after the threshold is crossed sees idle >= threshold
                    # and fires another cycle, pegging CPU on spreading
                    # activation indefinitely.
                    self._last_activity = time.time()

            except Exception as e:
                logger.debug("Dream loop error: %s", e)
                if self._stop_event.wait(timeout=60):
                    break

    # -- Dream Cycle ---------------------------------------------------------

    def _run_dream_cycle(self) -> Dict[str, Any]:
        """Execute a full NREM → REM → Insight cycle."""
        with self._lock:
            start = time.time()
            total_stats: Dict[str, Any] = {"nrem": {}, "rem": {}, "insights": {}}

            try:
                total_stats["nrem"] = self._phase_nrem()
                # REM + Insight phases gate themselves — community
                # installs see {"skipped": "pro_feature"} from those
                # methods.  NREM consolidation always runs.
                total_stats["rem"] = self._phase_rem()
                total_stats["insights"] = self._phase_insights()

                self._dream_count += 1
                if self._memory:
                    try:
                        s = self._memory.stats()
                        self._memory_count_at_last_dream = s.get("memories", 0)
                    except Exception:
                        pass

                total_stats["duration"] = time.time() - start
                total_stats["dream_id"] = self._dream_count

                logger.info(
                    "Dream #%d complete: %.1fs | NREM: %d+/ %d- / %d pruned | REM: %d bridges | Insights: %d",
                    self._dream_count, total_stats["duration"],
                    total_stats["nrem"].get("strengthened", 0),
                    total_stats["nrem"].get("weakened", 0),
                    total_stats["nrem"].get("pruned", 0),
                    total_stats["rem"].get("bridges", 0),
                    total_stats["insights"].get("insights", 0),
                )

            except Exception as e:
                logger.error("Dream cycle failed: %s", e)
                total_stats["error"] = str(e)
            finally:
                # Reset the idle timer after every cycle, regardless of how
                # _run_dream_cycle was invoked (loop poll OR explicit dream_now()).
                # Without this in the finally, dream_now() leaves _last_activity
                # untouched, so the very next 30s loop wake sees idle >>
                # idle_threshold and immediately fires another cycle — pegging
                # CPU on spreading activation. The duplicate reset in the loop
                # caller is now redundant but harmless.
                self._last_activity = time.time()

            return total_stats

    # -- Phase 1: NREM -------------------------------------------------------

    def _phase_nrem(self) -> Dict[str, Any]:
        """NREM: Replay, strengthen active, weaken inactive, prune dead.

        For each recent memory:
          1. Fire spreading activation
          2. Activated edges: batch strengthen
          3. Non-activated edges: bulk weaken (single SQL UPDATE)
          4. Edges below threshold: prune
        """
        stats = {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}
        session_id = self._backend.start_session("nrem")

        try:
            memories = self._backend.sample_for_dream(
                self._max_memories,
                recent_pct=self._sample_recent_pct,
                random_old_pct=self._sample_random_pct,
                low_salience_pct=self._sample_low_salience_pct,
            )
            if not memories:
                return stats

            # GPU PPR adjacency cache management:
            # Per-cycle invalidation is wasteful — full rebuild is ~2s on a
            # 2M-nnz tensor and grows with the corpus. The cache is allowed
            # to live across cycles; it goes stale (REM-added bridges and
            # NREM strengthen/weaken/prune are not reflected) but PPR is
            # already an approximation of activation strength, so a
            # single-cycle lag is fine. Force a fresh rebuild every Nth
            # cycle so drift can't accumulate forever. _dream_count is
            # incremented in _run_dream_cycle, not here — read-only.
            ppr_rebuild_every = getattr(self, "_ppr_rebuild_every", 10)
            if (
                self._memory is not None
                and hasattr(self._memory, "_invalidate_gpu_ppr_adjacency")
                and self._dream_count > 0
                and self._dream_count % max(1, ppr_rebuild_every) == 0
            ):
                try:
                    self._memory._invalidate_gpu_ppr_adjacency()
                    logger.info(
                        "Dream cycle %d — invalidating GPU PPR adjacency for refresh",
                        self._dream_count,
                    )
                except Exception:
                    pass

            activated_edges: Set[Tuple[int, int]] = set()

            # Prefer think_ids (no SQL get_many, no result-dict build, top-k
            # on GPU) on memory backends that support it. Falls back to the
            # full-think path on free-tier / non-CUDA setups.
            use_think_ids = self._memory is not None and hasattr(self._memory, "think_ids")

            for mem in memories:
                mid = mem["id"]
                if self._memory:
                    try:
                        if use_think_ids:
                            activated_ids = self._memory.think_ids(mid, k=20)
                            for aid in activated_ids:
                                if aid != mid:
                                    activated_edges.add(
                                        (min(mid, aid), max(mid, aid))
                                    )
                        else:
                            activated = self._memory.think(mid, depth=2)
                            for a in activated:
                                aid = a.get("id")
                                if aid and aid != mid:
                                    activated_edges.add(
                                        (min(mid, aid), max(mid, aid))
                                    )
                    except Exception:
                        pass
                stats["processed"] += 1

            # Batch strengthen activated edges (usually small set)
            if activated_edges:
                stats["strengthened"] = self._backend.batch_strengthen_connections(
                    list(activated_edges), 0.05, dream_session_id=session_id,
                )

            # Bulk weaken ALL non-activated connections above threshold
            # Single SQL UPDATE instead of per-row loop
            stats["weakened"] = self._backend.batch_weaken_connections(
                threshold=0.05, delta=0.01, dream_session_id=session_id,
            )

            # Prune dead connections
            stats["pruned"] = self._backend.prune_weak(0.05)

            # Every NREM cycle: prune old history + orphans
            try:
                pruned_hist = self._backend.prune_connection_history(keep_days=7)
                if pruned_hist:
                    logger.info("Pruned %d old connection_history entries", pruned_hist)
                pruned_sessions = self._backend.prune_old_dream_sessions(keep_days=30)
                if pruned_sessions:
                    logger.info("Pruned %d old dream sessions", pruned_sessions)
                pruned_orphans = self._backend.prune_orphans()
                if pruned_orphans:
                    # Roll orphan deletes into the per-session counter so
                    # `dream_sessions.connections_pruned` accounts for every
                    # path that drops rows from `connections`. Previously
                    # orphans only surfaced as a log line, which made the
                    # cumulative `edges pruned` stat under-count whenever a
                    # batch of orphan rows showed up after a memory delete.
                    stats["pruned"] = int(stats.get("pruned", 0) or 0) + int(pruned_orphans)
                    logger.info("Pruned %d orphan connections", pruned_orphans)

                # Time-bounded derived:cluster prune. Insight emits up
                # to MAX_CLUSTERS_PER_CYCLE (50) cluster summaries per
                # cycle — ephemeral output, not durable knowledge. With
                # centroid drift the exact-content dedup misses, and
                # cluster identity (which members are grouped) shifts
                # cycle-to-cycle anyway. A short TTL keeps the table
                # bounded around recent cycles' output.
                #   50 clusters × 277 cycles/h × (TTL_h) = steady-state size
                # 5 min TTL → ~1 150 cluster summaries on average, fresh.
                pruned_clusters = self._prune_old_derived_clusters(
                    keep_seconds=5 * 60,
                )
                if pruned_clusters:
                    logger.info(
                        "Pruned %d aged derived:cluster memories (>5min)",
                        pruned_clusters,
                    )
            except Exception as e:
                logger.debug("Maintenance cleanup error: %s", e)

        except Exception as e:
            logger.debug("NREM phase error: %s", e)
        finally:
            try:
                self._backend.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("NREM finish_session failed: %s", e)

        return stats

    # -- Phase 2: REM --------------------------------------------------------

    def _phase_rem(self) -> Dict[str, Any]:
        """REM: Explore isolated memories, discover bridges.

        1. Find isolated memories (few connections)
        2. Search via embedding similarity for unconnected but similar
        3. Create tentative bridge connections (weight 0.1-0.3)

        Pro feature.  Community installs early-return without touching
        the dream-session table — they get NREM-only consolidation.
        """
        if not has_feature("rem"):
            logger.info(
                "REM phase skipped — Pro feature.  Engine running "
                "NREM-only consolidation.  See "
                "https://mazemaker.online/#pricing"
            )
            return {"skipped": "pro_feature", "explored": 0,
                    "bridges": 0, "rejected": 0}

        stats = {"explored": 0, "bridges": 0, "rejected": 0}
        session_id = self._backend.start_session("rem")

        try:
            isolated = self._backend.sample_isolated_for_dream(
                self._max_isolated,
                max_connections=3,
                recent_pct=self._sample_recent_pct,
                random_old_pct=self._sample_random_pct,
                low_salience_pct=self._sample_low_salience_pct,
            )
            if not isolated:
                return stats

            if not self._memory:
                return stats

            # Collect inputs for one batched recall — drops empty-content
            # rows up-front so the batch index aligns with the kept memories.
            kept = [m for m in isolated if m.get("content")]
            if not kept:
                return stats

            queries = [m["content"][:200] for m in kept]
            stats["explored"] = len(kept)

            # ── Batched semantic search ──────────────────────────────────
            # Single embed_batch IPC + single matmul (B, 1024) @ (1024, N)
            # on CUDA, vs N sequential embed-server round-trips. On a 800-
            # batch / 193k-corpus this collapses ~6min of REM Python loop
            # into a few hundred ms of GPU compute.
            try:
                if hasattr(self._memory, "recall_batch"):
                    similar_lists = self._memory.recall_batch(queries, k=10)
                else:
                    similar_lists = [
                        self._memory.recall(q, k=10) for q in queries
                    ]
            except Exception as e:
                logger.debug("REM batch recall failed: %s — falling back", e)
                similar_lists = [self._memory.recall(q, k=10) for q in queries]

            # ── Build the bridge candidate list ──────────────────────────
            # Filter follows the same rule as the per-memory loop:
            # 0.3 < similarity < 0.95, no self-loops. We accumulate first
            # so the SQL write becomes a single bulk transaction below.
            candidate_bridges: List[Tuple[int, int, float]] = []
            for mem, similar in zip(kept, similar_lists):
                mid = mem["id"]
                for sim in similar:
                    sim_id = sim.get("id")
                    sim_score = sim.get("similarity", 0.0)
                    if not sim_id or sim_id == mid:
                        continue
                    if sim_score < 0.3 or sim_score > 0.95:
                        continue
                    bridge_weight = round(float(sim_score) * 0.3, 3)
                    candidate_bridges.append((int(mid), int(sim_id), bridge_weight))

            # ── One transaction for the whole batch ──────────────────────
            # add_bridges_batch handles canonicalisation, dedup against
            # existing edges, and logs connection_history in the same
            # transaction. ~6000 individual commits → 1 commit per cycle.
            if candidate_bridges:
                try:
                    new_bridges = self._backend.add_bridges_batch(
                        candidate_bridges,
                        dream_session_id=session_id,
                        reason="rem_bridge",
                    )
                    stats["bridges"] = int(new_bridges or 0)
                    stats["rejected"] = len(candidate_bridges) - stats["bridges"]
                except Exception as e:
                    logger.debug(
                        "REM batch bridge insert failed: %s — falling back to per-edge",
                        e,
                    )
                    for s, t, w in candidate_bridges:
                        if self._backend.add_bridge(s, t, w):
                            src, tgt = (s, t) if s < t else (t, s)
                            self._backend.log_connection_change(
                                src, tgt, 0.0, w, "rem_bridge",
                                dream_session_id=session_id,
                            )
                            stats["bridges"] += 1

        except Exception as e:
            logger.debug("REM phase error: %s", e)
        finally:
            try:
                self._backend.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("REM finish_session failed: %s", e)

        return stats

    # -- Phase 3: Insights ---------------------------------------------------

    def _phase_insights(self) -> Dict[str, Any]:
        """Insight: Community detection, bridge identification, abstraction.

        Pipeline (timings emitted at INFO so we can spot regressions):
          1. Load full connections table (~1.16 M rows, ~0.6 s).
          2. Build adjacency + node-set in numpy (vectorised, ~0.3 s on 1 M).
          3. C++ Louvain modularity-optimisation (~1.3 s on 1 M edges).
          4. Vectorised bridge-node scan via per-edge community lookup.
          5. Emit cluster insights with capped theme-sample SQL.
          6. Emit bridge insights, all batched into one transaction.
          7. Flush all dream_insights rows via add_insights_batch (1 commit).
        """
        import numpy as np
        if not has_feature("insight"):
            logger.info(
                "Insight phase skipped — Pro feature.  No derived:* "
                "cluster memories will be crystallised this cycle.  "
                "See https://mazemaker.online/#pricing"
            )
            return {"skipped": "pro_feature", "communities": 0,
                    "bridges": 0, "insights": 0, "derived_facts": 0}
        stats = {"communities": 0, "bridges": 0, "insights": 0, "derived_facts": 0}
        session_id = self._backend.start_session("insight")
        t_phase = time.perf_counter()

        try:
            t0 = time.perf_counter()
            edges = self._backend.get_connections()
            if not edges:
                return stats
            # Exclude derived_from edges. Those are synthetic links from
            # `derived:cluster` summary memories to their member nodes —
            # output of the previous cycle's Insight. Letting them into
            # Louvain creates a feedback loop: the previous cycle's
            # clusters become nodes themselves, the new cycle clusters
            # those clusters, the graph fragments into thousands of
            # tiny meta-components, and derived:cluster blows up by the
            # ~256 ratio per cycle. The signal-bearing edge types are
            # similar / bridge / causal — those go in.
            edges = [e for e in edges if (e.get("edge_type") or "similar") != "derived_from"]
            if not edges:
                return stats
            t_load = time.perf_counter() - t0

            # ── Vectorised adjacency build ────────────────────────────────
            # Pre-extract three flat int/float arrays so the rest of the
            # pipeline can index without re-walking dicts. NumPy's C loops
            # are 10-30x faster than the Python list-of-tuples adjacency
            # the original code built for 1 M edges.
            t0 = time.perf_counter()
            src_arr = np.fromiter((e["source_id"] for e in edges),
                                   dtype=np.int64, count=len(edges))
            dst_arr = np.fromiter((e["target_id"] for e in edges),
                                   dtype=np.int64, count=len(edges))
            w_arr   = np.fromiter((e.get("weight", 0.0) or 0.0 for e in edges),
                                   dtype=np.float32, count=len(edges))
            unique_nodes = np.unique(np.concatenate([src_arr, dst_arr]))
            nodes = set(unique_nodes.tolist())

            # Lazy adjacency for the bridge-community count below — only
            # needed for nodes that actually become bridges, so build per
            # endpoint after we know which nodes are bridges (cheaper than
            # a full O(E) Python adjacency dict for 1 M edges).
            t_adj = time.perf_counter() - t0

            # ── C++ Louvain ───────────────────────────────────────────────
            t0 = time.perf_counter()
            communities = self._detect_communities(edges, nodes, None)
            stats["communities"] = len(communities)
            t_louvain = time.perf_counter() - t0

            # node_to_comm[node_id] = community_index. Vectorise lookup:
            # build a parallel array indexed by position in unique_nodes.
            t0 = time.perf_counter()
            node_to_comm: Dict[int, int] = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i

            # ── Vectorised bridge-node scan ──────────────────────────────
            # Map every src/dst to its community in one pass over the edge
            # arrays, then mark endpoints whose communities differ.
            n_edges = src_arr.shape[0]
            src_comm = np.fromiter(
                (node_to_comm.get(int(s), -1) for s in src_arr),
                dtype=np.int32, count=n_edges,
            )
            dst_comm = np.fromiter(
                (node_to_comm.get(int(d), -1) for d in dst_arr),
                dtype=np.int32, count=n_edges,
            )
            cross = (src_comm != dst_comm) & (src_comm >= 0) & (dst_comm >= 0)
            bridge_nodes_arr = np.unique(
                np.concatenate([src_arr[cross], dst_arr[cross]])
            )
            bridge_nodes = bridge_nodes_arr.tolist()
            stats["bridges"] = len(bridge_nodes)
            t_bridges = time.perf_counter() - t0

            # ── Stage cluster + bridge insights into one buffer ─────────
            # Single dream_insights commit at the end via add_insights_batch.
            #
            # Cluster-emission budget. Two thresholds + a hard cap so we
            # never write more than ~50 derived:cluster memories per cycle:
            #   - MIN_CLUSTER_SIZE: clusters smaller than this are noise.
            #     A 195 k-memory corpus produces thousands of 3-5 member
            #     fragments when the graph is sparse; emitting all of them
            #     wallpapers the writer for 30+ s and starves recall.
            #   - MAX_CLUSTERS_PER_CYCLE: even with the size threshold the
            #     biggest N clusters carry the meaningful structure. Skip
            #     the tail.
            # Communities arrive sorted DESC by size (see _detect_communities)
            # so taking the prefix == taking the largest.
            MIN_CLUSTER_SIZE = 10
            MAX_CLUSTERS_PER_CYCLE = 50
            # Rotation window: skip communities whose anchor (lowest member
            # id, deterministic) has been emitted within this window.  6h
            # is large enough that the same Top-50 mega-clusters don't
            # re-emit every cycle, small enough that the corpus does
            # eventually cycle back through (after drift) for refresh.
            ANCHOR_REEMIT_WINDOW_S = 6 * 3600

            recent_anchors = self.backend.recent_cluster_anchors(
                ANCHOR_REEMIT_WINDOW_S
            )

            t0 = time.perf_counter()
            insight_rows: List[Tuple[int, str, int, str, float]] = []
            emitted = 0
            skipped_recent = 0

            for comm in communities:
                if len(comm) < MIN_CLUSTER_SIZE:
                    continue
                if emitted >= MAX_CLUSTERS_PER_CYCLE:
                    break
                # Stable anchor: lowest member id.  `comm[0]` is whatever
                # order the community detector returned, which drifts
                # cycle-to-cycle on the same underlying community; min(id)
                # is deterministic given the same membership set, so the
                # rotation window actually catches duplicates.
                anchor = min(int(x) for x in comm)
                if anchor in recent_anchors:
                    skipped_recent += 1
                    continue
                emitted += 1
                # _extract_theme caps its own SQL fan-out — see method.
                # Returns a real excerpt from the cluster's centroid member,
                # so the resulting derived:cluster memory's embedding
                # actually represents the cluster's semantic content (was
                # previously a keyword-frequency soup that embedded poorly).
                theme = self._extract_theme(comm)
                confidence = min(len(comm) / 10.0, 1.0)
                # Excerpt-first content layout: the theme leads, the
                # `[cluster:N]` suffix is purely informational and won't
                # dominate the embedding. A recall query like "trailer
                # status" now hits the cluster memory directly because
                # its embedding reflects the excerpt content.
                content = f"{theme}\n[cluster:{len(comm)}m]"
                insight_rows.append(
                    (session_id, "cluster", anchor, content, confidence)
                )
                stats["insights"] += 1
                if self._write_derived_cluster_memory(comm, content, confidence) is not None:
                    stats["derived_facts"] += 1
            stats["clusters_skipped_recent"] = skipped_recent
            t_clusters = time.perf_counter() - t0

            # Build the per-bridge community count without a full adjacency
            # dict: scan the cross-community edges once, group by endpoint,
            # and tally distinct neighbour-community counts via dict-of-sets
            # only for the bridge endpoints. ~10-50x cheaper than walking
            # the full Python adjacency for every bridge node.
            t0 = time.perf_counter()
            bridge_set = set(bridge_nodes)
            bridge_neighbors_comm: Dict[int, set] = defaultdict(set)
            cross_idx = np.flatnonzero(cross)
            for idx in cross_idx:
                s = int(src_arr[idx]); d = int(dst_arr[idx])
                sc = int(src_comm[idx]); dc = int(dst_comm[idx])
                if s in bridge_set:
                    bridge_neighbors_comm[s].add(dc)
                if d in bridge_set:
                    bridge_neighbors_comm[d].add(sc)

            for bnode in bridge_nodes:
                bcs = bridge_neighbors_comm.get(bnode)
                if bcs is None or len(bcs) < 2:
                    continue
                content = (
                    f"Bridge connecting {len(bcs)} communities, "
                    f"memory #{bnode}"
                )
                insight_rows.append(
                    (session_id, "bridge", int(bnode), content, 0.8)
                )
                stats["insights"] += 1
            t_bridge_scan = time.perf_counter() - t0

            # ── One commit for every dream_insights row of this cycle ───
            t0 = time.perf_counter()
            if insight_rows:
                if hasattr(self._backend, "add_insights_batch"):
                    self._backend.add_insights_batch(insight_rows)
                else:
                    for sid, ity, src, content, conf in insight_rows:
                        self._backend.add_insight(sid, ity, src, content, conf)
            t_flush = time.perf_counter() - t0

            logger.info(
                "Insight cycle: communities=%d bridges=%d insights=%d derived=%d "
                "[load=%.2fs adj=%.2fs louvain=%.2fs xcomm=%.2fs clusters=%.2fs "
                "bridge_scan=%.2fs flush=%.2fs total=%.2fs]",
                stats["communities"], stats["bridges"], stats["insights"],
                stats["derived_facts"],
                t_load, t_adj, t_louvain, t_bridges, t_clusters,
                t_bridge_scan, t_flush,
                time.perf_counter() - t_phase,
            )

        except Exception as e:
            logger.debug("Insight phase error: %s", e, exc_info=True)
        finally:
            try:
                self._backend.finish_session(session_id, stats)
            except Exception as e:
                logger.debug("Insight finish_session failed: %s", e)

        return stats

    def _detect_communities(self, edges: List[Dict[str, Any]], nodes: set,
                            adj: Dict[int, List[Tuple[int, float]]]) -> List[List[int]]:
        """Louvain community detection with three-tier fallback.

        Tier 1: C++ libmazemaker.so Louvain (cpp_bridge.detect_communities).
                Sub-second on 1M-edge graphs, modularity-optimal partition,
                C++26 implementation — preferred path on every cycle.
        Tier 2: NetworkX pure-Python Louvain. Used when the C++ shared lib
                isn't built or fails to load. Reasonable for <100k edges,
                stretches into minutes above that.
        Tier 3: BFS connected components (O(V+E)). Last-resort coarse
                partition that always finishes in ~1s.
        """
        # ── Tier 1: C++ Louvain ────────────────────────────────────────
        try:
            from cpp_bridge import detect_communities as _cpp_louvain
        except Exception:
            _cpp_louvain = None
        if _cpp_louvain is not None:
            try:
                cpp_edges = (
                    (e["source_id"], e["target_id"], float(e.get("weight", 0.0) or 0.0))
                    for e in edges
                )
                res = _cpp_louvain(cpp_edges, seed=42)
                logger.debug(
                    "C++ Louvain: %d edges → %d communities, Q=%.4f, %d iters, %.1f ms",
                    len(edges), len(res["communities"]), res["modularity"],
                    res["iterations"], res["elapsed_ms"],
                )
                if res["communities"]:
                    out = [sorted(int(n) for n in c) for c in res["communities"]]
                    out.sort(key=lambda c: (-len(c), c[0]))
                    return out
            except Exception as exc:
                logger.debug("C++ Louvain failed (%s); falling back to NetworkX/BFS", exc)

        # ── Tier 2: NetworkX Louvain (capped at 100k edges) ─────────────
        LOUVAIN_EDGE_LIMIT = 100_000
        if HAS_NETWORKX and len(edges) <= LOUVAIN_EDGE_LIMIT:
            try:
                graph = nx.Graph()
                graph.add_nodes_from(nodes)
                for e in edges:
                    graph.add_edge(e["source_id"], e["target_id"], weight=float(e.get("weight", 0.0) or 0.0))
                if hasattr(nx.algorithms.community, "louvain_communities"):
                    comms = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=42)
                    out = [sorted(int(n) for n in comm) for comm in comms if comm]
                    if out:
                        out.sort(key=lambda c: (-len(c), c[0]))
                        return out
            except Exception:
                pass

        # collections.deque for O(1) popleft instead of list.pop(0) which
        # is O(N) — the latter turns this BFS into O(V*E) on dense graphs
        # and was making Insight phase stretch into the minutes once the
        # graph crossed the LOUVAIN_EDGE_LIMIT threshold.
        # _phase_insights now passes adj=None and builds it lazily here so
        # the hot path (C++ Louvain) avoids the full O(E) Python adjacency
        # build entirely; only the cold-fallback BFS pays the cost.
        if adj is None:
            adj = defaultdict(list)
            for e in edges:
                s, t = e["source_id"], e["target_id"]
                w = float(e.get("weight", 0.0) or 0.0)
                adj[s].append((t, w))
                adj[t].append((s, w))
        from collections import deque
        visited = set()
        communities: List[List[int]] = []
        for node in nodes:
            if node in visited:
                continue
            component = []
            queue = deque([node])
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                component.append(curr)
                for neighbor, _ in adj.get(curr, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            communities.append(component)
        return communities

    def _write_derived_cluster_memory(self, comm: List[int], content: str, confidence: float) -> int | None:
        """Materialize a dream insight as a first-class derived memory.

        Dedup strategy:
        1) Reuse existing identical derived:cluster content when present.
        2) Otherwise create a new memory entry (with conflict detection enabled).
        """
        if not self._memory:
            return None

        store = getattr(self._memory, "store", None)
        if store is None and hasattr(self._memory, "_sqlite_memory"):
            store = self._memory._sqlite_memory.store

        derived_id: Optional[int] = None

        # Reuse exact duplicate first to prevent unbounded growth.
        if store is not None:
            try:
                lock = getattr(store, "_lock", None)
                if lock is not None:
                    with lock:
                        row = store.conn.execute(
                            "SELECT id FROM memories WHERE label = ? AND content = ? ORDER BY id DESC LIMIT 1",
                            ("derived:cluster", content),
                        ).fetchone()
                else:
                    row = store.conn.execute(
                        "SELECT id FROM memories WHERE label = ? AND content = ? ORDER BY id DESC LIMIT 1",
                        ("derived:cluster", content),
                    ).fetchone()

                if row is not None:
                    derived_id = int(row["id"] if isinstance(row, sqlite3.Row) else row[0])
                    try:
                        store.touch(derived_id)
                    except Exception:
                        pass
            except Exception:
                pass

        # No exact duplicate found: create one.
        #
        # detect_conflicts=False is deliberate: derived:cluster content has
        # a stable templated form ("Cluster of N related memories: theme")
        # that won't semantically collide with real memories, and the
        # exact-content reuse above already absorbs duplicates from prior
        # cycles. Conflict detection runs a full semantic recall against
        # the 193k-corpus per derived memory — that was ~500 ms per cluster
        # and dominated the Insight phase (cluster step at 138 s for 269
        # communities).
        if derived_id is None:
            try:
                created = self._memory.remember(
                    content,
                    label="derived:cluster",
                    auto_connect=False,
                    detect_conflicts=False,
                )
                if isinstance(created, list):
                    created = created[0]
                derived_id = int(created)
            except Exception:
                return None

        # Ensure derived_from links to source memories exist.
        #
        # Cap members per cluster: a single Louvain community can hold tens
        # of thousands of memories on a well-connected corpus, and writing
        # 20k+ derived_from edges per cycle drowns the SQLite writer for
        # minutes (each add_connection used to commit on its own — fsync
        # per row). The cap keeps the link skeleton useful for navigation
        # without turning Insight into a write-storm.
        DERIVED_LINK_CAP = 200
        try:
            if store is None:
                store = getattr(self._memory, "store", None)
                if store is None and hasattr(self._memory, "_sqlite_memory"):
                    store = self._memory._sqlite_memory.store
            if store is not None:
                link_weight = max(0.35, min(0.95, confidence))
                d_id = int(derived_id)
                members = comm[:DERIVED_LINK_CAP]
                pairs = [
                    (d_id, int(sid), link_weight)
                    for sid in members
                    if int(sid) != d_id
                ]
                if pairs and hasattr(store, "add_connections_batch"):
                    store.add_connections_batch(pairs, edge_type="derived_from")
                else:
                    # Fallback for backends without batch support.
                    for d, s, w in pairs:
                        store.add_connection(d, s, w, edge_type="derived_from")
        except Exception:
            pass

        return int(derived_id)

    # -- Helpers -------------------------------------------------------------

    def _prune_old_derived_clusters(self, keep_seconds: int = 24 * 3600) -> int:
        """Delete derived:cluster memories older than keep_seconds.

        Insight emits these as cycle-output cluster summaries; they're
        not curated knowledge. Without a TTL the synthetic-cluster table
        grows ~17 k rows per hour because centroid-driven content
        differs between cycles and exact-content dedup misses. The
        derived_from edges to source members are also dropped (cascade
        through prune_orphans on the next NREM pass), so the cluster
        skeleton rebuilds from scratch each day from current data —
        which is the whole point of cycling.
        """
        store = getattr(self._memory, "store", None) if self._memory else None
        if store is None and self._memory is not None and hasattr(self._memory, "_sqlite_memory"):
            store = getattr(self._memory._sqlite_memory, "store", None)
        if store is None:
            return 0
        cutoff = time.time() - keep_seconds
        try:
            with store._lock:
                cur = store.conn.execute(
                    "DELETE FROM memories "
                    "WHERE label = 'derived:cluster' AND created_at < ?",
                    (cutoff,),
                )
                n = int(cur.rowcount or 0)
                store.conn.commit()
                return n
        except Exception as exc:
            logger.debug("derived:cluster prune failed: %s", exc)
            return 0

    def _extract_theme(self, node_ids: List[int]) -> str:
        """Pick a representative excerpt from the cluster's most-central member.

        Replaces the old keyword-frequency soup (which produced unreadable
        word lists like "trailer, status, v13, scenes, audit") with
        content-aware selection:

        1. Sample up to 256 members. Cluster cohesion plateaus quickly;
           running on a 20 000-member community would burn compute since
           the central tendency converges within ~100 docs.
        2. Fetch contents + embeddings via store.get_many. Compute the
           cluster centroid (mean of L2-normalised embeddings), then the
           cosine similarity of every sampled member to the centroid.
           argmax = the most central member.
        3. Strip template lead-ins (session/role/turn headers + the
           `=== USER ===` / `=== ASSISTANT ===` conversation markers,
           skipping the user-question section because that's the echo
           noise we just filtered out of recall). Return the first ~240
           chars of the surviving signal.

        The derived:cluster memory's embedding now reflects the semantic
        mass of the cluster, so a recall query like "trailer status"
        matches the cluster memory directly — replacing the role that the
        now-filtered `auto:turn:*` echoes used to play.
        """
        THEME_SAMPLE = 256
        sample_ids = node_ids if len(node_ids) <= THEME_SAMPLE else node_ids[:THEME_SAMPLE]

        # Resolve the underlying SQLiteStore. The daemon passes a Memory
        # wrapper (mazemaker.py), the in-pod plugin passes a Mazemaker
        # engine directly (memory_client.py). Same dual-shape resolution
        # pattern that _write_derived_cluster_memory already uses.
        store = getattr(self._memory, "store", None) if self._memory else None
        if store is None and self._memory is not None and hasattr(self._memory, "_sqlite_memory"):
            store = getattr(self._memory._sqlite_memory, "store", None)
        if store is None:
            return f"{len(node_ids)} memories"

        try:
            mems = store.get_many(sample_ids, include_embedding=True)
        except Exception:
            mems = {}

        # Reject labels that are pure user-question echoes — they have no
        # answer content (just "whats the status of the trailer?" etc.)
        # and otherwise dominate the centroid for clusters that formed
        # around repeated questions. Also reject other derived:cluster
        # memories so we don't recurse on prior cycles' synthetic outputs;
        # we want the original signal at the leaves of the graph.
        # Paragraph-aware user-echo filter — same logic as Mazemaker.recall,
        # kept synced. User-question echoes carry a `:u` segment either as
        # the last token (`auto:hermes:<sid>:t<N>:u`) or directly before a
        # paragraph token when chunked (`...:t<N>:u:p<M>`). Also reject
        # `:t` and `:t:p<M>` segments — those are tool-output dumps which
        # are equally noisy when used as cluster representatives.
        import re as _re
        _PARA_RE = _re.compile(r"^p\d+$")

        def _has_role_segment(label: str, role: str) -> bool:
            parts = label.split(":")
            for i, seg in enumerate(parts):
                if seg != role:
                    continue
                if i == len(parts) - 1:
                    return True
                if i == len(parts) - 2 and _PARA_RE.match(parts[i + 1]):
                    return True
            return False

        def _is_signal_candidate(m: dict) -> bool:
            if not (m.get("content") and m.get("embedding")):
                return False
            label = m.get("label") or ""
            if _has_role_segment(label, "u"):
                return False
            if _has_role_segment(label, "t"):
                return False
            if label == "derived:cluster" or label.startswith("derived:cluster"):
                return False
            return True
        candidates = [m for m in mems.values() if _is_signal_candidate(m)]
        # If everything got filtered (e.g. cluster is purely echoes +
        # prior derived clusters), fall back to the unfiltered pool so
        # we still produce a non-trivial theme rather than the count.
        if not candidates:
            candidates = [m for m in mems.values() if m.get("content") and m.get("embedding")]
        if not candidates:
            return f"{len(node_ids)} memories"

        # Centroid pick. Falls back to first candidate on numerical edge
        # cases (zero-norm centroid means embeddings cancelled — should
        # never happen with non-trivial clusters but guard anyway).
        best_idx = 0
        try:
            import numpy as np
            embeds = np.asarray(
                [m["embedding"] for m in candidates], dtype=np.float32
            )
            norms = np.linalg.norm(embeds, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            embeds_n = embeds / norms
            centroid = embeds_n.mean(axis=0)
            c_norm = float(np.linalg.norm(centroid))
            if c_norm >= 1e-9:
                sims = embeds_n @ (centroid / c_norm)
                best_idx = int(np.argmax(sims))
        except Exception:
            pass

        raw = candidates[best_idx].get("content") or ""
        excerpt = self._strip_template_leadins(raw)
        if not excerpt:
            return f"{len(node_ids)} memories"

        LIMIT = 240
        if len(excerpt) > LIMIT:
            excerpt = excerpt[:LIMIT].rsplit(" ", 1)[0] + "…"
        return excerpt

    def _strip_template_leadins(self, content: str) -> str:
        """Remove session/turn framing from auto-saved conversation content.

        Auto-saved memories begin with template headers
        (``session:...``, ``role: user``, ``turn: 4``, ``paragraph: 1/2``)
        and conversation markers (``=== USER ===``, ``=== ASSISTANT ===``).
        Those are noise for theme extraction. We drop them, and when both
        sides of a conversation are present we keep only the assistant
        side — the user side is the question, which is the same kind of
        echo we filter out of recall.
        """
        if not isinstance(content, str):
            return ""
        skip_prefixes = (
            "session:", "role:", "turn:", "parent_id:",
            "parent_label:", "paragraph:",
        )
        out_lines: List[str] = []
        suppressing_user_section = False
        budget = 600
        for line in content.split("\n"):
            s = line.strip()
            if not s:
                continue
            if s.startswith(skip_prefixes):
                continue
            if s == "=== USER ===":
                suppressing_user_section = True
                continue
            if s == "=== ASSISTANT ===" or s == "=== SYSTEM ===":
                suppressing_user_section = False
                continue
            if suppressing_user_section:
                continue
            out_lines.append(s)
            budget -= len(s) + 1
            if budget <= 0:
                break
        return " ".join(out_lines).strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get dream engine statistics."""
        base = self._backend.get_dream_stats()
        base["engine_running"] = not self._stop_event.is_set() and (self._thread is not None and self._thread.is_alive())
        base["dream_cycles"] = self._dream_count
        return base
