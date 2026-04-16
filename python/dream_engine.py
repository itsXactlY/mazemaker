"""Dream Engine — autonomous background memory consolidation.

Implements three phases inspired by biological sleep:
  1. NREM — Replay & consolidation (strengthen, prune)
  2. REM  — Exploration & bridge discovery
  3. Insight — Abstraction & community detection

Runs as a background daemon during idle periods. Stores results
in the same SQLite DB as the main neural memory, extended with
dream-specific tables.

MSSQL support: if mssql_store is configured, dreams run against
the shared MSSQL backend (sneaky multi-agent consolidation).
Otherwise falls back to SQLite.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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
    changed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dream_insights_type
    ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session
    ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_nodes
    ON connection_history(source_id, target_id);
"""


# ---------------------------------------------------------------------------
# Abstract Dream Backend
# ---------------------------------------------------------------------------

class DreamBackend:
    """Interface for dream storage backends (SQLite or MSSQL)."""

    def start_session(self, phase: str) -> int:
        raise NotImplementedError

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_random_memories(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get a random sample of memories from the full history."""
        raise NotImplementedError

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50,
                               oldest_first: bool = False) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_connections(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        raise NotImplementedError

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        raise NotImplementedError

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        raise NotImplementedError

    def prune_weak(self, threshold: float = 0.05) -> int:
        raise NotImplementedError

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        raise NotImplementedError

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        raise NotImplementedError

    def get_dream_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SQLite Dream Backend
# ---------------------------------------------------------------------------

class SQLiteDreamBackend(DreamBackend):
    """Dream backend using the existing neural memory SQLite DB."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_DREAM_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
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

    def get_random_memories(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get a random sample of memories from the full history.

        Uses SQLite's random() for unbiased sampling across all eras.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, content FROM memories "
                "ORDER BY RANDOM() LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"id": r["id"], "content": r["content"] or ""} for r in rows]
        finally:
            conn.close()

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50,
                               oldest_first: bool = False) -> List[Dict[str, Any]]:
        order = "ASC" if oldest_first else "DESC"
        conn = self._connect()
        try:
            rows = conn.execute(f"""
                SELECT m.id, m.content,
                       (SELECT COUNT(*) FROM connections
                        WHERE source_id = m.id OR target_id = m.id) as cnt
                FROM memories m
                WHERE (SELECT COUNT(*) FROM connections
                       WHERE source_id = m.id OR target_id = m.id) < ?
                ORDER BY m.created_at {order} LIMIT ?
            """, (max_connections, limit)).fetchall()
            return [
                {"id": r["id"], "content": r["content"] or "", "connection_count": r["cnt"]}
                for r in rows
            ]
        finally:
            conn.close()

    def get_connections(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, target_id, weight FROM connections "
                "WHERE weight >= 0.05"
            ).fetchall()
            return [
                {"source_id": r["source_id"], "target_id": r["target_id"], "weight": r["weight"]}
                for r in rows
            ]
        finally:
            conn.close()

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
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

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id FROM connections "
                "WHERE (source_id = ? AND target_id = ?) "
                "OR (source_id = ? AND target_id = ?)",
                (source_id, target_id, target_id, source_id)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO connections (source_id, target_id, weight, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (source_id, target_id, weight, time.time())
                )
                conn.commit()
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
                               reason: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO connection_history "
                "(source_id, target_id, old_weight, new_weight, reason, changed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, target_id, old_weight, new_weight, reason, time.time())
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
    """Autonomous background consolidation for neural memory.

    Three phases:
      NREM  — Replay recent memories, strengthen active, prune dead
      REM   — Explore isolated memories, discover bridges via embedding
      Insight — Community detection, bridge identification, abstraction

    Deep dream: every DEEP_DREAM_INTERVAL cycles, runs a deeper
    consolidation pass with expanded memory window and full-history
    random sampling to prevent long-term memory erosion.
    """

    DEEP_DREAM_INTERVAL = 10  # every Nth cycle is a deep dream
    HISTORY_SAMPLE_RATIO = 0.3  # 30% of NREM budget from random history
    STRONG_EDGE_THRESHOLD = 0.4  # edges above this resist decay
    STRONG_EDGE_DECAY_FACTOR = 0.2  # strong edges decay 5x slower

    def __init__(
        self,
        backend: DreamBackend,
        neural_memory: Optional[Any] = None,
        idle_threshold: float = 600.0,     # 10 min idle (was 5min — too aggressive)
        memory_threshold: int = 50,         # dream every N new memories
        max_memories_per_cycle: int = 100,
        min_dream_interval: float = 600.0,  # 10 min cooldown between cycles
    ):
        self._backend = backend
        self._memory = neural_memory        # NeuralMemory instance for think/recall
        self._idle_threshold = idle_threshold
        self._memory_threshold = memory_threshold
        self._max_memories = max_memories_per_cycle
        self._min_dream_interval = min_dream_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._last_dream_time = 0.0         # cooldown tracking
        self._memory_count_at_last_dream = 0
        self._dream_count = 0
        self._bridge_node_ids: set = set()  # populated by insight phase

    @classmethod
    def sqlite(cls, db_path: str, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with SQLite backend."""
        backend = SQLiteDreamBackend(db_path)
        return cls(backend, neural_memory, **kwargs)

    @classmethod
    def mssql(cls, mssql_config: dict, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with MSSQL backend."""
        from dream_mssql_store import DreamMSSQLStore
        backend = DreamMSSQLStore.from_config(mssql_config)
        return cls(backend, neural_memory, **kwargs)

    def start(self) -> None:
        """Start the background dream daemon."""
        if self._running:
            return
        self._running = True
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
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Dream engine stopped after %d cycles", self._dream_count)

    def touch(self) -> None:
        """Signal activity — resets idle timer."""
        self._last_activity = time.time()

    def dream_now(self) -> Dict[str, Any]:
        """Force an immediate dream cycle. Returns stats."""
        return self._run_dream_cycle()

    # -- Main loop -----------------------------------------------------------

    def _dream_loop(self) -> None:
        """Background daemon: dream when idle or threshold reached.

        Cooldown prevents rapid re-triggering — at least min_dream_interval
        seconds must pass between cycles, even if idle.
        """
        while self._running:
            try:
                time.sleep(30)
                if not self._running:
                    break

                idle = time.time() - self._last_activity
                since_last_dream = time.time() - self._last_dream_time

                # Cooldown: don't dream again too soon
                if since_last_dream < self._min_dream_interval:
                    continue

                try:
                    stats = self._memory.stats() if self._memory else {"memories": 0}
                    total = stats.get("memories", 0)
                except Exception:
                    total = 0
                new_since_last = total - self._memory_count_at_last_dream

                should_dream = (
                    idle >= self._idle_threshold
                    and new_since_last > 0  # Only if there are new memories
                ) or new_since_last >= self._memory_threshold

                if should_dream:
                    logger.info(
                        "Dream cycle triggered: idle=%.0fs, new=%d",
                        idle, new_since_last,
                    )
                    self._run_dream_cycle()

            except Exception as e:
                logger.debug("Dream loop error: %s", e)
                time.sleep(60)

    # -- Dream Cycle ---------------------------------------------------------

    def _is_deep_dream(self) -> bool:
        """Check if the current cycle should be a deep dream."""
        return (self._dream_count + 1) % self.DEEP_DREAM_INTERVAL == 0

    def _run_dream_cycle(self) -> Dict[str, Any]:
        """Execute a full NREM → REM → Insight cycle.

        Every DEEP_DREAM_INTERVAL cycles, runs a deep dream with:
        - 3x memory window for NREM
        - All-history random sampling (no recency bias)
        - Oldest-first REM exploration
        """
        with self._lock:
            start = time.time()
            is_deep = self._is_deep_dream()
            total_stats: Dict[str, Any] = {
                "nrem": {}, "rem": {}, "insights": {},
                "deep_dream": is_deep,
            }

            try:
                total_stats["nrem"] = self._phase_nrem(deep=is_deep)
                total_stats["rem"] = self._phase_rem(deep=is_deep)
                total_stats["insights"] = self._phase_insights()

                self._dream_count += 1
                self._last_dream_time = time.time()  # cooldown tracking
                if self._memory:
                    try:
                        s = self._memory.stats()
                        self._memory_count_at_last_dream = s.get("memories", 0)
                    except Exception:
                        pass

                total_stats["duration"] = time.time() - start
                total_stats["dream_id"] = self._dream_count

                mode = "DEEP " if is_deep else ""
                logger.info(
                    "%sDream #%d complete: %.1fs | NREM: %d+/ %d- / %d pruned / %d protected"
                    " | REM: %d bridges | Insights: %d",
                    mode, self._dream_count, total_stats["duration"],
                    total_stats["nrem"].get("strengthened", 0),
                    total_stats["nrem"].get("weakened", 0),
                    total_stats["nrem"].get("pruned", 0),
                    total_stats["nrem"].get("protected", 0),
                    total_stats["rem"].get("bridges", 0),
                    total_stats["insights"].get("insights", 0),
                )

            except Exception as e:
                logger.error("Dream cycle failed: %s", e)
                total_stats["error"] = str(e)

            return total_stats

    # -- Phase 1: NREM -------------------------------------------------------

    def _phase_nrem(self, deep: bool = False) -> Dict[str, Any]:
        """NREM: Replay, strengthen active, weaken inactive, prune dead.

        Hybrid sampling: mixes recent memories with random historical samples
        so old connections get replayed and don't silently decay.

        Decay protection:
        - Strong edges (weight > STRONG_EDGE_THRESHOLD) decay 5x slower
        - Bridge edges (connecting communities) are immune to decay
        - Deep dream cycles use 3x the memory window

        Batched for performance: all updates in one transaction.
        """
        stats = {
            "processed": 0, "strengthened": 0,
            "weakened": 0, "pruned": 0, "protected": 0,
        }
        session_id = self._backend.start_session("deep-nrem" if deep else "nrem")

        try:
            # --- Hybrid sampling: recent + random historical ---
            budget = self._max_memories * (3 if deep else 1)
            recent_budget = int(budget * (1 - self.HISTORY_SAMPLE_RATIO))
            history_budget = budget - recent_budget

            memories = self._backend.get_recent_memories(recent_budget)
            seen_ids = {m["id"] for m in memories}

            # Add random historical memories (deduped against recent)
            try:
                historical = self._backend.get_random_memories(history_budget * 2)
                for m in historical:
                    if m["id"] not in seen_ids:
                        memories.append(m)
                        seen_ids.add(m["id"])
                    if len(memories) >= budget:
                        break
            except NotImplementedError:
                pass  # Backend doesn't support random sampling

            if not memories:
                return stats

            activated_edges = set()

            for mem in memories:
                mid = mem["id"]
                if self._memory:
                    try:
                        activated = self._memory.think(mid, depth=2)
                        for a in activated:
                            aid = a.get("id")
                            if aid and aid != mid:
                                activated_edges.add((min(mid, aid), max(mid, aid)))
                    except Exception:
                        pass
                stats["processed"] += 1

            # --- Batch update with decay protection ---
            all_conns = self._backend.get_connections()
            to_strengthen = []
            to_weaken = []

            for conn in all_conns:
                src, tgt = conn["source_id"], conn["target_id"]
                key = (min(src, tgt), max(src, tgt))
                old_w = conn["weight"]

                if key in activated_edges:
                    new_w = min(old_w + 0.05, 1.0)
                    to_strengthen.append((new_w, src, tgt))
                    stats["strengthened"] += 1
                elif old_w > 0.05:
                    # --- Decay protection ---
                    # Bridge edges: immune to decay
                    if src in self._bridge_node_ids and tgt in self._bridge_node_ids:
                        stats["protected"] += 1
                        continue
                    # Strong edges: decay 5x slower
                    if old_w >= self.STRONG_EDGE_THRESHOLD:
                        decay = 0.01 * self.STRONG_EDGE_DECAY_FACTOR
                    else:
                        decay = 0.01
                    new_w = max(old_w - decay, 0.0)
                    to_weaken.append((new_w, src, tgt))
                    stats["weakened"] += 1

            # Execute batch updates
            if isinstance(self._backend, SQLiteDreamBackend):
                conn = self._backend._connect()
                try:
                    conn.execute("BEGIN")
                    if to_strengthen:
                        conn.executemany(
                            "UPDATE connections SET weight = ? WHERE source_id = ? AND target_id = ?",
                            to_strengthen
                        )
                    if to_weaken:
                        conn.executemany(
                            "UPDATE connections SET weight = ? WHERE source_id = ? AND target_id = ?",
                            to_weaken
                        )
                    conn.commit()
                finally:
                    conn.close()
            else:
                # MSSQL batch updates — single commit, no per-row logging
                try:
                    batch_method = getattr(self._backend, 'batch_strengthen_connections', None)
                    if batch_method and to_strengthen:
                        batch_method(to_strengthen)
                    elif to_strengthen:
                        # Fallback: execute in single transaction
                        cursor = self._backend.conn.cursor()
                        for new_w, src, tgt in to_strengthen:
                            cursor.execute(
                                "UPDATE connections SET weight = ? WHERE source_id = ? AND target_id = ?",
                                new_w, src, tgt
                            )
                        self._backend.conn.commit()

                    batch_weaken = getattr(self._backend, 'batch_weaken_connections', None)
                    if batch_weaken and to_weaken:
                        batch_weaken(to_weaken)
                    elif to_weaken:
                        cursor = self._backend.conn.cursor()
                        for new_w, src, tgt in to_weaken:
                            cursor.execute(
                                "UPDATE connections SET weight = ? WHERE source_id = ? AND target_id = ?",
                                new_w, src, tgt
                            )
                        self._backend.conn.commit()
                except Exception as e:
                    logger.warning("MSSQL batch update failed: %s", e)

            # Prune dead connections
            stats["pruned"] = self._backend.prune_weak(0.05)

            # Periodic cleanup: prune old history + orphans every 50 cycles
            if self._dream_count % 50 == 0:
                try:
                    prune_fn = getattr(self._backend, 'prune_connection_history', None)
                    if prune_fn:
                        pruned = prune_fn(keep_days=7)
                        if pruned:
                            logger.info("Pruned %d old connection_history entries", pruned)
                    prune_sessions = getattr(self._backend, 'prune_old_dream_sessions', None)
                    if prune_sessions:
                        pruned_s = prune_sessions(keep_days=30)
                        if pruned_s:
                            logger.info("Pruned %d old dream sessions", pruned_s)
                    prune_orphans = getattr(self._backend, 'prune_orphans', None)
                    if prune_orphans:
                        pruned_o = prune_orphans()
                        if pruned_o:
                            logger.info("Pruned %d orphan connections", pruned_o)
                except Exception as e:
                    logger.debug("Maintenance cleanup error: %s", e)

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("NREM phase error: %s", e)

        return stats

    # -- Phase 2: REM --------------------------------------------------------

    def _phase_rem(self, deep: bool = False) -> Dict[str, Any]:
        """REM: Explore isolated memories, discover bridges.

        1. Find isolated memories (few connections)
        2. Search via embedding similarity for unconnected but similar
        3. Create tentative bridge connections (weight 0.1-0.3)

        Alternates between recent-first and oldest-first on each cycle
        to ensure historical orphans get explored too. Deep dreams
        always explore oldest-first and with a larger budget.
        """
        stats = {"explored": 0, "bridges": 0, "rejected": 0}
        session_id = self._backend.start_session("deep-rem" if deep else "rem")

        try:
            # Alternate direction: even cycles = recent-first, odd = oldest-first
            # Deep dreams always explore oldest-first with expanded budget
            oldest_first = deep or (self._dream_count % 2 == 1)
            limit = 100 if deep else 50

            try:
                isolated = self._backend.get_isolated_memories(
                    max_connections=3, limit=limit, oldest_first=oldest_first,
                )
            except TypeError:
                # Backend doesn't support oldest_first parameter
                isolated = self._backend.get_isolated_memories(
                    max_connections=3, limit=limit,
                )
            if not isolated:
                return stats

            for mem in isolated:
                mid = mem["id"]
                content = mem.get("content", "")

                if not self._memory or not content:
                    continue

                try:
                    similar = self._memory.recall(content[:200], k=10)
                    stats["explored"] += 1

                    for sim in similar:
                        sim_id = sim.get("id")
                        sim_score = sim.get("similarity", 0)

                        if not sim_id or sim_id == mid:
                            continue
                        if sim_score < 0.3 or sim_score > 0.95:
                            continue

                        bridge_weight = round(sim_score * 0.3, 3)
                        self._backend.add_bridge(mid, sim_id, bridge_weight)
                        self._backend.log_connection_change(
                            mid, sim_id, 0.0, bridge_weight, "rem_bridge"
                        )
                        stats["bridges"] += 1

                except Exception:
                    pass

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("REM phase error: %s", e)

        return stats

    # -- Phase 3: Insights ---------------------------------------------------

    def _phase_insights(self) -> Dict[str, Any]:
        """Insight: Community detection, bridge identification, abstraction.

        1. Find connected components (communities)
        2. Identify bridge nodes connecting communities
        3. Create insight memories for dense clusters
        """
        stats = {"communities": 0, "bridges": 0, "insights": 0}
        session_id = self._backend.start_session("insight")

        try:
            edges = self._backend.get_connections()
            if not edges:
                return stats

            # Build adjacency
            adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            nodes = set()
            for e in edges:
                s, t, w = e["source_id"], e["target_id"], e["weight"]
                adj[s].append((t, w))
                adj[t].append((s, w))
                nodes.add(s)
                nodes.add(t)

            # Connected components (BFS)
            visited = set()
            communities: List[List[int]] = []
            for node in nodes:
                if node in visited:
                    continue
                component = []
                queue = [node]
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    component.append(curr)
                    for neighbor, _ in adj.get(curr, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
                communities.append(component)
            stats["communities"] = len(communities)

            # Map nodes to communities
            node_to_comm = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i

            # Find bridge nodes
            bridge_nodes = set()
            for e in edges:
                s_comm = node_to_comm.get(e["source_id"], -1)
                t_comm = node_to_comm.get(e["target_id"], -1)
                if s_comm != t_comm and s_comm >= 0 and t_comm >= 0:
                    bridge_nodes.add(e["source_id"])
                    bridge_nodes.add(e["target_id"])
            stats["bridges"] = len(bridge_nodes)

            # Cache bridge nodes for NREM decay protection
            self._bridge_node_ids = bridge_nodes

            # Create cluster insights
            for i, comm in enumerate(communities):
                if len(comm) < 3:
                    continue
                theme = self._extract_theme(comm)
                confidence = min(len(comm) / 10.0, 1.0)
                content = f"Cluster of {len(comm)} related memories: {theme}"
                self._backend.add_insight(session_id, "cluster", comm[0], content, confidence)
                stats["insights"] += 1

            # Create bridge insights
            for bnode in bridge_nodes:
                bridging_communities = set()
                for e in edges:
                    if e["source_id"] == bnode or e["target_id"] == bnode:
                        other = e["target_id"] if e["source_id"] == bnode else e["source_id"]
                        bridging_communities.add(node_to_comm.get(other, -1))
                bridging_communities.discard(-1)

                if len(bridging_communities) >= 2:
                    content = (
                        f"Bridge connecting {len(bridging_communities)} communities, "
                        f"memory #{bnode}"
                    )
                    self._backend.add_insight(session_id, "bridge", bnode, content, 0.8)
                    stats["insights"] += 1

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("Insight phase error: %s", e)

        return stats

    # -- Helpers -------------------------------------------------------------

    def _extract_theme(self, node_ids: List[int]) -> str:
        """Extract common themes from node IDs (simple keyword frequency)."""
        # If we have memory access, get contents via store (thread-safe)
        contents = []
        if self._memory and hasattr(self._memory, 'store'):
            try:
                placeholders = ",".join("?" * len(node_ids))
                with self._memory.store._lock:
                    rows = self._memory.store.conn.execute(
                        f"SELECT content FROM memories WHERE id IN ({placeholders})",
                        tuple(node_ids)
                    ).fetchall()
                    contents = [r[0] for r in rows if r[0]]
            except Exception:
                pass

        if not contents:
            return f"{len(node_ids)} memories"

        word_freq: Dict[str, int] = defaultdict(int)
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "it", "its", "this", "that", "user", "assistant", "and", "or",
            "but", "not", "if", "then", "so", "just", "also", "very",
            "really", "like", "get", "got", "want", "need", "think",
            "know", "see", "look", "make", "let", "use", "still",
        }
        for c in contents:
            for w in c.lower().split():
                w = w.strip(".,!?;:'\"()[]{}#@")
                if len(w) > 3 and w not in stopwords:
                    word_freq[w] += 1

        top = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
        return ", ".join(w for w, _ in top) if top else "mixed topics"

    def get_stats(self) -> Dict[str, Any]:
        """Get dream engine statistics."""
        base = self._backend.get_dream_stats()
        base["engine_running"] = self._running
        base["dream_cycles"] = self._dream_count
        return base
