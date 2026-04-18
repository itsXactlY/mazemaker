#!/usr/bin/env python3
"""
dlm_adapter.py - JackrabbitDLM adapter for Neural Memory.

Routes memory operations through JackrabbitDLM's JSON-over-TCP protocol
via the DLMLocker library. Keeps SQLite for fast local reads and indexing.

Architecture:
  SQLite (local reads + key index)  -- fast path for recall/think
  DLM client (writes + locking)    -- JackrabbitDLM server -> volatile key-value
  DLM client (retrieval)           -- direct GET for known keys

DLM server address from env vars:
  DLM_HOST  (default: 127.0.0.1)
  DLM_PORT  (default: 37373)

The DLM protocol (via DLMLocker):
  - Put(expire, data)  -> store value with TTL
  - Get()              -> retrieve value
  - Erase()            -> delete value
  - Lock(expire)       -> acquire distributed lock
  - Unlock()           -> release lock
  - Version()          -> server health check

Both the main Memory class and the Dream Engine can use DLM mode.
SQLite dream_sessions are kept locally for tracking.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Add JackrabbitDLM to path
sys.path.insert(0, '/home/JackrabbitDLM')

# ---------------------------------------------------------------------------#
# DLM Connection (wraps DLMLocker)
# ---------------------------------------------------------------------------#

class DLMConnection:
    """Low-level JackrabbitDLM connection using DLMLocker library."""

    def __init__(self, host: str = "127.0.0.1", port: int = 37373,
                 timeout: float = 10.0, identity: str = "neural-memory"):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._identity = identity
        self._Locker = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        """Load DLMLocker and verify DLM server is reachable."""
        try:
            from DLMLocker import Locker
            self._Locker = Locker
            # Health check via Version
            test = Locker("health-check", Host=self._host, Port=self._port,
                          ID=self._identity, Timeout=int(self._timeout))
            ver = test.Version()
            if 'JackrabbitDLM' in str(ver):
                self._connected = True
                logger.info("DLM connected to %s:%s", self._host, self._port)
            else:
                logger.warning("DLM version check failed: %s", ver)
                self._connected = False
        except ImportError:
            logger.warning("DLMLocker not found at /home/JackrabbitDLM")
            self._connected = False
        except Exception as e:
            logger.warning("DLM connection failed: %s", e)
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._Locker is not None

    def _make_locker(self, key: str):
        """Create a DLMLocker instance for a given key."""
        return self._Locker(key, Host=self._host, Port=self._port,
                            ID=self._identity, Timeout=int(self._timeout))

    def health_check(self) -> bool:
        """Verify DLM server is reachable."""
        if not self.is_connected:
            return False
        try:
            lock = self._make_locker("health-check")
            return 'JackrabbitDLM' in str(lock.Version())
        except Exception:
            return False

    def put(self, key: str, data: str, ttl: int = 3600) -> bool:
        """Store data under key with TTL (seconds)."""
        if not self.is_connected:
            return False
        try:
            lock = self._make_locker(key)
            resp = lock.Put(expire=ttl, data=data)
            if isinstance(resp, bytes):
                resp = resp.decode('utf-8')
            return "Done" in str(resp)
        except Exception as e:
            logger.debug("DLM put(%s) failed: %s", key, e)
            return False

    def get(self, key: str) -> Optional[str]:
        """Retrieve data by key."""
        if not self.is_connected:
            return None
        try:
            lock = self._make_locker(key)
            resp = lock.Get()
            if isinstance(resp, dict) and resp.get("Status") == "Done":
                return resp.get("DataStore")
            return None
        except Exception as e:
            logger.debug("DLM get(%s) failed: %s", key, e)
            return None

    def erase(self, key: str) -> bool:
        """Delete data by key."""
        if not self.is_connected:
            return False
        try:
            lock = self._make_locker(key)
            resp = lock.Erase()
            if isinstance(resp, bytes):
                resp = resp.decode('utf-8')
            return "Done" in str(resp)
        except Exception as e:
            logger.debug("DLM erase(%s) failed: %s", key, e)
            return False

    def lock(self, key: str, expire: int = 300) -> bool:
        """Acquire a distributed lock."""
        if not self.is_connected:
            return False
        try:
            lock = self._make_locker(f"lock-{key}")
            resp = lock.Lock(expire=expire)
            return resp == "locked"
        except Exception as e:
            logger.debug("DLM lock(%s) failed: %s", key, e)
            return False

    def unlock(self, key: str) -> bool:
        """Release a distributed lock."""
        if not self.is_connected:
            return False
        try:
            lock = self._make_locker(f"lock-{key}")
            resp = lock.Unlock()
            return resp == "unlocked"
        except Exception as e:
            logger.debug("DLM unlock(%s) failed: %s", key, e)
            return False

    def close(self) -> None:
        """Close connection."""
        self._connected = False


# ---------------------------------------------------------------------------#
# DLM Store - Neural Memory store using DLM as backend
# ---------------------------------------------------------------------------#

class DLMStore:
    """
    Neural Memory store that uses JackrabbitDLM for distributed key-value
    storage, with SQLite as a local index for fast reads and search.

    Memory data lives in DLM (volatile, TTL-bound).
    SQLite holds: key index, content snippets for search, connection graph.
    """

    def __init__(self, dlm: DLMConnection, db_path: Optional[str] = None,
                 default_ttl: int = 86400):
        import sqlite3

        self._dlm = dlm
        self._default_ttl = default_ttl
        self._db_path = db_path or str(
            Path.home() / ".neural_memory" / "dlm_index.db"
        )
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Local SQLite index for fast lookups
        conn = sqlite3.connect(self._db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dlm_keys (
                key TEXT PRIMARY KEY,
                label TEXT,
                content_snippet TEXT,
                created_at REAL NOT NULL,
                last_accessed REAL,
                ttl INTEGER DEFAULT 86400
            );
            CREATE TABLE IF NOT EXISTS dlm_connections (
                source_key TEXT NOT NULL,
                target_key TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                edge_type TEXT DEFAULT 'similar',
                created_at REAL NOT NULL,
                PRIMARY KEY (source_key, target_key)
            );
            CREATE INDEX IF NOT EXISTS idx_dlm_keys_label ON dlm_keys(label);
            CREATE INDEX IF NOT EXISTS idx_dlm_conn_source ON dlm_connections(source_key);
            CREATE INDEX IF NOT EXISTS idx_dlm_conn_target ON dlm_connections(target_key);
        """)
        conn.commit()
        conn.close()

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _dlm_key(self, mem_id: int) -> str:
        """Generate DLM key from memory ID."""
        return f"nm-mem-{mem_id}"

    # -- Write operations (DLM + local index) --

    def store(self, label: str, content: str, embedding: list[float],
              ttl: Optional[int] = None) -> int:
        """Store a memory in DLM and index locally. Returns memory ID."""
        import time as _time

        ttl = ttl or self._default_ttl
        now = _time.time()

        # Generate next ID from local index
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(oid), 0) as max_id FROM dlm_keys"
            ).fetchone()
            # Use a hash-based ID for distributed uniqueness
            next_id = int(now * 1000) % 100000000 + (row[0] if row else 0) + 1
        except Exception:
            next_id = int(now * 1000) % 100000000
        finally:
            conn.close()

        dlm_key = self._dlm_key(next_id)
        memory_data = json.dumps({
            "id": next_id,
            "label": label,
            "content": content,
            "embedding": embedding,
            "created_at": now,
        })

        # Store in DLM
        ok = self._dlm.put(dlm_key, memory_data, ttl=ttl)

        # Index locally regardless of DLM result (for resilience)
        conn = self._connect()
        try:
            snippet = content[:200] if content else ""
            conn.execute(
                "INSERT OR REPLACE INTO dlm_keys (key, label, content_snippet, created_at, ttl) "
                "VALUES (?, ?, ?, ?, ?)",
                (dlm_key, label, snippet, now, ttl)
            )
            conn.commit()
        finally:
            conn.close()

        if not ok:
            logger.warning("DLM put failed for key %s, stored in index only", dlm_key)

        return next_id

    def retrieve(self, mem_id: int) -> Optional[dict]:
        """Retrieve a memory by ID from DLM."""
        dlm_key = self._dlm_key(mem_id)
        data = self._dlm.get(dlm_key)
        if data:
            try:
                mem = json.loads(data)
                # Update last_accessed in index
                conn = self._connect()
                try:
                    conn.execute(
                        "UPDATE dlm_keys SET last_accessed = ? WHERE key = ?",
                        (time.time(), dlm_key)
                    )
                    conn.commit()
                finally:
                    conn.close()
                return mem
            except json.JSONDecodeError:
                return None
        # Fallback: try local index
        return None

    def delete(self, mem_id: int) -> bool:
        """Delete a memory from DLM and local index."""
        dlm_key = self._dlm_key(mem_id)
        ok = self._dlm.erase(dlm_key)
        conn = self._connect()
        try:
            conn.execute("DELETE FROM dlm_keys WHERE key = ?", (dlm_key,))
            conn.execute(
                "DELETE FROM dlm_connections WHERE source_key = ? OR target_key = ?",
                (dlm_key, dlm_key)
            )
            conn.commit()
        finally:
            conn.close()
        return ok

    # -- Connection operations (local SQLite + DLM for distributed sync) --

    def add_connection(self, source_id: int, target_id: int,
                       weight: float = 0.5, edge_type: str = "similar") -> None:
        """Add a graph edge."""
        src_key = self._dlm_key(source_id)
        tgt_key = self._dlm_key(target_id)
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO dlm_connections "
                "(source_key, target_key, weight, edge_type, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (src_key, tgt_key, weight, edge_type, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def strengthen_connection(self, source_id: int, target_id: int,
                              delta: float = 0.05) -> None:
        """Strengthen an edge."""
        src_key = self._dlm_key(source_id)
        tgt_key = self._dlm_key(target_id)
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE dlm_connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_key = ? AND target_key = ?",
                (delta, src_key, tgt_key)
            )
            conn.commit()
        finally:
            conn.close()

    def weaken_connection(self, source_id: int, target_id: int,
                          delta: float = 0.01) -> None:
        """Weaken an edge."""
        src_key = self._dlm_key(source_id)
        tgt_key = self._dlm_key(target_id)
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE dlm_connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE source_key = ? AND target_key = ?",
                (delta, src_key, tgt_key)
            )
            conn.commit()
        finally:
            conn.close()

    def get_connections(self, mem_id: int) -> List[dict]:
        """Get connections for a memory."""
        dlm_key = self._dlm_key(mem_id)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_key, target_key, weight, edge_type "
                "FROM dlm_connections WHERE source_key = ? OR target_key = ?",
                (dlm_key, dlm_key)
            ).fetchall()
            results = []
            for r in rows:
                # Convert key back to ID
                src = r["source_key"].replace("nm-mem-", "")
                tgt = r["target_key"].replace("nm-mem-", "")
                results.append({
                    "source": int(src),
                    "target": int(tgt),
                    "weight": r["weight"],
                    "edge_type": r["edge_type"],
                })
            return results
        finally:
            conn.close()

    def get_connections_raw(self) -> List[Dict[str, Any]]:
        """Get all edges."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_key, target_key, weight "
                "FROM dlm_connections WHERE weight >= 0.05"
            ).fetchall()
            return [
                {
                    "source_id": int(r["source_key"].replace("nm-mem-", "")),
                    "target_id": int(r["target_key"].replace("nm-mem-", "")),
                    "weight": r["weight"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def batch_strengthen(self, edges: List[Tuple[int, int]],
                         delta: float = 0.05) -> int:
        """Batch strengthen edges."""
        if not edges:
            return 0
        conn = self._connect()
        try:
            conn.executemany(
                "UPDATE dlm_connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_key = ? AND target_key = ?",
                [(delta, self._dlm_key(s), self._dlm_key(t)) for s, t in edges]
            )
            conn.commit()
            return len(edges)
        finally:
            conn.close()

    def batch_weaken(self, threshold: float = 0.05,
                     delta: float = 0.01) -> int:
        """Bulk weaken all connections above threshold."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE dlm_connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE weight > ?",
                (delta, threshold)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def add_bridge(self, source_id: int, target_id: int,
                   weight: float = 0.3) -> None:
        """Add a bridge edge."""
        src_key = self._dlm_key(source_id)
        tgt_key = self._dlm_key(target_id)
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT 1 FROM dlm_connections "
                "WHERE (source_key = ? AND target_key = ?) "
                "OR (source_key = ? AND target_key = ?)",
                (src_key, tgt_key, tgt_key, src_key)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO dlm_connections "
                    "(source_key, target_key, weight, edge_type, created_at) "
                    "VALUES (?, ?, ?, 'bridge', ?)",
                    (src_key, tgt_key, weight, time.time())
                )
                conn.commit()
        finally:
            conn.close()

    def prune_weak(self, threshold: float = 0.05) -> int:
        """Delete weak connections."""
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM dlm_connections WHERE weight < ?",
                (threshold,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def log_connection_change(self, source_id: int, target_id: int,
                              old_weight: float, new_weight: float,
                              reason: str) -> None:
        """Log connection change (no-op for DLM, use SQLite dream tables)."""
        pass

    # -- Read operations (local index + DLM fallback) --

    def recall(self, embedding: list[float], k: int = 5) -> List[dict]:
        """
        Search by iterating local index, loading from DLM.
        No native vector search — uses brute-force cosine on indexed keys.
        """
        import math

        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT key, label, content_snippet FROM dlm_keys "
                "ORDER BY created_at DESC LIMIT 500"
            ).fetchall()
        finally:
            conn.close()

        results = []
        for r in rows:
            mem_id = int(r["key"].replace("nm-mem-", ""))
            full = self.retrieve(mem_id)
            if not full or not full.get("embedding"):
                continue
            emb = full["embedding"]
            if len(emb) != len(embedding):
                continue
            # Cosine similarity
            dot = sum(a * b for a, b in zip(embedding, emb))
            norm_a = math.sqrt(sum(a * a for a in embedding))
            norm_b = math.sqrt(sum(b * b for b in emb))
            if norm_a == 0 or norm_b == 0:
                continue
            sim = dot / (norm_a * norm_b)
            results.append({
                "id": mem_id,
                "label": full.get("label", ""),
                "content": full.get("content", ""),
                "embedding": emb,
                "similarity": round(sim, 4),
                "created_at": full.get("created_at", 0),
            })

        results.sort(key=lambda x: -x["similarity"])
        return results[:k]

    def stats(self) -> dict:
        """Get store statistics."""
        conn = self._connect()
        try:
            mem_count = conn.execute("SELECT COUNT(*) FROM dlm_keys").fetchone()[0]
            edge_count = conn.execute("SELECT COUNT(*) FROM dlm_connections").fetchone()[0]
            return {"memories": mem_count, "connections": edge_count}
        finally:
            conn.close()

    def close(self) -> None:
        """Shutdown."""
        pass


# ---------------------------------------------------------------------------#
# DLM Dream Backend - for Dream Engine integration
# ---------------------------------------------------------------------------#

class DLMDreamBackend:
    """
    Dream backend that routes graph operations through DLM.

    Dream sessions are tracked locally in SQLite (fast, no DLM dependency).
    Graph operations (strengthen, weaken, prune, bridge) use DLMStore's
    local SQLite index (same DB).
    """

    def __init__(self, store: DLMStore, db_path: Optional[str] = None):
        import sqlite3

        self._store = store
        self._db_path = db_path or store._db_path

        # Ensure dream tables exist in the same DB
        conn = sqlite3.connect(self._db_path)
        conn.executescript("""
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
        """)
        conn.commit()
        conn.close()

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- Dream Sessions (local SQLite) --

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
                "UPDATE dream_sessions SET finished_at=?, memories_processed=?, "
                "connections_strengthened=?, connections_pruned=?, "
                "bridges_found=?, insights_created=? WHERE id=?",
                (time.time(),
                 stats.get("processed", stats.get("explored", 0)),
                 stats.get("strengthened", 0),
                 stats.get("pruned", 0),
                 stats.get("bridges", 0),
                 stats.get("insights", 0),
                 session_id)
            )
            conn.commit()
        finally:
            conn.close()

    # -- Graph Operations (through DLMStore) --

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memories from local index."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT key FROM dlm_keys ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"id": int(r["key"].replace("nm-mem-", ""))} for r in rows]
        except Exception as e:
            logger.debug("DLM get_recent_memories failed: %s", e)
            return []
        finally:
            conn.close()

    def get_isolated_memories(self, max_connections: int = 3,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """Find memories with few edges."""
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT k.key, k.content_snippet,
                       (SELECT COUNT(*) FROM dlm_connections
                        WHERE source_key = k.key OR target_key = k.key) as cnt
                FROM dlm_keys k
                WHERE (SELECT COUNT(*) FROM dlm_connections
                       WHERE source_key = k.key OR target_key = k.key) < ?
                ORDER BY k.created_at DESC LIMIT ?
            """, (max_connections, limit)).fetchall()
            return [
                {"id": int(r["key"].replace("nm-mem-", "")),
                 "content": r["content_snippet"] or "",
                 "connection_count": r["cnt"]}
                for r in rows
            ]
        except Exception as e:
            logger.debug("DLM get_isolated_memories failed: %s", e)
            return []
        finally:
            conn.close()

    def get_connections(self) -> List[Dict[str, Any]]:
        """Get all edges."""
        return self._store.get_connections_raw()

    def strengthen_connection(self, source_id: int, target_id: int,
                              delta: float = 0.05) -> None:
        self._store.strengthen_connection(source_id, target_id, delta)

    def weaken_connection(self, source_id: int, target_id: int,
                          delta: float = 0.01) -> None:
        self._store.weaken_connection(source_id, target_id, delta)

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                      delta: float = 0.05) -> int:
        return self._store.batch_strengthen(edges, delta)

    def batch_weaken_connections(self, threshold: float = 0.05,
                                  delta: float = 0.01) -> int:
        return self._store.batch_weaken(threshold, delta)

    def add_bridge(self, source_id: int, target_id: int,
                   weight: float = 0.3) -> None:
        self._store.add_bridge(source_id, target_id, weight)

    def prune_weak(self, threshold: float = 0.05) -> int:
        return self._store.prune_weak(threshold)

    def log_connection_change(self, source_id: int, target_id: int,
                              old_weight: float, new_weight: float,
                              reason: str) -> None:
        pass

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        """Store insight in local SQLite."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO dream_insights "
                "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, insight_type, source_memory_id,
                 content, confidence, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """No-op (no connection_history table in DLM mode)."""
        return 0

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Prune old dream sessions from local SQLite."""
        conn = self._connect()
        try:
            cutoff = time.time() - (keep_days * 86400)
            count = conn.execute(
                "DELETE FROM dream_sessions WHERE started_at < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent keys."""
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM dlm_connections "
                "WHERE source_key NOT IN (SELECT key FROM dlm_keys) "
                "OR target_key NOT IN (SELECT key FROM dlm_keys)"
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        """Get dream stats from local SQLite."""
        conn = self._connect()
        try:
            s = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(memories_processed),0), "
                "COALESCE(SUM(connections_strengthened),0), "
                "COALESCE(SUM(connections_pruned),0), "
                "COALESCE(SUM(bridges_found),0), "
                "COALESCE(SUM(insights_created),0) FROM dream_sessions"
            ).fetchone()
            return {
                "sessions": s[0] if s else 0,
                "total_processed": s[1] if s else 0,
                "total_strengthened": s[2] if s else 0,
                "total_pruned": s[3] if s else 0,
                "total_bridges": s[4] if s else 0,
                "total_insights": s[5] if s else 0,
                "backend": "dlm",
            }
        finally:
            conn.close()

    def close(self) -> None:
        """Shutdown."""
        pass


# ---------------------------------------------------------------------------#
# Top-level convenience: check if DLM is available
# ---------------------------------------------------------------------------#

def check_dlm_available(host: str = "", port: int = 0) -> bool:
    """Quick check if DLM server is reachable."""
    host = host or os.environ.get("DLM_HOST", "127.0.0.1")
    port = port or int(os.environ.get("DLM_PORT", "37373"))
    try:
        conn = DLMConnection(host=host, port=port)
        ok = conn.is_connected
        conn.close()
        return ok
    except Exception:
        return False
