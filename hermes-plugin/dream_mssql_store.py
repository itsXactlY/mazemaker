"""
Dream MSSQL Store — Dream-specific tables on MSSQL backend.

Extends the existing MSSQLStore pattern with dream_sessions,
dream_insights, and connection_history tables. Falls back to
SQLite if MSSQL is unavailable.

Credentials resolution order:
  1. Environment variables: MSSQL_SERVER, MSSQL_DATABASE, MSSQL_USERNAME, MSSQL_PASSWORD
  2. .env file in project or ~/.hermes/.env
  3. Config dict (from config.yaml)
  4. Defaults (localhost, NeuralMemory, SA)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loader (lightweight, no python-dotenv dependency)
# ---------------------------------------------------------------------------

def _load_dotenv(paths: list[str]) -> dict:
    """Load .env files from multiple paths, env vars take precedence."""
    env = {}
    for p in paths:
        path = Path(p).expanduser()
        if path.is_file():
            try:
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip("\"'")
                        if key and val:
                            env.setdefault(key, val)  # first wins
            except Exception:
                pass
    return env


_dotenv = _load_dotenv([
    ".env",                           # CWD
    str(Path.home() / ".hermes" / ".env"),  # ~/.hermes/.env
    str(Path(__file__).parent / ".env"),    # plugin dir
])


def _env(key: str, fallback: str = "") -> str:
    """Get env var: OS env > .env > fallback."""
    return os.environ.get(key) or _dotenv.get(key, fallback)

_DREAM_MSSQL_SCHEMA = """
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dream_sessions')
CREATE TABLE dream_sessions (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    started_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
    finished_at DATETIME2(3),
    phase NVARCHAR(20) NOT NULL,
    memories_processed INT DEFAULT 0,
    connections_strengthened INT DEFAULT 0,
    connections_pruned INT DEFAULT 0,
    bridges_found INT DEFAULT 0,
    insights_created INT DEFAULT 0
);

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dream_insights')
CREATE TABLE dream_insights (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    session_id BIGINT,
    insight_type NVARCHAR(50) NOT NULL,
    source_memory_id BIGINT,
    content NVARCHAR(MAX),
    confidence FLOAT DEFAULT 0.0,
    created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'connection_history')
CREATE TABLE connection_history (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    source_id BIGINT NOT NULL,
    target_id BIGINT NOT NULL,
    old_weight FLOAT,
    new_weight FLOAT,
    reason NVARCHAR(100),
    changed_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_insights_type')
CREATE INDEX idx_dream_insights_type ON dream_insights(insight_type);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_insights_session')
CREATE INDEX idx_dream_insights_session ON dream_insights(session_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_conn_history')
CREATE INDEX idx_dream_conn_history ON connection_history(source_id, target_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_sessions_started')
CREATE INDEX idx_dream_sessions_started ON dream_sessions(started_at);
"""


class DreamMSSQLStore:
    """MSSQL-backed dream store. Creates dream tables on first use.

    Credentials: env vars > .env > config dict > defaults.
    """

    def __init__(self, server='', database='', username='', password='',
                 driver=''):
        try:
            import pyodbc
        except ImportError:
            raise ImportError("pyodbc required for MSSQL dream backend")

        # Resolve: explicit args > env vars > defaults
        server = server or _env('MSSQL_SERVER', '127.0.0.1')
        # Fix IPv6 localhost issue — MSSQL only listens on IPv4
        if server == 'localhost':
            server = '127.0.0.1'
        database = database or _env('MSSQL_DATABASE', 'NeuralMemory')
        username = username or _env('MSSQL_USERNAME', 'SA')
        password = password or _env('MSSQL_PASSWORD', '')
        driver = driver or _env('MSSQL_DRIVER', '{ODBC Driver 18 for SQL Server}')

        if not password:
            logger.warning(
                "MSSQL_PASSWORD not set — add it to ~/.hermes/.env "
                "or set MSSQL_PASSWORD env var"
            )

        self.conn_str = (
            f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
            f'TrustServerCertificate=yes;'
        )
        self.conn = pyodbc.connect(self.conn_str, autocommit=True)
        self._ensure_schema()

    @classmethod
    def from_config(cls, config: dict) -> 'DreamMSSQLStore':
        """Create from config dict (mssql section). Env vars override config."""
        return cls(
            server=config.get('server', ''),
            database=config.get('database', ''),
            username=config.get('username', ''),
            password=config.get('password', ''),
            driver=config.get('driver', ''),
        )

    def _ensure_schema(self):
        """Create dream tables if they don't exist."""
        cursor = self.conn.cursor()
        for stmt in _DREAM_MSSQL_SCHEMA.split(';'):
            stmt = stmt.strip()
            if stmt and 'GO' not in stmt:
                try:
                    cursor.execute(stmt)
                except Exception:
                    pass
        self.conn.commit()

    # -- Dream Sessions ------------------------------------------------------

    def start_session(self, phase: str) -> int:
        """Record the start of a dream session. Returns session ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO dream_sessions (phase) "
            "OUTPUT INSERTED.id VALUES (?)",
            phase
        )
        row = cursor.fetchone()
        self.conn.commit()
        return row[0] if row else -1

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        """Record the end of a dream session."""
        if session_id < 0:
            return
        self.conn.execute(
            "UPDATE dream_sessions SET "
            "finished_at = SYSUTCDATETIME(), "
            "memories_processed = ?, "
            "connections_strengthened = ?, "
            "connections_pruned = ?, "
            "bridges_found = ?, "
            "insights_created = ? "
            "WHERE id = ?",
            stats.get("processed", stats.get("explored", 0)),
            stats.get("strengthened", 0),
            stats.get("pruned", 0),
            stats.get("bridges", 0),
            stats.get("insights", 0),
            session_id
        )
        self.conn.commit()

    # -- Connections (read from MSSQL) ---------------------------------------

    def get_connections(self) -> List[Dict[str, Any]]:
        """Get all active connections."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT source_id, target_id, weight FROM connections "
            "WHERE weight >= 0.05 ORDER BY weight DESC"
        )
        return [
            {"source_id": r[0], "target_id": r[1], "weight": r[2]}
            for r in cursor.fetchall()
        ]

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50,
                               oldest_first: bool = False) -> List[Dict[str, Any]]:
        """Find memories with few connections (isolated nodes)."""
        order = "ASC" if oldest_first else "DESC"
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT m.id, m.content, m.embedding, m.vector_dim,
                   ISNULL(conn.cnt, 0) as conn_count
            FROM memories m
            LEFT JOIN (
                SELECT source_id as mid, COUNT(*) as cnt
                FROM connections GROUP BY source_id
            ) conn ON m.id = conn.mid
            WHERE ISNULL(conn.cnt, 0) < ?
            ORDER BY m.id {order}
        """, max_connections)
        results = []
        for row in cursor.fetchall():
            mem_id, content, blob, dim, cnt = row
            results.append({
                "id": mem_id,
                "content": content or "",
                "connection_count": cnt,
            })
            if len(results) >= limit:
                break
        return results

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        """Strengthen a connection by delta (capped at 1.0)."""
        self.conn.execute(
            "UPDATE connections SET weight = CASE "
            "WHEN weight + ? > 1.0 THEN 1.0 ELSE weight + ? END "
            "WHERE source_id = ? AND target_id = ?",
            delta, delta, source_id, target_id
        )
        self.conn.commit()

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        """Weaken a connection by delta (floored at 0.0)."""
        self.conn.execute(
            "UPDATE connections SET weight = CASE "
            "WHEN weight - ? < 0.0 THEN 0.0 ELSE weight - ? END "
            "WHERE source_id = ? AND target_id = ?",
            delta, delta, source_id, target_id
        )
        self.conn.commit()

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        """Add a new bridge connection."""
        # Check if exists
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM connections "
            "WHERE (source_id = ? AND target_id = ?) "
            "OR (source_id = ? AND target_id = ?)",
            source_id, target_id, target_id, source_id
        )
        if cursor.fetchone():
            return
        self.conn.execute(
            "INSERT INTO connections (source_id, target_id, weight) "
            "VALUES (?, ?, ?)",
            source_id, target_id, weight
        )
        self.conn.commit()

    def prune_weak(self, threshold: float = 0.05) -> int:
        """Delete connections below threshold. Returns count deleted."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM connections WHERE weight < ?", threshold
        )
        self.conn.commit()
        return cursor.rowcount

    # -- Connection History ---------------------------------------------------

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        """Log a connection weight change."""
        self.conn.execute(
            "INSERT INTO connection_history "
            "(source_id, target_id, old_weight, new_weight, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            source_id, target_id, old_weight, new_weight, reason
        )
        self.conn.commit()

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        """Store a dream insight."""
        self.conn.execute(
            "INSERT INTO dream_insights "
            "(session_id, insight_type, source_memory_id, content, "
            "confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            session_id, insight_type, source_memory_id,
            content, confidence
        )
        self.conn.commit()

    def get_insights(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent dream insights."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, session_id, insight_type, source_memory_id, "
            "content, confidence, created_at "
            "FROM dream_insights ORDER BY created_at DESC",
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0], "session_id": row[1],
                "type": row[2], "memory_id": row[3],
                "content": row[4], "confidence": row[5],
                "created_at": row[6],
            })
            if len(results) >= limit:
                break
        return results

    # -- Memory access (for NREM replay) -------------------------------------

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memories for replay."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT TOP (?) id, content FROM memories "
            "ORDER BY created_at DESC",
            limit
        )
        return [{"id": r[0], "content": r[1] or ""} for r in cursor.fetchall()]

    def get_random_memories(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get a random sample of memories from the full history.

        Uses TABLESAMPLE for efficient random sampling on MSSQL,
        falling back to NEWID() ordering if table is too small.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT TOP (?) id, content FROM memories "
            "ORDER BY NEWID()",
            limit
        )
        return [{"id": r[0], "content": r[1] or ""} for r in cursor.fetchall()]

    # -- Stats ---------------------------------------------------------------

    def get_dream_stats(self) -> Dict[str, Any]:
        """Return statistics about past dream sessions."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as total,
                   ISNULL(SUM(memories_processed), 0),
                   ISNULL(SUM(connections_strengthened), 0),
                   ISNULL(SUM(connections_pruned), 0),
                   ISNULL(SUM(bridges_found), 0),
                   ISNULL(SUM(insights_created), 0)
            FROM dream_sessions
        """)
        row = cursor.fetchone()
        if not row:
            return {"sessions": 0}

        cursor.execute(
            "SELECT insight_type, COUNT(*) FROM dream_insights "
            "GROUP BY insight_type"
        )
        insight_types = {r[0]: r[1] for r in cursor.fetchall()}

        return {
            "sessions": row[0],
            "total_processed": row[1],
            "total_strengthened": row[2],
            "total_pruned": row[3],
            "total_bridges": row[4],
            "total_insights": row[5],
            "insight_types": insight_types,
        }

    # -- Batch Operations (for NREM deadlock prevention) --------------------

    def batch_strengthen_connections(self, edges: list) -> int:
        """Batch strengthen connections — single transaction with CASE WHEN.

        Args:
            edges: list of (new_weight, source_id, target_id) tuples
        Returns:
            Number of rows updated
        """
        if not edges:
            return 0
        cursor = self.conn.cursor()
        # Use CASE WHEN instead of MIN() (MSSQL doesn't support MIN as scalar)
        cursor.execute(
            "UPDATE connections SET weight = ? "
            "WHERE source_id = ? AND target_id = ?",
            edges[0][0], edges[0][1], edges[0][2]
        )
        for new_w, src, tgt in edges[1:]:
            cursor.execute(
                "UPDATE connections SET weight = ? "
                "WHERE source_id = ? AND target_id = ?",
                new_w, src, tgt
            )
        self.conn.commit()
        return len(edges)

    def batch_weaken_connections(self, edges: list) -> int:
        """Batch weaken connections — single transaction.

        Args:
            edges: list of (new_weight, source_id, target_id) tuples
        Returns:
            Number of rows updated
        """
        if not edges:
            return 0
        cursor = self.conn.cursor()
        for new_w, src, tgt in edges:
            cursor.execute(
                "UPDATE connections SET weight = ? "
                "WHERE source_id = ? AND target_id = ?",
                new_w, src, tgt
            )
        self.conn.commit()
        return len(edges)

    def prune_connection_history(self, keep_days: int = 7) -> int:
        """Prune old connection history entries to prevent unbounded growth.

        Keeps only entries from the last N days.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM connection_history "
            "WHERE changed_at < DATEADD(day, -?, SYSUTCDATETIME())",
            keep_days
        )
        self.conn.commit()
        return cursor.rowcount

    def prune_old_dream_sessions(self, keep_days: int = 30) -> int:
        """Prune old dream sessions and their insights.

        Keeps only sessions from the last N days.
        """
        cursor = self.conn.cursor()
        # Delete insights first (no CASCADE)
        cursor.execute(
            "DELETE di FROM dream_insights di "
            "INNER JOIN dream_sessions ds ON di.session_id = ds.id "
            "WHERE ds.started_at < DATEADD(day, -?, SYSUTCDATETIME())",
            keep_days
        )
        cursor.execute(
            "DELETE FROM dream_sessions "
            "WHERE started_at < DATEADD(day, -?, SYSUTCDATETIME())",
            keep_days
        )
        self.conn.commit()
        return cursor.rowcount

    def prune_orphans(self) -> int:
        """Delete connections pointing to non-existent memories in MSSQL."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM connections
            WHERE from_memory_id NOT IN (SELECT id FROM memories)
               OR to_memory_id NOT IN (SELECT id FROM memories)
        """)
        self.conn.commit()
        return cursor.rowcount

    def close(self):
        """Close the connection."""
        try:
            self.conn.close()
        except Exception:
            pass
