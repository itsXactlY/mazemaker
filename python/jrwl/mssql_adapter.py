"""
JRWL MSSQL Adapter - Connection pool for the broker.

Wraps pyodbc with asyncio-compatible connection pooling.
Pure stdlib + pyodbc (optional, only imported at broker runtime).

Usage:
    pool = MSSQLAdapter(config)
    await pool.start()
    rows = await pool.query("SELECT * FROM GraphNodes")
    await pool.stop()
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger("jrwl.mssql")


class MSSQLAdapter:
    """Async connection pool for MSSQL via pyodbc.

    Maintains a pool of reusable pyodbc connections.
    Thread-safe via asyncio lock (pyodbc connections are not thread-safe,
    but we serialize access through the event loop).
    """

    def __init__(self, config):
        """Initialize adapter with JRWLConfig."""
        self._config = config
        self._pool: deque = deque()
        self._pool_size = config.mssql_pool_size
        self._lock = asyncio.Lock()
        self._available = asyncio.Semaphore(config.mssql_pool_size)
        self._conn_str = config.mssql_conn_str
        self._started = False
        self._total_created = 0
        self._total_queries = 0
        self._total_errors = 0

    async def start(self):
        """Initialize the connection pool."""
        if self._started:
            return

        # Verify pyodbc is available
        try:
            import pyodbc
            self._pyodbc = pyodbc
        except ImportError:
            raise RuntimeError(
                "pyodbc is required for MSSQL adapter. "
                "Install it with: pip install pyodbc"
            )

        logger.info(
            "MSSQL pool starting: %d connections to %s",
            self._pool_size,
            self._config.mssql_server,
        )

        # Pre-create connections
        for _ in range(self._pool_size):
            try:
                conn = await self._create_connection()
                self._pool.append(conn)
            except Exception as e:
                logger.warning("Failed to pre-create connection: %s", e)
                break

        if not self._pool:
            raise RuntimeError(
                f"Could not create any MSSQL connections to "
                f"{self._config.mssql_server}"
            )

        self._started = True
        logger.info(
            "MSSQL pool ready: %d/%d connections",
            len(self._pool),
            self._pool_size,
        )

    async def stop(self):
        """Close all connections."""
        if not self._started:
            return

        logger.info("MSSQL pool shutting down...")
        self._started = False

        async with self._lock:
            while self._pool:
                conn = self._pool.popleft()
                try:
                    conn.close()
                except Exception:
                    pass

        logger.info("MSSQL pool closed")

    async def _create_connection(self):
        """Create a new pyodbc connection."""
        loop = asyncio.get_event_loop()
        conn = await loop.run_in_executor(
            None,
            lambda: self._pyodbc.connect(self._conn_str, timeout=self._config.mssql_conn_timeout),
        )
        conn.autocommit = True
        self._total_created += 1
        logger.debug("New MSSQL connection created (#%d)", self._total_created)
        return conn

    async def _get_connection(self):
        """Acquire a connection from the pool."""
        await self._available.acquire()

        async with self._lock:
            if self._pool:
                return self._pool.popleft()

        # Pool was empty despite semaphore — create a new one
        try:
            return await self._create_connection()
        except Exception:
            self._available.release()
            raise

    async def _return_connection(self, conn):
        """Return a connection to the pool, or close if broken."""
        if conn is None:
            return

        try:
            # Quick health check
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: conn.cursor().execute("SELECT 1").fetchone(),
            )
            async with self._lock:
                self._pool.append(conn)
            self._available.release()
        except Exception:
            # Connection is dead — close and don't return to pool
            try:
                conn.close()
            except Exception:
                pass
            self._available.release()
            logger.debug("Dead connection removed from pool")

    async def query(self, sql: str, params: list = None) -> list[dict]:
        """Execute a SELECT query and return rows as list of dicts.

        Args:
            sql: SQL query string (can use ? placeholders)
            params: Optional list of parameter values

        Returns:
            List of dicts, one per row.
        """
        if not self._started:
            raise RuntimeError("MSSQL adapter not started")

        conn = None
        try:
            conn = await self._get_connection()
            loop = asyncio.get_event_loop()

            def _execute():
                cursor = conn.cursor()
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)

                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        val = row[i]
                        # Convert non-serializable types
                        if isinstance(val, (bytes, bytearray)):
                            val = val.hex()
                        elif hasattr(val, "isoformat"):
                            val = val.isoformat()
                        row_dict[col] = val
                    result.append(row_dict)
                return result

            result = await asyncio.wait_for(
                loop.run_in_executor(None, _execute),
                timeout=self._config.mssql_query_timeout,
            )
            self._total_queries += 1
            return result

        except Exception as e:
            self._total_errors += 1
            logger.error("MSSQL query error: %s", e)
            raise
        finally:
            await self._return_connection(conn)

    async def exec(self, sql: str, params: list = None) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected row count.

        Args:
            sql: SQL statement
            params: Optional parameter values

        Returns:
            Number of affected rows.
        """
        if not self._started:
            raise RuntimeError("MSSQL adapter not started")

        conn = None
        try:
            conn = await self._get_connection()
            loop = asyncio.get_event_loop()

            def _execute():
                cursor = conn.cursor()
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                return cursor.rowcount

            affected = await asyncio.wait_for(
                loop.run_in_executor(None, _execute),
                timeout=self._config.mssql_query_timeout,
            )
            self._total_queries += 1
            return affected

        except Exception as e:
            self._total_errors += 1
            logger.error("MSSQL exec error: %s", e)
            raise
        finally:
            await self._return_connection(conn)

    async def health_check(self) -> dict:
        """Run a health check against MSSQL.

        Returns:
            Dict with status info.
        """
        info = {
            "status": "ok",
            "pool_size": self._pool_size,
            "pool_available": len(self._pool),
            "total_created": self._total_created,
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "server": self._config.mssql_server,
            "database": self._config.mssql_database,
        }

        if not self._started:
            info["status"] = "not_started"
            return info

        try:
            result = await self.query("SELECT 1 AS health")
            info["db_reachable"] = True
        except Exception as e:
            info["status"] = "degraded"
            info["db_reachable"] = False
            info["error"] = str(e)

        return info

    @property
    def stats(self) -> dict:
        """Return pool statistics."""
        return {
            "pool_size": self._pool_size,
            "pool_available": len(self._pool),
            "total_created": self._total_created,
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "started": self._started,
        }
