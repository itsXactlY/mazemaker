"""
JRWL Client - VM-side async message bus client.

Connects to JRWL Broker over UNIX socket or TCP.
Provides query(), exec(), health() methods with auto-reconnect.

Pure Python + asyncio, zero dependencies.

Usage:
    from jrwl import JRWLClient, JRWLConfig

    client = JRWLClient()
    await client.connect()
    rows = await client.query("SELECT TOP 5 * FROM GraphNodes")
    affected = await client.exec("UPDATE T SET x=1 WHERE id=?", [42])
    status = await client.health()
    await client.disconnect()
"""

import asyncio
import logging
from typing import Optional

from jrwl.config import JRWLConfig
from jrwl.protocol import JRWLProtocol, JRWLRequest, JRWLResponse

logger = logging.getLogger("jrwl.client")


class JRWLClient:
    """Async client for JRWL message bus.

    Features:
        - UNIX socket or TCP transport
        - Auto-reconnect on connection loss
        - Configurable timeouts
        - Connection pooling for concurrent requests
    """

    def __init__(self, config: JRWLConfig = None):
        self._config = config or JRWLConfig()
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self):
        """Connect to the JRWL broker.

        Raises:
            ConnectionError: if connection fails after retries.
        """
        if self._connected:
            return

        last_error = None
        for attempt in range(1, self._config.reconnect_attempts + 1):
            try:
                await self._do_connect()
                self._connected = True
                logger.info("Connected to JRWL broker")
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "Connection attempt %d/%d failed: %s",
                    attempt,
                    self._config.reconnect_attempts,
                    e,
                )
                if attempt < self._config.reconnect_attempts:
                    await asyncio.sleep(self._config.reconnect_delay)

        raise ConnectionError(
            f"Failed to connect after {self._config.reconnect_attempts} "
            f"attempts: {last_error}"
        )

    async def _do_connect(self):
        """Perform the actual connection."""
        if self._config.transport_type == "unix":
            self._reader, self._writer = await asyncio.open_unix_connection(
                self._config.unix_socket
            )
        else:
            self._reader, self._writer = await asyncio.open_connection(
                self._config.tcp_host, self._config.tcp_port
            )

    async def disconnect(self):
        """Disconnect from the broker."""
        if not self._connected:
            return

        self._connected = False
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        logger.debug("Disconnected from JRWL broker")

    async def _ensure_connected(self):
        """Ensure we have a live connection, reconnect if needed."""
        if self._connected and self._reader and not self._reader.at_eof():
            return

        logger.info("Connection lost, reconnecting...")
        self._connected = False
        await self.disconnect()
        await self.connect()

    async def _request(self, req: JRWLRequest) -> JRWLResponse:
        """Send a request and wait for the response.

        Handles reconnection and serialization of requests.
        """
        timeout = req.timeout or self._config.client_timeout

        async with self._lock:
            await self._ensure_connected()

            try:
                await asyncio.wait_for(
                    JRWLProtocol.send_request(self._writer, req),
                    timeout=timeout,
                )
                resp = await asyncio.wait_for(
                    JRWLProtocol.read_response(self._reader),
                    timeout=timeout,
                )
                return resp

            except (asyncio.IncompleteReadError, ConnectionError, BrokenPipeError):
                # Connection dropped — try once more
                logger.warning("Connection error during request, retrying...")
                self._connected = False
                await self._ensure_connected()

                await JRWLProtocol.send_request(self._writer, req)
                resp = await JRWLProtocol.read_response(self._reader)
                return resp

    async def query(self, sql: str, params: list = None) -> list[dict]:
        """Execute a SELECT query.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            List of row dicts.
        """
        req = JRWLRequest(cmd="query", sql=sql, params=params or [])
        resp = await self._request(req)

        if not resp.ok:
            raise RuntimeError(f"JRWL query error: {resp.error}")

        return resp.rows

    async def exec(self, sql: str, params: list = None) -> int:
        """Execute INSERT/UPDATE/DELETE.

        Args:
            sql: SQL statement
            params: Optional parameters

        Returns:
            Number of affected rows.
        """
        req = JRWLRequest(cmd="exec", sql=sql, params=params or [])
        resp = await self._request(req)

        if not resp.ok:
            raise RuntimeError(f"JRWL exec error: {resp.error}")

        return resp.affected

    async def health(self) -> dict:
        """Check broker and backend health.

        Returns:
            Dict with health status info.
        """
        req = JRWLRequest(cmd="health")
        resp = await self._request(req)

        if not resp.ok:
            return {"status": "error", "error": resp.error}

        return resp.meta

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self._connected and self._reader is not None and not self._reader.at_eof()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    def __repr__(self):
        mode = self._config.transport_type
        addr = self._config.transport_addr
        state = "connected" if self._connected else "disconnected"
        return f"JRWLClient({mode}={addr}, {state})"
