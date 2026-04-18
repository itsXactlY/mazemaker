"""
JRWL Broker - Host-side async message bus server.

Accepts connections from JRWLClients over UNIX socket or TCP.
Routes SQL queries to MSSQL via the connection pool adapter.

Pure Python + asyncio (MSSQL requires pyodbc at runtime).

Usage:
    from jrwl import JRWLBroker, JRWLConfig
    config = JRWLConfig()
    broker = JRWLBroker(config)
    await broker.serve_forever()
"""

import asyncio
import logging
import os
import signal
import socket
import time
from pathlib import Path
from typing import Optional

from jrwl.config import JRWLConfig
from jrwl.protocol import JRWLProtocol, JRWLRequest, JRWLResponse
from jrwl.mssql_adapter import MSSQLAdapter

logger = logging.getLogger("jrwl.broker")


class JRWLBroker:
    """Async JRWL message bus broker.

    Listens for client connections and dispatches SQL commands
    to the MSSQL backend.
    """

    def __init__(self, config: JRWLConfig = None):
        self._config = config or JRWLConfig()
        self._server: Optional[asyncio.AbstractServer] = None
        self._adapter: Optional[MSSQLAdapter] = None
        self._clients: set = set()
        self._running = False
        self._start_time = 0.0
        self._requests_handled = 0

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self._config.log_level, logging.INFO),
            format="[%(name)s] %(levelname)s: %(message)s",
        )

    async def start(self):
        """Initialize broker: start MSSQL adapter, bind listener."""
        if self._running:
            return

        logger.info("JRWL Broker starting: %s", self._config)

        # Start MSSQL adapter
        self._adapter = MSSQLAdapter(self._config)
        try:
            await self._adapter.start()
        except Exception as e:
            logger.error("Failed to start MSSQL adapter: %s", e)
            raise

        # Create server
        if self._config.transport_type == "unix":
            await self._start_unix_server()
        else:
            await self._start_tcp_server()

        self._start_time = time.time()
        self._running = True
        logger.info("JRWL Broker ready")

    async def _start_unix_server(self):
        """Start UNIX socket server."""
        sock_path = self._config.unix_socket
        # Ensure parent directory exists
        Path(sock_path).parent.mkdir(parents=True, exist_ok=True)
        # Remove stale socket file
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        self._server = await asyncio.start_unix_server(
            self._handle_client, path=sock_path
        )
        # Set socket permissions
        os.chmod(sock_path, 0o660)
        logger.info("Listening on UNIX socket: %s", sock_path)

    async def _start_tcp_server(self):
        """Start TCP server."""
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self._config.tcp_host,
            port=self._config.tcp_port,
        )
        logger.info(
            "Listening on TCP: %s:%d",
            self._config.tcp_host,
            self._config.tcp_port,
        )

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Handle a connected client."""
        peer = writer.get_extra_info("peername") or "unix"
        client_id = id(writer)
        self._clients.add(client_id)
        logger.debug("Client connected: %s (active: %d)", peer, len(self._clients))

        try:
            while not reader.at_eof():
                try:
                    req = await asyncio.wait_for(
                        JRWLProtocol.read_request(reader),
                        timeout=self._config.request_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.debug("Client timeout: %s", peer)
                    break
                except (asyncio.IncompleteReadError, ConnectionError):
                    break

                resp = await self._dispatch(req)
                self._requests_handled += 1

                try:
                    await JRWLProtocol.send_response(writer, resp)
                except (ConnectionError, BrokenPipeError):
                    break

        except Exception as e:
            logger.error("Client handler error (%s): %s", peer, e)
        finally:
            self._clients.discard(client_id)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("Client disconnected: %s", peer)

    async def _dispatch(self, req: JRWLRequest) -> JRWLResponse:
        """Dispatch a request to the appropriate handler."""
        try:
            handler = {
                "query": self._handle_query,
                "exec": self._handle_exec,
                "health": self._handle_health,
            }.get(req.cmd)

            if handler is None:
                return JRWLResponse.error_response(
                    req.id, f"Unknown command: {req.cmd}"
                )

            return await handler(req)

        except asyncio.TimeoutError:
            return JRWLResponse.error_response(req.id, "Query timeout")
        except Exception as e:
            logger.error("Dispatch error for cmd=%s: %s", req.cmd, e)
            return JRWLResponse.error_response(req.id, str(e))

    async def _handle_query(self, req: JRWLRequest) -> JRWLResponse:
        """Handle a SELECT query."""
        if not req.sql:
            return JRWLResponse.error_response(req.id, "No SQL provided")

        rows = await self._adapter.query(req.sql, req.params)
        return JRWLResponse(id=req.id, ok=True, rows=rows)

    async def _handle_exec(self, req: JRWLRequest) -> JRWLResponse:
        """Handle INSERT/UPDATE/DELETE."""
        if not req.sql:
            return JRWLResponse.error_response(req.id, "No SQL provided")

        affected = await self._adapter.exec(req.sql, req.params)
        return JRWLResponse(id=req.id, ok=True, affected=affected)

    async def _handle_health(self, req: JRWLRequest) -> JRWLResponse:
        """Handle health check."""
        info = await self._adapter.health_check()
        info["uptime_seconds"] = round(time.time() - self._start_time, 1)
        info["active_clients"] = len(self._clients)
        info["requests_handled"] = self._requests_handled
        return JRWLResponse.health_ok(req.id, info)

    async def serve_forever(self):
        """Start the broker and run until interrupted."""
        await self.start()

        logger.info("JRWL Broker serving. Press Ctrl+C to stop.")

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def _signal_handler():
            logger.info("Shutdown signal received")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Graceful shutdown."""
        if not self._running:
            return

        logger.info("JRWL Broker stopping...")
        self._running = False

        # Close server listener
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Stop MSSQL adapter
        if self._adapter:
            await self._adapter.stop()

        # Clean up UNIX socket
        if (
            self._config.transport_type == "unix"
            and os.path.exists(self._config.unix_socket)
        ):
            os.unlink(self._config.unix_socket)

        logger.info("JRWL Broker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "uptime": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "active_clients": len(self._clients),
            "requests_handled": self._requests_handled,
            "adapter_stats": self._adapter.stats if self._adapter else None,
        }
