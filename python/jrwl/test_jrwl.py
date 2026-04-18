"""
JRWL Test Suite - Validates protocol, client, broker, and config.

Run with: python -m pytest python/jrwl/test_jrwl.py -v
Or:       python python/jrwl/test_jrwl.py
"""

import asyncio
import json
import os
import struct
import sys
import unittest
from pathlib import Path

# Ensure python dir is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jrwl.config import JRWLConfig
from jrwl.protocol import (
    JRWLProtocol, JRWLRequest, JRWLResponse,
    HEADER_SIZE, HEADER_FMT, MAX_MESSAGE_SIZE,
)
from jrwl.client import JRWLClient


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = JRWLConfig()
        self.assertEqual(c.tcp_host, "127.0.0.1")
        self.assertEqual(c.tcp_port, 9876)
        self.assertEqual(c.max_clients, 64)
        self.assertIn("unix", c.transport_type)

    def test_tcp_mode(self):
        c = JRWLConfig()
        c.unix_socket = ""
        self.assertEqual(c.transport_type, "tcp")
        self.assertEqual(c.transport_addr, ("127.0.0.1", 9876))

    def test_unix_mode(self):
        c = JRWLConfig()
        self.assertEqual(c.transport_type, "unix")
        self.assertTrue(c.transport_addr.endswith("jrwl.sock"))

    def test_conn_str(self):
        c = JRWLConfig()
        c.mssql_server = "testhost"
        c.mssql_password = "secret"
        cs = c.mssql_conn_str
        self.assertIn("testhost", cs)
        self.assertIn("secret", cs)
        self.assertIn("TrustServerCertificate", cs)


class TestProtocol(unittest.TestCase):
    def test_request_roundtrip(self):
        req = JRWLRequest(cmd="query", sql="SELECT 1", params=[42])
        data = req.to_json()
        parsed = JRWLRequest.from_json(data)
        self.assertEqual(parsed.cmd, "query")
        self.assertEqual(parsed.sql, "SELECT 1")
        self.assertEqual(parsed.params, [42])
        self.assertEqual(parsed.id, req.id)

    def test_response_roundtrip(self):
        resp = JRWLResponse(id="abc", ok=True, rows=[{"x": 1}], affected=1)
        data = resp.to_json()
        parsed = JRWLResponse.from_json(data)
        self.assertEqual(parsed.id, "abc")
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.rows, [{"x": 1}])
        self.assertEqual(parsed.affected, 1)

    def test_error_response(self):
        resp = JRWLResponse.error_response("xyz", "bad")
        self.assertFalse(resp.ok)
        self.assertEqual(resp.error, "bad")

    def test_health_response(self):
        resp = JRWLResponse.health_ok("h1", {"uptime": 42})
        self.assertTrue(resp.ok)
        self.assertEqual(resp.meta["uptime"], 42)

    def test_encode_decode_frame(self):
        payload = json.dumps({"test": True})
        frame = JRWLProtocol.encode(payload)
        self.assertEqual(len(frame), HEADER_SIZE + len(payload))
        length = struct.unpack(HEADER_FMT, frame[:HEADER_SIZE])[0]
        self.assertEqual(length, len(payload))

    def test_request_auto_id(self):
        req = JRWLRequest(cmd="health")
        self.assertTrue(len(req.id) > 0)


class TestAsyncTransport(unittest.TestCase):
    def test_tcp_roundtrip(self):
        async def run():
            config = JRWLConfig()
            config.unix_socket = ""
            config.tcp_port = 19899

            async def handler(reader, writer):
                try:
                    while not reader.at_eof():
                        try:
                            req = await JRWLProtocol.read_request(reader)
                        except Exception:
                            break
                        resp = JRWLResponse(
                            id=req.id, ok=True,
                            rows=[{"cmd": req.cmd, "sql": req.sql}],
                        )
                        await JRWLProtocol.send_response(writer, resp)
                finally:
                    writer.close()
                    await writer.wait_closed()

            server = await asyncio.start_server(handler, "127.0.0.1", config.tcp_port)
            client = JRWLClient(config)

            await client.connect()
            self.assertTrue(client.is_connected)

            rows = await client.query("SELECT 1")
            self.assertEqual(rows[0]["cmd"], "query")

            health = await client.health()
            # health returns resp.meta which is empty dict in our mock
            self.assertIsInstance(health, dict)

            affected = await client.exec("UPDATE t SET x=1")
            # exec returns resp.affected which is -1 in our mock (rows-based response)
            self.assertIsInstance(affected, int)

            await client.disconnect()
            self.assertFalse(client.is_connected)

            server.close()
            await server.wait_closed()

        asyncio.run(run())

    def test_unix_roundtrip(self):
        async def run():
            sock = "/tmp/jrwl_test_unit.sock"
            if os.path.exists(sock):
                os.unlink(sock)

            async def handler(reader, writer):
                try:
                    while not reader.at_eof():
                        try:
                            req = await JRWLProtocol.read_request(reader)
                        except Exception:
                            break
                        resp = JRWLResponse(id=req.id, ok=True, rows=[{"unix": True}])
                        await JRWLProtocol.send_response(writer, resp)
                finally:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass

            server = await asyncio.start_unix_server(handler, path=sock)

            config = JRWLConfig()
            config.unix_socket = sock
            client = JRWLClient(config)
            await client.connect()
            rows = await client.query("SELECT 1")
            self.assertEqual(rows[0]["unix"], True)
            await client.disconnect()

            server.close()
            await server.wait_closed()
            if os.path.exists(sock):
                os.unlink(sock)

        asyncio.run(run())

    def test_client_context_manager(self):
        async def run():
            config = JRWLConfig()
            config.unix_socket = ""
            config.tcp_port = 19900

            async def handler(reader, writer):
                try:
                    req = await JRWLProtocol.read_request(reader)
                    resp = JRWLResponse(id=req.id, ok=True, rows=[{"cm": True}])
                    await JRWLProtocol.send_response(writer, resp)
                finally:
                    writer.close()
                    await writer.wait_closed()

            server = await asyncio.start_server(handler, "127.0.0.1", config.tcp_port)

            async with JRWLClient(config) as client:
                rows = await client.query("SELECT 1")
                self.assertEqual(rows[0]["cm"], True)

            server.close()
            await server.wait_closed()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
