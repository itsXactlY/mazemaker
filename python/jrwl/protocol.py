"""
JRWL Protocol - Length-prefixed JSON over stream transport.

Wire format:
    [4 bytes big-endian length][JSON payload]

Request payload:
    {"id": "uuid", "cmd": "query|exec|health|subscribe", "sql": "...", "params": [...]}

Response payload:
    {"id": "uuid", "ok": true, "rows": [...], "affected": N, "error": null}
    {"id": "uuid", "ok": false, "error": "message"}

Pure Python, zero dependencies.
"""

import json
import struct
import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# Max message size: 16 MB
MAX_MESSAGE_SIZE = 16 * 1024 * 1024
HEADER_FMT = "!I"  # 4 bytes, big-endian unsigned int
HEADER_SIZE = struct.calcsize(HEADER_FMT)


@dataclass
class JRWLRequest:
    """A JRWL request message."""

    cmd: str  # "query", "exec", "health", "subscribe"
    sql: str = ""
    params: list = field(default_factory=list)
    id: str = ""
    timeout: float = 0.0  # 0 = use default

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.cmd:
            raise ValueError("cmd is required")

    def to_json(self) -> str:
        d = {"id": self.id, "cmd": self.cmd}
        if self.sql:
            d["sql"] = self.sql
        if self.params:
            d["params"] = self.params
        if self.timeout > 0:
            d["timeout"] = self.timeout
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> "JRWLRequest":
        d = json.loads(data)
        return cls(
            cmd=d["cmd"],
            sql=d.get("sql", ""),
            params=d.get("params", []),
            id=d.get("id", ""),
            timeout=d.get("timeout", 0.0),
        )


@dataclass
class JRWLResponse:
    """A JRWL response message."""

    id: str
    ok: bool = True
    rows: list = field(default_factory=list)
    affected: int = -1
    error: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = {"id": self.id, "ok": self.ok}
        if self.rows:
            d["rows"] = self.rows
        if self.affected >= 0:
            d["affected"] = self.affected
        if self.error:
            d["error"] = self.error
        if self.meta:
            d["meta"] = self.meta
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> "JRWLResponse":
        d = json.loads(data)
        return cls(
            id=d["id"],
            ok=d.get("ok", True),
            rows=d.get("rows", []),
            affected=d.get("affected", -1),
            error=d.get("error", ""),
            meta=d.get("meta", {}),
        )

    @classmethod
    def error_response(cls, req_id: str, error: str) -> "JRWLResponse":
        """Create an error response."""
        return cls(id=req_id, ok=False, error=error)

    @classmethod
    def health_ok(cls, req_id: str, info: dict = None) -> "JRWLResponse":
        """Create a health-check OK response."""
        return cls(id=req_id, ok=True, meta=info or {"status": "ok"})


class JRWLProtocol:
    """Low-level protocol helpers for length-prefixed JSON over asyncio streams.

    All methods are static/stateless — usable on both client and broker side.
    """

    @staticmethod
    def encode(payload: str) -> bytes:
        """Encode a JSON string into a length-prefixed frame."""
        data = payload.encode("utf-8")
        if len(data) > MAX_MESSAGE_SIZE:
            raise ValueError(
                f"Message too large: {len(data)} > {MAX_MESSAGE_SIZE}"
            )
        header = struct.pack(HEADER_FMT, len(data))
        return header + data

    @staticmethod
    async def read_frame(reader: asyncio.StreamReader) -> str:
        """Read one length-prefixed JSON frame from stream.

        Returns the decoded JSON string.
        Raises:
            ConnectionError: if the connection is closed.
            ValueError: if the frame is invalid.
        """
        # Read 4-byte header
        header = await reader.readexactly(HEADER_SIZE)
        (length,) = struct.unpack(HEADER_FMT, header)

        if length > MAX_MESSAGE_SIZE:
            raise ValueError(f"Frame too large: {length} > {MAX_MESSAGE_SIZE}")
        if length == 0:
            raise ValueError("Empty frame")

        # Read payload
        payload = await reader.readexactly(length)
        return payload.decode("utf-8")

    @staticmethod
    async def write_frame(writer: asyncio.StreamWriter, payload: str):
        """Write one length-prefixed JSON frame to stream."""
        frame = JRWLProtocol.encode(payload)
        writer.write(frame)
        await writer.drain()

    @staticmethod
    async def send_request(
        writer: asyncio.StreamWriter, req: JRWLRequest
    ):
        """Encode and send a request."""
        await JRWLProtocol.write_frame(writer, req.to_json())

    @staticmethod
    async def send_response(
        writer: asyncio.StreamWriter, resp: JRWLResponse
    ):
        """Encode and send a response."""
        await JRWLProtocol.write_frame(writer, resp.to_json())

    @staticmethod
    async def read_request(reader: asyncio.StreamReader) -> JRWLRequest:
        """Read and parse a request."""
        data = await JRWLProtocol.read_frame(reader)
        return JRWLRequest.from_json(data)

    @staticmethod
    async def read_response(reader: asyncio.StreamReader) -> JRWLResponse:
        """Read and parse a response."""
        data = await JRWLProtocol.read_frame(reader)
        return JRWLResponse.from_json(data)
