#!/usr/bin/env python3
"""
dlm_mssql_bridge.py — JackrabbitDLM server with MSSQL backend.

Runs on the host, listens on TCP port 37373 (JackrabbitDLM standard).
Accepts DLM node registrations, routes queries to MSSQL, and bridges
embedding requests to the SmolVM embed server.

Protocol: JackrabbitDLM (DLMX wire format)
  Header: magic(4='DLMX') + version(1) + msg_type(1) + flags(1) + reserved(1) + payload_len(4) = 12 bytes
  Body: JSON-encoded payload
"""

import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DLM_BIND = os.environ.get('DLM_BIND', '0.0.0.0')
DLM_PORT = int(os.environ.get('DLM_PORT', '37373'))
LOG_FILE = os.environ.get('LOG_FILE', 'dlm_bridge.log')
DATA_DIR = os.environ.get('SMOLVM_DATA_DIR', './data')

MSSQL_DRIVER = os.environ.get('MSSQL_DRIVER', 'ODBC Driver 18 for SQL Server')
MSSQL_HOST = os.environ.get('MSSQL_HOST', 'localhost')
MSSQL_PORT = os.environ.get('MSSQL_PORT', '1433')
MSSQL_USER = os.environ.get('MSSQL_USER', 'sa')
MSSQL_PASS = os.environ.get('MSSQL_PASS', '')
MSSQL_DB = os.environ.get('MSSQL_DB', 'NeuralMemory')

EMBED_PORT = os.environ.get('SMOLVM_EMBED_PORT', '8501')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [dlm-bridge] %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ]
)
log = logging.getLogger('dlm-bridge')

# ---------------------------------------------------------------------------
# JackrabbitDLM Protocol
# ---------------------------------------------------------------------------

DLM_MAGIC = b'DLMX'
DLM_VERSION = 1

# Message types
MSG_REGISTER   = 0x01
MSG_HEARTBEAT  = 0x02
MSG_DATA       = 0x03
MSG_QUERY      = 0x04
MSG_RESPONSE   = 0x10
MSG_ERROR      = 0x11
MSG_ACK        = 0x12

# Header: magic(4) + version(1) + msg_type(1) + flags(1) + reserved(1) + payload_len(4) = 12 bytes
HEADER_FMT = '>4sBBBI'

# Registered nodes
registered_nodes: Dict[str, Dict[str, Any]] = {}
node_counter = 0
node_lock = threading.Lock()


def encode_message(msg_type: int, payload: dict, flags: int = 0) -> bytes:
    """Encode a JackrabbitDLM message."""
    body = json.dumps(payload).encode('utf-8')
    header = struct.pack(HEADER_FMT, DLM_MAGIC, DLM_VERSION, msg_type, flags, len(body))
    return header + body


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('Connection closed')
        buf.extend(chunk)
    return bytes(buf)


def decode_message(sock: socket.socket) -> Tuple[int, int, dict]:
    """Read and decode a JackrabbitDLM message."""
    header = recv_exact(sock, 12)
    magic, version, msg_type, flags, payload_len = struct.unpack(HEADER_FMT, header)
    if magic != DLM_MAGIC:
        raise ValueError(f'Bad DLM magic: {magic!r}')
    if version != DLM_VERSION:
        raise ValueError(f'Unsupported DLM version: {version}')
    payload = {}
    if payload_len > 0:
        body = recv_exact(sock, payload_len)
        payload = json.loads(body.decode('utf-8'))
    return msg_type, flags, payload


# ---------------------------------------------------------------------------
# MSSQL Backend
# ---------------------------------------------------------------------------

class MSSQLBackend:
    """Handles MSSQL operations for the DLM bridge."""

    def __init__(self):
        self.conn_str = (
            f'DRIVER={{{MSSQL_DRIVER}}};'
            f'SERVER={MSSQL_HOST},{MSSQL_PORT};'
            f'DATABASE={MSSQL_DB};'
            f'UID={MSSQL_USER};'
            f'PWD={MSSQL_PASS};'
            'TrustServerCertificate=yes;'
        )

    def get_connection(self):
        import pyodbc
        try:
            return pyodbc.connect(self.conn_str, timeout=10)
        except Exception as e:
            log.error(f'MSSQL connection failed: {e}')
            return None

    def execute_query(self, sql: str, params: list = None) -> dict:
        conn = self.get_connection()
        if not conn:
            return {'error': 'MSSQL unavailable'}
        try:
            cur = conn.cursor()
            cur.execute(sql, params or [])
            if cur.description:
                columns = [d[0] for d in cur.description]
                rows = [dict(zip(columns, row)) for row in cur.fetchall()]
                return {'rows': rows, 'count': len(rows)}
            else:
                conn.commit()
                return {'affected': cur.rowcount}
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()

    def handle_store(self, payload: dict) -> dict:
        sql = (
            "INSERT INTO memories (content, embedding, metadata, namespace) "
            "OUTPUT INSERTED.id "
            "VALUES (?, ?, ?, ?)"
        )
        return self.execute_query(sql, [
            payload.get('content', ''),
            json.dumps(payload.get('embedding', [])),
            json.dumps(payload.get('metadata', {})),
            payload.get('namespace', 'default'),
        ])

    def handle_query(self, payload: dict) -> dict:
        sql = payload.get('sql', '')
        params = payload.get('params', [])
        if not sql:
            return {'error': 'No SQL provided'}
        return self.execute_query(sql, params)

    def health_check(self) -> dict:
        conn = self.get_connection()
        if conn:
            conn.close()
            return {'status': 'ok', 'mssql': True}
        return {'status': 'degraded', 'mssql': False}


mssql = MSSQLBackend()


# ---------------------------------------------------------------------------
# Node Registration
# ---------------------------------------------------------------------------

def register_node(identity: str, node_type: str, capabilities: list,
                  endpoints: dict, version: str) -> dict:
    """Register a DLM node and return node_id."""
    global node_counter
    with node_lock:
        node_counter += 1
        node_id = f'node-{node_counter:04d}'
        registered_nodes[node_id] = {
            'identity': identity,
            'node_type': node_type,
            'capabilities': capabilities,
            'endpoints': endpoints,
            'version': version,
            'registered_at': time.time(),
            'last_heartbeat': time.time(),
        }
    log.info(f'Node registered: {node_id} ({identity}) caps={capabilities}')
    return {'status': 'registered', 'node_id': node_id}


def update_heartbeat(identity: str) -> None:
    """Update last heartbeat for a node."""
    with node_lock:
        for nid, info in registered_nodes.items():
            if info['identity'] == identity:
                info['last_heartbeat'] = time.time()
                break


# ---------------------------------------------------------------------------
# Request Handler
# ---------------------------------------------------------------------------

def handle_request(msg_type: int, payload: dict) -> Tuple[int, dict]:
    """Route a DLM message to the appropriate handler."""

    if msg_type == MSG_REGISTER:
        result = register_node(
            identity=payload.get('identity', 'unknown'),
            node_type=payload.get('node_type', 'unknown'),
            capabilities=payload.get('capabilities', []),
            endpoints=payload.get('endpoints', {}),
            version=payload.get('version', '0.0.0'),
        )
        return MSG_ACK, result

    elif msg_type == MSG_HEARTBEAT:
        identity = payload.get('identity', '')
        update_heartbeat(identity)
        return MSG_ACK, {'status': 'ok', 'timestamp': time.time()}

    elif msg_type == MSG_QUERY:
        action = payload.get('action', '')
        if action == 'health':
            result = mssql.health_check()
            result['nodes'] = len(registered_nodes)
            return MSG_RESPONSE, result
        elif action == 'store':
            return MSG_RESPONSE, mssql.handle_store(payload)
        elif action == 'query':
            return MSG_RESPONSE, mssql.handle_query(payload)
        elif action == 'nodes':
            with node_lock:
                return MSG_RESPONSE, {
                    'nodes': [
                        {'node_id': nid, **info}
                        for nid, info in registered_nodes.items()
                    ]
                }
        else:
            return MSG_ERROR, {'message': f'Unknown query action: {action}'}

    elif msg_type == MSG_DATA:
        action = payload.get('action', '')
        if action == 'embed':
            # Forward to embed server
            import urllib.request
            try:
                texts = payload.get('texts', [])
                req = urllib.request.Request(
                    f'http://localhost:{EMBED_PORT}/embed',
                    data=json.dumps({'texts': texts}).encode(),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return MSG_RESPONSE, json.loads(resp.read())
            except Exception as e:
                return MSG_ERROR, {'message': f'Embed error: {e}'}
        elif action == 'sql':
            return MSG_RESPONSE, mssql.handle_query(payload)
        else:
            return MSG_ERROR, {'message': f'Unknown data action: {action}'}

    else:
        return MSG_ERROR, {'message': f'Unknown message type: 0x{msg_type:02x}'}


# ---------------------------------------------------------------------------
# Connection Handler
# ---------------------------------------------------------------------------

def handle_client(conn: socket.socket, addr: tuple) -> None:
    """Handle a single DLM client connection."""
    log.info(f'Client connected: {addr}')
    try:
        while True:
            msg_type, flags, payload = decode_message(conn)
            log.debug(f'RX from {addr}: type=0x{msg_type:02x} action={payload.get("action", "n/a")}')

            resp_type, resp_payload = handle_request(msg_type, payload)
            conn.sendall(encode_message(resp_type, resp_payload))
    except ConnectionError:
        log.info(f'Client disconnected: {addr}')
    except Exception as e:
        log.error(f'Client handler error ({addr}): {e}')
        try:
            conn.sendall(encode_message(MSG_ERROR, {'message': str(e)}))
        except Exception:
            pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def main() -> None:
    log.info('============================================')
    log.info('  JackrabbitDLM Bridge Server')
    log.info(f'  Listening: {DLM_BIND}:{DLM_PORT}')
    log.info(f'  MSSQL: {MSSQL_HOST}:{MSSQL_PORT}/{MSSQL_DB}')
    log.info('============================================')

    # Test MSSQL on startup
    health = mssql.health_check()
    if health['mssql']:
        log.info('MSSQL connection: OK')
    else:
        log.warning('MSSQL not reachable — will retry on requests')

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((DLM_BIND, DLM_PORT))
    server.listen(20)
    log.info(f'JackrabbitDLM server listening on {DLM_BIND}:{DLM_PORT}')

    def shutdown(signum, frame):
        log.info('Shutting down DLM server...')
        server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        try:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            log.error(f'Accept error: {e}')


if __name__ == '__main__':
    main()
