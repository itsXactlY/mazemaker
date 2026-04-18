#!/usr/bin/env bash
# ============================================================
# Start DLM Client — connects to JackrabbitDLM server on host
# Registers as 'neural-memory-smolvm' node
# Port: 37373 (JackrabbitDLM standard)
# ============================================================
set -euo pipefail

DLM_HOST="${DLM_HOST:-host.docker.internal}"
DLM_PORT="${DLM_PORT:-37373}"
DATA_DIR="${SMOLVM_DATA_DIR:-/app/data}"
LOG_FILE="${SMOLVM_LOG_DIR:-/app/logs}/dlm_client.log"
REG_FLAG="$DATA_DIR/dlm_registered"

mkdir -p "$(dirname "$LOG_FILE")" "$DATA_DIR"
rm -f "$REG_FLAG"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [dlm-client] $*" | tee -a "$LOG_FILE"
}

log "Starting DLM client — target=${DLM_HOST}:${DLM_PORT}"

python3 -c "
import os
import sys
import json
import socket
import hashlib
import logging
import time
import struct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [dlm-client] %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.environ.get('SMOLVM_LOG_DIR', '/app/logs') + '/dlm_client.log')
    ]
)
log = logging.getLogger('dlm-client')

# JackrabbitDLM protocol constants
DLM_MAGIC = b'DLMX'
DLM_VERSION = 1

# Message types
MSG_REGISTER = 0x01
MSG_HEARTBEAT = 0x02
MSG_DATA = 0x03
MSG_QUERY = 0x04
MSG_RESPONSE = 0x10
MSG_ERROR = 0x11
MSG_ACK = 0x12

# Header format: magic(4) + version(1) + msg_type(1) + flags(1) + reserved(1) + payload_len(4) = 12 bytes
HEADER_FMT = '>4sBBBI'

NODE_IDENTITY = 'neural-memory-smolvm'
NODE_CAPABILITIES = ['embedding', 'sqlite-memory', 'recall', 'similarity']


def encode_message(msg_type, payload, flags=0):
    \"\"\"Encode a JackrabbitDLM message.\"\"\"
    body = json.dumps(payload).encode('utf-8')
    header = struct.pack(HEADER_FMT, DLM_MAGIC, DLM_VERSION, msg_type, flags, len(body))
    return header + body


def recv_exact(sock, n):
    \"\"\"Read exactly n bytes from socket.\"\"\"
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('DLM connection closed')
        buf.extend(chunk)
    return bytes(buf)


def decode_message(sock):
    \"\"\"Read and decode a JackrabbitDLM message.\"\"\"
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


class DLMClient:
    def __init__(self):
        self.host = os.environ.get('DLM_HOST', '${DLM_HOST}')
        self.port = int(os.environ.get('DLM_PORT', '${DLM_PORT}'))
        self.data_dir = os.environ.get('SMOLVM_DATA_DIR', '${DATA_DIR}')
        self.sock = None
        self.connected = False
        self.registered = False

    def connect(self):
        \"\"\"Connect to JackrabbitDLM server.\"\"\"
        log.info(f'Connecting to DLM server at {self.host}:{self.port}')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect((self.host, self.port))
        self.connected = True
        log.info('TCP connection established')

    def register(self):
        \"\"\"Register this node with the DLM server.\"\"\"
        payload = {
            'identity': NODE_IDENTITY,
            'node_type': 'neural-memory',
            'capabilities': NODE_CAPABILITIES,
            'endpoints': {
                'embed': f'http://localhost:{os.environ.get(\"SMOLVM_EMBED_PORT\", \"8501\")}',
                'health': f'http://localhost:{os.environ.get(\"SMOLVM_EMBED_PORT\", \"8501\")}/health'
            },
            'version': '1.0.0'
        }
        self.sock.sendall(encode_message(MSG_REGISTER, payload))
        msg_type, flags, resp = decode_message(self.sock)

        if msg_type == MSG_ACK:
            self.registered = True
            log.info(f'Registration ACK — node_id={resp.get(\"node_id\", \"unknown\")}')

            # Write registration flag for init.sh
            flag_path = os.path.join(self.data_dir, 'dlm_registered')
            with open(flag_path, 'w') as f:
                f.write(json.dumps({
                    'identity': NODE_IDENTITY,
                    'registered_at': time.time(),
                    'node_id': resp.get('node_id', ''),
                    'server': f'{self.host}:{self.port}'
                }))
            return True
        elif msg_type == MSG_ERROR:
            log.error(f'Registration rejected: {resp.get(\"message\", \"unknown error\")}')
            return False
        else:
            log.warning(f'Unexpected response type: {msg_type}')
            return False

    def health_loop(self):
        \"\"\"Send periodic heartbeats and handle incoming requests.\"\"\"
        last_heartbeat = 0
        heartbeat_interval = 30

        while True:
            try:
                now = time.time()

                # Send heartbeat
                if now - last_heartbeat >= heartbeat_interval:
                    self.sock.settimeout(2)
                    try:
                        self.sock.sendall(encode_message(MSG_HEARTBEAT, {
                            'identity': NODE_IDENTITY,
                            'timestamp': now,
                            'status': 'healthy'
                        }))
                        msg_type, _, _ = decode_message(self.sock)
                        if msg_type == MSG_ACK:
                            log.debug('Heartbeat ACK')
                        last_heartbeat = now
                    except socket.timeout:
                        pass

                # Check for incoming requests (non-blocking)
                self.sock.settimeout(1.0)
                try:
                    msg_type, flags, payload = decode_message(self.sock)
                    self._handle_request(msg_type, payload)
                except socket.timeout:
                    continue

            except ConnectionError:
                log.error('DLM connection lost')
                self.connected = False
                break
            except Exception as e:
                log.error(f'Health loop error: {e}')
                time.sleep(5)

    def _handle_request(self, msg_type, payload):
        \"\"\"Handle incoming DLM requests.\"\"\"
        action = payload.get('action', 'unknown')
        log.info(f'Received request: type=0x{msg_type:02x} action={action}')

        if action == 'embed':
            # Forward to embed server
            import urllib.request
            try:
                texts = payload.get('texts', [])
                embed_port = os.environ.get('SMOLVM_EMBED_PORT', '8501')
                req = urllib.request.Request(
                    f'http://localhost:{embed_port}/embed',
                    data=json.dumps({'texts': texts}).encode(),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                self.sock.sendall(encode_message(MSG_RESPONSE, result))
            except Exception as e:
                self.sock.sendall(encode_message(MSG_ERROR, {'message': str(e)}))

        elif action == 'health':
            self.sock.sendall(encode_message(MSG_RESPONSE, {
                'status': 'ok',
                'identity': NODE_IDENTITY,
                'registered': self.registered
            }))

        else:
            self.sock.sendall(encode_message(MSG_ERROR, {
                'message': f'Unknown action: {action}'
            }))

    def run(self):
        \"\"\"Main loop with reconnection.\"\"\"
        retries = 0
        max_retries = 10

        while retries < max_retries:
            try:
                self.connect()
                if self.register():
                    log.info('DLM client fully operational')
                    retries = 0
                    self.health_loop()
                else:
                    log.warning('Registration failed, will retry')
            except ConnectionError as e:
                log.error(f'Connection error: {e}')
            except Exception as e:
                log.error(f'Unexpected error: {e}')

            retries += 1
            wait = min(2 ** retries, 60)
            log.info(f'Retry {retries}/{max_retries} in {wait}s')
            time.sleep(wait)

        log.error('Max retries exceeded, DLM client exiting')
        sys.exit(1)


if __name__ == '__main__':
    client = DLMClient()
    client.run()
" 2>&1 | tee -a "$LOG_FILE"
