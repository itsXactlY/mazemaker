"""
JRWL Configuration - all settings in one place.
Zero dependencies, pure stdlib.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JRWLConfig:
    """JackRabbitWonderland configuration.

    Reads from environment variables with sensible defaults.
    """

    # --- Transport ---
    # UNIX socket path (default: ~/.neural_memory/jrwl.sock)
    unix_socket: str = ""
    # TCP host:port (used when unix_socket is empty)
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 9876

    # --- Broker ---
    # Max concurrent client connections
    max_clients: int = 64
    # Request timeout in seconds
    request_timeout: float = 30.0
    # Graceful shutdown drain timeout
    shutdown_timeout: float = 5.0

    # --- MSSQL connection pool (broker-side) ---
    mssql_server: str = ""
    mssql_database: str = "NeuralMemory"
    mssql_username: str = "sa"
    mssql_password: str = ""
    mssql_driver: str = "ODBC Driver 18 for SQL Server"
    mssql_pool_size: int = 4
    mssql_conn_timeout: int = 10
    mssql_query_timeout: int = 60
    mssql_trust_cert: bool = True

    # --- Client ---
    # Reconnect attempts on connection loss
    reconnect_attempts: int = 3
    # Delay between reconnect attempts (seconds)
    reconnect_delay: float = 1.0
    # Client-side request timeout
    client_timeout: float = 30.0

    # --- Logging ---
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

    def __post_init__(self):
        """Load from environment variables."""
        self.unix_socket = os.environ.get(
            "JRWL_SOCKET",
            self.unix_socket or str(Path.home() / ".neural_memory" / "jrwl.sock"),
        )
        self.tcp_host = os.environ.get("JRWL_TCP_HOST", self.tcp_host)
        self.tcp_port = int(os.environ.get("JRWL_TCP_PORT", str(self.tcp_port)))
        self.max_clients = int(os.environ.get("JRWL_MAX_CLIENTS", str(self.max_clients)))
        self.request_timeout = float(
            os.environ.get("JRWL_REQUEST_TIMEOUT", str(self.request_timeout))
        )

        # MSSQL settings (align with existing env vars)
        self.mssql_server = os.environ.get("MSSQL_SERVER", self.mssql_server)
        self.mssql_database = os.environ.get("MSSQL_DATABASE", self.mssql_database)
        self.mssql_username = os.environ.get("MSSQL_USER", self.mssql_username)
        self.mssql_password = os.environ.get("MSSQL_PASSWORD", self.mssql_password)
        self.mssql_driver = os.environ.get("MSSQL_DRIVER", self.mssql_driver)
        self.mssql_pool_size = int(
            os.environ.get("MSSQL_POOL_SIZE", str(self.mssql_pool_size))
        )
        self.mssql_trust_cert = os.environ.get(
            "MSSQL_TRUST_CERT", str(self.mssql_trust_cert)
        ).lower() in ("1", "true", "yes")

        # Client
        self.reconnect_attempts = int(
            os.environ.get("JRWL_RECONNECT_ATTEMPTS", str(self.reconnect_attempts))
        )
        self.client_timeout = float(
            os.environ.get("JRWL_CLIENT_TIMEOUT", str(self.client_timeout))
        )

        self.log_level = os.environ.get("JRWL_LOG_LEVEL", self.log_level).upper()

    @property
    def mssql_conn_str(self) -> str:
        """Build ODBC connection string."""
        parts = [
            f"DRIVER={{{self.mssql_driver}}}",
            f"SERVER={self.mssql_server}",
            f"DATABASE={self.mssql_database}",
            f"UID={self.mssql_username}",
            f"PWD={self.mssql_password}",
        ]
        if self.mssql_trust_cert:
            parts.append("TrustServerCertificate=yes")
        parts.append(f"Connection Timeout={self.mssql_conn_timeout}")
        return ";".join(parts)

    @property
    def transport_type(self) -> str:
        """Return 'unix' or 'tcp' based on config."""
        if self.unix_socket:
            return "unix"
        return "tcp"

    @property
    def transport_addr(self) -> tuple | str:
        """Return bind/connect address."""
        if self.transport_type == "unix":
            return self.unix_socket
        return (self.tcp_host, self.tcp_port)

    def __repr__(self):
        mode = self.transport_type
        addr = self.transport_addr
        return f"JRWLConfig({mode}={addr})"
