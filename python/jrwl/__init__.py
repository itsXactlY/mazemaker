"""
JRWL - JackRabbitWonderland: Lightweight async message bus.

Pure Python (stdlib + asyncio), zero pip dependencies.
Bridges SmolVM <-> Host for MSSQL access over UNIX socket or TCP.

Usage:
    # Host side (broker)
    from jrwl import JRWLBroker
    broker = JRWLBroker()
    await broker.serve()

    # VM side (client)
    from jrwl import JRWLClient
    client = JRWLClient()
    await client.connect()
    rows = await client.query("SELECT TOP 5 * FROM GraphNodes")
    await client.disconnect()
"""

from jrwl.config import JRWLConfig
from jrwl.protocol import JRWLRequest, JRWLResponse, JRWLProtocol
from jrwl.client import JRWLClient
from jrwl.broker import JRWLBroker

__version__ = "0.1.0"
__all__ = [
    "JRWLConfig",
    "JRWLRequest",
    "JRWLResponse",
    "JRWLProtocol",
    "JRWLClient",
    "JRWLBroker",
    "JRWLMSSQLStore",
]

# Lazy import to avoid requiring pyodbc when not needed
def __getattr__(name):
    if name == "JRWLMSSQLStore":
        from jrwl.store import JRWLMSSQLStore
        return JRWLMSSQLStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
