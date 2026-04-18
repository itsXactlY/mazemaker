#!/usr/bin/env python3
"""
dlm_store.py - JackrabbitDLM storage backend for Neural Memory
Uses DLMLocker library for JSON-over-TCP communication with DLM server.

Key scheme:
  nm:memory:{id}          -> JSON memory record
  nm:connection:{s}:{t}   -> JSON edge record
  nm:stats                -> JSON stats
  nm:next_id              -> next memory ID (integer as string)
  nm:lock:{id}            -> per-memory lock for atomic updates

All data stored as DLM values (string). TTL defaults to 3600s (1 hour) for
volatility; extend on access.
"""

import sys
import json
import time
import struct
import threading
from typing import Optional, List, Dict, Any

# Robert's convention: DLMLocker.py lives in /home/JackrabbitDLM
sys.path.insert(0, '/home/JackrabbitDLM')


class DLMStore:
    """DLM-backed memory store, drop-in replacement for MSSQLStore."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 37373,
                 identity: str = "neural-memory"):
        self.host = host
        self.port = port
        self.identity = identity
        self._check_dlm()
        self._lock = threading.Lock()
        self._ensure_schema()
    
    def _check_dlm(self):
        """Verify DLMLocker importable and DLM server reachable."""
        try:
            from DLMLocker import Locker
        except ImportError:
            raise ImportError(
                "DLMLocker.py not found. Install JackrabbitDLM to /home/JackrabbitDLM.\n"
                "See: https://github.com/rapmd73/JackrabbitDLM"
            )
        self._Locker = Locker
    
    def _make_locker(self, name: str, ttl: int = 300):
        """Create a DLMLocker instance."""
        return self._Locker(name, Host=self.host, Port=self.port,
                            ID=self.identity, Timeout=ttl)
    
    def _acquire_lock(self, lock, expire: int = 10) -> bool:
        """Acquire lock, return True if locked."""
        resp = lock.Lock(expire=expire)
        return resp == "locked"
    
    def health_check(self) -> bool:
        """Check if DLM server is reachable."""
        try:
            lock = self._make_locker("nm:health")
            v = lock.Version()
            return "JackrabbitDLM" in str(v)
        except Exception:
            return False
    
    def _ensure_schema(self):
        """Initialize stats and next_id if missing."""
        # Ensure nm:stats exists
        stats_lock = self._make_locker("nm:stats")
        resp = stats_lock.Get()
        if not (isinstance(resp, dict) and resp.get("Status") == "Done" and resp.get("DataStore")):
            # Initialize stats
            stats = {"memories": 0, "connections": 0}
            stats_lock.Put(expire=3600, data=json.dumps(stats))
        
        # Ensure nm:next_id exists
        id_lock = self._make_locker("nm:next_id")
        resp = id_lock.Get()
        if not (isinstance(resp, dict) and resp.get("Status") == "Done" and resp.get("DataStore")):
            id_lock.Put(expire=3600, data="1")
    
    def _get_json(self, key: str) -> Optional[dict]:
        """Get and parse JSON from DLM."""
        lock = self._make_locker(key)
        resp = lock.Get()
        if isinstance(resp, dict) and resp.get("Status") == "Done":
            data = resp.get("DataStore")
            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return None
        return None
    
    def _put_json(self, key: str, data: dict, ttl: int = 3600) -> bool:
        """Store JSON in DLM."""
        lock = self._make_locker(key)
        resp = lock.Put(expire=ttl, data=json.dumps(data))
        print(f"[DLM] Put response for {key}: {resp!r}")
        if isinstance(resp, bytes):
            resp = resp.decode('utf-8')
        return "Done" in str(resp)
    
    def _get_string(self, key: str) -> Optional[str]:
        """Get raw string from DLM."""
        lock = self._make_locker(key)
        resp = lock.Get()
        if isinstance(resp, dict) and resp.get("Status") == "Done":
            return resp.get("DataStore")
        return None
    
    def _put_string(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Store raw string in DLM."""
        lock = self._make_locker(key)
        resp = lock.Put(expire=ttl, data=value)
        if isinstance(resp, bytes):
            resp = resp.decode('utf-8')
        return "Done" in str(resp)
    
    def _next_id(self) -> int:
        """Atomically get and increment next memory ID."""
        with self._lock:
            lock = self._make_locker("nm:next_id")
            # Lock for atomic increment
            if not self._acquire_lock(lock, expire=10):
                raise RuntimeError("Could not acquire lock for nm:next_id")
            try:
                resp = lock.Get()
                current = 1
                if isinstance(resp, dict) and resp.get("Status") == "Done":
                    data = resp.get("DataStore")
                    if data:
                        try:
                            current = int(data)
                        except ValueError:
                            current = 1
                # Increment and store
                lock.Put(expire=3600, data=str(current + 1))
                return current
            finally:
                lock.Unlock()
    
    def _update_stats(self, delta_memories: int = 0, delta_connections: int = 0):
        """Atomically update stats."""
        with self._lock:
            lock = self._make_locker("nm:stats")
            if not self._acquire_lock(lock, expire=10):
                raise RuntimeError("Could not acquire lock for nm:stats")
            try:
                resp = lock.Get()
                stats = {"memories": 0, "connections": 0}
                if isinstance(resp, dict) and resp.get("Status") == "Done":
                    data = resp.get("DataStore")
                    if data:
                        try:
                            stats = json.loads(data)
                        except json.JSONDecodeError:
                            pass
                stats["memories"] = max(0, stats.get("memories", 0) + delta_memories)
                stats["connections"] = max(0, stats.get("connections", 0) + delta_connections)
                lock.Put(expire=3600, data=json.dumps(stats))
            finally:
                lock.Unlock()
    
    def store(self, label: str, content: str, embedding: List[float]) -> int:
        """Store a memory. Returns memory ID."""
        mem_id = self._next_id()
        print(f"[DLM] _next_id returned {mem_id}")
        record = {
            "id": mem_id,
            "label": label,
            "content": content,
            "embedding": embedding,  # Store as JSON list (floats)
            "vector_dim": len(embedding),
            "salience": 1.0,
            "access_count": 0,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }
        key = f"nm:memory:{mem_id}"
        success = self._put_json(key, record)
        print(f"[DLM] _put_json success: {success}")
        if success:
            self._add_memory_id(mem_id)
            self._update_stats(delta_memories=1)
            return mem_id
        return 0
    
    def get_all(self) -> List[dict]:
        """Get all memories (iterate via pattern matching)."""
        # DLM doesn't support pattern matching via DLMLocker directly.
        # We'll maintain an index key nm:memory:ids with list of IDs.
        # For simplicity, we'll store IDs in a separate key.
        ids = self._get_memory_ids()
        results = []
        for mid in ids:
            m = self.get(mid)
            if m:
                results.append(m)
        return results
    
    def _get_memory_ids(self) -> List[int]:
        """Get list of stored memory IDs."""
        data = self._get_json("nm:memory:ids")
        if data and isinstance(data, list):
            return data
        return []
    
    def _add_memory_id(self, mem_id: int):
        """Add ID to index."""
        ids = self._get_memory_ids()
        if mem_id not in ids:
            ids.append(mem_id)
            self._put_json("nm:memory:ids", ids)
    
    def get(self, id_: int) -> Optional[dict]:
        """Get a memory by ID."""
        key = f"nm:memory:{id_}"
        return self._get_json(key)
    
    def touch(self, id_: int):
        """Update last_accessed and increment access_count."""
        key = f"nm:memory:{id_}"
        lock = self._make_locker(key)
        if not self._acquire_lock(lock, expire=10):
            raise RuntimeError(f"Could not acquire lock for {key}")
        try:
            resp = lock.Get()
            if isinstance(resp, dict) and resp.get("Status") == "Done":
                data = resp.get("DataStore")
                if data:
                    record = json.loads(data)
                    record["last_accessed"] = time.time()
                    record["access_count"] = record.get("access_count", 0) + 1
                    lock.Put(expire=3600, data=json.dumps(record))
        finally:
            lock.Unlock()
    
    def add_connection(self, source: int, target: int, weight: float,
                       edge_type: str = "similar"):
        """Add or update a connection between memories."""
        # Ensure consistent key ordering
        if source > target:
            source, target = target, source
        key = f"nm:connection:{source}:{target}"
        
        with self._lock:
            lock = self._make_locker(key)
            if not self._acquire_lock(lock, expire=10):
                raise RuntimeError(f"Could not acquire lock for {key}")
            try:
                # Check if exists
                resp = lock.Get()
                existing_weight = 0.0
                is_new = True
                if isinstance(resp, dict) and resp.get("Status") == "Done":
                    data = resp.get("DataStore")
                    if data:
                        try:
                            existing = json.loads(data)
                            existing_weight = existing.get("weight", 0.0)
                            is_new = False
                        except json.JSONDecodeError:
                            pass
                
                # Take max weight
                final_weight = max(weight, existing_weight)
                record = {
                    "source": source,
                    "target": target,
                    "weight": final_weight,
                    "edge_type": edge_type,
                    "created_at": time.time(),
                }
                lock.Put(expire=3600, data=json.dumps(record))
                
                if is_new:
                    self._add_connection_index(source, target)
                    self._update_stats(delta_connections=1)
            finally:
                lock.Unlock()
    
    def get_connections(self, node_id: int) -> List[dict]:
        """Get all connections for a node."""
        # Since DLM doesn't support pattern matching, we maintain a connection index
        # nm:connections:{node_id} -> list of {source, target} pairs
        index_key = f"nm:connections:{node_id}"
        index = self._get_json(index_key)
        if not index or not isinstance(index, list):
            return []
        
        results = []
        for pair in index:
            s = pair.get("source")
            t = pair.get("target")
            if s is not None and t is not None:
                edge_key = f"nm:connection:{s}:{t}"
                edge = self._get_json(edge_key)
                if edge:
                    results.append(edge)
        return results
    
    def _add_connection_index(self, source: int, target: int):
        """Update connection index for both nodes."""
        # Ensure consistent ordering for edge key
        if source > target:
            s, t = target, source
        else:
            s, t = source, target
        
        pair = {"source": s, "target": t}
        
        # Update source index
        src_key = f"nm:connections:{source}"
        src_index = self._get_json(src_key) or []
        if pair not in src_index:
            src_index.append(pair)
            self._put_json(src_key, src_index)
        
        # Update target index (if different)
        if source != target:
            tgt_key = f"nm:connections:{target}"
            tgt_index = self._get_json(tgt_key) or []
            if pair not in tgt_index:
                tgt_index.append(pair)
                self._put_json(tgt_key, tgt_index)
    
    def stats(self) -> dict:
        """Get store statistics."""
        stats = self._get_json("nm:stats")
        if stats:
            return stats
        return {"memories": 0, "connections": 0}
    
    def close(self):
        """Clean up (DLM connections are per-call, nothing to close)."""
        pass


# Quick test
if __name__ == "__main__":
    import os
    try:
        store = DLMStore()
        if not store.health_check():
            print("DLM server not reachable on port 37373")
            sys.exit(1)
        
        mid = store.store("test", "Hello DLM", [0.1] * 384)
        print(f"Stored: {mid}")
        m = store.get(mid)
        print(f"Retrieved: {m['label']}")
        s = store.stats()
        print(f"Stats: {s}")
        store.close()
        print("DLM: OK")
    except Exception as e:
        print(f"DLM error: {e}")
        sys.exit(1)