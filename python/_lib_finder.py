"""Shared resolver for libmazemaker.so.

Three call sites (memory_client.py, cpp_bridge.py, lstm_knn_bridge.py)
used to carry their own copy of this logic — duplicated paths drifted
over time (memory_client lacked the ctypes.util.find_library fallback,
the others lacked debug-build precedence). One canonical implementation
keeps them in sync.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
from pathlib import Path


def find_lib() -> str:
    """Return the absolute path to libmazemaker.so or raise FileNotFoundError.

    Search order:
      1. <repo>/build/libmazemaker.so   (in-tree CMake build)
      2. ~/projects/mazemaker-adapter/build/libmazemaker.so  (legacy dev path)
      3. /usr/local/lib/libmazemaker.so
      4. /usr/lib/libmazemaker.so
      5. ctypes.util.find_library("mazemaker")  — falls through LD_LIBRARY_PATH
    """
    # __file__ here is python/_lib_finder.py — repo root is two parents up.
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "build" / "libmazemaker.so",
        Path.home() / "projects" / "mazemaker-adapter" / "build" / "libmazemaker.so",
        Path("/usr/local/lib/libmazemaker.so"),
        Path("/usr/lib/libmazemaker.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    found = ctypes.util.find_library("mazemaker")
    if found:
        return found
    raise FileNotFoundError(
        "libmazemaker.so not found. Build first:\n"
        "  mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(nproc)"
    )


# Process-wide ctypes.CDLL cache. Each call to ctypes.CDLL(path) returns
# a new Python handle even when the dynamic linker has already mapped
# the .so — argtypes/restype tables would have to be configured on
# every handle. With three call sites (cpp_bridge, lstm_knn_bridge,
# memory_client), we used to load the same .so three times per process.
_LIB_CACHE: dict[str, "ctypes.CDLL"] = {}
_LIB_CACHE_LOCK = threading.Lock()


def shared_cdll(lib_path: str) -> "ctypes.CDLL":
    """Return the process-wide ctypes.CDLL for `lib_path`, loading once."""
    with _LIB_CACHE_LOCK:
        h = _LIB_CACHE.get(lib_path)
        if h is None:
            h = ctypes.CDLL(lib_path)
            _LIB_CACHE[lib_path] = h
        return h
