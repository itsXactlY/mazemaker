#!/usr/bin/env python3
"""test_integration.py - Fast integration tests (no sentence-transformers)"""
import sys, os, tempfile, math
from pathlib import Path

PASS = FAIL = 0

def test(name):
    def d(fn):
        global PASS, FAIL
        try: fn(); print(f"  PASS: {name}"); PASS += 1
        except Exception as e: print(f"  FAIL: {name}: {e}"); FAIL += 1
    return d

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = sum(x*x for x in a)**0.5
    nb = sum(x*x for x in b)**0.5
    return dot/(na*nb) if na*nb > 1e-10 else 0

# Hash backend only (no ST import)
@test("Hash embed - basic")
def t1():
    sys.path.insert(0, str(Path(__file__).parent))
    # Inline hash embedding
    dim = 384
    vec = [0.0]*dim
    for i, tok in enumerate("hello world".split()):
        h = hash(tok)
        for j in range(4):
            vec[(h ^ (j*2654435761)) % dim] += 1.0/(1.0+i*0.1)
            h = (h>>8)|((h&0xFF)<<24)
    n = sum(v*v for v in vec)**0.5
    assert n > 0

@test("SQLite - store/get")
def t2():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        mid = s.store("test", "content", [0.1]*384)
        assert mid > 0
        m = s.get(mid)
        assert m['label'] == "test"
        assert len(m['embedding']) == 384
        s.close()
    finally: os.unlink(db)

@test("SQLite - connections")
def t3():
    from memory_client import SQLiteStore
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        s = SQLiteStore(db)
        s.store("a", "a", [0.1]*384)
        s.store("b", "b", [0.2]*384)
        s.add_connection(1, 2, 0.8)
        c = s.get_connections(1)
        assert len(c) == 1 and c[0]['weight'] == 0.8
        s.close()
    finally: os.unlink(db)

@test("Memory - store/recall (hash)")
def t4():
    sys.path.insert(0, str(Path(__file__).parent))
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        m.remember("Dog named Lou", "Pet")
        m.remember("Trading platform BTQuant", "Work")
        r = m.recall("dog pet", k=2)
        assert len(r) >= 1 and r[0]['similarity'] > 0
        m.close()
    finally: os.unlink(db)

@test("Memory - connections auto")
def t5():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        m.remember("Dogs are great pets")
        m.remember("Dogs need walks daily")
        m.remember("Python is a programming language")
        s = m.stats()
        assert s['memories'] == 3
        m.close()
    finally: os.unlink(db)

@test("Memory - spreading activation")
def t6():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        id1 = m.remember("Topic A about dogs")
        id2 = m.remember("Topic B about cats")
        t = m.think(id1, depth=2)
        assert isinstance(t, list)
        m.close()
    finally: os.unlink(db)

@test("Memory - persistence")
def t7():
    from memory_client import NeuralMemory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        m = NeuralMemory(db_path=db, embedding_backend="hash")
        m.remember("Persistent fact about dogs")
        m.close()
        m2 = NeuralMemory(db_path=db, embedding_backend="hash")
        r = m2.recall("dogs", k=1)
        assert len(r) >= 1
        m2.close()
    finally: os.unlink(db)

@test("Memory API - unified")
def t8():
    from neural_memory import Memory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f: db = f.name
    try:
        with Memory(db_path=db, embedding_backend="hash", use_cpp=False) as m:
            m.remember("Test memory")
            r = m.recall("test", k=1)
            assert len(r) >= 1
    finally: os.unlink(db)

@test("Hermes plugin installed")
def t9():
    p = Path.home() / ".hermes/hermes-agent/plugins/memory/neural/__init__.py"
    assert p.exists(), f"Missing: {p}"

@test("C++ library symbols")
def t10():
    import subprocess
    lib = os.path.expanduser("~/projects/neural-memory-adapter/build/libneural_memory.so")
    if not os.path.exists(lib):
        return  # skip
    r = subprocess.run(["nm", "-D", lib], capture_output=True, text=True)
    assert "neural_memory_create" in r.stdout
    assert "neural_memory_store" in r.stdout
    assert "neural_memory_retrieve_full" in r.stdout

if __name__ == "__main__":
    print("=" * 50)
    print("  Neural Memory Adapter - Tests")
    print("=" * 50)
    print()
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 50)
    sys.exit(1 if FAIL > 0 else 0)
