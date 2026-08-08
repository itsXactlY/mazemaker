import os, sys
for p in ("/patch","/app/core","/app/shared","/app"):
    if os.path.isdir(p) and p not in sys.path: sys.path.insert(0,p)
from dream_postgres_store import DreamPostgresStore
be = DreamPostgresStore()
with be._cursor() as (_c, cur):
    cur.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL LIMIT 2")
    for mid, vec in cur.fetchall():
        print(f"id={mid} type(embedding)={type(vec).__name__} sample={repr(vec)[:60]}")
        try:
            parsed=[float(x) for x in vec][:3]
            print("   [float(x) for x in vec] ->", parsed, "(len", len([float(x) for x in vec]),")")
        except Exception as e:
            print("   parse FAILS:", type(e).__name__, str(e)[:50])
