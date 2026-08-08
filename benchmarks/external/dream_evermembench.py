"""Run the FULL formation layer (dream cycles) over the already-ingested
EverMemBench corpus in the isolated `evermembench` PG DB.

The MazemakerAdapter ingested in sponge mode (auto_connect=False) — CORRECT
production pattern — but the graph/supersedes/insight/AFE only exist AFTER the
dream engine runs. This builds them. Replicates inception-bench run_dream_cycle:
DreamEngine must be constructed with an explicit DreamPostgresStore (its
convenience path defaults to SQLite via nm._db_path=/dev/null → write error).
"""
import os, sys, time
for p in ("/patch", "/app/core", "/app/shared", "/app"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import memory_client as mc
from dream_engine import DreamEngine
from dream_postgres_store import DreamPostgresStore

N_CYCLES = int(os.environ.get("MM_DREAM_CYCLES", "4"))

nm = mc.Mazemaker(db_path="/dev/null", embedding_backend="auto", use_cpp=True,
                  retrieval_mode="advanced", use_hnsw="auto", lazy_graph=True,
                  think_engine="ppr", rerank=True, retrieval_candidates=512,
                  channel_weights={"colbert": 1.5, "dae": 1.0})
nm.store._ensure_embedding_column(1024)

def edge_count():
    try:
        with nm.store._cursor() as (_c, cur):
            cur.execute("SELECT count(*) FROM connections")
            return int(cur.fetchone()[0])
    except Exception as e:
        return f"?({type(e).__name__})"

n_corpus = 10222
print(f"[dream] corpus ~{n_corpus} | edges before: {edge_count()} | cycles: {N_CYCLES}", flush=True)

PHASES = [("_phase_nrem", "NREM"), ("_phase_rem", "REM"), ("_phase_insights", "Insight"),
          ("_phase_supersedes", "Supersedes"), ("_phase_afe", "AFE"), ("_phase_dae", "DAE")]

for c in range(1, N_CYCLES + 1):
    backend = DreamPostgresStore()
    de = DreamEngine(backend, neural_memory=nm,
                     max_memories_per_cycle=min(n_corpus, 4000),
                     max_isolated_per_cycle=min(n_corpus, 3000))
    row = []
    for attr, label in PHASES:
        if not hasattr(de, attr):
            continue
        t0 = time.perf_counter()
        try:
            r = getattr(de, attr)()
            r = (r if isinstance(r, (int, float, dict)) else "ok")
        except Exception as e:
            r = f"ERR:{type(e).__name__}:{str(e)[:60]}"
        row.append(f"{label}={r}({time.perf_counter()-t0:.1f}s)")
    print(f"[dream] cycle {c}/{N_CYCLES}: edges={edge_count()} | " + " ".join(row), flush=True)

print(f"[dream] DONE. edges after: {edge_count()}", flush=True)
# verify derived/AFE memories formed
try:
    with nm.store._cursor() as (_c, cur):
        cur.execute("SELECT split_part(label,':',1) AS p, count(*) FROM memories GROUP BY 1 ORDER BY 2 DESC LIMIT 12")
        print("[dream] label prefixes now:", [(r[0], r[1]) for r in cur.fetchall()], flush=True)
except Exception as e:
    print("[dream] prefix check err:", e, flush=True)
