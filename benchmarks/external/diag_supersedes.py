"""Diagnose why _phase_supersedes found 0 pairs (pairs_checked=0) on the
evermembench corpus — check each early-return condition step by step."""
import os, sys, re
for p in ("/patch", "/app/core", "/app/shared", "/app"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import memory_client as mc
from dream_engine import DreamEngine, _NUMERIC_TOKEN_RE
from dream_postgres_store import DreamPostgresStore

nm = mc.Mazemaker(db_path="/dev/null", embedding_backend="auto", use_cpp=True,
                  retrieval_mode="advanced", use_hnsw="auto", lazy_graph=True)
nm.store._ensure_embedding_column(1024)
be = DreamPostgresStore()
de = DreamEngine(be, neural_memory=nm, max_memories_per_cycle=4000, max_isolated_per_cycle=3000)

print("sim_threshold =", getattr(de, "_supersedes_sim_threshold", "?"))
mems = be.sample_for_dream(de._max_memories, recent_pct=de._sample_recent_pct,
                           random_old_pct=de._sample_random_pct, low_salience_pct=de._sample_low_salience_pct)
print("1) sample_for_dream ->", len(mems) if mems else 0, "memories")
if not mems or len(mems) < 2:
    sys.exit("BAIL @ sample")
mem_ids = [m["id"] for m in mems]
vecs = be.get_memory_vectors(mem_ids)
print("2) get_memory_vectors ->", len(vecs) if vecs else 0)
if not vecs:
    sys.exit("BAIL @ vectors")
try:
    meta = be.get_memory_metadata(mem_ids)
    print("3) get_memory_metadata ->", len(meta) if meta else 0, "| sample keys:",
          list(next(iter(meta.values())).keys()) if meta else None)
except Exception as e:
    sys.exit(f"BAIL @ get_memory_metadata EXC: {type(e).__name__}: {e}")
if not meta:
    sys.exit("BAIL @ meta empty")
# numeric filter
num_ids = []
for mid in mem_ids:
    if mid not in vecs or mid not in meta:
        continue
    content = meta[mid]["content"]
    toks = set(_NUMERIC_TOKEN_RE.findall(content or ""))
    if toks:
        num_ids.append(mid)
print("4) numeric_ids ->", len(num_ids), "of", len(mem_ids), "sampled")
# created_at diversity
cats = [meta[m]["created_at"] for m in meta if "created_at" in meta[m]]
print("5) distinct created_at values:", len(set(map(str, cats))), "(of", len(cats), ")")
print("   sample created_at:", list(map(str, cats[:3])))
if num_ids:
    ex = meta[num_ids[0]]["content"]
    print("   example numeric mem:", (ex or "")[:120])
