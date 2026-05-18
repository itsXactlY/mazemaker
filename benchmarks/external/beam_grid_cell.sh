#!/usr/bin/env bash
# beam_grid_cell.sh — one grid cell: (k, cycles) on conv-1.
#
# Usage:
#   beam_grid_cell.sh <k> <cycles> <tag>
#
# Per cell:
#   1. Wipe connections / dream / dae tables (keep memories untouched)
#   2. Nuke + rebuild gpu_cache from clean state
#   3. Run dream_worker for <cycles> NREM+REM+Insight cycles
#   4. DAE bulk-compute
#   5. emit-prompts-only with k=<k>, write JSONL to results/grid/<tag>/
#
# Output: results/grid/<tag>/prompts.jsonl + metadata
set -euo pipefail

K="${1:?usage: $0 <k> <cycles> <tag>}"
CYCLES="${2:?usage: $0 <k> <cycles> <tag>}"
TAG="${3:?usage: $0 <k> <cycles> <tag>}"

HERE="$(cd "$(dirname "$0")" && pwd)"
DB="/tmp/beam-10m/conv-1/memory.db"
GRID_DIR="${HERE}/results/grid/${TAG}"
mkdir -p "${GRID_DIR}"

PY_DIR="/home/alca/projects/mazemaker/python"
GPU_CACHE_DIR="${HOME}/.mazemaker/engine/gpu_cache"

echo "[grid] cell tag=${TAG} k=${K} cycles=${CYCLES}"
echo "[grid] DB=${DB}"
echo "[grid] output=${GRID_DIR}"

# ── 1. wipe consolidation tables (keep memories) ──────────────────────────
echo "[grid] wipe consolidation tables"
sqlite3 "${DB}" "
DELETE FROM connections;
DELETE FROM connection_history;
DELETE FROM dream_sessions;
DELETE FROM dream_insights;
DELETE FROM memory_dae_embeddings;
VACUUM;
"

# ── 2. nuke + rebuild gpu_cache ───────────────────────────────────────────
echo "[grid] nuke gpu_cache"
rm -f "${GPU_CACHE_DIR}/embeddings.npy" "${GPU_CACHE_DIR}/metadata.pkl"

# ── 3. dream cycles ───────────────────────────────────────────────────────
echo "[grid] running ${CYCLES} dream cycle(s)"
for i in $(seq 1 "${CYCLES}"); do
    echo "[grid]   cycle ${i}/${CYCLES}"
    cd "${PY_DIR}"
    PYTHONUNBUFFERED=1 python -u dream_worker.py \
        --db "${DB}" \
        --max-memories 2000 \
        --max-isolated 800 \
        --once \
        --log-level WARNING \
        2>&1 | tail -3 | sed 's/^/[grid]     /'
done

# ── 4. DAE bulk-compute ────────────────────────────────────────────────────
echo "[grid] DAE bulk-compute"
cd "${PY_DIR}"
PYTHONUNBUFFERED=1 python -c "
import os
os.environ['MM_DAE_ENABLED'] = '1'
os.environ['MM_COLBERT_ENABLED'] = '1'
from memory_client import Mazemaker
from dae import dae_bulk_compute
nm = Mazemaker(db_path='${DB}', embedding_backend='auto', use_cpp=False,
               retrieval_mode='hybrid', use_hnsw=False, lazy_graph=True,
               rerank=False, channel_weights={'colbert': 1.5, 'dae': 1.0})
r = dae_bulk_compute(nm, self_weight=0.4, neighbour_k=20)
print(f'[grid] DAE: {r}')
nm.close()
" 2>&1 | grep -E "DAE|error|Error" | head -5

# ── 5. emit prompts ────────────────────────────────────────────────────────
echo "[grid] emit prompts (k=${K})"
cd "${HERE}"
python -u beam_10m.py \
    --convs 1 \
    --skip-ingest \
    --k "${K}" \
    --emit-prompts-only \
    --tag "grid-${TAG}" \
    --quiet 2>&1 | grep -E "wrote generations|recall=" | tail -5

# Find the emitted file
EMIT_FILE=$(ls -t "${HERE}/results/beam_10m_generations_grid-${TAG}_"*.json | head -1)
echo "[grid] emit JSON: ${EMIT_FILE}"

# Extract single-question files into the grid cell dir for Haiku dispatch
python3 -c "
import json, os
src = json.load(open('${EMIT_FILE}'))
entries = src['generations_by_model']['<prompts-only>']
out_dir = '${GRID_DIR}/inputs'
os.makedirs(out_dir, exist_ok=True)
for e in entries:
    qid = f\"{e['ability']}_{e['q_idx']}\"
    fn = f'{out_dir}/q_{qid}.json'
    with open(fn, 'w') as f:
        json.dump({
            'qid': qid,
            'ability': e['ability'],
            'q_idx': e['q_idx'],
            'question': e['question'],
            'prompt': e['prompt'],
        }, f)
print(f'[grid] wrote {len(entries)} question files to {out_dir}')
" 2>&1

# Metadata
cat > "${GRID_DIR}/cell.json" <<EOF
{
  "tag": "${TAG}",
  "k": ${K},
  "cycles": ${CYCLES},
  "emit_file": "${EMIT_FILE}",
  "inputs_dir": "${GRID_DIR}/inputs",
  "haiku_outputs_dir": "${GRID_DIR}/haiku_outputs",
  "scored_path": "${GRID_DIR}/scored.jsonl"
}
EOF

mkdir -p "${GRID_DIR}/haiku_outputs"
echo "[grid] cell ${TAG} ready: inputs in ${GRID_DIR}/inputs/"
echo "[grid] next: dispatch Haiku sub-agents on those inputs, then judge"
