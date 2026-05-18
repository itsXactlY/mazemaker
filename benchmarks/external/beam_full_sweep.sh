#!/usr/bin/env bash
# beam_full_sweep.sh — drives conv 2..10 through the full Mazemaker
# pipeline (ingest → 1× dream cycle → DAE bulk-compute → emit prompts
# at k=42). Conv 1 is already done. Multi-recall + judge are dispatched
# from the operator's session after this script finishes.
#
# Wall time estimate: ~30-35 min per conv × 9 = ~4.5-5 h sequential.
# Runs in background; monitor /tmp/beam-sweep.log.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY_DIR="/home/alca/projects/mazemaker/python"
LOG="/tmp/beam-sweep.log"

echo "[sweep] starting at $(date -Iseconds)" | tee -a "${LOG}"

for CONV in 2 3 4 5 6 7 8 9 10; do
    DB="/tmp/beam-10m/conv-${CONV}/memory.db"
    PROMPTS_DIR="${HERE}/results/grid/sweep_conv${CONV}/inputs"

    echo "" | tee -a "${LOG}"
    echo "============================================================" | tee -a "${LOG}"
    echo "[sweep] CONV ${CONV} starting at $(date -Iseconds)" | tee -a "${LOG}"
    echo "============================================================" | tee -a "${LOG}"

    # ── ingest ─────────────────────────────────────────────────────────
    if [ -f "${DB}" ]; then
        EXISTING=$(sqlite3 "${DB}" "SELECT COUNT(*) FROM memories" 2>/dev/null || echo "0")
        if [ "${EXISTING}" -gt "10000" ]; then
            echo "[sweep] conv ${CONV}: skip ingest (${EXISTING} memories already present)" | tee -a "${LOG}"
        else
            echo "[sweep] conv ${CONV}: removing stale DB" | tee -a "${LOG}"
            rm -rf "/tmp/beam-10m/conv-${CONV}"
            EXISTING=0
        fi
    fi

    if [ ! -f "${DB}" ]; then
        echo "[sweep] conv ${CONV}: ingest starting at $(date -Iseconds)" | tee -a "${LOG}"
        cd "${HERE}"
        python -u beam_10m.py --convs "${CONV}" --ingest-only --quiet 2>&1 \
            | tee -a "${LOG}" \
            | grep -E "ingest done|ingested [0-9]+/" \
            | tail -3
        echo "[sweep] conv ${CONV}: ingest done at $(date -Iseconds)" | tee -a "${LOG}"
    fi

    # ── wipe consolidation tables (memories untouched) ────────────────
    echo "[sweep] conv ${CONV}: wipe dream/connection/dae tables" | tee -a "${LOG}"
    sqlite3 "${DB}" "
DELETE FROM connections;
DELETE FROM connection_history;
DELETE FROM dream_sessions;
DELETE FROM dream_insights;
DELETE FROM memory_dae_embeddings;
VACUUM;
" 2>&1 | tee -a "${LOG}" || true

    # ── nuke + rebuild gpu_cache fresh ────────────────────────────────
    rm -f "${HOME}/.mazemaker/engine/gpu_cache/embeddings.npy" \
          "${HOME}/.mazemaker/engine/gpu_cache/metadata.pkl"

    # ── 1× full dream cycle ──────────────────────────────────────────
    echo "[sweep] conv ${CONV}: dream --once" | tee -a "${LOG}"
    cd "${PY_DIR}"
    PYTHONUNBUFFERED=1 python -u dream_worker.py \
        --db "${DB}" \
        --max-memories 2000 \
        --max-isolated 800 \
        --once \
        --log-level WARNING 2>&1 \
        | grep -E "Dream #|complete" | tee -a "${LOG}" | tail -3

    # ── DAE bulk-compute ─────────────────────────────────────────────
    echo "[sweep] conv ${CONV}: DAE bulk-compute" | tee -a "${LOG}"
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
print(f'[sweep] conv ${CONV} DAE: {r}')
nm.close()
" 2>&1 | grep -E "DAE:" | tee -a "${LOG}"

    # ── emit prompts at k=42 ──────────────────────────────────────────
    echo "[sweep] conv ${CONV}: emit prompts (k=42)" | tee -a "${LOG}"
    cd "${HERE}"
    python -u beam_10m.py \
        --convs "${CONV}" \
        --skip-ingest \
        --k 42 \
        --emit-prompts-only \
        --tag "sweep-conv${CONV}" \
        --quiet 2>&1 \
        | grep -E "wrote generations" | tee -a "${LOG}"

    # locate the emitted file
    EMIT_FILE=$(ls -t "${HERE}/results/beam_10m_generations_sweep-conv${CONV}_"*.json | head -1)
    echo "[sweep] conv ${CONV}: emit file = ${EMIT_FILE}" | tee -a "${LOG}"

    # split into per-question files for sub-agent dispatch
    mkdir -p "${PROMPTS_DIR}"
    python3 -c "
import json, os
src = json.load(open('${EMIT_FILE}'))
entries = src['generations_by_model']['<prompts-only>']
for e in entries:
    qid = f\"{e['ability']}_{e['q_idx']}\"
    fn = '${PROMPTS_DIR}/q_' + qid + '.json'
    with open(fn, 'w') as f:
        json.dump({
            'qid': qid, 'ability': e['ability'], 'q_idx': e['q_idx'],
            'question': e['question'], 'prompt': e['prompt'],
        }, f)
print(f'[sweep] conv ${CONV}: wrote {len(entries)} q-files')
" 2>&1 | tee -a "${LOG}"

    # write per-conv metadata
    mkdir -p "${HERE}/results/grid/sweep_conv${CONV}/haiku_outputs"
    cat > "${HERE}/results/grid/sweep_conv${CONV}/cell.json" <<EOF
{
  "conv": ${CONV},
  "k": 42,
  "cycles": 1,
  "emit_file": "${EMIT_FILE}",
  "inputs_dir": "${PROMPTS_DIR}",
  "ready_at": "$(date -Iseconds)"
}
EOF

    echo "[sweep] conv ${CONV} DONE at $(date -Iseconds)" | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"
echo "[sweep] ALL DONE at $(date -Iseconds)" | tee -a "${LOG}"
echo "[sweep] convs 2-10 ready for multi-recall + judge dispatch" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"
