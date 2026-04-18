#!/usr/bin/env bash
# ============================================================
# Start Embed Server — loads bge-m3 on CPU, serves /embed + /health
# ============================================================
set -euo pipefail

PORT="${SMOLVM_EMBED_PORT:-8501}"
MODEL_DIR="${SMOLVM_MODEL_DIR:-/app/models/bge-m3}"
MODEL_NAME="${EMBED_MODEL_NAME:-BAAI/bge-m3}"
LOG_FILE="${SMOLVM_LOG_DIR:-/app/logs}/embed_server.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting embed server on port $PORT" | tee -a "$LOG_FILE"

python3 -c "
import os
import sys
import json
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [embed] %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.environ.get('SMOLVM_LOG_DIR', '/app/logs') + '/embed_server.log')
    ]
)
log = logging.getLogger('embed')

app = Flask(__name__)
model = None

def load_model():
    global model
    model_name = os.environ.get('EMBED_MODEL_NAME', '${MODEL_NAME}')
    model_dir = os.environ.get('SMOLVM_MODEL_DIR', '${MODEL_DIR}')

    log.info(f'Loading model: {model_name}')
    log.info(f'Cache dir: {model_dir}')

    try:
        model = SentenceTransformer(model_name, cache_folder=model_dir, device='cpu')
        log.info(f'Model loaded: {model.get_sentence_embedding_dimension()}d embeddings')
    except Exception as e:
        log.error(f'Model load failed: {e}')
        # Try loading from cache only (offline mode)
        try:
            model = SentenceTransformer(model_dir, device='cpu')
            log.info(f'Model loaded from cache: {model.get_sentence_embedding_dimension()}d')
        except Exception as e2:
            log.error(f'Cache load also failed: {e2}')
            sys.exit(1)

@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({'status': 'loading'}), 503
    return jsonify({
        'status': 'ok',
        'model': os.environ.get('EMBED_MODEL_NAME', '${MODEL_NAME}'),
        'dimension': model.get_sentence_embedding_dimension()
    }), 200

@app.route('/embed', methods=['POST'])
def embed():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing \"texts\" field'}), 400

    texts = data['texts']
    if isinstance(texts, str):
        texts = [texts]

    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return jsonify({
            'embeddings': embeddings.tolist(),
            'dimension': embeddings.shape[1],
            'count': len(texts)
        }), 200
    except Exception as e:
        log.error(f'Embed error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def similarity():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data or 'query' not in data or 'candidates' not in data:
        return jsonify({'error': 'Missing \"query\" or \"candidates\"'}), 400

    try:
        query_emb = model.encode([data['query']], normalize_embeddings=True)
        cand_embs = model.encode(data['candidates'], normalize_embeddings=True)
        scores = np.dot(cand_embs, query_emb.T).flatten().tolist()
        ranked = sorted(zip(data['candidates'], scores), key=lambda x: -x[1])
        return jsonify({
            'results': [{'text': t, 'score': s} for t, s in ranked]
        }), 200
    except Exception as e:
        log.error(f'Similarity error: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('SMOLVM_EMBED_PORT', ${PORT}))
    log.info(f'Serving on 0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, threaded=True)
" 2>&1 | tee -a "$LOG_FILE"
