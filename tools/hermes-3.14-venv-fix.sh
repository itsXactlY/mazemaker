#!/usr/bin/env bash
# Bring the hermes 3.14 venv up to the full neural-memory feature set.
# Safe to re-run; pip skips already-installed packages.
set -e

VENV=/home/alca/.hermes/hermes-agent/venv
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

echo "[1/5] setuptools (required by pip itself + setup_fast)"
$PIP install --quiet setuptools

echo "[2/5] hnswlib (ANN — fixes the 50s brute-force recall on 2660 rows)"
$PIP install --quiet hnswlib

echo "[3/5] sentence-transformers + transformers (rerank=true in config currently silently no-ops without these)"
$PIP install --quiet sentence-transformers transformers

# Detect CUDA before deciding torch flavour
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "[4/5] torch + CUDA (GPU detected)"
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1 | cut -d. -f1)
    if [ "$CUDA_VERSION" = "12" ]; then
        $PIP install --quiet torch --index-url https://download.pytorch.org/whl/cu121
    else
        $PIP install --quiet torch --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "[4/5] torch CPU (no NVIDIA GPU detected)"
    $PIP install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "[5/5] verification"
$PY -c "
import importlib, sys
for m in ['setuptools','hnswlib','sentence_transformers','transformers','torch']:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, '__version__', '?')
        print(f'  OK  {m:25s} {v}')
    except ImportError as e:
        print(f'  FAIL {m:25s} {e}')
        sys.exit(1)
print()
print('venv is now at full feature set. Restart hermes to load HNSW + rerank.')
"
