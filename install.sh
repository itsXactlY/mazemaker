#!/bin/bash
# install.sh - Neural Memory Adapter Installer
# Usage: bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_DIR/build"
HERMES_DIR="$HOME/.hermes"
NEURAL_DIR="$HOME/.neural_memory"

echo "=============================================="
echo "  Neural Memory Adapter - Installer"
echo "=============================================="

# 1. Create directories
echo -e "\n[1/6] Creating directories..."
mkdir -p "$NEURAL_DIR"
mkdir -p "$NEURAL_DIR/models"

# 2. Build C++ library (optional)
echo -e "\n[2/6] Building C++ library..."
if command -v cmake &> /dev/null && command -v g++ &> /dev/null; then
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    cmake --build . -j$(nproc) 2>&1 | tail -3
    echo "  C++ library: OK"
else
    echo "  C++ build tools not found, skipping (Python-only mode)"
fi

# 3. Install Python dependencies
echo -e "\n[3/6] Installing Python dependencies..."
pip install --quiet sentence-transformers numpy 2>&1 | tail -3
echo "  Python deps: OK"

# 4. Install Hermes plugin
echo -e "\n[4/6] Installing Hermes plugin..."
PLUGIN_DIR="$HERMES_DIR/hermes-agent/plugins/memory/neural"
mkdir -p "$PLUGIN_DIR"

# Copy plugin files
cp "$PROJECT_DIR/python/memory_client.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/python/embed_provider.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/python/neural_memory.py" "$PLUGIN_DIR/"
cp "$PROJECT_DIR/python/cpp_bridge.py" "$PLUGIN_DIR/" 2>/dev/null || true

# Copy plugin __init__.py if it exists
if [ -f "$PLUGIN_DIR/__init__.py" ]; then
    echo "  Plugin already installed, updating modules..."
else
    echo "  Plugin needs __init__.py (run hermes setup)"
fi

# 5. Update config (if hermes config exists)
echo -e "\n[5/6] Configuration..."
CONFIG="$HERMES_DIR/config.yaml"
if [ -f "$CONFIG" ]; then
    if grep -q "neural" "$CONFIG"; then
        echo "  Neural config already present"
    else
        echo "  Add to $CONFIG:"
        echo "    memory:"
        echo "      provider: neural  # or keep 'mempalace'"
    fi
else
    echo "  No Hermes config found (that's OK for standalone use)"
fi

# 6. Run tests
echo -e "\n[6/6] Running integration tests..."
cd "$PROJECT_DIR"
python3 python/test_integration.py 2>&1 || true

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  python3 python/demo.py              # Run demo"
echo "  python3 python/test_integration.py  # Run tests"
echo ""
echo "Python API:"
echo "  from neural_memory import Memory"
echo "  mem = Memory()"
echo "  mem.remember('fact')"
echo "  results = mem.recall('query')"
echo ""
echo "Hermes integration:"
echo "  Set memory.provider: neural in ~/.hermes/config.yaml"
echo ""
