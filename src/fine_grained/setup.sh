#!/bin/bash
# Fine-Grained Data Valuation Setup Script
# Usage: bash src/fine_grained/setup.sh

set -e

echo "=========================================="
echo "Fine-Grained Data Valuation Setup"
echo "=========================================="

# Detect repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "[1/6] Cloning GhostSuite..."
if [ ! -d "GhostSuite" ]; then
    git clone https://github.com/Jiachen-T-Wang/GhostSuite.git
else
    echo "  → GhostSuite already cloned"
fi

echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "finegrained_valuation" ]; then
    uv venv finegrained_valuation
    echo "  → Created finegrained_valuation"
else
    echo "  → Environment already exists"
fi

# Activate venv
source finegrained_valuation/bin/activate

echo ""
echo "[3/6] Installing PyPI packages (logix-ai + 198 dependencies)..."
uv pip install -r requirements.txt -q

echo ""
echo "[4/6] Installing GhostSuite (editable from GitHub)..."
pip install -e GhostSuite -q

echo ""
echo "[5/6] Installing modified opendataval..."
pip install -e src/fine_grained/opendataval -q

echo ""
echo "[6/6] Verifying installation..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || echo "  ✗ PyTorch failed"
python -c "import ghostsuite; print('  ✓ GhostSuite installed')" || echo "  ✗ GhostSuite failed"
python -c "from logix import *; print('  ✓ logix-ai installed')" || echo "  ✗ logix-ai failed"
python -c "from src.fine_grained.opendataval import *; print('  ✓ opendataval installed')" || echo "  ✗ opendataval failed"

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:"
echo "     source finegrained_valuation/bin/activate"
echo ""
echo "  2. Run experiments:"
echo "     python run_resnet9_dataval.py"
echo ""
