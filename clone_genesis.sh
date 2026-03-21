#!/usr/bin/env bash
# Clone Genesis with all submodules and set up GPU support
# Usage: bash clone_genesis.sh [install_dir]
#
# Single command to clone everything:
#   git clone --recursive https://github.com/skurtyyskirts/Genesis.git && cd Genesis && pip install -e ".[dev]"

set -euo pipefail

INSTALL_DIR="${1:-Genesis}"

echo "=== Cloning Genesis with all submodules ==="
git clone --recursive https://github.com/skurtyyskirts/Genesis.git "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo ""
echo "=== Verifying submodules ==="
git submodule status

echo ""
echo "=== Installing Genesis with GPU support ==="
pip install -e ".[dev,render]"

echo ""
echo "=== Installing PyTorch with CUDA support ==="
pip install torch --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "=== Detecting GPU ==="
python genesis_gpu_config.py

echo ""
echo "=== Installing GUI dependencies ==="
pip install PyQt6

echo ""
echo "=== Setup complete! ==="
echo "Run the Genesis GUI with: python genesis_gui.py"
echo "Run examples with GPU:    python -c \"from genesis_gpu_config import init_genesis_gpu; init_genesis_gpu()\""
