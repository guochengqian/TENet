#!/usr/bin/env bash
# Install the TENet environment.
#
# Default recipe targets MODERN NVIDIA GPUs, including Blackwell (RTX 50-series /
# RTX 5090, sm_120), using a uv-managed Python 3.12 venv and PyTorch >= 2.7 built
# against CUDA 12.8 (cu128 wheels). Blackwell requires CUDA >= 12.8 / PyTorch >= 2.7.
# TENet has no custom CUDA/C++ ops, so no compilation is needed.
#
# Usage:  source install.sh
set -e

# bicubic_pytorch submodule
git submodule update --init --recursive

# Install uv first if missing:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
source .venv/bin/activate

# PyTorch with CUDA 12.8 (Blackwell / sm_120). See https://pytorch.org for other CUDA builds.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt

echo ""
echo "Done. Quick check:"
echo "  python -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\""


# ===========================================================================
# LEGACY recipe (CUDA 11.1 / PyTorch 1.10, conda). Kept for reference only.
# Does NOT support Blackwell (RTX 5090). Uncomment and adapt for older GPUs.
# ===========================================================================
# export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"
# conda create -n tenet -y python=3.7 numpy=1.20 numba
# conda activate tenet
# conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
# pip install -r requirements.txt
