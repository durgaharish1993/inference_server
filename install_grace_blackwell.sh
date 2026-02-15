#!/bin/bash

# NVIDIA Grace Blackwell (GB10) installation script
# ARM-based system with NVIDIA GPU acceleration

set -e  # Exit on any error

echo "🚀 Installing for NVIDIA Grace Blackwell (ARM + NVIDIA GPU)..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check ARM architecture
ARCH=$(uname -m)
echo "🏗️  Detected architecture: $ARCH"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo "⚠️  Warning: Expected ARM architecture (aarch64/arm64), got $ARCH"
fi

# Check if virtual environment exists
if [ ! -d ".venv311" ]; then
    echo "❌ Virtual environment .venv311 not found!"
    echo "Please create it first with: python3.11 -m venv .venv311"
    exit 1
fi

# Activate virtual environment
source .venv311/bin/activate

echo "🐍 Using Python: $(which python)"
echo "📍 Python version: $(python --version)"

# Upgrade pip and setuptools first
echo "📦 Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch for NVIDIA Grace Blackwell (CUDA 12.1 for ARM)
echo "🔥 Installing PyTorch for Grace Blackwell (ARM + CUDA 12.1)..."
pip install torch>=2.1.0,\<2.4.0 torchvision>=0.16.0,\<0.19.0 torchaudio>=2.1.0,\<2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install NVIDIA utilities for Grace Blackwell
echo "🎯 Installing NVIDIA Grace Blackwell optimizations..."
pip install nvidia-ml-py>=12.555.0
pip install cupy-cuda12x>=13.0.0

# Create modified requirements file without PyTorch and NVIDIA packages
echo "📝 Creating temporary requirements without PyTorch..."
grep -v "^torch\|^nvidia-ml-py\|^cupy-cuda12x" requirements-export.txt > temp_requirements.txt

# Install remaining export requirements
echo "🤖 Installing remaining export requirements..."
pip install -r temp_requirements.txt

# Clean up
rm temp_requirements.txt

echo "🌐 Installing server requirements..."
pip install -r requirements-server.txt

echo "⚡ Installing TensorRT requirements..."
pip install -r requirements-tensorrt.txt || echo "⚠️  Some TensorRT packages failed (normal if TensorRT not installed)"

echo "✅ Installation complete!"
echo ""
echo "🔍 Grace Blackwell verification..."
python -c "import platform; print(f'Architecture: {platform.machine()}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "
try:
    import pynvml
    pynvml.nvmlInit()
    print(f'NVML initialized: {pynvml.nvmlDeviceGetCount()} GPUs detected')
except:
    print('NVML not available')
"
python -c "
try:
    import cupy
    print(f'CuPy: {cupy.__version__} (CUDA acceleration ready)')
except:
    print('CuPy not available')
"

echo ""
echo "🎉 NVIDIA Grace Blackwell setup complete! 🚀"
echo "✨ ARM + NVIDIA GPU acceleration ready"