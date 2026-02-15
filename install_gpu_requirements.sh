#!/bin/bash

# GPU-enabled PyTorch installation script for inference server
# This ensures CUDA-enabled PyTorch is installed properly

set -e  # Exit on any error

echo "🚀 Installing GPU-enabled inference server requirements..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# CRITICAL: Install PyTorch with CUDA support FIRST (before other packages)
echo "🔥 Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify GPU PyTorch installation
echo "🔍 Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✅ GPU PyTorch successfully installed!"
else
    echo "⚠️  Warning: PyTorch installed but CUDA not available"
fi

# Create modified requirements file without PyTorch (since we installed GPU version)
echo "📝 Creating temporary requirements without PyTorch..."
grep -v "^torch" requirements-export.txt > temp_requirements.txt

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
echo "🔍 Final verification..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "⚠️  Transformers not installed"
python -c "import onnx; print(f'ONNX: {onnx.__version__}')" 2>/dev/null || echo "⚠️  ONNX not installed"

echo ""
echo "🎉 GPU-enabled setup complete!"