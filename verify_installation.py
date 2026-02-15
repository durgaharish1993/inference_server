#!/usr/bin/env python3
"""
Verification script for inference server installation
Run this after installation to check if everything is working properly
"""

import sys
import subprocess
import importlib.util
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.11+)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, f"❌ {package_name} (not found)"
        
        # Try to import
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✅ {package_name} {version}"
    except Exception as e:
        return False, f"❌ {package_name} (error: {str(e)[:50]})"

def check_cuda_support() -> Tuple[bool, str]:
    """Check CUDA support in PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            return True, f"✅ CUDA available ({device_count} GPUs, {device_name})"
        else:
            return False, "❌ CUDA not available"
    except ImportError:
        return False, "❌ PyTorch not installed"

def main():
    """Run all verification checks"""
    print("🔍 Verifying inference server installation...\n")
    
    # Core packages to check
    core_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"), 
        ("onnx", "onnx"),
        ("onnxruntime-gpu", "onnxruntime"),
        ("fastapi", "fastapi"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
    ]
    
    optional_packages = [
        ("tensorrt", "tensorrt"),
        ("pycuda", "pycuda"),
        ("tritonclient", "tritonclient"),
        ("optimum", "optimum"),
        ("datasets", "datasets"),
        ("sentence-transformers", "sentence_transformers"),
    ]
    
    # Run checks
    print("📋 System Check:")
    python_ok, python_msg = check_python_version()
    print(f"  {python_msg}")
    
    print("\n📦 Core Packages:")
    core_ok = True
    for pkg_name, import_name in core_packages:
        ok, msg = check_package(pkg_name, import_name)
        print(f"  {msg}")
        if not ok:
            core_ok = False
    
    print("\n🔧 Optional Packages:")
    for pkg_name, import_name in optional_packages:
        ok, msg = check_package(pkg_name, import_name)
        print(f"  {msg}")
    
    print("\n🚀 GPU Support:")
    cuda_ok, cuda_msg = check_cuda_support()
    print(f"  {cuda_msg}")
    
    # Summary
    print("\n" + "="*50)
    if python_ok and core_ok:
        print("🎉 Installation verification PASSED!")
        print("✨ Your system is ready for inference serving")
        if not cuda_ok:
            print("⚠️  Note: CUDA not available - CPU inference only")
    else:
        print("❌ Installation verification FAILED!")
        print("🔧 Please check the errors above and reinstall missing packages")
        sys.exit(1)
    
    print("\n📚 Next steps:")
    print("  - Export models: python export/text_embedding/export_onnx.py")
    print("  - Start server: python server/main.py")  
    print("  - Run tests: python -m pytest benchmarks/")

if __name__ == "__main__":
    main()