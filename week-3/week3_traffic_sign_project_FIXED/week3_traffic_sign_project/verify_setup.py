#!/usr/bin/env python3
"""
Verify that the Week 3 AI Traffic Sign Recognition project is properly set up.
Checks all dependencies, configurations, and module imports.

Usage:
    python verify_setup.py
"""

import sys
import subprocess
from pathlib import Path
import importlib

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    import_name = import_name or package_name.replace("-", "_")
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False

def check_dependencies():
    """Check all required dependencies."""
    print("\n📦 Checking dependencies...")
    
    required = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("PyYAML", "yaml"),
        ("seaborn", "seaborn"),
        ("streamlit", "streamlit"),
        ("altair", "altair"),
    ]
    
    all_ok = True
    for package, import_name in required:
        if not check_package(package, import_name):
            all_ok = False
    
    return all_ok

def check_files():
    """Check all necessary files exist."""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "app.py",
        "train_models.py",
        "configs/experiments.yaml",
        "requirements.txt",
        "src/config.py",
        "src/data.py",
        "src/models.py",
        "src/engine.py",
        "src/experiments.py",
        "src/metrics.py",
        "src/utils.py",
        "src/visualization.py",
        "scripts/run_experiments.py",
        "scripts/predict.py",
    ]
    
    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check necessary directories exist."""
    print("\n📂 Checking directories...")
    
    dirs = ["outputs", "data", "configs", "src", "scripts"]
    all_ok = True
    for dir_name in dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"✓ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ not found")
            all_ok = False
    
    return all_ok

def check_modules():
    """Check all source modules can be imported."""
    print("\n🔧 Checking Python modules...")
    
    sys.path.insert(0, str(Path.cwd()))
    
    modules = [
        "src.config",
        "src.data",
        "src.models",
        "src.engine",
        "src.experiments",
        "src.metrics",
        "src.utils",
        "src.visualization",
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            all_ok = False
    
    return all_ok

def check_config():
    """Check experiment configuration."""
    print("\n⚙️  Checking configuration...")
    
    try:
        from src.config import load_config
        cfg = load_config("configs/experiments.yaml")
        
        print(f"✓ Config loaded")
        print(f"  - Experiments: {len(cfg.get('experiments', {}))}")
        print(f"  - Epochs: {cfg.get('epochs', '?')}")
        print(f"  - Batch size: {cfg.get('batch_size', '?')}")
        print(f"  - Device: {cfg.get('device', '?')}")
        
        return True
    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return False

def check_device():
    """Check CUDA availability."""
    print("\n🖥️  Checking device...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
        else:
            print(f"⚠️  CUDA not available, using CPU")
        
        return True
    except Exception as e:
        print(f"❌ Device check failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("🚦 AI Traffic Sign Recognition System - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("File Structure", check_files),
        ("Directories", check_directories),
        ("Python Modules", check_modules),
        ("Configuration", check_config),
        ("Device/CUDA", check_device),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"\n❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓" if result else "❌"
        print(f"{status} {check_name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ Setup verification passed! Ready to proceed.")
        print("\nNext steps:")
        print("1. Train models:  python train_models.py")
        print("2. Launch app:    streamlit run app.py")
        return 0
    else:
        print("\n❌ Some checks failed. Please review the errors above.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
