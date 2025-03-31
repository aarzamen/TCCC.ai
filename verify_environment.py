#!/usr/bin/env python3
"""
Quick environment verification script for TCCC project.
Checks for critical dependencies and correct setup.
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

# Add project to path
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, src_dir)

# Required packages for core functionality
CORE_PACKAGES = [
    "numpy", 
    "yaml", 
    "torch", 
    "sounddevice", 
    "soundfile"
]

# Required project modules
TCCC_MODULES = [
    "tccc.audio_pipeline.audio_pipeline",
    "tccc.stt_engine.stt_engine",
    "tccc.utils.config_manager",
    "tccc.utils.vad_manager"
]

def check_python_packages():
    """Check if required Python packages are installed."""
    print("\n=== Checking Python packages ===")
    missing = []
    for package in CORE_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    return missing

def check_tccc_modules():
    """Check if TCCC modules can be imported."""
    print("\n=== Checking TCCC modules ===")
    missing = []
    for module in TCCC_MODULES:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} ({e})")
            missing.append(module)
    
    return missing

def check_audio_devices():
    """Check audio devices."""
    print("\n=== Checking audio devices ===")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        print(f"Found {len(input_devices)} input devices:")
        
        for i, device in enumerate(input_devices):
            print(f"  {i}: {device['name']} (channels: {device['max_input_channels']})")
            
        return len(input_devices) > 0
    except Exception as e:
        print(f"Error checking audio devices: {e}")
        return False

def check_file_resources():
    """Check critical file resources."""
    print("\n=== Checking file resources ===")
    resources = {
        "Config files": [
            "config/audio_pipeline.yaml",
            "config/stt_engine.yaml",
        ],
        "Test data": [
            "test_data/test_speech.wav",
            "test_data/sample_call.wav"
        ]
    }
    
    missing = []
    for category, files in resources.items():
        print(f"\n{category}:")
        for file_path in files:
            full_path = os.path.join(project_dir, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (missing)")
                missing.append(file_path)
    
    return missing

def install_missing_packages(packages):
    """Install missing packages."""
    if not packages:
        return True
    
    print("\n=== Installing missing packages ===")
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    
    try:
        subprocess.check_call(cmd)
        print("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def fix_tccc_path():
    """Fix TCCC module path by creating .pth file in site-packages."""
    print("\n=== Fixing TCCC module path ===")
    try:
        # Get site-packages directory
        import site
        site_packages = site.getsitepackages()[0]
        
        # Create .pth file
        pth_file = os.path.join(site_packages, 'tccc.pth')
        with open(pth_file, 'w') as f:
            f.write(src_dir)
        
        print(f"Created {pth_file} pointing to {src_dir}")
        return True
    except Exception as e:
        print(f"Error fixing TCCC path: {e}")
        return False

def main():
    """Main verification function."""
    print("\n=== TCCC Environment Verification ===")
    
    is_venv = sys.prefix != sys.base_prefix
    print(f"\nVirtual environment: {'Active' if is_venv else 'Not active'}")
    
    if not is_venv:
        print("Warning: Not running in a virtual environment")
        venv_path = os.path.join(project_dir, 'venv')
        if os.path.exists(venv_path):
            print(f"Virtual environment found at {venv_path}")
            print(f"Activate with: source {os.path.join(venv_path, 'bin', 'activate')}")
        else:
            print("Virtual environment not found")
            print("Create with: python -m venv venv")
    
    # Check Python packages
    missing_packages = check_python_packages()
    
    # Install missing packages if any
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        install_missing_packages(missing_packages)
    
    # Check TCCC modules
    missing_modules = check_tccc_modules()
    
    # Fix TCCC path if needed
    if missing_modules and any("No module named 'tccc'" in str(m) for m in missing_modules):
        fix_tccc_path()
    
    # Check audio devices
    has_audio_devices = check_audio_devices()
    
    # Check file resources
    missing_resources = check_file_resources()
    
    # Summarize results
    print("\n=== Verification Summary ===")
    print(f"Python packages: {'✓ All present' if not missing_packages else f'✗ Missing {len(missing_packages)}'}")
    print(f"TCCC modules: {'✓ All importable' if not missing_modules else f'✗ Issues with {len(missing_modules)}'}")
    print(f"Audio devices: {'✓ Available' if has_audio_devices else '✗ No input devices found'}")
    print(f"File resources: {'✓ All present' if not missing_resources else f'✗ Missing {len(missing_resources)}'}")
    
    # Overall status
    if not missing_packages and not missing_modules and has_audio_devices and not missing_resources:
        print("\n✅ Environment is ready for TCCC development")
        return 0
    else:
        print("\n⚠️ Environment has issues that need to be fixed")
        return 1

if __name__ == "__main__":
    sys.exit(main())