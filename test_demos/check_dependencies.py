#!/usr/bin/env python3
"""
Dependency checker for TCCC.ai project.

This script checks for the presence of all required dependencies and provides
information about their availability and status.
"""

import importlib
import logging
import os
import sys
import subprocess
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DependencyChecker")

# Define core dependencies with their minimum versions
CORE_DEPENDENCIES = {
    "numpy": "1.20.0",
    "torch": "2.0.0",
    "torchaudio": "2.0.0",
    "transformers": "4.36.0",
    "optimum": "1.16.0",
    "accelerate": "0.24.0",
    "faster-whisper": "0.9.0",
    "onnxruntime": "1.15.0",
    "faiss-cpu": "1.10.0",
    "nltk": "3.8.1",
    "pyaudio": "0.2.13",
    "sounddevice": "0.4.5",
    "librosa": "0.10.0",
    "bitsandbytes": "0.41.0",
    "soundfile": "0.12.0",
    "pydantic": "2.0.0",
    "fastapi": "0.100.0",
    "pymongo": "4.4.1",
    "uvicorn": "0.22.0",
    "websockets": "11.0.3",
}

# Specialized dependencies that may not be available on all platforms
OPTIONAL_DEPENDENCIES = {
    "onnxruntime-gpu": "1.15.0",
    "nvidia-cuda-runtime-cu12": "12.1.105",
    "nvidia-cublas-cu12": "12.1.3.1",
    "nvidia-cudnn-cu12": "8.9.2.26",
    "tensorrt": "8.6.0",
    "jetson-stats": "4.2.3",
    "jtop": "4.2.3",
    "jetson-inference": "2.1.0",
    "jetson-utils": "2.1.0",
    "pyannote.audio": "3.0.0",
}

def check_dependency(name: str, min_version: str = None) -> Tuple[bool, str, str]:
    """
    Check if a dependency is available and meets the minimum version requirement.
    
    Args:
        name: Name of the dependency
        min_version: Minimum required version (optional)
        
    Returns:
        Tuple of (is_available, installed_version, status_message)
    """
    try:
        # Special case for CUDA
        if name == "cuda":
            return check_cuda_availability()
            
        # Special case for faiss-cpu
        if name == "faiss-cpu":
            try:
                import faiss
                return True, faiss.__version__, "Available"
            except ImportError:
                return False, "Not installed", "Not available"
        
        # Import the module
        module = importlib.import_module(name.replace("-", "_"))
        
        # Get version
        if hasattr(module, "__version__"):
            version = module.__version__
        elif hasattr(module, "version"):
            version = module.version()
        else:
            version = "Unknown"
        
        # Check version if required
        if min_version and version != "Unknown":
            try:
                meets_version = tuple(map(int, version.split("."))) >= tuple(map(int, min_version.split(".")))
                if not meets_version:
                    return False, version, f"Installed version {version} does not meet minimum requirement {min_version}"
            except ValueError:
                # Can't compare versions properly, just warn
                logger.warning(f"Could not compare versions for {name}: {version} vs {min_version}")
        
        return True, version, "Available"
    
    except ImportError:
        return False, "Not installed", "Not available"
    
    except Exception as e:
        return False, "Error", f"Error checking: {str(e)}"

def check_cuda_availability() -> Tuple[bool, str, str]:
    """Check if CUDA is available."""
    try:
        import torch
        
        if torch.cuda.is_available():
            return True, torch.version.cuda, "Available"
        else:
            return False, "N/A", "CUDA not available"
    
    except ImportError:
        return False, "N/A", "PyTorch not installed"
    
    except Exception as e:
        return False, "Error", f"Error checking CUDA: {str(e)}"

def check_gpu_details() -> Dict[str, Any]:
    """Get GPU details if available."""
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": []
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["count"]):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "name": props.name,
                    "memory": f"{props.total_memory / 1024**3:.2f} GB",
                    "compute_capability": f"{props.major}.{props.minor}"
                })
        
        return gpu_info
    
    except Exception as e:
        logger.error(f"Error retrieving GPU details: {str(e)}")
        return gpu_info

def check_all_dependencies() -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    Check all dependencies and return their status.
    
    Returns:
        Dictionary with dependency categories and their status information
    """
    results = {
        "core": [],
        "optional": [],
        "hardware": []
    }
    
    # Check core dependencies
    for name, min_version in CORE_DEPENDENCIES.items():
        available, version, status = check_dependency(name, min_version)
        category = "core"
        results[category].append((name, min_version, version, status))
    
    # Check optional dependencies
    for name, min_version in OPTIONAL_DEPENDENCIES.items():
        available, version, status = check_dependency(name, min_version)
        category = "optional"
        results[category].append((name, min_version, version, status))
    
    # Check hardware dependencies
    available, version, status = check_cuda_availability()
    results["hardware"].append(("CUDA", "N/A", version, status))
    
    # Check for ARM/Jetson environment
    is_arm = "aarch64" in os.uname().machine
    jetson_status = "Detected" if is_arm else "Not detected"
    results["hardware"].append(("Jetson/ARM", "N/A", os.uname().machine, jetson_status))
    
    return results

def print_results(results: Dict[str, List[Tuple[str, str, str, str]]]):
    """Print dependency check results in a formatted way."""
    print("\nDependency Check Results\n")
    
    categories = [
        ("Core Dependencies", results["core"]),
        ("Optional Dependencies", results["optional"]),
        ("Hardware", results["hardware"])
    ]
    
    for category_name, dependencies in categories:
        print(f"\n{category_name}:")
        print("-" * 80)
        print(f"{'Dependency':<30} {'Min Version':<15} {'Installed':<15} {'Status':<20}")
        print("-" * 80)
        
        for name, min_version, version, status in dependencies:
            print(f"{name:<30} {min_version:<15} {version:<15} {status:<20}")
    
    # Print GPU details if available
    gpu_info = check_gpu_details()
    if gpu_info["available"]:
        print("\nGPU Information:")
        print("-" * 80)
        print(f"GPU Count: {gpu_info['count']}")
        
        for i, device in enumerate(gpu_info["devices"]):
            print(f"\nGPU {i}:")
            print(f"  Name: {device['name']}")
            print(f"  Memory: {device['memory']}")
            print(f"  Compute Capability: {device['compute_capability']}")
    else:
        print("\nNo GPU detected")

def main():
    """Main entry point for dependency checker."""
    try:
        print("\nChecking TCCC.ai dependencies...")
        results = check_all_dependencies()
        print_results(results)
        
        # Check if all core dependencies are available
        missing_core = [name for name, _, version, _ in results["core"] if version == "Not installed"]
        
        if missing_core:
            print("\n⚠️  Missing core dependencies:", ", ".join(missing_core))
            print("Run: pip install " + " ".join(missing_core))
            return 1
        else:
            print("\n✅ All core dependencies are installed!")
            return 0
            
    except Exception as e:
        logger.error(f"Error during dependency check: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())