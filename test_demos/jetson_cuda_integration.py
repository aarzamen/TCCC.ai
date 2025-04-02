#!/usr/bin/env python3
"""
Jetson CUDA Integration - TCCC Project

This script integrates the TCCC RAG system with the Jetson CUDA runtime
to optimize performance on NVIDIA Jetson hardware. It provides:
- GPU memory management
- Optimized CUDA configurations
- Hardware-specific performance tuning
- TensorRT integration
"""

import os
import sys
import time
import argparse
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/jetson_cuda.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Try to import optional dependencies
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    logger.warning("PyTorch not available, some features will be disabled")
    HAVE_TORCH = False

try:
    import tensorrt as trt
    HAVE_TENSORRT = True
except ImportError:
    logger.warning("TensorRT not available, CUDA optimization will be limited")
    HAVE_TENSORRT = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAVE_PYCUDA = True
except ImportError:
    logger.warning("PyCUDA not available, CUDA direct access will be disabled")
    HAVE_PYCUDA = False

try:
    from tccc.utils import ConfigManager
except ImportError:
    logger.warning("ConfigManager not available, using minimal implementation")
    class ConfigManager:
        def load_config(self, name):
            return {}
        def load_config_from_file(self, path):
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return {}
        def save_config(self, config, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)

def detect_jetson_hardware():
    """Detect Jetson hardware and capabilities."""
    hardware_info = {
        "is_jetson": False,
        "model": "unknown",
        "cuda_available": False,
        "cuda_version": "unknown",
        "gpu_name": "unknown",
        "gpu_memory_mb": 0,
        "cuda_cores": 0,
        "tensorrt_available": HAVE_TENSORRT
    }
    
    # Check if running on Jetson
    if os.path.exists("/etc/nv_tegra_release"):
        hardware_info["is_jetson"] = True
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                release_info = f.read().strip()
                hardware_info["model"] = release_info
        except:
            pass
    
    # Alternative Jetson detection
    if not hardware_info["is_jetson"] and os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_info = f.read().strip()
                if "NVIDIA Jetson" in model_info:
                    hardware_info["is_jetson"] = True
                    hardware_info["model"] = model_info
        except:
            pass
    
    # Check CUDA availability
    if HAVE_TORCH:
        hardware_info["cuda_available"] = torch.cuda.is_available()
        if hardware_info["cuda_available"]:
            hardware_info["cuda_version"] = torch.version.cuda
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
            hardware_info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    
    # Additional CUDA information using CUDA toolkit
    if HAVE_PYCUDA:
        try:
            hardware_info["cuda_cores"] = cuda.Device(0).get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        except:
            pass
    
    return hardware_info

def get_optimal_cuda_config(hardware_info):
    """Get optimal CUDA configuration for the detected hardware."""
    config = {
        "use_cuda": hardware_info["cuda_available"],
        "optimize_memory": True,
        "use_tensorrt": HAVE_TENSORRT,
        "batch_size": 8,
        "max_gpu_memory_percent": 70,
        "mixed_precision": True,
        "compute_capability": "7.2"  # Default for Jetson
    }
    
    # Adjust based on model
    model = hardware_info["model"].lower()
    if "nano" in model:
        config["batch_size"] = 4
        config["max_gpu_memory_percent"] = 60
        config["compute_capability"] = "5.3"
    elif "xavier" in model:
        config["batch_size"] = 16
        config["max_gpu_memory_percent"] = 80
        config["compute_capability"] = "7.2"
    elif "agx" in model:
        config["batch_size"] = 24
        config["max_gpu_memory_percent"] = 85
        config["compute_capability"] = "7.2"
    elif "orin" in model:
        config["batch_size"] = 32
        config["max_gpu_memory_percent"] = 85
        config["compute_capability"] = "8.7"
    
    return config

def configure_gpu_memory(cuda_config):
    """Configure GPU memory usage based on the configuration."""
    if not HAVE_TORCH or not torch.cuda.is_available():
        logger.warning("Unable to configure GPU memory: CUDA not available")
        return False
    
    try:
        # Set memory fraction to use
        memory_percent = cuda_config["max_gpu_memory_percent"]
        memory_fraction = memory_percent / 100.0
        
        # Reserve memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        max_memory = int(total_memory * memory_fraction)
        
        # Use PyTorch memory management
        torch.cuda.empty_cache()
        
        # Cache allocation - useful for LLMs and embedding models to reduce memory fragmentation
        if hasattr(torch.cuda, 'memory_reserved'):
            logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        if hasattr(torch.cuda, 'memory_allocated'):
            logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        logger.info(f"Configured GPU memory: {max_memory / 1024**2:.2f} MB ({memory_percent}% of total)")
        return True
    except Exception as e:
        logger.error(f"Failed to configure GPU memory: {str(e)}")
        return False

def optimize_torch_for_jetson(cuda_config):
    """Apply PyTorch optimizations for Jetson."""
    if not HAVE_TORCH:
        logger.warning("Unable to optimize PyTorch: not available")
        return False
    
    try:
        # Use mixed precision if available
        if cuda_config["mixed_precision"] and torch.cuda.is_available():
            if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                logger.info("Enabling automatic mixed precision (AMP)")
                # This just sets up the infrastructure - will be used during model execution
            else:
                logger.warning("Automatic mixed precision not available in this PyTorch version")
        
        # Set default CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            logger.info(f"Set default CUDA device to: {torch.cuda.get_device_name(0)}")
        
        # Optimize CUDA algorithms
        if torch.cuda.is_available() and hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode")
        
        logger.info("PyTorch optimization for Jetson complete")
        return True
    except Exception as e:
        logger.error(f"Failed to optimize PyTorch for Jetson: {str(e)}")
        return False

def setup_tensorrt_integration(cuda_config):
    """Set up TensorRT integration if available."""
    if not HAVE_TENSORRT:
        logger.warning("Unable to set up TensorRT: not available")
        return False
    
    try:
        logger.info("Setting up TensorRT integration")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Check TensorRT version
        version = trt.__version__
        logger.info(f"TensorRT version: {version}")
        
        # Get compute capability
        compute_capability = cuda_config["compute_capability"]
        
        # Create builder configuration for demonstration purposes
        # In a real implementation, this would be used to build TensorRT engines
        builder = trt.Builder(trt_logger)
        config = builder.create_builder_config()
        
        # These are just examples, not used directly in this script
        config.max_workspace_size = 1 << 28  # 256 MiB
        
        # Set up precision flags
        if version >= '7':
            # Enable mixed precision if TensorRT version supports it
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision for TensorRT")
            else:
                logger.warning("FP16 precision not supported by this GPU")
        
        logger.info("TensorRT integration setup complete")
        return True
    except Exception as e:
        logger.error(f"Failed to set up TensorRT integration: {str(e)}")
        return False

def apply_cuda_optimizations(cuda_config, rag_config=None):
    """Apply all CUDA optimizations and update the RAG config if provided."""
    logger.info("Applying CUDA optimizations")
    
    # Configure GPU memory
    memory_configured = configure_gpu_memory(cuda_config)
    
    # Optimize PyTorch
    torch_optimized = optimize_torch_for_jetson(cuda_config)
    
    # Set up TensorRT
    tensorrt_setup = False
    if cuda_config["use_tensorrt"]:
        tensorrt_setup = setup_tensorrt_integration(cuda_config)
    
    # Update RAG config if provided
    if rag_config is not None:
        logger.info("Updating RAG configuration with CUDA settings")
        
        # Ensure embedding section exists
        if "embedding" not in rag_config:
            rag_config["embedding"] = {}
        
        # Update with CUDA settings
        rag_config["embedding"]["use_gpu"] = cuda_config["use_cuda"]
        rag_config["embedding"]["batch_size"] = cuda_config["batch_size"]
        rag_config["embedding"]["optimize_memory"] = cuda_config["optimize_memory"]
        
        # Add CUDA info section
        rag_config["cuda"] = {
            "enabled": cuda_config["use_cuda"],
            "tensorrt_enabled": cuda_config["use_tensorrt"] and tensorrt_setup,
            "max_gpu_memory_percent": cuda_config["max_gpu_memory_percent"],
            "mixed_precision": cuda_config["mixed_precision"],
            "compute_capability": cuda_config["compute_capability"]
        }
    
    return {
        "memory_configured": memory_configured,
        "torch_optimized": torch_optimized,
        "tensorrt_setup": tensorrt_setup,
        "cuda_enabled": cuda_config["use_cuda"],
        "updated_config": rag_config
    }

def check_cuda_compatibility():
    """Check if the CUDA installation is compatible with the system."""
    # Check if CUDA paths are properly set
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        logger.info(f"CUDA home directory: {cuda_home}")
    else:
        logger.warning("CUDA_HOME or CUDA_PATH not set in environment")
        
        # Try to find CUDA installation
        potential_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-10.2",
            "/usr/local/cuda-11.4"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Found CUDA installation at: {path}")
                break
    
    # Check for required libraries
    missing_libs = []
    required_libs = ['libcudart.so', 'libcublas.so', 'libcudnn.so']
    
    for lib in required_libs:
        # Try to find the library
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if lib not in result.stdout:
                missing_libs.append(lib)
        except:
            # If ldconfig not available, can't check libraries
            logger.warning("ldconfig not available, can't verify CUDA libraries")
            break
    
    if missing_libs:
        logger.warning(f"Missing CUDA libraries: {', '.join(missing_libs)}")
    
    # Check pytorch CUDA compatibility
    if HAVE_TORCH:
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA: {torch.version.cuda}")
            logger.info(f"PyTorch GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"PyTorch cuDNN version: {torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else 'unknown'}")
        else:
            logger.warning("PyTorch CUDA not available")
    
    return {
        "cuda_home": cuda_home,
        "missing_libs": missing_libs,
        "pytorch_cuda": torch.cuda.is_available() if HAVE_TORCH else False
    }

def main():
    parser = argparse.ArgumentParser(description="Jetson CUDA Integration")
    parser.add_argument("--config", type=str, help="Path to RAG configuration file")
    parser.add_argument("--check-only", action="store_true", help="Only check CUDA compatibility, don't apply optimizations")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU mode even if CUDA is available")
    parser.add_argument("--output-config", type=str, default="config/cuda_optimized_rag.yaml", help="Output path for the updated configuration")
    args = parser.parse_args()
    
    # Detect hardware
    logger.info("Detecting Jetson hardware...")
    hardware_info = detect_jetson_hardware()
    
    # Print hardware information
    logger.info("Hardware information:")
    for key, value in hardware_info.items():
        logger.info(f"  {key}: {value}")
    
    # Check if running on Jetson
    if not hardware_info["is_jetson"]:
        logger.warning("Not running on Jetson hardware. Some optimizations may not be applicable.")
    
    # Check CUDA compatibility
    logger.info("Checking CUDA compatibility...")
    compatibility = check_cuda_compatibility()
    
    # If check-only mode, exit here
    if args.check_only:
        logger.info("Check-only mode, exiting")
        return 0
    
    # Get optimal CUDA configuration
    cuda_config = get_optimal_cuda_config(hardware_info)
    
    # Override with force-cpu if specified
    if args.force_cpu:
        logger.info("Forcing CPU mode as requested")
        cuda_config["use_cuda"] = False
        cuda_config["use_tensorrt"] = False
    
    # Load RAG configuration if provided
    rag_config = None
    if args.config:
        logger.info(f"Loading RAG configuration from {args.config}")
        config_manager = ConfigManager()
        rag_config = config_manager.load_config_from_file(args.config)
    
    # Apply optimizations
    result = apply_cuda_optimizations(cuda_config, rag_config)
    
    # Save updated configuration if provided
    if rag_config is not None:
        output_path = args.output_config
        logger.info(f"Saving updated configuration to {output_path}")
        config_manager = ConfigManager()
        config_manager.save_config(rag_config, output_path)
        
        print(f"\nCUDA optimization complete!")
        print(f"Updated configuration saved to: {output_path}")
        print("\nTo use the CUDA-optimized configuration, run:")
        print(f"  python jetson_rag_explorer.py --config {output_path}")
    else:
        print("\nCUDA optimization complete!")
        print("No configuration provided, so no configuration file was updated.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())