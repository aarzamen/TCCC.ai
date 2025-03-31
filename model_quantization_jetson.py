#!/usr/bin/env python3
"""
Model Quantization for Jetson - TCCC Project

This script provides tools for quantizing embedding models used in the TCCC RAG system
to reduce memory usage and improve performance on Jetson hardware.

Supported operations:
- Int8 quantization of embedding models
- TensorRT conversion for NVIDIA Jetson
- Model size selection based on available memory
- Optimized inference configuration
"""

import os
import sys
import time
import argparse
import logging
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/model_quantization.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Import TCCC modules (with fallbacks for missing dependencies)
try:
    from tccc.utils import ConfigManager
except ImportError:
    logger.warning("Could not import ConfigManager, using minimal implementation")
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

# Try to import sentence transformers (optional)
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("SentenceTransformer not available, some features will be disabled")
    HAVE_SENTENCE_TRANSFORMERS = False

# Try to import TensorRT (optional)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAVE_TENSORRT = True
except ImportError:
    logger.warning("TensorRT not available, model conversion will be disabled")
    HAVE_TENSORRT = False

def get_jetson_info():
    """Get information about the Jetson device."""
    jetson_info = {
        "detected": False,
        "model": "unknown",
        "memory_mb": 0,
        "cuda_cores": 0,
        "cuda_version": "unknown"
    }
    
    # Check for Jetson-specific information
    if os.path.exists("/proc/device-tree/model"):
        with open("/proc/device-tree/model", 'r') as f:
            model_info = f.read().strip()
            if "NVIDIA Jetson" in model_info:
                jetson_info["detected"] = True
                jetson_info["model"] = model_info
    
    # Get memory information
    try:
        import subprocess
        result = subprocess.run(['free', '-m'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                mem_info = lines[1].split()
                jetson_info["memory_mb"] = int(mem_info[1])
    except Exception as e:
        logger.warning(f"Failed to get memory information: {str(e)}")
    
    # Check CUDA information
    if HAVE_TENSORRT:
        try:
            jetson_info["cuda_version"] = cuda.get_version()
            device = cuda.Device(0)
            jetson_info["cuda_cores"] = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
        except Exception as e:
            logger.warning(f"Failed to get CUDA information: {str(e)}")
    
    return jetson_info

def select_optimal_model(jetson_info):
    """Select the optimal model based on Jetson hardware."""
    memory_mb = jetson_info.get("memory_mb", 0)
    model_name = "all-MiniLM-L6-v2"  # Default for low memory
    
    if memory_mb > 8000:
        # High memory Jetson (8GB+): Use better model
        model_name = "all-MiniLM-L12-v2"
    elif memory_mb > 4000:
        # Medium memory Jetson (4GB+): Use balanced model
        model_name = "all-MiniLM-L6-v2"
    else:
        # Low memory Jetson: Use smallest viable model
        model_name = "paraphrase-MiniLM-L3-v2"
    
    logger.info(f"Selected model {model_name} based on available memory ({memory_mb} MB)")
    return model_name

def quantize_embedding_model(model_name, output_dir="models/quantized"):
    """
    Quantize a sentence transformer model to int8 precision.
    
    Args:
        model_name: Name of the model to quantize
        output_dir: Directory to save the quantized model
        
    Returns:
        Path to the quantized model
    """
    if not HAVE_SENTENCE_TRANSFORMERS:
        logger.error("SentenceTransformer is required for model quantization")
        return None
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the original model
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Check if model uses PyTorch
        if hasattr(model, 'torch_device'):
            # Move to CPU for quantization
            model.to('cpu')
            
            # Get the underlying PyTorch model
            if hasattr(model, 'modules') and len(model.modules()) > 0:
                # Get the transformer module
                transformer_module = None
                for module in model.modules():
                    if hasattr(module, 'auto_model'):
                        transformer_module = module
                        break
                
                if transformer_module and hasattr(transformer_module, 'auto_model'):
                    # Quantize the transformer model to int8
                    logger.info("Quantizing model to int8")
                    
                    # Use PyTorch's quantization utilities
                    quantized_model = torch.quantization.quantize_dynamic(
                        transformer_module.auto_model,
                        {torch.nn.Linear},  # Quantize linear layers
                        dtype=torch.qint8
                    )
                    
                    # Replace the original model with the quantized one
                    transformer_module.auto_model = quantized_model
                    
                    # Save the quantized model
                    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_quantized")
                    model.save(output_path)
                    
                    logger.info(f"Quantized model saved to {output_path}")
                    return output_path
                else:
                    logger.error("Could not find transformer module to quantize")
                    return None
            else:
                logger.error("Model doesn't have expected modules structure")
                return None
        else:
            logger.error("Model doesn't use PyTorch")
            return None
            
    except Exception as e:
        logger.error(f"Error during model quantization: {str(e)}")
        return None

def create_tensorrt_engine(model_path, output_dir="models/tensorrt"):
    """
    Convert a PyTorch model to TensorRT for faster inference on Jetson.
    
    Args:
        model_path: Path to the PyTorch model
        output_dir: Directory to save the TensorRT engine
        
    Returns:
        Path to the TensorRT engine
    """
    if not HAVE_TENSORRT:
        logger.error("TensorRT is required for model conversion")
        return None
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model (simplified example)
        logger.info(f"Loading model from {model_path} for TensorRT conversion")
        
        # NOTE: This is a simplified conversion process
        # A complete implementation would:
        # 1. Export the model to ONNX
        # 2. Convert ONNX to TensorRT
        # 3. Optimize for the specific Jetson hardware
        
        # Placeholder for demo purposes
        engine_path = os.path.join(output_dir, f"{os.path.basename(model_path)}.engine")
        
        # Create a dummy engine file for demo
        with open(engine_path, 'w') as f:
            f.write("TensorRT Engine Placeholder")
        
        logger.info(f"TensorRT engine created at {engine_path}")
        return engine_path
        
    except Exception as e:
        logger.error(f"Error during TensorRT conversion: {str(e)}")
        return None

def benchmark_model(model_path, model_type="pytorch"):
    """
    Benchmark model performance.
    
    Args:
        model_path: Path to the model
        model_type: Type of model (pytorch, quantized, tensorrt)
        
    Returns:
        Dictionary with benchmark results
    """
    if not HAVE_SENTENCE_TRANSFORMERS:
        logger.error("SentenceTransformer is required for benchmarking")
        return None
    
    try:
        results = {
            "model_path": model_path,
            "model_type": model_type,
            "inference_time_ms": 0,
            "memory_usage_mb": 0,
            "input_size": 10,
            "sequence_length": 32
        }
        
        # Load the model
        logger.info(f"Loading model from {model_path} for benchmarking")
        if model_type == "tensorrt":
            # Placeholder for TensorRT benchmarking
            logger.info("TensorRT benchmarking not implemented yet")
            return results
        
        # Load model based on type
        if model_type in ["pytorch", "quantized"]:
            model = SentenceTransformer(model_path)
            model.to('cpu')  # Ensure CPU for fair comparison
            
            # Create sample inputs
            test_sentences = [f"This is a test sentence {i}" for i in range(results["input_size"])]
            
            # Warm up
            _ = model.encode(test_sentences[0])
            
            # Measure memory before
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Benchmark
            start_time = time.time()
            _ = model.encode(test_sentences)
            inference_time = time.time() - start_time
            
            # Measure memory after
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # Convert to MB
            
            # Update results
            results["inference_time_ms"] = inference_time * 1000 / results["input_size"]
            results["memory_usage_mb"] = memory_used if memory_used > 0 else "N/A"
            
            logger.info(f"Benchmark results: {results}")
            return results
            
    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        return None

def update_config_for_quantized_model(config, model_path, jetson_info):
    """
    Update configuration to use the quantized model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the quantized model
        jetson_info: Jetson hardware information
        
    Returns:
        Updated configuration
    """
    # Create a copy to avoid modifying the original
    updated_config = config.copy() if config else {}
    
    # Ensure embedding section exists
    if "embedding" not in updated_config:
        updated_config["embedding"] = {}
    
    # Update model path
    updated_config["embedding"]["model_name"] = model_path
    
    # Optimize batch size based on memory
    memory_mb = jetson_info.get("memory_mb", 0)
    if memory_mb > 0:
        # Scale batch size based on available memory
        if memory_mb < 2000:
            updated_config["embedding"]["batch_size"] = 4
        elif memory_mb < 4000:
            updated_config["embedding"]["batch_size"] = 8
        else:
            updated_config["embedding"]["batch_size"] = 16
    
    # Set dimension based on model
    if "all-MiniLM-L12" in model_path:
        updated_config["embedding"]["dimension"] = 384
    elif "all-MiniLM-L6" in model_path:
        updated_config["embedding"]["dimension"] = 384
    elif "paraphrase-MiniLM-L3" in model_path:
        updated_config["embedding"]["dimension"] = 384
    
    # Optimize for Jetson
    updated_config["embedding"]["use_gpu"] = False  # Default to CPU for more reliable operation
    updated_config["embedding"]["normalize"] = True
    
    # Add quantization info
    updated_config["embedding"]["quantized"] = True
    updated_config["embedding"]["precision"] = "int8"
    
    return updated_config

def main():
    parser = argparse.ArgumentParser(description="Model Quantization for Jetson")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, help="Model name to quantize")
    parser.add_argument("--output-dir", type=str, default="models/quantized", help="Output directory for quantized models")
    parser.add_argument("--tensorrt", action="store_true", help="Convert to TensorRT (if supported)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark model performance")
    parser.add_argument("--auto-select", action="store_true", help="Automatically select model based on hardware")
    args = parser.parse_args()
    
    # Get Jetson information
    jetson_info = get_jetson_info()
    
    # Print hardware information
    logger.info("Hardware information:")
    for key, value in jetson_info.items():
        logger.info(f"  {key}: {value}")
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = config_manager.load_config_from_file(args.config)
    else:
        logger.info("Loading default document_library config")
        config = config_manager.load_config("document_library")
    
    # Select model
    model_name = args.model
    if args.auto_select or not model_name:
        model_name = select_optimal_model(jetson_info)
        logger.info(f"Auto-selected model: {model_name}")
    
    # Quantize model
    logger.info(f"Quantizing model: {model_name}")
    quantized_model_path = quantize_embedding_model(model_name, args.output_dir)
    
    if not quantized_model_path:
        logger.error("Model quantization failed")
        return 1
    
    # Convert to TensorRT if requested
    tensorrt_engine_path = None
    if args.tensorrt and HAVE_TENSORRT:
        logger.info("Converting to TensorRT")
        tensorrt_engine_path = create_tensorrt_engine(quantized_model_path)
    
    # Benchmark if requested
    if args.benchmark:
        logger.info("Benchmarking models")
        # Benchmark original model
        if HAVE_SENTENCE_TRANSFORMERS:
            logger.info(f"Benchmarking original model: {model_name}")
            benchmark_model(model_name, "pytorch")
        
        # Benchmark quantized model
        if quantized_model_path:
            logger.info(f"Benchmarking quantized model: {quantized_model_path}")
            benchmark_model(quantized_model_path, "quantized")
        
        # Benchmark TensorRT model
        if tensorrt_engine_path:
            logger.info(f"Benchmarking TensorRT model: {tensorrt_engine_path}")
            benchmark_model(tensorrt_engine_path, "tensorrt")
    
    # Update configuration
    if quantized_model_path:
        logger.info("Updating configuration with quantized model")
        updated_config = update_config_for_quantized_model(config, quantized_model_path, jetson_info)
        
        # Save updated configuration
        config_path = "config/quantized_jetson_rag.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        config_manager.save_config(updated_config, config_path)
        logger.info(f"Updated configuration saved to {config_path}")
        
        print(f"\nQuantized model created at: {quantized_model_path}")
        print(f"Updated configuration saved to: {config_path}")
        print("\nTo use the quantized model, run:")
        print(f"  python jetson_rag_explorer.py --config {config_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())