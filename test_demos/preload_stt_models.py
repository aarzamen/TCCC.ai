#!/usr/bin/env python3
"""
Speech-to-Text Model Preloader for TCCC.ai system.

This script preloads speech recognition models in the background to reduce
startup time for the main application. It should be run at system startup
or as part of the initialization process.
"""

import os
import sys
import time
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/preload_models.log')
    ]
)
logger = logging.getLogger("ModelPreloader")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preload STT models")
    parser.add_argument("--models", nargs="+", default=["tiny.en"], 
                        choices=["tiny", "tiny.en", "base", "small", "medium", "large"],
                        help="Model sizes to preload")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Device to load models on")
    parser.add_argument("--compute-type", choices=["float16", "float32", "int8"], 
                        default="float16", help="Compute type for model inference")
    return parser.parse_args()

def preload_models(models, device, compute_type):
    """
    Preload STT models into the model cache.
    
    Args:
        models: List of model sizes to preload
        device: Device to load models on (cpu or cuda)
        compute_type: Compute type for model inference
    """
    try:
        from tccc.stt_engine.model_cache_manager import get_model_cache_manager
        
        logger.info(f"Starting preload for models: {models}")
        
        # Get model cache manager
        cache_manager = get_model_cache_manager()
        initial_status = cache_manager.get_status()
        logger.info(f"Initial cache status: {initial_status['cache_size']} models cached")
        
        # Preload each model
        for model_size in models:
            logger.info(f"Preloading model: {model_size}")
            
            # Set up model configuration
            model_config = {
                "size": model_size,
                "device": device,
                "compute_type": compute_type,
                "language": "en",
                "type": "faster-whisper"
            }
            
            # Force manual cleanup before preloading large models
            if model_size in ["medium", "large"] and initial_status['cache_size'] > 0:
                logger.info("Cleaning up cache before loading large model")
                cache_manager._cleanup_unused_models()
            
            # Preload the model
            start_time = time.time()
            success = cache_manager.preload_model("faster-whisper", model_config)
            end_time = time.time()
            
            if success:
                logger.info(f"Successfully preloaded {model_size} in {end_time - start_time:.2f}s")
            else:
                logger.error(f"Failed to preload {model_size}")
        
        # Get final cache status
        final_status = cache_manager.get_status()
        logger.info(f"Final cache status: {final_status['cache_size']} models cached")
        logger.info("Models preloaded successfully")
        
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Error preloading models: {e}")
        return False
        
    return True

def main():
    """Main entry point."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Parse command line arguments
    args = parse_args()
    
    logger.info(f"Starting STT model preloader with models: {args.models}")
    
    # Check for virtual environment
    venv_active = True if os.environ.get('VIRTUAL_ENV') else False
    if not venv_active:
        logger.warning("Virtual environment not active, attempting to activate")
        venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'activate')
        if os.path.exists(venv_path):
            try:
                # This doesn't actually work within the script, but we can try
                os.system(f"source {venv_path}")
                logger.info("Attempted to activate virtual environment")
            except Exception:
                pass
    
    # Check dependencies
    try:
        import torch
        from faster_whisper import WhisperModel
        logger.info("Dependencies check passed")
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Please run: pip install faster-whisper torch")
        return 1
    
    # Check CUDA availability if requested
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                args.device = "cpu"
                # For CPU, use int8 for better performance
                if args.compute_type == "float16":
                    args.compute_type = "int8"
                    logger.info("Switching to int8 compute type for CPU")
        except ImportError:
            logger.warning("Could not check CUDA availability, assuming CPU")
            args.device = "cpu"
    
    # Check available RAM and adjust models if needed
    try:
        import psutil
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / (1024 * 1024 * 1024)
        
        logger.info(f"System has {available_gb:.2f}GB available RAM")
        
        # Adjust models based on available memory
        if available_gb < 4.0 and "large" in args.models:
            logger.warning("Less than 4GB RAM available, removing large model")
            args.models.remove("large")
            
        if available_gb < 2.0 and "medium" in args.models:
            logger.warning("Less than 2GB RAM available, removing medium model")
            args.models.remove("medium")
            
        if available_gb < 1.0 and len(args.models) > 1:
            logger.warning("Less than 1GB RAM available, keeping only smallest model")
            smallest_model = min(args.models, key=lambda x: ["tiny", "tiny.en", "base", "small", "medium", "large"].index(x))
            args.models = [smallest_model]
    except ImportError:
        logger.warning("Could not check system memory, continuing with requested models")
    
    # Preload the models
    success = preload_models(args.models, args.device, args.compute_type)
    
    if success:
        logger.info("Model preloading completed successfully")
        return 0
    else:
        logger.error("Model preloading failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())