"""
Phi-2 GGUF model implementation for TCCC.ai system.

This module provides a real implementation of Microsoft's Phi-2 model 
using GGUF format for LLM analysis optimized for Jetson Orin Nano hardware.
"""

import os
import json
import time
import logging
import traceback
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import contextlib

# Local imports
from tccc.utils.logging import get_logger

logger = get_logger(__name__)

# Try importing llama-cpp-python for GGUF support
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. Please install with: pip install llama-cpp-python")

class PhiGGUFModel:
    """
    Implementation of Microsoft's Phi-2 model for low-resource deployment
    on Jetson Orin Nano hardware using GGUF format through llama-cpp-python.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Phi-2 GGUF model.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model_path = Path(config.get("gguf_model_path", "models/phi-2-gguf/phi-2.Q4_K_M.gguf"))
        self.use_gpu = config.get("use_gpu", True)
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.num_threads = config.get("num_threads", os.cpu_count())
        
        # Initialize model state
        self.model = None
        
        # Metrics for tracking
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency": 0.0
        }
        
        # Load model
        self._setup_gpu()
        self._load_model()
    
    def _setup_gpu(self):
        """Set up GPU configuration for model inference."""
        if not self.use_gpu:
            logger.info("GPU usage disabled, using CPU")
            return
        
        try:
            import torch
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                logger.info(f"Using GPU: {device_props.name}")
                logger.info(f"GPU Memory: {device_props.total_memory / 1024**2:.0f}MB")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self.use_gpu = False
        except Exception as e:
            logger.error(f"Error setting up GPU: {str(e)}")
            logger.warning("Falling back to CPU-only mode")
            self.use_gpu = False
    
    def _load_model(self):
        """Load the Phi-2 GGUF model."""
        try:
            # Check if GGUF support is available
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("llama-cpp-python not available")
                
            # Check if model file exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            logger.info(f"Loading Phi-2 GGUF model from {self.model_path}")
            
            # Set up GPU options
            if self.use_gpu:
                n_gpu_layers = -1  # Use all layers on GPU
                logger.info("Using GPU for GGUF model")
            else:
                n_gpu_layers = 0  # CPU only
                logger.info("Using CPU for GGUF model")
            
            # Load model with llama-cpp-python
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,  # Context window size
                n_parts=1,  # Number of parts to split the model into
                n_gpu_layers=n_gpu_layers,
                n_threads=self.num_threads,
                verbose=False
            )
            
            logger.info(f"Phi-2 GGUF model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Phi-2 GGUF model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load Phi-2 GGUF model: {str(e)}")
    
    def _prepare_prompt_for_phi(self, prompt: str) -> str:
        """Format the prompt for Phi-2 model.
        
        Args:
            prompt: Raw input prompt
            
        Returns:
            Formatted prompt for the model
        """
        # Phi-2 works well with this instruct format
        formatted_prompt = f"""<|system|>
You are an AI medical assistant for military medics, specializing in Tactical Combat Casualty Care.
Analyze the following transcript and extract the requested information accurately.
<|user|>
{prompt}
<|assistant|>"""
        
        return formatted_prompt
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None) -> Dict[str, Any]:
        """Generate text based on the prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling value
            
        Returns:
            Dictionary with generated text
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        # Set generation parameters
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        try:
            # Prepare prompt
            formatted_prompt = self._prepare_prompt_for_phi(prompt)
            
            # Generate text with llama-cpp
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False  # Don't include the prompt in the response
            )
            
            # Extract generated text
            generated_text = response["choices"][0]["text"].strip()
            
            # Update token metrics
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            total_tokens = prompt_tokens + completion_tokens
            
            self.metrics["total_tokens"] += total_tokens
            
            # Update average latency
            elapsed = time.time() - start_time
            self.metrics["avg_latency"] = (
                (self.metrics["avg_latency"] * (self.metrics["total_requests"] - 1) + elapsed) / 
                self.metrics["total_requests"]
            )
            
            return {
                "id": str(uuid.uuid4()),
                "choices": [{"text": generated_text}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "model": "phi-2-gguf",
                "latency": elapsed
            }
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model usage metrics.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            "total_requests": self.metrics["total_requests"],
            "total_tokens": int(self.metrics["total_tokens"]),
            "avg_latency": round(self.metrics["avg_latency"], 3),
            "model": "phi-2-gguf",
            "use_gpu": self.use_gpu,
            "num_threads": self.num_threads
        }

def get_phi_gguf_model(config: Dict[str, Any]) -> Union[PhiGGUFModel, 'MockPhiModel']:
    """Factory function to get the appropriate Phi model implementation.
    
    Tries to load the real PhiGGUFModel implementation, but falls back to
    the MockPhiModel if the real model can't be loaded or if mock mode
    is explicitly enabled.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PhiGGUFModel or MockPhiModel instance
    """
    # Check if mock mode is explicitly enabled
    use_mock = os.environ.get("TCCC_USE_MOCK_LLM", "0") == "1"
    
    if use_mock:
        logger.info("Mock mode enabled through environment variable")
        from tccc.llm_analysis.mock_llm import MockPhiModel
        return MockPhiModel(config)
    
    try:
        # Try loading the real model
        return PhiGGUFModel(config)
    except Exception as e:
        logger.warning(f"Failed to load real Phi GGUF model: {str(e)}")
        logger.info("Falling back to mock implementation")
        
        # Fall back to mock implementation
        from tccc.llm_analysis.mock_llm import MockPhiModel
        return MockPhiModel(config)