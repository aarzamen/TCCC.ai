"""
Phi-2 model implementation for TCCC.ai system.

This module provides a real implementation of Microsoft's Phi-2 Instruct model 
for LLM analysis optimized for Jetson Orin Nano hardware.
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

import torch
import numpy as np

# Try importing optimized libraries
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Local imports
from tccc.utils.logging import get_logger

logger = get_logger(__name__)

class PhiModel:
    """
    Implementation of Microsoft's Phi-2 model for low-resource deployment
    on Jetson Orin Nano hardware.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Phi-2 model.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model_path = Path(config["model_path"])
        self.use_gpu = config.get("use_gpu", True)
        self.quantization = config.get("quantization", "4-bit")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        
        # Initialize model state
        self.model = None
        self.tokenizer = None
        
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
            if torch.cuda.is_available():
                # Set up TensorRT if available
                if TENSORRT_AVAILABLE and self.config.get("use_tensorrt", True):
                    logger.info("TensorRT available, will use for acceleration")
                    self._setup_tensorrt()
                    
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
    
    def _setup_tensorrt(self):
        """Set up TensorRT for optimized inference on Jetson."""
        try:
            if TENSORRT_AVAILABLE:
                logger.info(f"TensorRT version: {trt.__version__}")
                
                # Check for Jetson specific optimizations
                jetson_devices = ["tegra", "orin", "xavier"]
                device_name = torch.cuda.get_device_name(0).lower()
                is_jetson = any(device in device_name for device in jetson_devices)
                
                if is_jetson:
                    logger.info(f"Detected Jetson hardware: {device_name}")
                    # Apply Jetson-specific optimizations here
                    # For Orin Nano, we want to use FP16 or INT8 precision
                    if self.quantization == "8-bit":
                        logger.info("Using INT8 precision for TensorRT on Jetson")
                    else:
                        logger.info("Using FP16 precision for TensorRT on Jetson")
        except Exception as e:
            logger.error(f"Error setting up TensorRT: {str(e)}")
            logger.warning("Will proceed without TensorRT acceleration")
    
    def _load_model(self):
        """Load the Phi-2 model based on available optimizations."""
        try:
            # Check if model files exist
            tokenizer_path = self.model_path / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            
            # First try loading optimized ONNX model if available
            onnx_path = self.model_path / "model.onnx"
            if ONNX_AVAILABLE and onnx_path.exists():
                self._load_onnx_model(onnx_path)
            else:
                # Fall back to regular transformers loading
                self._load_transformers_model()
                
            logger.info(f"Phi-2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Phi-2 model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load Phi-2 model: {str(e)}")
    
    def _load_onnx_model(self, onnx_path: Path):
        """Load model using ONNX Runtime with optimizations for Jetson.
        
        Args:
            onnx_path: Path to the ONNX model file
        """
        logger.info(f"Loading Phi-2 model with ONNX Runtime from {onnx_path}")
        
        try:
            # Set up ONNX Runtime session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set execution providers based on hardware
            exec_providers = []
            if self.use_gpu:
                if TENSORRT_AVAILABLE and self.config.get("use_tensorrt", True):
                    exec_providers.append('TensorrtExecutionProvider')
                exec_providers.append('CUDAExecutionProvider')
            exec_providers.append('CPUExecutionProvider')
            
            # Create inference session
            self.model = ort.InferenceSession(
                str(onnx_path), 
                sess_options=sess_options,
                providers=exec_providers
            )
            
            # Load tokenizer
            self._load_tokenizer()
            
            logger.info(f"Model loaded with ONNX Runtime, providers: {exec_providers}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            logger.debug(traceback.format_exc())
            # Fall back to transformers
            logger.warning("Falling back to regular transformers model")
            self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Load model using Hugging Face Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Phi-2 model with transformers from {self.model_path}")
            
            # Determine precision based on quantization setting and hardware
            load_in_8bit = self.quantization in ["8-bit", "8bit", "int8"]
            load_in_4bit = self.quantization in ["4-bit", "4bit", "int4"]
            
            # Set dtype based on hardware capability
            if self.use_gpu:
                dtype = torch.float16  # Use FP16 on GPU
            else:
                dtype = torch.float32  # Use FP32 on CPU
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # Load model with appropriate quantization
            if load_in_8bit:
                # Handle 8-bit quantization
                logger.info("Loading model in 8-bit precision")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    device_map="auto" if self.use_gpu else "cpu",
                    torch_dtype=dtype,
                    load_in_8bit=True,
                    trust_remote_code=True
                )
            elif load_in_4bit:
                # Handle 4-bit quantization
                logger.info("Loading model in 4-bit precision")
                # Need to import specialized libraries for 4-bit
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    device_map="auto" if self.use_gpu else "cpu",
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
            else:
                # Regular loading
                logger.info(f"Loading model in {dtype} precision")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    device_map="auto" if self.use_gpu else "cpu",
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
            
            logger.info(f"Model loaded with transformers, quantization: {self.quantization}")
            
        except Exception as e:
            logger.error(f"Failed to load transformers model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load Phi-2 model: {str(e)}")
    
    def _load_tokenizer(self):
        """Load the tokenizer for the model."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer_path = self.model_path
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True
            )
            
            logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
    
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
            
            # Generate text based on model type
            if isinstance(self.model, ort.InferenceSession):
                # ONNX Runtime generation
                generated_text = self._generate_onnx(formatted_prompt, max_tokens, temperature, top_p)
            else:
                # Transformers generation
                generated_text = self._generate_transformers(formatted_prompt, max_tokens, temperature, top_p)
            
            # Update token metrics
            prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
            completion_tokens = len(self.tokenizer.encode(generated_text)) - prompt_tokens
            if completion_tokens < 0:
                completion_tokens = len(self.tokenizer.encode(generated_text))
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
                "model": "phi-2-instruct",
                "latency": elapsed
            }
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def _generate_onnx(self, prompt: str, max_tokens: int, 
                      temperature: float, top_p: float) -> str:
        """Generate text using ONNX Runtime.
        
        Args:
            prompt: Formatted prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling value
            
        Returns:
            Generated text
        """
        # This would implement generation with ONNX Runtime
        # For a proper implementation, this would handle tokenization,
        # feed tokens through the model, and sample new tokens repeatedly
        
        # For our implementation, we'll use a simplified approach
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        
        # Get input and output names
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        # Generate tokens one by one
        for _ in range(max_tokens):
            # Run model
            outputs = self.model.run(
                [output_name], 
                {input_name: input_ids}
            )[0]
            
            # Get next token (simplified sampling)
            next_token_logits = outputs[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-7)
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = np.sort(next_token_logits)[::-1], np.argsort(next_token_logits)[::-1]
            cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            next_token_logits[sorted_indices[sorted_indices_to_remove]] = -float('inf')
            
            # Sample token
            probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
            next_token = np.random.choice(len(probs), p=probs)
            
            # Append token
            input_ids = np.append(input_ids, [[next_token]], axis=1)
            
            # Check for end of generation
            if next_token == self.tokenizer.eos_token_id:
                break
        
        # Decode and return
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in generated_text:
            generated_text = generated_text.split("<|assistant|>")[1].strip()
        
        return generated_text
    
    def _generate_transformers(self, prompt: str, max_tokens: int, 
                              temperature: float, top_p: float) -> str:
        """Generate text using Hugging Face Transformers.
        
        Args:
            prompt: Formatted prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling value
            
        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to GPU if available
        if self.use_gpu:
            input_ids = input_ids.to("cuda")
        
        # Set up generation config
        from transformers import GenerationConfig
        
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                generation_config=generation_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in generated_text:
            generated_text = generated_text.split("<|assistant|>")[1].strip()
        
        return generated_text
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model usage metrics.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            "total_requests": self.metrics["total_requests"],
            "total_tokens": int(self.metrics["total_tokens"]),
            "avg_latency": round(self.metrics["avg_latency"], 3),
            "model": "phi-2-instruct",
            "quantization": self.quantization,
            "use_gpu": self.use_gpu
        }

def get_phi_model(config: Dict[str, Any]) -> Union[PhiModel, 'MockPhiModel']:
    """Factory function to get the appropriate Phi model implementation.
    
    Tries to load the real PhiModel implementation, but falls back to
    the MockPhiModel if the real model can't be loaded or if mock mode
    is explicitly enabled.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PhiModel or MockPhiModel instance
    """
    # Check if mock mode is explicitly enabled
    use_mock = os.environ.get("TCCC_USE_MOCK_LLM", "0") == "1"
    
    if use_mock:
        logger.info("Mock mode enabled through environment variable")
        from tccc.llm_analysis.mock_llm import MockPhiModel
        return MockPhiModel(config)
    
    try:
        # Try loading the real model
        return PhiModel(config)
    except Exception as e:
        logger.warning(f"Failed to load real Phi model: {str(e)}")
        logger.info("Falling back to mock implementation")
        
        # Fall back to mock implementation
        from tccc.llm_analysis.mock_llm import MockPhiModel
        return MockPhiModel(config)