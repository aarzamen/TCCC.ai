"""
LLM Analysis module for TCCC.ai system.

This module provides advanced conversation analysis and agent recommendations using LLMs.
"""

from .llm_analysis import LLMAnalysis

# Conditionally import mock implementation if available
try:
    from .mock_llm import MockPhiModel
    __has_mock = True
except ImportError:
    __has_mock = False

def get_phi_model(config, use_mock=None):
    """
    Factory function to get appropriate Phi model implementation.
    
    Args:
        config: Configuration dictionary
        use_mock: Override to force mock usage (or force real implementation)
            If None, will use mock if available and fallback to real implementation
            
    Returns:
        Phi model implementation (either real or mock)
    """
    # Determine whether to use mock
    should_use_mock = use_mock if use_mock is not None else False
    
    # Check environment variable override
    import os
    if os.environ.get("TCCC_USE_MOCK_LLM") == "1":
        should_use_mock = True
    elif os.environ.get("TCCC_USE_MOCK_LLM") == "0":
        should_use_mock = False
    
    # Use mock if available and allowed
    if should_use_mock and __has_mock:
        from .mock_llm import MockPhiModel
        return MockPhiModel(config)
    
    # Fall back to real implementation
    try:
        # Import here to avoid dependency requirements when using mock
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from optimum.onnxruntime import ORTModelForCausalLM
        
        # Load tokenizer and model
        # For Phi-2, we need special handling
        try:
            # Try to load model from config path
            model_path = config.get("model_path", "microsoft/phi-2")
            
            # First try ONNX optimized implementation
            model = ORTModelForCausalLM.from_pretrained(
                model_path,
                provider="CUDAExecutionProvider" if config.get("use_gpu", True) else "CPUExecutionProvider"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create wrapper class that mimics the mock implementation's API
            class PhiModel:
                def __init__(self, config, model, tokenizer):
                    self.config = config
                    self.model = model
                    self.tokenizer = tokenizer
                    self.metrics = {"total_requests": 0, "total_tokens": 0, "avg_latency": 0.0}
                
                def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
                    import time
                    start_time = time.time()
                    self.metrics["total_requests"] += 1
                    
                    # Set generation parameters
                    if max_tokens is None:
                        max_tokens = 1024
                    if temperature is None:
                        temperature = 0.7
                    if top_p is None:
                        top_p = 0.9
                    
                    # Tokenize input
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    
                    # Generate output
                    outputs = self.model.generate(
                        **inputs,
                        max_length=len(inputs["input_ids"][0]) + max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                    )
                    
                    # Decode output
                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract completion part (remove the prompt)
                    completion_text = text[len(prompt):].strip()
                    
                    # Update metrics
                    elapsed = time.time() - start_time
                    self.metrics["avg_latency"] = (
                        (self.metrics["avg_latency"] * (self.metrics["total_requests"] - 1) + elapsed) / 
                        self.metrics["total_requests"]
                    )
                    
                    return {
                        "id": str(uuid.uuid4()),
                        "choices": [{"text": completion_text}],
                        "usage": {
                            "prompt_tokens": len(inputs["input_ids"][0]),
                            "completion_tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
                            "total_tokens": len(outputs[0])
                        },
                        "model": "phi-2-instruct",
                        "latency": elapsed
                    }
                
                def get_metrics(self):
                    return {
                        "total_requests": self.metrics["total_requests"],
                        "total_tokens": self.metrics["total_tokens"],
                        "avg_latency": round(self.metrics["avg_latency"], 3),
                        "model": "phi-2-instruct"
                    }
            
            return PhiModel(config, model, tokenizer)
            
        except (ImportError, RuntimeError, OSError) as e:
            # If loading fails and mock is available, use mock instead
            if __has_mock:
                from .mock_llm import MockPhiModel
                return MockPhiModel(config)
            else:
                # Re-raise the exception if we can't fall back
                raise
    
    except ImportError:
        # If optimum.onnxruntime or transformers not available and mock is available,
        # use mock implementation
        if __has_mock:
            from .mock_llm import MockPhiModel
            return MockPhiModel(config)
        else:
            # Can't import real implementation and mock isn't available
            raise ImportError(
                "Cannot import transformers or optimum.onnxruntime and mock implementation is not available. "
                "Install with pip install transformers optimum onnx onnxruntime or create a valid mock_llm.py"
            )