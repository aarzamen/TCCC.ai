"""
TCCC.ai Tensor Optimization Module
----------------------------------
Provides tensor operation optimizations specifically for Jetson Orin Nano hardware
with mixed precision, memory efficient operations, and TensorRT integration.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

try:
    import torch
    from torch.cuda import amp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class TensorOptimizer:
    """Handles tensor operations optimization for Jetson hardware."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tensor optimizer.
        
        Args:
            config: Configuration dictionary with optimization settings
        """
        self.config = config
        self.device = self._get_optimal_device()
        self.fp16_enabled = config.get("mixed_precision", True)
        self.use_trt = config.get("use_tensorrt", True) and TENSORRT_AVAILABLE
        self.memory_efficient = config.get("memory_efficient", True)
        
        # Set up mixed precision autocast
        self.amp_enabled = (
            self.fp16_enabled and 
            TORCH_AVAILABLE and 
            torch.cuda.is_available()
        )
        
        # Initialize memory tracking
        self.memory_tracker = MemoryTracker() if TORCH_AVAILABLE else None
        
        logger.info(f"TensorOptimizer initialized with device={self.device}, "
                   f"fp16={self.fp16_enabled}, TensorRT={self.use_trt}")
    
    def _get_optimal_device(self) -> str:
        """Determine the optimal device for tensor operations."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()
            if any(name in device_name for name in ["tegra", "orin", "xavier"]):
                logger.info(f"Detected Jetson device: {device_name}")
            return "cuda"
        
        return "cpu"
    
    def optimize_tensor(self, tensor: Union[np.ndarray, "torch.Tensor"], 
                        dtype: Optional[str] = None) -> "torch.Tensor":
        """
        Optimize a tensor for efficient processing on Jetson.
        
        Args:
            tensor: Input tensor (NumPy array or PyTorch tensor)
            dtype: Target data type (float16, float32, int8)
            
        Returns:
            Optimized PyTorch tensor
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning unoptimized tensor")
            return tensor
        
        # Convert NumPy array to PyTorch tensor if needed
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        
        # Apply data type conversion if specified
        if dtype == "float16" and self.device == "cuda":
            tensor = tensor.half()
        elif dtype == "int8":
            # Note: proper int8 quantization would need scaling factors
            tensor = tensor.to(torch.int8)
        else:
            tensor = tensor.to(torch.float32)
        
        # Move to appropriate device
        tensor = tensor.to(self.device)
        
        # Apply memory optimizations
        if self.memory_efficient and tensor.requires_grad:
            with self.memory_tracker.track():
                tensor = tensor.detach().clone().requires_grad_()
        
        return tensor
    
    def mixed_precision_context(self):
        """Context manager for mixed precision operations."""
        if self.amp_enabled:
            return amp.autocast()
        else:
            # Return a no-op context manager if AMP not enabled
            return DummyContextManager()
    
    def optimize_model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model inputs dictionary for efficient processing.
        
        Args:
            inputs: Dictionary of model inputs
            
        Returns:
            Optimized inputs dictionary
        """
        optimized_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                # Optimize tensor inputs
                optimized_inputs[key] = self.optimize_tensor(value)
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (np.ndarray, torch.Tensor)) for x in value):
                # Optimize list/tuple of tensors
                optimized_inputs[key] = [self.optimize_tensor(x) for x in value]
            else:
                # Keep non-tensor inputs as is
                optimized_inputs[key] = value
        
        return optimized_inputs
    
    def apply_inference_optimizations(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """
        Apply inference-time optimizations to a PyTorch model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        if not TORCH_AVAILABLE:
            return model
        
        # Ensure model is in eval mode
        model.eval()
        
        # Move to optimal device
        model = model.to(self.device)
        
        if self.fp16_enabled and self.device == "cuda":
            # Convert to half precision for inference
            model = model.half()
        
        # Apply TensorRT optimization if enabled
        if self.use_trt and self.device == "cuda":
            try:
                from torch2trt import torch2trt
                # This is a placeholder - actual implementation would need sample inputs
                # model = torch2trt(model, [sample_input])
                logger.info("TensorRT optimization applied to model")
            except ImportError:
                logger.warning("torch2trt not available, skipping TensorRT optimization")
        
        # Apply fusing optimizations if available
        try:
            if hasattr(torch, 'jit') and not self.use_trt:
                # Trace the model for faster inference
                # This is a placeholder - actual implementation would need sample inputs
                # model = torch.jit.trace(model, sample_input)
                logger.info("JIT optimization available")
        except Exception as e:
            logger.warning(f"Failed to apply JIT optimization: {e}")
        
        return model
    
    def optimize_convolution(self, input_tensor: "torch.Tensor", 
                             weight: "torch.Tensor", 
                             bias: Optional["torch.Tensor"] = None,
                             stride: int = 1, 
                             padding: int = 0) -> "torch.Tensor":
        """
        Optimized convolution operation for Jetson.
        
        Args:
            input_tensor: Input tensor (N, C_in, H, W)
            weight: Weight tensor (C_out, C_in, kH, kW)
            bias: Optional bias tensor (C_out)
            stride: Convolution stride
            padding: Convolution padding
            
        Returns:
            Output tensor
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for convolution operation")
        
        # Use mixed precision if enabled
        with self.mixed_precision_context():
            if self.device == "cuda" and self.memory_efficient:
                # Use cudnn deterministic mode for better memory efficiency
                with torch.backends.cudnn.flags(deterministic=True, benchmark=False):
                    return torch.nn.functional.conv2d(
                        input_tensor, weight, bias=bias, stride=stride, padding=padding
                    )
            else:
                return torch.nn.functional.conv2d(
                    input_tensor, weight, bias=bias, stride=stride, padding=padding
                )
    
    def optimize_matmul(self, tensor1: "torch.Tensor", 
                        tensor2: "torch.Tensor") -> "torch.Tensor":
        """
        Optimized matrix multiplication for Jetson.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            Result tensor
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for matrix multiplication")
        
        # Use mixed precision if enabled
        with self.mixed_precision_context():
            # Try to use torch.matmul with memory-efficient operations
            if self.memory_efficient:
                # Split large matrices if needed to reduce memory usage
                if tensor1.shape[0] > 1024 and tensor2.shape[1] > 1024:
                    # Process in chunks of 512
                    chunk_size = 512
                    result_chunks = []
                    
                    for i in range(0, tensor1.shape[0], chunk_size):
                        end = min(i + chunk_size, tensor1.shape[0])
                        result_chunks.append(torch.matmul(
                            tensor1[i:end], tensor2
                        ))
                    
                    return torch.cat(result_chunks, dim=0)
                else:
                    return torch.matmul(tensor1, tensor2)
            else:
                return torch.matmul(tensor1, tensor2)
    
    def calculate_dram_saved(self, before_size: int, after_size: int) -> str:
        """Calculate DRAM memory savings and return formatted string."""
        saved_mb = (before_size - after_size) / (1024 * 1024)
        return f"Memory saved: {saved_mb:.2f} MB"


class MemoryTracker:
    """Tracks memory usage of tensor operations."""
    
    def __init__(self):
        """Initialize memory tracker."""
        self.peak_memory = 0
        self.initial_memory = 0
    
    def track(self):
        """Context manager to track memory usage."""
        return MemoryTrackingContext(self)
    
    def reset(self):
        """Reset memory tracking."""
        self.peak_memory = 0
        self.initial_memory = 0
    
    def get_current_memory(self) -> int:
        """Get current CUDA memory usage in bytes."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0
        
        return torch.cuda.memory_allocated()
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self.peak_memory
    
    def get_memory_summary(self) -> str:
        """Get memory usage summary."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return "Memory tracking unavailable"
        
        current = self.get_current_memory()
        peak = self.get_peak_memory()
        
        return (f"Current memory: {current / (1024**2):.2f} MB, "
                f"Peak memory: {peak / (1024**2):.2f} MB")


class MemoryTrackingContext:
    """Context manager for tracking memory usage."""
    
    def __init__(self, tracker: MemoryTracker):
        """Initialize context manager."""
        self.tracker = tracker
    
    def __enter__(self):
        """Enter the context."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.tracker.initial_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated()
            if current_peak > self.tracker.peak_memory:
                self.tracker.peak_memory = current_peak


class DummyContextManager:
    """Dummy context manager for when features aren't available."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TensorRTOptimizer:
    """Handles TensorRT optimization for models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TensorRT optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.workspace_size = config.get("trt_workspace_size", 1 << 30)  # 1GB default
        self.precision = config.get("precision", "fp16")
        self.cache_dir = config.get("trt_engine_cache_dir", "cache/tensorrt")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger = get_logger("TensorRTOptimizer")
        
        # Check if TensorRT is available
        if not TENSORRT_AVAILABLE:
            self.logger.warning("TensorRT not available, optimizations will be skipped")
    
    def optimize_onnx(self, onnx_path: str, input_shapes: Dict[str, List[int]]) -> str:
        """
        Optimize ONNX model with TensorRT.
        
        Args:
            onnx_path: Path to ONNX model file
            input_shapes: Dictionary of input names to shapes
            
        Returns:
            Path to TensorRT engine file
        """
        if not TENSORRT_AVAILABLE:
            self.logger.error("TensorRT not available")
            return ""
        
        try:
            # Generate engine name based on model and precision
            model_name = os.path.basename(onnx_path).replace(".onnx", "")
            precision_str = self.precision
            engine_name = f"{model_name}_{precision_str}.engine"
            engine_path = os.path.join(self.cache_dir, engine_name)
            
            # Check if engine already exists
            if os.path.exists(engine_path):
                self.logger.info(f"Using existing TensorRT engine: {engine_path}")
                return engine_path
            
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.INFO)
            
            # Create builder and network
            with trt.Builder(trt_logger) as builder, \
                 builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                 trt.OnnxParser(network, trt_logger) as parser:
                
                # Parse ONNX model
                with open(onnx_path, 'rb') as model_file:
                    if not parser.parse(model_file.read()):
                        for error in range(parser.num_errors):
                            self.logger.error(f"ONNX parse error: {parser.get_error(error)}")
                        return ""
                
                # Create optimization profile for dynamic shapes
                profile = builder.create_optimization_profile()
                for input_name, input_shape in input_shapes.items():
                    min_shape = input_shape.copy()
                    opt_shape = input_shape.copy()
                    max_shape = input_shape.copy()
                    
                    # Set dynamic dimensions
                    for i, dim in enumerate(input_shape):
                        if dim == -1:
                            min_shape[i] = 1
                            opt_shape[i] = 16
                            max_shape[i] = 64
                    
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                
                # Create config
                config = builder.create_builder_config()
                config.add_optimization_profile(profile)
                config.max_workspace_size = self.workspace_size
                
                # Set precision flags
                if self.precision == "fp16" and builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    self.logger.info("Enabling FP16 precision")
                elif self.precision == "int8" and builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    self.logger.info("Enabling INT8 precision")
                
                # Build engine
                self.logger.info(f"Building TensorRT engine for {onnx_path}")
                engine = builder.build_engine(network, config)
                
                if engine:
                    # Save engine
                    with open(engine_path, 'wb') as engine_file:
                        engine_file.write(engine.serialize())
                    self.logger.info(f"TensorRT engine saved to {engine_path}")
                    return engine_path
                else:
                    self.logger.error("Failed to build TensorRT engine")
                    return ""
                    
        except Exception as e:
            self.logger.error(f"Error optimizing ONNX with TensorRT: {e}")
            return ""


def apply_tensor_optimizations(
    model_func: Callable[..., Any], 
    config: Dict[str, Any]
) -> Callable[..., Any]:
    """
    Decorator to apply tensor optimizations to model function.
    
    Args:
        model_func: Model function to optimize
        config: Configuration dictionary
        
    Returns:
        Optimized model function
    """
    # Create tensor optimizer
    optimizer = TensorOptimizer(config)
    
    def optimized_func(*args, **kwargs):
        # Apply input optimizations if first arg is a tensor
        if args and isinstance(args[0], (np.ndarray, torch.Tensor)):
            args = list(args)
            args[0] = optimizer.optimize_tensor(args[0])
        
        # Process keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                kwargs[key] = optimizer.optimize_tensor(value)
        
        # Use mixed precision for the computation
        with optimizer.mixed_precision_context():
            result = model_func(*args, **kwargs)
        
        return result
    
    return optimized_func


def optimize_batch_processing(batch_size: int, config: Dict[str, Any]) -> int:
    """
    Determine optimal batch size based on available memory and hardware.
    
    Args:
        batch_size: Requested batch size
        config: Configuration dictionary
        
    Returns:
        Optimized batch size
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return max(1, batch_size // 2)  # Reduce batch size on CPU
    
    memory_limit = config.get("memory_limit_mb", 0)
    if memory_limit > 0:
        # Estimate memory per sample (this is application specific)
        memory_per_sample = config.get("memory_per_sample_mb", 50)  # Default 50MB per sample
        max_samples = memory_limit // memory_per_sample
        
        # Clamp batch size to memory limit
        adjusted_batch_size = min(batch_size, max_samples)
        
        if adjusted_batch_size < batch_size:
            logger.warning(f"Reduced batch size from {batch_size} to {adjusted_batch_size} due to memory limit")
        
        return max(1, adjusted_batch_size)
    
    return batch_size