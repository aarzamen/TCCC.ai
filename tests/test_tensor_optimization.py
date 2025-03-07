"""
Tests for tensor optimization module.

These tests validate the tensor optimization capabilities for Jetson hardware,
ensuring proper memory management and precision handling.
"""

import os
import pytest
import numpy as np
from unittest import mock

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

# Import the module under test
from tccc.utils.tensor_optimization import (
    TensorOptimizer, 
    MemoryTracker,
    TensorRTOptimizer,
    apply_tensor_optimizations,
    optimize_batch_processing
)


class TestTensorOptimizer:
    """Test the TensorOptimizer class."""

    def test_initialization(self):
        """Test initialization with default config."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        optimizer = TensorOptimizer(config)
        
        assert optimizer.fp16_enabled is True
        assert optimizer.memory_efficient is True
        assert optimizer.device in ["cuda", "cpu"]
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimize_tensor_cuda(self):
        """Test tensor optimization on CUDA."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        optimizer = TensorOptimizer(config)
        
        # Create a test tensor
        input_tensor = torch.randn(100, 100)
        
        # Optimize tensor
        optimized = optimizer.optimize_tensor(input_tensor, dtype="float16")
        
        # Check properties
        assert optimized.device.type == "cuda"
        assert optimized.dtype == torch.float16
    
    def test_optimize_tensor_cpu(self):
        """Test tensor optimization on CPU."""
        # Force CPU usage
        config = {
            "mixed_precision": False,
            "memory_efficient": True
        }
        
        with mock.patch("torch.cuda.is_available", return_value=False):
            optimizer = TensorOptimizer(config)
            
            # Create a test tensor
            input_tensor = torch.randn(100, 100)
            
            # Optimize tensor
            optimized = optimizer.optimize_tensor(input_tensor)
            
            # Check properties
            assert optimized.device.type == "cpu"
            assert optimized.dtype == torch.float32
    
    def test_optimize_numpy_input(self):
        """Test optimization of NumPy arrays."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        optimizer = TensorOptimizer(config)
        
        # Create a numpy array
        numpy_array = np.random.randn(100, 100).astype(np.float32)
        
        # Optimize tensor
        optimized = optimizer.optimize_tensor(numpy_array)
        
        # Check that it was converted to torch
        assert isinstance(optimized, torch.Tensor)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_context(self):
        """Test mixed precision context manager."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        optimizer = TensorOptimizer(config)
        
        # Test within context
        with optimizer.mixed_precision_context():
            # Should use half precision when available
            if hasattr(torch.cuda, "amp"):
                assert torch.cuda.amp.autocast.is_enabled()
    
    def test_optimize_model_inputs(self):
        """Test optimization of model input dictionaries."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        optimizer = TensorOptimizer(config)
        
        # Create input dictionary
        inputs = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "images": [torch.randn(3, 224, 224) for _ in range(2)],
            "metadata": {"non_tensor_key": "value"}
        }
        
        # Optimize inputs
        optimized = optimizer.optimize_model_inputs(inputs)
        
        # Check that tensors were optimized
        assert optimized["input_ids"].device == torch.device(optimizer.device)
        assert optimized["attention_mask"].device == torch.device(optimizer.device)
        assert isinstance(optimized["images"], list)
        assert optimized["images"][0].device == torch.device(optimizer.device)
        # Non-tensor data should be unchanged
        assert optimized["metadata"] == inputs["metadata"]
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimize_matmul(self):
        """Test optimized matrix multiplication."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        optimizer = TensorOptimizer(config)
        
        # Create test tensors
        a = torch.randn(100, 200).to(optimizer.device)
        b = torch.randn(200, 50).to(optimizer.device)
        
        # Standard matmul
        expected = torch.matmul(a, b)
        
        # Optimized matmul
        result = optimizer.optimize_matmul(a, b)
        
        # Results should be close
        assert torch.allclose(expected, result, rtol=1e-2, atol=1e-2)  # Looser tolerance for FP16
    
    def test_memory_tracking(self):
        """Test memory tracking functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        tracker = MemoryTracker()
        
        # Track memory usage
        with tracker.track():
            # Allocate some tensors
            tensors = [torch.randn(1000, 1000, device="cuda") for _ in range(5)]
            
        # Should have recorded peak memory
        assert tracker.get_peak_memory() > 0
        
        # Summary should contain memory values
        summary = tracker.get_memory_summary()
        assert "Current memory" in summary
        assert "Peak memory" in summary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTensorOptimizationDecorator:
    """Test the tensor optimization decorator."""
    
    def test_decorator_function(self):
        """Test the decorator on a simple function."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        
        @apply_tensor_optimizations(config)
        def test_func(x):
            return x + 1
        
        # Create input tensor
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # Call function
        result = test_func(input_tensor)
        
        # Check result
        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))
        
        # Should work with numpy inputs too
        numpy_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = test_func(numpy_input)
        assert torch.is_tensor(result)
    
    def test_keyword_args(self):
        """Test the decorator with keyword arguments."""
        config = {
            "mixed_precision": True,
            "memory_efficient": True
        }
        
        @apply_tensor_optimizations(config)
        def test_func(x=None, y=None):
            if x is not None and y is not None:
                return x + y
            return x
        
        # Create input tensors
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        
        # Call with keyword args
        result = test_func(x=x, y=y)
        
        # Check result
        assert torch.allclose(result, torch.tensor([5.0, 7.0, 9.0]))


class TestBatchProcessing:
    """Test batch processing optimization."""
    
    def test_optimize_batch_size(self):
        """Test batch size optimization based on memory."""
        # Test with unlimited memory
        config = {"memory_limit_mb": 0}
        batch_size = 32
        result = optimize_batch_processing(batch_size, config)
        assert result == 32  # Should be unchanged
        
        # Test with limited memory
        config = {"memory_limit_mb": 100, "memory_per_sample_mb": 10}
        batch_size = 32
        result = optimize_batch_processing(batch_size, config)
        assert result == 10  # Limited by memory
        
        # Test with CPU (should reduce)
        with mock.patch("torch.cuda.is_available", return_value=False):
            config = {"memory_limit_mb": 0}
            batch_size = 32
            result = optimize_batch_processing(batch_size, config)
            assert result == 16  # Half on CPU


@pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not available")
class TestTensorRTOptimizer:
    """Test TensorRT optimization."""
    
    def test_initialization(self):
        """Test initialization of TensorRT optimizer."""
        config = {
            "precision": "fp16",
            "trt_workspace_size": 1 << 30,
            "trt_engine_cache_dir": "test_cache"
        }
        
        # Clean up any existing test directory
        if os.path.exists("test_cache"):
            import shutil
            shutil.rmtree("test_cache")
        
        optimizer = TensorRTOptimizer(config)
        
        # Check properties
        assert optimizer.precision == "fp16"
        assert optimizer.workspace_size == 1 << 30
        assert os.path.exists(optimizer.cache_dir)
        
        # Clean up
        if os.path.exists("test_cache"):
            import shutil
            shutil.rmtree("test_cache")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])