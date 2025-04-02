# TCCC.ai Tensor Optimization Guide

This document explains how tensor operations are optimized for the Jetson Orin Nano platform in the TCCC.ai system.

## Overview

TCCC.ai's tensor optimization module (`tccc.utils.tensor_optimization`) provides specialized operations for maximizing performance on edge hardware. These optimizations focus on:

1. **Memory Efficiency**: Reducing memory footprint to fit within Jetson's 8GB RAM
2. **Mixed Precision**: Using FP16/INT8 operations where appropriate
3. **Chunked Processing**: Breaking large operations into manageable pieces
4. **TensorRT Integration**: Accelerating model inference through TensorRT

## Using the Tensor Optimizer

### Basic Usage

```python
from tccc.utils.tensor_optimization import TensorOptimizer

# Create optimizer with config
config = {
    "mixed_precision": True,
    "memory_efficient": True,
    "use_tensorrt": True
}
optimizer = TensorOptimizer(config)

# Optimize input tensor
input_tensor = torch.randn(1000, 1000)
optimized_tensor = optimizer.optimize_tensor(input_tensor, dtype="float16")

# Use mixed precision context for operations
with optimizer.mixed_precision_context():
    result = model(optimized_tensor)
```

### Decorator for Automatic Optimization

```python
from tccc.utils.tensor_optimization import apply_tensor_optimizations

@apply_tensor_optimizations(config)
def process_audio(audio_data):
    # Will automatically optimize tensors
    return model(audio_data)
```

### Memory Tracking

```python
from tccc.utils.tensor_optimization import MemoryTracker

tracker = MemoryTracker()

with tracker.track():
    # Perform tensor operations
    result = large_model(large_input)

print(tracker.get_memory_summary())
```

## Configuration Options

The tensor optimization system uses configuration from `config/jetson_optimizer.yaml`. Key settings:

```yaml
tensor_optimizations:
  enabled: true                # Master switch
  memory_efficient: true       # Enable memory-saving operations
  mixed_precision: true        # Use FP16 where possible
  use_tensorrt: true           # Convert models to TensorRT when possible
  trt_workspace_size: 1073741824  # 1GB TensorRT workspace
  target_precision: "fp16"     # Target precision mode
  chunk_large_tensors: true    # Process large tensors in chunks
  chunk_size: 512              # Chunk size for large operations
```

## Performance Profiles

TCCC.ai includes three performance profiles that affect tensor operations:

1. **Emergency Mode**:
   - Maximizes speed at the cost of memory efficiency
   - Uses maximum TensorRT acceleration
   - Disables chunking for large operations

2. **Field Mode**:
   - Balanced performance and memory usage
   - Default for most operations
   - Enables adaptive tensor optimizations

3. **Training Mode**:
   - Maximizes battery life
   - Disables TensorRT
   - Uses higher precision (FP32) for more accurate results
   - Maximizes memory efficiency

To switch profiles:

```python
from tccc.utils.config import Config

# Load config with specific profile
config = Config.load_with_profile("field")
```

## TensorRT Optimization

The TensorRTOptimizer converts ONNX models to optimized TensorRT engines:

```python
from tccc.utils.tensor_optimization import TensorRTOptimizer

optimizer = TensorRTOptimizer(config)

# Optimize ONNX model
engine_path = optimizer.optimize_onnx(
    "models/whisper_encoder.onnx",
    input_shapes={"input": [1, 80, 3000]}
)
```

Engine files are cached at `cache/tensorrt/` for faster loading on subsequent runs.

## Implementation Details

### Memory Efficiency Strategies

1. **Tensor Chunking**: Large matrix multiplications are split into smaller chunks
2. **Progressive Loading**: Models and data are loaded in stages to minimize peak memory
3. **In-place Operations**: Using in-place operations where possible to avoid duplicates
4. **Gradient-free Inference**: Detaching tensors when gradients aren't needed

### Mixed Precision Details

1. **Automatic Detection**: FP16 is used on GPU, INT8 on CPU when possible
2. **Context Management**: `mixed_precision_context()` provides a PyTorch amp-compatible context
3. **Model Conversion**: Automatically converts models to FP16 when appropriate

### Jetson-Specific Optimizations

1. **CUDA Stream Management**: Proper stream synchronization to maximize GPU utilization
2. **cuDNN Configuration**: Tuned for Ampere architecture in Orin
3. **Tensor Core Usage**: Operations sized to benefit from Tensor Cores
4. **Memory Capping**: Limiting allocation to prevent OOM errors

## Best Practices

1. Use the decorator pattern for the simplest integration
2. Track memory usage during development to identify bottlenecks
3. Prefer batched operations when possible
4. Use the appropriate profile for your use case
5. Test all optimizations thoroughly - behavior can differ between development and deployment hardware