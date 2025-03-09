# Phi-2 GGUF Implementation for TCCC

This document provides information about the Phi-2 GGUF implementation for the TCCC (Tactical Combat Casualty Care) project. The implementation uses the GGUF format to efficiently run the Phi-2 model on Jetson hardware.

## What is GGUF?

GGUF (GPT-Generated Unified Format) is a file format designed for storing and efficiently loading large language models. It offers several advantages over other formats:

- Smaller file sizes through efficient quantization
- Faster loading and inference
- Better support for different hardware configurations
- Standardized format supported by multiple libraries

## Implementation Overview

Our implementation uses the `llama-cpp-python` library to load and run Phi-2 models in GGUF format. The main components are:

1. `phi_gguf_model.py`: Implementation of the `PhiGGUFModel` class
2. Integration with the LLM analysis module
3. Configuration options in `llm_analysis.yaml`
4. Test script in `test_phi_gguf.py`

## Getting Started

### Prerequisites

- Python 3.8+
- TCCC project environment
- llama-cpp-python library: `pip install llama-cpp-python`
- Phi-2 GGUF model file

### Download the Model

To download the Phi-2 GGUF model, run:

```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id='TheBloke/phi-2-GGUF', filename='phi-2.Q4_K_M.gguf')
```

Or use a one-liner:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='TheBloke/phi-2-GGUF', filename='phi-2.Q4_K_M.gguf'))"
```

### Configuration

Update your `llm_analysis.yaml` configuration to use the GGUF model:

```yaml
model:
  primary:
    provider: "local"
    name: "phi-2-gguf"
    path: "models/llm/phi-2/"
    gguf_model_path: "models/phi-2-gguf/phi-2.Q4_K_M.gguf"
    use_gguf: true
    force_real: true
```

### Testing

Run the test script to verify the GGUF implementation:

```bash
python test_phi_gguf.py --use-gpu
```

For testing with the mock implementation:

```bash
python test_phi_gguf.py --mock
```

## Implementation Details

### PhiGGUFModel Class

The `PhiGGUFModel` class provides the core functionality for the GGUF implementation:

- Loading the model with appropriate parameters
- Handling GPU acceleration when available
- Generating text responses with the proper format
- Tracking usage metrics
- Graceful error handling and fallback options

### Integration with TCCC

The GGUF implementation integrates seamlessly with the existing TCCC architecture:

- Factory pattern for model instantiation
- Consistent API with other model implementations
- Automatic fallback to mock implementation
- Proper logging and error handling

### Performance Considerations

The GGUF implementation is optimized for Jetson hardware:

- Quantized model (Q4_K_M) for reduced memory footprint
- Optional GPU acceleration
- Configurable thread count for CPU inference
- Adjustable context window size

## Advanced Usage

### Environment Variables

- `TCCC_USE_MOCK_LLM=1`: Force using the mock implementation

### GPU Acceleration

Enable GPU acceleration by setting:

```yaml
hardware:
  enable_acceleration: true
  cuda_device: 0
```

And use the `--use-gpu` flag when running the test script.

### Quantization Options

Different quantization options are available for different performance/quality tradeoffs:

- `phi-2.Q4_K_M.gguf`: Balanced performance/quality (recommended)
- `phi-2.Q8_0.gguf`: Higher quality, larger model
- `phi-2.Q5_K_M.gguf`: Middle ground between Q4 and Q8

## Troubleshooting

### Model Not Found

If you encounter a "Model not found" error, make sure:

1. You've downloaded the model file
2. The path in the configuration file is correct
3. The file has proper permissions

### CUDA Issues

If GPU acceleration isn't working:

1. Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Verify the GPU is recognized: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
3. Make sure llama-cpp-python is installed with CUDA support