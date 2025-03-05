# STT Engine Implementation Guide

This document describes the implementation of the Speech-to-Text (STT) engine in the TCCC.ai system, focusing on the ONNX Runtime approach.

## Current Implementation

We've chosen to implement the STT engine using OpenAI's Whisper models with ONNX Runtime for optimal performance on Jetson hardware. The implementation includes:

- ONNX Runtime integration with encoder-decoder architecture
- English-only models for better performance
- Medical vocabulary handling for improved transcription accuracy
- Mock implementation for testing without dependencies

## Model Configuration

The system supports various model sizes:

| Model | Size | Memory | Accuracy | Notes |
|-------|------|--------|----------|-------|
| tiny-en | ~150MB | ~350MB | Good | Fastest, suitable for simple commands |
| base-en | ~300MB | ~500MB | Better | Good for general transcription |
| small-en | ~500MB | ~1GB | Better+ | Recommended for medical terminology |
| medium-en | ~1.5GB | ~3GB | Best | Highest accuracy, requires more resources |

## Performance Considerations

The ONNX Runtime implementation provides several advantages:
- 1.5-3x faster inference compared to PyTorch implementation
- Lower memory usage during inference
- GPU acceleration via CUDA and TensorRT on Jetson
- FP16 precision for faster inference with minimal accuracy loss

## Using the Implementation

### Basic Usage

```python
from tccc.stt_engine import STTEngine
from tccc.utils import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("stt_engine")

# Initialize STT engine
engine = STTEngine()
engine.initialize(config)

# Transcribe audio
audio_data = load_audio("sample.wav")
result = engine.transcribe_segment(audio_data)
print(result["text"])
```

### Testing with Mock Implementation

For testing without dependencies, use the mock implementation:

```python
# Set environment variable before importing
import os
os.environ["USE_MOCK_STT"] = "1"

from tccc.stt_engine import STTEngine
```

## Medical Terminology Support

The engine includes a medical term processor that improves transcription quality for TCCC-specific terminology:

- Common medical and tactical combat care terms
- Corrections for commonly misheard terms (e.g., "tore nick it" → "tourniquet")
- Medical abbreviations (e.g., "TQ" → "Tourniquet", "MARCH" → "Massive Hemorrhage, Airway, Respiration, Circulation, Hypothermia")

## Future Considerations

For future development, we should evaluate:

1. **Faster-Whisper Models**
   - 4x faster than official implementation
   - 2x lower memory usage
   - Drop-in API replacement
   - Excellent on Jetson with GPU acceleration

2. **NVIDIA Nexa Edge Models**
   - Specifically optimized for Jetson platforms
   - Qwen2Audio for edge-optimized performance
   - Native TensorRT integration

3. **Quantization Improvements**
   - INT8 quantization for further speed/memory improvements
   - Weight pruning for smaller model size
   - KV cache optimization for streaming inference

## Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [OpenAI Whisper Repository](https://github.com/openai/whisper)
- [Faster-Whisper Repository](https://github.com/guillaumekln/faster-whisper)
- [NVIDIA Nexa Models Documentation](https://developer.nvidia.com/nexa)