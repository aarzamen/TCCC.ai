# TCCC.ai Speech Recognition Models

This directory contains models for the TCCC.ai speech recognition system. The system uses OpenAI's Whisper models
converted to ONNX format for improved performance on the Jetson platform.

## Model Structure

Each Whisper model is split into two ONNX files:
- `encoder.onnx`: Processes audio features
- `decoder.onnx`: Generates text from encoded audio features 

## English-Only Models

The following English-only models are supported:

| Model | Size | Disk Space | Memory Usage | Accuracy | Speed (RTF) |
|-------|------|------------|--------------|----------|-------------|
| whisper-tiny-en | ~150MB | ~350MB | Good | 1.5-3x faster |
| whisper-base-en | ~300MB | ~500MB | Better | 1.5-3x faster |
| whisper-small-en | ~500MB | ~1GB | Better+ | 1.5-3x faster |
| whisper-medium-en | ~1.5GB | ~3GB | Best | 1.5-3x faster |

## ONNX Runtime Acceleration

The models use ONNX Runtime for inference acceleration with the following features:
- FP16 mixed precision support
- TensorRT integration for Jetson platforms
- CPU fallback for deployment without GPU

## Adding New Models

To add a new model:
1. Create a directory with the model name format `whisper-{size}-{lang}`
2. Convert PyTorch model to ONNX using the ModelManager._convert_whisper_to_onnx() method
3. Add model configuration to config/stt_engine.yaml

## Future Considerations

Future improvements may include:
1. Faster-Whisper models for improved speed with comparable accuracy
2. NVIDIA Nexa edge-optimized models like Qwen2Audio for optimized performance
3. Quantized models (INT8) for even more efficiency on edge devices

This implementation uses the ONNX Runtime approach that's already integrated in the codebase.