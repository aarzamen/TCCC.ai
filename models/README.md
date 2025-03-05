# TCCC.ai Models

This directory contains models for the TCCC.ai system. The models are optimized for the Jetson platform using ONNX Runtime and quantization techniques for efficient inference.

## Speech Recognition Models

The system uses OpenAI's Whisper models converted to ONNX format for improved performance.

### Whisper Model Structure

Each Whisper model is split into two ONNX files:
- `encoder.onnx`: Processes audio features
- `decoder.onnx`: Generates text from encoded audio features 

### English-Only Whisper Models

The following English-only models are supported:

| Model | Size | Disk Space | Memory Usage | Accuracy | Speed (RTF) |
|-------|------|------------|--------------|----------|-------------|
| whisper-tiny-en | ~150MB | ~350MB | Good | 1.5-3x faster |
| whisper-base-en | ~300MB | ~500MB | Better | 1.5-3x faster |
| whisper-small-en | ~500MB | ~1GB | Better+ | 1.5-3x faster |
| whisper-medium-en | ~1.5GB | ~3GB | Best | 1.5-3x faster |

## LLM Analysis Models

The system uses Microsoft's Phi-2 model (2.7B parameters) for medical text analysis, optimized for edge deployment.

### Phi-2 Model Structure

The Phi-2 model is converted to ONNX format for efficient inference:
- `model.onnx`: The full model in ONNX format
- `config.json`: Model configuration

### Phi-2 Model Options

| Model | Parameters | Quantization | Disk Space | Memory Usage | Performance |
|-------|------------|--------------|------------|--------------|-------------|
| phi-2-instruct | 2.7B | FP16 | ~2.5GB | ~5GB | Full accuracy |
| phi-2-instruct-int8 | 2.7B | INT8 | ~1.3GB | ~2.5GB | Good accuracy |
| phi-2-instruct-int4 | 2.7B | INT4 | ~700MB | ~1.3GB | Acceptable accuracy |

## ONNX Runtime Acceleration

All models use ONNX Runtime for inference acceleration with the following features:
- FP16 mixed precision support
- TensorRT integration for Jetson platforms
- CPU fallback for deployment without GPU
- Quantization options (FP16, INT8, INT4)

## Adding New Models

### Adding Speech Recognition Models
1. Create a directory with the model name format `whisper-{size}-{lang}`
2. Convert PyTorch model to ONNX using the ModelManager._convert_whisper_to_onnx() method
3. Add model configuration to config/stt_engine.yaml

### Adding LLM Models
1. Create a directory with the model name (e.g., `phi-2-instruct`)
2. Convert model to ONNX using optimum.onnxruntime
3. Add model configuration to config/llm_analysis.yaml

## Future Considerations

Future improvements may include:
1. Faster-Whisper models for improved speech recognition speed
2. NVIDIA Nexa edge-optimized models for Jetson
3. Phi-3 models for improved LLM capabilities
4. Custom domain-specific fine-tuning for medical terminology