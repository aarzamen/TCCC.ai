# TCCC.ai Models

This directory contains models for the TCCC.ai system. The models are optimized for the Jetson platform using ONNX Runtime and quantization techniques for efficient inference.

## Speech Recognition Models

The system supports two implementations of OpenAI's Whisper models: the standard ONNX-based implementation and Nexa AI's faster-whisper implementation.

### Nexa AI's faster-whisper Models

The system now uses Nexa AI's faster-whisper, which offers significant performance improvements:

| Model | Parameters | Size | Memory (FP16) | Accuracy | Speed (RTF) |
|-------|------------|------|---------------|----------|-------------|
| faster-whisper-tiny | ~39M | ~80MB | ~200MB | Good | 0.15-0.2x (5-7x real-time) |
| faster-whisper-small | ~244M | ~500MB | ~1GB | Better | 0.3-0.4x (2.5-3x real-time) |
| faster-whisper-medium | ~769M | ~1.5GB | ~2.5GB | Better+ | 0.6-0.7x (1.5x real-time) |
| faster-whisper-large-v2 | ~1.5B | ~3GB | ~4.5GB | Best | 0.9-1.0x (1x real-time) |

Each faster-whisper model utilizes the CTranslate2 backend and can run in various precision modes:
- `float16`: Best balance of accuracy and speed on GPU
- `int8`: Optimized for CPU-only deployment with minimal accuracy loss
- `int8_float16`: Mixed quantization for balanced performance

### Legacy Whisper ONNX Models

The system also supports the original Whisper models converted to ONNX format:

Each Whisper model is split into two ONNX files:
- `encoder.onnx`: Processes audio features
- `decoder.onnx`: Generates text from encoded audio features 

| Model | Size | Disk Space | Memory Usage | Accuracy | Speed (RTF) |
|-------|------|------------|--------------|----------|-------------|
| whisper-tiny-en | ~150MB | ~350MB | Good | 0.5-0.7x (1.5-2x real-time) |
| whisper-base-en | ~300MB | ~500MB | Better | 0.6-0.8x (1.2-1.5x real-time) |
| whisper-small-en | ~500MB | ~1GB | Better+ | 0.7-0.9x (1.1-1.4x real-time) |
| whisper-medium-en | ~1.5GB | ~3GB | Best | 0.8-1.0x (1-1.2x real-time) |

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

## Document Embedding Models

The system uses Nexa AI's all-MiniLM-L12-v2 embedding model for document retrieval and semantic search capabilities.

### all-MiniLM-L12-v2 Model Structure

The all-MiniLM-L12-v2 model uses a distilled BERT architecture and is used through the sentence-transformers framework:
- `model.safetensors`: The PyTorch model weights (can be converted to ONNX for further optimization)
- `tokenizer.json`: Tokenizer configuration
- `config.json`: Model configuration

### all-MiniLM-L12-v2 Model Specifications

| Feature | Value |
|---------|-------|
| Embedding Dimension | 384 |
| Model Size | ~130MB |
| Memory Usage | ~500MB |
| Max Sequence Length | 512 tokens |
| Speed | ~1ms per embedding on GPU, ~10ms on CPU |
| Performance | High quality for semantic search, 78.6% semantic similarity benchmark score |

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

### Adding Embedding Models
1. Create a directory with the model name (e.g., `all-MiniLM-L12-v2`)
2. Download via Hugging Face or sentence-transformers
3. Optionally convert to ONNX for accelerated inference
4. Add model configuration to config/document_library.yaml

## Future Considerations

Future improvements may include:
1. Faster-Whisper models for improved speech recognition speed
2. NVIDIA Nexa edge-optimized models for Jetson
3. Phi-3 models for improved LLM capabilities
4. Custom domain-specific fine-tuning for medical terminology
5. Multilingual embedding models for international deployment