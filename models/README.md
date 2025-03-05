# TCCC.ai Models

## IMPORTANT: Model Files Not Included

⚠️ **This directory only contains configuration information. The actual model files are NOT included in the repository for security, licensing, and size reasons.**

Model weights must be downloaded separately according to the instructions below.

## Required Models

For full functionality, the TCCC.ai system requires these models:

| Component | Model | Size | Format | Source |
|-----------|-------|------|--------|--------|
| STT Engine | faster-whisper-large-v3 | ~3-4GB | INT8 quantized | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3) |  
| LLM Analysis | Phi-2 | ~700MB-1.3GB | 4-bit quantized | [Hugging Face](https://huggingface.co/microsoft/phi-2) |
| Document Library | all-MiniLM-L12-v2 | ~90MB | FP16/INT8 | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) |

## Model Performance on Jetson

All models are optimized for Jetson edge deployment:

### Speech Recognition (faster-whisper)

| Model | Parameters | Quantization | Memory | Latency | Accuracy |
|-------|------------|--------------|--------|---------|----------|
| tiny | ~39M | INT8 | ~200MB | 0.15x RTF | Good |
| base | ~74M | INT8 | ~300MB | 0.25x RTF | Better |
| small | ~244M | INT8 | ~600MB | 0.4x RTF | Better+ |
| medium | ~769M | INT8 | ~1.2GB | 0.7x RTF | Excellent |
| large-v3 | ~1.5B | INT8 | ~2.5GB | 1.0x RTF | Superior |

*RTF = Real-Time Factor (lower is faster)*

### LLM Analysis (Phi-2)

| Quantization | Size | Memory | Tokens/sec | Quality |
|--------------|------|--------|------------|---------|
| FP16 | ~2.5GB | ~5GB | ~2-5 | 100% |
| INT8 | ~1.3GB | ~2.5GB | ~5-10 | 98% | 
| INT4 | ~700MB | ~1.3GB | ~10-20 | 95% |

### Document Embeddings (all-MiniLM-L12-v2)

| Feature | Specification |
|---------|---------------|
| Embedding Dimension | 384 |
| Size | ~90MB |
| Memory Usage | ~200-500MB |
| Latency | ~1-10ms per embedding |
| Performance | 78.6% semantic similarity score |

## Directory Structure

The system expects models to be organized as follows:

```
models/
├── stt/
│   └── faster-whisper-large-v3/
│       └── ... model files ...
├── llm/
│   └── phi-2/
│       └── ... model files ...
└── embeddings/
    └── all-MiniLM-L12-v2/
        └── ... model files ...
```

## Configuration

Model paths and parameters must be specified in your local config files:

```yaml
# Example STT engine configuration
model:
  provider: local
  name: faster-whisper-large-v3
  path: /path/to/models/stt/faster-whisper-large-v3
  quantization: int8
  compute_type: int8_float16
```

See the `config/templates/` directory for complete configuration examples.

## Security and Licensing

- Never commit model weights to this repository
- Review model licenses before deployment
- All models used in TCCC.ai are configured to run completely on-device
- No model weights or embeddings are transmitted externally

## Further Documentation

For detailed instructions on model conversion, optimization, and deployment, refer to:
- [Edge Deployment Guide](../references/best_practices/development_guide.md)
- [Model Optimization](../references/architecture/model_optimization.md)
- [Jetson Deployment](../references/architecture/jetson_deployment.md)