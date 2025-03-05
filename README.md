# TCCC.ai 🧠

<p align="center">
  <img src="images/green_logo.png" alt="TCCC.ai Logo" width="200"/>
</p>

> Edge-deployed intelligence for mission-critical communications

[![Project Status: Active](https://img.shields.io/badge/Status-Active-green.svg)](https://github.com/tccc-ai/tccc-project)
[![Jetson Optimized](https://img.shields.io/badge/Jetson-Optimized-76B900.svg)](https://developer.nvidia.com/embedded/jetson-orin)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

TCCC.ai is an edge-first AI system delivering privacy-preserving transcription, analysis, and decision support capabilities on resource-constrained hardware. Built specifically for the NVIDIA Jetson Orin Nano platform, it processes sensitive communications with zero cloud dependencies.

## Key Capabilities

⚡ **Real-time processing** with sub-200ms latency on edge hardware  
🛡️ **Zero-cloud operation** for air-gapped and sensitive environments  
🧠 **Fully quantized models** optimized for Jetson's limited resources  
🔄 **Adaptive resource allocation** balancing performance and power  
🔌 **Modular architecture** supporting component customization

## System Architecture

<p align="center">
  <img src="images/blue_logo.png" alt="TCCC.ai Architecture" width="250"/>
</p>

TCCC.ai employs a modular edge-optimized architecture with six specialized components:

1. **Audio Pipeline** - Low-latency audio capture and preprocessing pipeline
2. **STT Engine** - On-device transcription using Nexa AI's faster-whisper-5 (quantized)
3. **Processing Core** - Real-time NLP with context-aware entity and intent extraction
4. **LLM Analysis** - Edge-optimized Phi-2 (4-bit quantized) for advanced reasoning
5. **Data Store** - Efficient on-device persistence with encryption capabilities
6. **Document Library** - Vector search using all-MiniLM-L12-v2 embeddings and FAISS

## Technology Stack

<p align="center">
  <img src="https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetsonNano/jetson-nano-som.png" alt="NVIDIA Jetson" width="180"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" width="120"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://developer.nvidia.com/blog/wp-content/uploads/2018/11/NV_TensorRT_Visual_2C_RGB-625x625-1.png" alt="TensorRT" width="120"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://onnxruntime.ai/images/ONNX-Runtime-logo.png" alt="ONNX Runtime" width="120"/>
</p>

## Installation & Setup

### Hardware Requirements

- NVIDIA Jetson Orin Nano (4GB+ RAM)
- JetPack 5.1.1 or later
- 15GB+ storage (SSD recommended)
- Optional: USB microphone for audio input

### Quick Start

```bash
# Clone repository
git clone https://github.com/tccc-ai/tccc-project.git
cd tccc-project

# Environment setup
python -m venv venv
source venv/bin/activate

# Installation options
pip install -e .              # Basic installation
pip install -e ".[dev]"       # Development mode
pip install -e ".[gpu]"       # With GPU acceleration

# Start system
python -m tccc.cli start
```

### Configuration

The system uses YAML configuration in the `config/` directory with sensible defaults for Jetson hardware. Key settings:

```yaml
# config/stt_engine.yaml
model:
  provider: local
  name: faster-whisper-5-int8
  path: /path/to/models/whisper
  quantization: int8
  compute_type: int8_float16
  
hardware:
  enable_acceleration: true
  cuda_device: 0
  memory_limit_mb: 2048
```

### Component Control

You can launch specific components as needed:

```bash
# Run only audio pipeline and STT
python -m tccc.cli start --modules audio_pipeline stt_engine

# Run with detailed logging
python -m tccc.cli start --log-level debug
```

## Core Components

### 🔍 RAG Database & Document Library

The TCCC.ai system leverages a fully-optimized Retrieval-Augmented Generation system for edge devices:

- **Vector Database**: FAISS-powered embedding store (99% smaller than traditional vector DBs)
- **Document Processing**: Efficient chunking with 1000-character/200-overlap strategy
- **Embedding Model**: all-MiniLM-L12-v2 with INT8 quantization (~90MB)
- **Query Optimization**: Multi-strategy retrieval (semantic, keyword, hybrid)
- **Context Management**: Dynamic context window handling with adaptive sizing
- **Medical Vocabulary**: Specialized medical term processing for TCCC applications

```python
# Access the RAG system
from tccc.document_library import DocumentLibrary

# Initialize with defaults optimized for Jetson
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# Query with different strategies
results = doc_lib.advanced_query(
    "How to treat a tension pneumothorax?",
    strategy="hybrid",
    limit=3
)
```

### 🧠 Edge-Optimized LLM Analysis

The LLM Analysis module provides advanced reasoning with strict resource constraints:

- **Primary Model**: Microsoft Phi-2 (4-bit quantized) for <600MB footprint
- **Context Integration**: Dynamically sized context window adapting to device capabilities
- **Specialized Extraction**: Targeted medical entity and procedure recognition
- **Offline Operation**: Zero-reliance on cloud APIs for sensitive environments
- **Adaptive Processor**: Automatically scales to available compute resources

### 📋 TCCC Casualty Card Implementation (In Progress)

Our current development focus is the TCCC Casualty Card (DD Form 1380) implementation:

- Automated form completion from transcribed communications
- Temporal sequence extraction for medical timeline construction
- Procedure and medication tracking with digital documentation
- Integration with established medical evacuation protocols
- Conforms to DoD documentation requirements while operating fully offline

## Project Structure

```
tccc-project/
├── config/                # Hardware-specific configurations
├── src/tccc/              # Core module implementations
│   ├── audio_pipeline/    # Audio processing optimized for Jetson
│   ├── stt_engine/        # faster-whisper integration (INT8)
│   ├── processing_core/   # Lightweight NLP pipeline
│   ├── llm_analysis/      # Edge-optimized Phi-2 integration
│   ├── data_store/        # Efficient on-device storage
│   └── document_library/  # RAG database implementation
├── tests/                 # Test suite with hardware simulation
└── references/            # Technical specifications
```

## Performance Optimization

TCCC.ai achieves edge performance through targeted optimizations:

- **Model Quantization**: INT8/FP16 precision for 60-80% size reduction
- **Inference Batching**: Grouped operations for higher throughput
- **Memory Management**: Progressive loading with configurable limits
- **CUDA Acceleration**: Custom CUDA kernels for Jetson architecture
- **Benchmarking**: Continuous performance monitoring with power metrics

## Contributing

We welcome contributions that align with our focus on edge AI deployment. See [CONTRIBUTING.md](CONTRIBUTING.md) for our development guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [System Architecture](references/architecture/system_architecture.md)
- [Edge Deployment Guide](references/best_practices/development_guide.md)
- [TCCC Protocol Standards](references/module_specs/)