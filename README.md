# TCCC.ai: Transcription, Compliance, and Customer Care AI System

TCCC.ai is an AI-powered system for call center operations, providing real-time transcription, compliance monitoring, and agent assistance capabilities optimized for the NVIDIA Jetson Orin Nano platform.

## Overview

The TCCC.ai system monitors call center interactions in real-time, provides transcripts, analyzes conversations, ensures compliance with regulations, and assists agents with actionable recommendations. The system is designed to run efficiently on edge hardware like the NVIDIA Jetson Orin Nano.

## Architecture

The system consists of six core modules:

1. **Audio Pipeline**: Captures and processes audio streams from call center interactions
2. **STT Engine**: Converts audio to text with speaker diarization capabilities
3. **Processing Core**: Analyzes transcriptions for entities, intents, and sentiment
4. **LLM Analysis**: Provides advanced analysis and agent recommendations
5. **Data Store**: Persists and manages conversation data and analysis results
6. **Document Library**: Manages reference documents and knowledge base

## Getting Started

### Prerequisites

- NVIDIA Jetson Orin Nano (or compatible Jetson device)
- JetPack 5.1.1 or later
- Python 3.9 or later
- ~15GB of storage for models and dependencies

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tccc-ai/tccc-project.git
   cd tccc-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

   For development:
   ```bash
   pip install -e ".[dev]"
   ```

   For GPU acceleration:
   ```bash
   pip install -e ".[gpu]"
   ```

### Configuration

The system uses YAML configuration files located in the `config/` directory. You can modify these files to adjust various parameters for each module.

### Running the System

To start the TCCC.ai system:

```bash
python -m tccc.cli start
```

To run specific modules:

```bash
python -m tccc.cli start --modules audio_pipeline stt_engine
```

## Development

### Current Features

- [x] **Document Library RAG Implementation**
  - Embedding-based semantic search using Nexa AI's all-MiniLM-L12-v2 model
  - Efficient document chunking and indexing with FAISS vector database
  - In-memory and disk-based caching with TTL and size limits
  - Support for text files with extensible architecture for PDF, DOCX, HTML

### Next Development Tasks

- [ ] **TCCC Casualty Card (DD Form 1380) Implementation** - HIGH PRIORITY
  - Purpose: Core functionality for documenting battlefield casualties
  - Reference: https://tccc.org.ua/files/downloads/tccc-cpp-skill-card-55-dd-1380-tccc-casualty-card-en.pdf
  - Requirements:
    - Automated completion based on audio input and LLM analysis
    - Digital form generation matching DD Form 1380 format
    - Equal importance to 9-line MEDEVAC and ZMIST record capabilities
  - Integration: Must connect with Document Library and LLM Analysis modules

### Project Structure

```
tccc-project/
├── config/                # Configuration files
├── references/            # Reference documentation
│   ├── architecture/      # System architecture documentation
│   ├── interfaces/        # Module interface definitions
│   └── best_practices/    # Development best practices
├── src/
│   └── tccc/              # Main package
│       ├── audio_pipeline/    # Audio capture and processing
│       ├── stt_engine/        # Speech-to-text conversion
│       ├── processing_core/   # NLP and text analysis
│       ├── llm_analysis/      # LLM-based recommendations
│       ├── data_store/        # Data persistence
│       └── document_library/  # Document management
├── tests/                 # Test suite
└── docs/                  # Documentation
```

### Running Tests

```bash
pytest
```

### Performance Optimization for Jetson

The system is optimized for the NVIDIA Jetson Orin Nano platform, using:

- TensorRT for model acceleration
- FP16/INT8 quantization
- CUDA optimization for parallel processing
- Model compression techniques
- Streaming architecture for real-time processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Reference

For detailed information about the system's architecture and module interfaces, refer to the `references/` directory.

- [System Architecture](references/architecture/system_architecture.md)
- [Module Interfaces](references/interfaces/module_interfaces.md)
- [Development Best Practices](references/best_practices/development_guide.md)