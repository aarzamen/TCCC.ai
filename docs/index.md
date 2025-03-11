# TCCC.ai Documentation

Welcome to the TCCC.ai documentation. This site contains comprehensive information about the TCCC.ai project, its architecture, components, and usage guidelines.

## What is TCCC.ai?

TCCC.ai (Tactical Combat Casualty Care AI) is an edge-deployed AI system designed to function as a "black box recorder" for combat medicine. Operating completely offline on the NVIDIA Jetson platform, it captures and processes audio during medical procedures to create structured documentation and timeline records without requiring network connectivity.

## Key Features

- **Offline Speech Processing**: Captures and transcribes medical narratives in battlefield conditions
- **Battlefield Audio Enhancement**: Multi-stage noise reduction optimized for combat environments
- **Medical Entity Recognition**: Extracts medical procedures, timelines, and vital information
- **Tactical Field Documentation**: Generates standardized casualty documentation for evacuation
- **Edge Deployment**: Fully operational on NVIDIA Jetson platforms with optimized resource usage

## Getting Started

To get started with TCCC.ai, please follow these guides:

- [Installation](guides/getting_started.md)
- [Deployment Guide](guides/deployment.md)
- [Hardware Setup](guides/display_setup.md)

## Architecture Overview

TCCC.ai follows a modular architecture with clean interfaces between components:

- [System Architecture](architecture/system_architecture.md)
- [Component Interactions](architecture/components.md)

## Module Documentation

Detailed documentation for each module:

- [Audio Pipeline](modules/audio_pipeline.md)
- [Speech-to-Text Engine](modules/stt_engine.md)
- [LLM Analysis](modules/llm_analysis.md)
- [Document Library](modules/document_library.md)
- [Processing Core](modules/processing_core.md)
- [Data Store](modules/data_store.md)

## Development

For developers looking to contribute to TCCC.ai:

- [Contribution Guide](development/contribution_guide.md)
- [Development Guide](development/development_guide.md)
- [Testing Guide](development/testing_guide.md)

## License

TCCC.ai is released under the [MIT License](https://github.com/tccc-ai/tccc-project/blob/main/LICENSE).