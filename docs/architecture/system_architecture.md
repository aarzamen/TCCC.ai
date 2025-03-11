# System Architecture

## Overview

TCCC.ai follows a modular architecture with clean interfaces between components. This approach allows for independent development, testing, and optimization of each component while ensuring they work together seamlessly in the integrated system.

## Architecture Diagram

![System Architecture](images/system_architecture.png)

## Core Components

### Audio Pipeline

The Audio Pipeline is responsible for capturing and processing audio from the environment. It includes:

- Audio capture from microphone input
- Voice Activity Detection (VAD) using Silero VAD
- Battlefield noise filtering
- Neural speech enhancement with FullSubNet
- Audio chunk management for efficient processing

### Speech-to-Text (STT) Engine

The STT Engine transcribes processed audio into text. Key features:

- Implements optimized Whisper model (faster-whisper-tiny.en)
- Battlefield-specific configurations for improved accuracy
- Adaptive processing based on audio quality
- Support for medical terminology

### LLM Analysis

The LLM Analysis module extracts medical information from transcribed text:

- Utilizes edge-optimized Phi-2 model
- Extracts medical procedures, medications, and vital signs
- Identifies timeline information and event sequences
- Maintains context across multiple transcription segments

### Document Library

The Document Library provides medical reference information:

- Implements RAG (Retrieval-Augmented Generation) system
- Uses all-MiniLM-L12-v2 embedding model
- Local vector database (FAISS/Chroma) for document retrieval
- Stores tactical medical protocols and guidelines

### Processing Core

The Processing Core coordinates all system components:

- Manages system resources and memory allocation
- Coordinates event flow between components
- Implements plugin architecture for extensibility
- Provides state management across the system

### Data Store

The Data Store maintains all system data:

- SQLite database with WAL mode
- Structured event storage
- Timeline database for medical events
- Optimized for Jetson's NVMe storage

## Data Flow

1. Audio is captured by the Audio Pipeline from microphone input
2. Processed audio is sent to the STT Engine for transcription
3. Transcribed text is analyzed by the LLM Analysis module
4. Extracted medical information is stored in the Data Store
5. The Document Library provides context for medical analysis
6. The Processing Core coordinates the entire process

## Memory Management

TCCC.ai implements a sophisticated memory management system to operate within the constraints of edge devices:

- Models are quantized (INT8/INT4) for reduced memory footprint
- TensorRT optimization for improved inference performance
- Dynamic model loading based on system requirements
- Efficient tensor operations with mixed precision

## Communication Protocol

Components communicate through a standardized event system:

- ZeroMQ for lightweight inter-module communication
- JSON-serialized event payloads
- Typed event schemas with validation
- Publish-subscribe pattern for event distribution

## Resource Allocation

The system implements dynamic resource allocation:

```
┌─────────────────────────────────────────────┐
│ Total System Memory (8GB on Jetson Orin)    │
├─────────────┬───────────────┬───────────────┤
│ STT Engine  │ LLM Analysis  │  Doc Library  │
│ (1-2GB)     │ (2-3GB)       │  (1GB)        │
├─────────────┴───────────────┴───────────────┤
│ System Overhead + Processing Core (2-3GB)   │
└─────────────────────────────────────────────┘
```

## Integration Points

TCCC.ai provides several integration points:

- Display interface for visual output
- Audio capture from various microphone sources
- External storage interface for data export
- Configuration system for customization