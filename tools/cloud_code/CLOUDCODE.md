# TCCC.ai Project Guide for Claude Code CLI

This document is structured to optimize Claude Code CLI interactions for the TCCC.ai project. It provides essential context, module-specific guidance, and structured workflows for efficient AI collaboration.

---

## 📋 Document Structure

This document is organized into modular sections to optimize Claude's context window usage:

- **Core Sections:** Always include these sections in every Claude session
- **Module Sections:** Include only sections relevant to your current task
- **Reference Sections:** Include when working with specific technologies or patterns
- **Workflow Sections:** Include when performing specific development activities

**Usage Instructions:**
1. For each Claude Code session, include relevant sections based on your task
2. Always include Core Sections for essential context
3. Include only Module Sections for components you're actively developing
4. Add Reference Sections only when needed for specific implementations

---

## 🌐 Project Context [CORE]

### Project Overview

TCCC.ai (Tactical Combat Casualty Care AI) is an edge-deployed artificial intelligence system designed to support combat medics in documenting medical procedures and interventions. The system:

- Captures audio during combat medical procedures
- Processes and enhances audio in challenging environments
- Transcribes speech to text with medical terminology awareness
- Extracts medical entities and events
- Generates structured documentation and timelines
- Provides reference information from medical protocols
- Operates completely offline on Jetson Orin Nano hardware

### System Architecture

The system uses an event-driven architecture with modular components:

```
┌─────────────┐     ┌────────────┐     ┌─────────────────┐     ┌────────────────┐
│ Audio Input │────▶│ Audio      │────▶│ Speech-to-Text  │────▶│ Processing     │
│ (Microphone)│     │ Pipeline   │     │ Engine          │     │ Core           │
└─────────────┘     └────────────┘     └─────────────────┘     └────────┬───────┘
                                                                        │
                      ┌────────────────────────────────────────────┐    │
                      │                                            │    │
                      ▼                                            ▼    ▼
               ┌─────────────┐     ┌────────────────┐     ┌───────────────┐
               │ Document    │◀───▶│ LLM            │◀───▶│ Data          │
               │ Library     │     │ Analysis       │     │ Store         │
               └─────────────┘     └────────────────┘     └───────────────┘
                                           │
                                           ▼
                                   ┌───────────────┐
                                   │ Display       │
                                   │ Interface     │
                                   └───────────────┘
```

### Hardware Constraints

- **Platform:** NVIDIA Jetson Orin Nano 8GB
- **CPU:** 6-core Arm® Cortex®-A78AE v8.2 (1.5 GHz)
- **GPU:** 512-core NVIDIA Ampere architecture (897 MHz)
- **Memory:** 8GB LPDDR5 (shared between CPU and GPU)
- **Storage:** 64GB eMMC
- **Power:** Battery-operated with thermal constraints
- **Peripherals:**
  - Razer Seiren V3 Mini microphone (USB)
  - WaveShare 7" display (1280x800 resolution)

### Key Performance Requirements

- **Latency:** Audio processing < 500ms
- **Transcription:** Near real-time (< 2x audio duration)
- **LLM Analysis:** < 1 second per analysis operation
- **Battery Life:** 6-8 hours of continuous operation
- **Memory Usage:** < 6GB total system memory

---

## 💻 Development Environment [CORE]

### Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Verify system dependencies
python check_dependencies.py

# Run all verification scripts
./run_all_verifications.sh
```

### Project Structure

```
src/tccc/               # Main package directory
├── audio_pipeline/     # Audio capture and processing
├── stt_engine/         # Speech-to-text conversion
├── processing_core/    # Event processing and orchestration
├── llm_analysis/       # LLM-based analysis with Phi-2
├── document_library/   # Document storage and retrieval
├── data_store/         # Data persistence
├── display/            # Display interface
└── utils/              # Shared utilities
    ├── vad_manager.py  # Voice activity detection
    ├── jetson_optimizer.py # Hardware acceleration
    └── tensor_optimization.py # Model optimization
```

### Core Development Principles

1. **Production Quality:** All code must be fully functional, robust, and error-resistant
2. **Resource Efficiency:** Optimize for memory, CPU/GPU, and power constraints
3. **Graceful Degradation:** System should continue functioning with reduced capabilities when resources are constrained
4. **Self-Contained Functionality:** Components should initialize automatically when possible
5. **Hardware Awareness:** All code should accommodate Jetson hardware limitations

---

## 🧠 Claude Code CLI Workflow [CORE]

### Session Structure

Follow this structure for effective Claude Code CLI sessions:

1. **Context Establishment:**
   - Include relevant CloudCode.md sections based on your task
   - Reference relevant existing code and interfaces
   - Specify module(s) being worked on

2. **Task Definition:**
   - Define specific task with clear scope boundaries
   - Include acceptance criteria or requirements
   - Reference any design patterns or architectural constraints

3. **Implementation Request:**
   - Ask for specific implementation details
   - Include interface requirements and expectations
   - Reference existing code patterns when applicable

4. **Review & Refine:**
   - Review Claude's implementation critically
   - Request specific improvements or optimizations
   - Test on target hardware when possible

### Effective Prompting Patterns

When requesting implementations, use this structure:

```
claude "Implement [specific functionality] for the [module] that [purpose].

Requirements:
- Must implement this interface: [interface code]
- Must handle these errors: [error conditions]
- Should optimize for [specific constraints]
- Must follow project patterns for [relevant pattern]

Example of similar functionality:
```python
[example code]
```

Context:
This component interacts with [related components] and is called during [usage scenario]."
```

### Context Management

To maximize Claude's context window efficiency:

1. **Use Progressive Detail:**
   - Start with high-level interface requirements
   - Add implementation details in subsequent messages
   - Focus initial implementation on core functionality

2. **Use /compact Command:**
   - Use `/compact` when conversation becomes lengthy
   - Start new sessions for substantially different tasks
   - Provide summary of previous work when starting new sessions

3. **Reference Previous Work:**
   - Reference file paths and commit hashes
   - Use `git diff` to show specific changes
   - Save iterations with meaningful commit messages

---

## 🎤 Audio Pipeline Module [MODULE]

### Overview

The Audio Pipeline captures and processes audio from physical microphones or file sources, enhances audio quality, detects voice activity, and delivers processed audio segments to the STT Engine.

### Key Components

- **Audio Source Interface:** Abstracts different audio inputs (microphone, file, network)
- **Stream Buffer:** Thread-safe buffer for audio segment management
- **Voice Activity Detection:** Identifies speech segments using Silero VAD
- **Audio Processor:** Applies noise reduction and audio enhancement
- **Battlefield Audio Enhancer:** Specialized audio enhancement for combat environments

### Interface Contract

```python
class AudioPipeline:
    def initialize(self, config: dict) -> bool:
        """Initialize the audio pipeline with configuration"""
        pass
        
    def start_capture(self, source_name: str = None) -> bool:
        """Start audio capture from specified source"""
        pass
        
    def stop_capture() -> bool:
        """Stop audio capture"""
        pass
        
    def get_audio_stream() -> StreamBuffer:
        """Get stream buffer for continuous reading"""
        pass
        
    def get_audio(timeout_ms: int = None) -> tuple[np.ndarray, dict]:
        """Get next processed audio segment with metadata"""
        pass
        
    def set_quality_parameters(params: dict) -> bool:
        """Update audio quality parameters dynamically"""
        pass
        
    def get_status() -> dict:
        """Get current status and diagnostics"""
        pass
```

### Implementation Considerations

1. **Thread Safety:**
   - Audio capture runs in separate thread
   - Buffer access must be thread-safe
   - Proper thread cleanup on shutdown

2. **Resource Management:**
   - Monitor CPU usage during processing
   - Use efficient numpy operations
   - Implement timeout mechanisms

3. **Error Handling:**
   - Handle device initialization failures
   - Recover from audio device disconnection
   - Implement retry mechanisms with backoff

4. **Performance Optimization:**
   - Use numpy vectorized operations
   - Minimize memory copies
   - Batch process when possible

### Example Implementation Request

```
claude "Implement the StreamBuffer class for the AudioPipeline that provides thread-safe audio segment management.

Requirements:
- Must be thread-safe for multi-producer/consumer access
- Must support configurable buffer size and timeout
- Must handle audio data as numpy arrays with sample rate and format metadata
- Should implement peek() functionality for non-destructive reads
- Must handle buffer overflow conditions gracefully

Interface:
```python
class StreamBuffer:
    def __init__(self, max_size: int = 100):
        """Initialize buffer with maximum size"""
        pass
        
    def write(self, data: np.ndarray, metadata: dict = None, timeout: float = None) -> bool:
        """Write data to buffer with optional timeout"""
        pass
        
    def read(self, timeout: float = None) -> tuple[np.ndarray, dict]:
        """Read and remove data from buffer with optional timeout"""
        pass
        
    def peek(self) -> tuple[np.ndarray, dict]:
        """Read without removing data from buffer"""
        pass
        
    def clear(self) -> None:
        """Clear all data from buffer"""
        pass
        
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        pass
        
    def is_full(self) -> bool:
        """Check if buffer is full"""
        pass
        
    def get_level(self) -> float:
        """Get buffer fill level (0.0-1.0)"""
        pass
```

Use a queue.Queue implementation with proper locking mechanisms. Consider implementing overflow handling by dropping oldest segments when necessary."
```

---

## 🗣️ STT Engine Module [MODULE]

### Overview

The STT Engine converts audio segments to text using the faster-whisper implementation of the Whisper model, optimized for medical terminology and battlefield conditions.

### Key Components

- **Model Manager:** Handles model loading, caching, and optimization
- **Transcription Engine:** Processes audio and returns text with timestamps
- **Medical Terminology Processor:** Improves accuracy for medical terms
- **Diarization:** Speaker identification (when multiple speakers)
- **Performance Monitor:** Tracks transcription speed and resource usage

### Interface Contract

```python
class STTEngine:
    def initialize(self, config: dict) -> bool:
        """Initialize the STT engine with configuration"""
        pass
        
    def transcribe_segment(self, audio: np.ndarray, metadata: dict = None) -> dict:
        """
        Transcribe an audio segment to text
        
        Returns dict with:
        - text: Full transcription text
        - segments: List of timed segments
        - words: Word-level detail with timestamps
        - confidence: Overall confidence score
        - processing_time: Time taken to transcribe
        """
        pass
        
    def transcribe_file(self, file_path: str, config: dict = None) -> dict:
        """Transcribe audio from file"""
        pass
        
    def get_status(self) -> dict:
        """Get engine status and diagnostics"""
        pass
        
    def shutdown(self) -> bool:
        """Release resources and shut down engine"""
        pass
```

### Implementation Considerations

1. **Model Optimization:**
   - INT8 quantization for Whisper model
   - TensorRT acceleration when available
   - Dynamic batch sizing based on available memory

2. **Medical Accuracy:**
   - Custom medical vocabulary incorporation
   - Post-processing for medical acronyms and terminology
   - Confidence scoring for medical terms

3. **Resource Management:**
   - Progressive model loading (small → medium → large)
   - GPU memory monitoring
   - Model offloading when idle

4. **Performance Optimization:**
   - Chunk-based processing for long audio
   - Caching of frequently used terms
   - Beam search parameters tuning

### Example Implementation Request

```
claude "Implement the medical_term_processor module for the STT Engine that improves transcription accuracy for medical terminology.

Requirements:
- Must integrate with faster-whisper output
- Must handle common TCCC medical abbreviations and terms
- Should correct common misrecognitions of medical terms
- Must maintain original timestamps and segmentation
- Should accept custom vocabulary additions

Interface:
```python
def process_medical_terms(
    transcription_result: dict,
    custom_vocabulary: List[str] = None
) -> dict:
    """
    Process transcription results to improve medical terminology accuracy
    
    Args:
        transcription_result: Original faster-whisper output
        custom_vocabulary: Additional terms to recognize
        
    Returns:
        Processed transcription with improved medical terminology
    """
    pass
```

Implementation should use a combination of pattern matching and custom vocabulary lookup with edit distance for error tolerance. Include common TCCC abbreviations like 'TQ' (tourniquet), 'GSW' (gunshot wound), etc. Consider implementing a confidence-based replacement strategy."
```

---

## 🔄 Processing Core Module [MODULE]

### Overview

The Processing Core orchestrates the entire system, managing event flow between components, tracking system state, and allocating resources dynamically.

### Key Components

- **Plugin Manager:** Handles dynamic loading of processing modules
- **State Manager:** Maintains system state and session information
- **Resource Monitor:** Tracks and allocates system resources
- **Event Router:** Directs events to appropriate handlers
- **Entity Extractor:** Identifies entities in text
- **Intent Classifier:** Determines user intent from input

### Interface Contract

```python
class ProcessingCore:
    def initialize(self, config: dict) -> bool:
        """Initialize processing core with configuration"""
        pass
        
    def register_module(self, 
                       name: str, 
                       module_type: str, 
                       instance: Any, 
                       dependencies: List[str] = None,
                       config: dict = None) -> bool:
        """Register a module with the core"""
        pass
        
    def process_event(self, event: dict) -> dict:
        """Process an event and return result"""
        pass
        
    def get_state(self) -> dict:
        """Get current system state"""
        pass
        
    def shutdown(self) -> bool:
        """Release resources and shut down"""
        pass
```

### Implementation Considerations

1. **Event Schema:**
   - Standardized event format with type, data, metadata
   - Event validation and normalization
   - Event sequence tracking

2. **Resource Management:**
   - Dynamic resource allocation
   - Priority-based scheduling
   - Memory usage monitoring and limits

3. **Error Handling:**
   - Component isolation for fault tolerance
   - Error propagation with severity levels
   - Recovery strategies for non-critical failures

4. **Module Dependencies:**
   - Dependency graph validation
   - Circular dependency detection
   - Initialization order determination

### Example Implementation Request

```
claude "Implement the ResourceMonitor class for the ProcessingCore that manages and allocates system resources.

Requirements:
- Must track CPU, GPU, and memory usage
- Must support resource reservation and release
- Must implement priority-based resource allocation
- Should handle resource contention with configurable strategies
- Must detect and respond to resource exhaustion

Interface:
```python
class ResourceMonitor:
    def __init__(self, config: dict = None):
        """Initialize resource monitor with configuration"""
        pass
        
    def get_resource_usage(self) -> dict:
        """Get current resource usage statistics"""
        pass
        
    def reserve_resources(self, 
                         component: str, 
                         resources: dict,
                         priority: int = 0) -> bool:
        """
        Request resources allocation with priority
        
        Args:
            component: Requesting component name
            resources: Dict of resource types and amounts
            priority: Request priority (higher = more important)
            
        Returns:
            Success flag
        """
        pass
        
    def release_resources(self, component: str) -> None:
        """Release resources allocated to component"""
        pass
        
    def check_availability(self, resources: dict) -> bool:
        """Check if requested resources are available"""
        pass
        
    def set_resource_strategy(self, strategy: str) -> None:
        """Set resource allocation strategy"""
        pass
```

Implement for Jetson-specific resource monitoring using CUDA tools for GPU and psutil for CPU/memory. Include support for preemptively releasing low-priority allocations when high-priority requests arrive."
```

---

## 📝 LLM Analysis Module [MODULE]

### Overview

The LLM Analysis module uses the Phi-2 language model to extract medical information, generate structured reports, and provide context-aware analysis of transcribed content.

### Key Components

- **Phi-2 Model:** Local LLM optimized for Jetson platform
- **Medical Entity Extractor:** Identifies medical entities and events
- **Report Generator:** Creates structured medical documentation
- **Context Manager:** Integrates document knowledge with analysis
- **Cache Manager:** Optimizes repeated analyses

### Interface Contract

```python
class LLMAnalysis:
    def initialize(self, config: dict) -> bool:
        """Initialize LLM analysis with configuration"""
        pass
        
    def process_text(self, 
                    text: str, 
                    context: dict = None,
                    options: dict = None) -> dict:
        """
        Process text through LLM analysis
        
        Args:
            text: Text to analyze
            context: Additional context information
            options: Processing options
            
        Returns:
            Analysis results with identified entities and insights
        """
        pass
        
    def generate_report(self, 
                       report_type: str, 
                       events: List[dict],
                       options: dict = None) -> dict:
        """
        Generate structured report from events
        
        Args:
            report_type: Type of report to generate
            events: List of events to include
            options: Report generation options
            
        Returns:
            Structured report
        """
        pass
        
    def get_status(self) -> dict:
        """Get analysis engine status"""
        pass
        
    def set_document_library(self, document_library: Any) -> None:
        """Set document library for context enhancement"""
        pass
```

### Implementation Considerations

1. **Model Optimization:**
   - INT8 quantization for Phi-2
   - TensorRT integration for GPU acceleration
   - Context window management for limited memory

2. **Medical Knowledge:**
   - Prompt engineering for medical extraction
   - Structured output formatting
   - Confidence scoring for extracted information

3. **Performance Optimization:**
   - Response caching for similar queries
   - Staged processing for complex analyses
   - Memory-efficient prompt construction

4. **Integration with Document Library:**
   - Context-aware analysis with relevant documents
   - Just-in-time retrieval of document information
   - Knowledge fusion from multiple sources

### Example Implementation Request

```
claude "Implement the medical_entity_extractor function for the LLM Analysis module that identifies medical entities from transcribed text.

Requirements:
- Must use the Phi-2 model efficiently with optimized prompts
- Must extract TCCC-relevant medical entities (injuries, medications, procedures)
- Should extract temporal relationships between events
- Must return structured data in consistent format
- Should handle ambiguity with confidence scores

Interface:
```python
def extract_medical_entities(
    text: str,
    phi_model: Any,
    options: dict = None
) -> dict:
    """
    Extract medical entities from text using Phi-2 model
    
    Args:
        text: Transcribed text to analyze
        phi_model: Initialized Phi-2 model instance
        options: Extraction options and parameters
        
    Returns:
        Dictionary containing:
        - entities: List of identified medical entities
        - procedures: List of medical procedures
        - timeline: Temporal sequence of events
        - confidence: Confidence scores for extractions
    """
    pass
```

Use structured prompting techniques with few-shot examples for the Phi-2 model. Implement prompt template for efficient extraction with minimal token usage. Include entity categories for injuries, vital signs, medications, procedures, and equipment."
```

---

## 📚 Document Library Module [MODULE]

### Overview

The Document Library manages medical reference documents, providing semantic search capabilities, context retrieval, and domain-specific knowledge for the LLM Analysis module.

### Key Components

- **Vector Store:** FAISS-based embedding storage
- **Document Processor:** Handles document ingestion and chunking
- **Query Engine:** Performs semantic and keyword search
- **Medical Vocabulary:** Manages domain-specific terminology
- **Cache Manager:** Optimizes frequent queries

### Interface Contract

```python
class DocumentLibrary:
    def initialize(self, config: dict) -> bool:
        """Initialize document library with configuration"""
        pass
        
    def add_document(self, document_data: dict) -> str:
        """
        Add document to library
        
        Args:
            document_data: Document with text and metadata
            
        Returns:
            Document ID
        """
        pass
        
    def query(self, query_text: str, n_results: int = 5) -> List[dict]:
        """
        Query library for relevant documents
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with scores
        """
        pass
        
    def advanced_query(self,
                      query_text: str,
                      strategy: str = "hybrid",
                      limit: int = 5,
                      min_similarity: float = 0.0,
                      filter_metadata: dict = None) -> List[dict]:
        """Advanced query with multiple strategies and filtering"""
        pass
        
    def get_status(self) -> dict:
        """Get library status and statistics"""
        pass
```

### Implementation Considerations

1. **Vector Embeddings:**
   - all-MiniLM-L12-v2 for embeddings
   - Batch embedding generation
   - Vector normalization and preprocessing

2. **Chunking Strategy:**
   - Semantic chunking for medical content
   - Overlapping windows for context preservation
   - Metadata enrichment for chunks

3. **Query Optimization:**
   - Hybrid search (vector + keyword)
   - Medical vocabulary expansion
   - Cache frequently accessed results

4. **Memory Efficiency:**
   - Progressive loading of index
   - Disk-based storage for vector data
   - Memory-mapped access for large indices

### Example Implementation Request

```
claude "Implement the hybrid_search function for the Document Library that combines vector similarity and keyword matching.

Requirements:
- Must combine semantic search with keyword-based retrieval
- Must normalize and rerank combined results
- Should handle medical terminology effectively
- Must support filtering by metadata
- Should implement response caching for performance

Interface:
```python
def hybrid_search(
    query_text: str,
    vector_index: Any,
    document_store: Any,
    limit: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    min_score: float = 0.0,
    filter_metadata: dict = None,
    cache_manager: Any = None
) -> List[dict]:
    """
    Perform hybrid search combining vector similarity and keyword matching
    
    Args:
        query_text: Query text
        vector_index: FAISS vector index
        document_store: Document storage
        limit: Maximum number of results
        semantic_weight: Weight for semantic search results
        keyword_weight: Weight for keyword search results
        min_score: Minimum score threshold
        filter_metadata: Metadata filters
        cache_manager: Optional cache manager
        
    Returns:
        List of document chunks with scores
    """
    pass
```

Implement efficient algorithms for combining vector similarity scores with keyword match scores. Include BM25-style weighting for keyword matches and cosine similarity for vector matches."
```

---

## 💾 Data Store Module [MODULE]

### Overview

The Data Store provides persistent storage for system events, transcriptions, analyses, and reports using an SQLite database with optimizations for the Jetson platform.

### Key Components

- **Event Storage:** Stores system events with timestamps
- **Session Manager:** Tracks and manages user sessions
- **Report Storage:** Manages generated reports
- **Timeline Generator:** Creates chronological event sequences
- **Query Engine:** Retrieves stored data with filtering

### Interface Contract

```python
class DataStore:
    def initialize(self, config: dict) -> bool:
        """Initialize data store with configuration"""
        pass
        
    def store_event(self, event: dict) -> str:
        """
        Store event data
        
        Args:
            event: Event data to store
            
        Returns:
            Event ID
        """
        pass
        
    def store_report(self, report: dict) -> str:
        """Store generated report"""
        pass
        
    def query_events(self, filters: dict = None) -> List[dict]:
        """Query events with optional filters"""
        pass
        
    def get_timeline(self, 
                    start_time: float = None, 
                    end_time: float = None) -> List[dict]:
        """Get timeline of events"""
        pass
        
    def get_context(self, 
                   current_time: float, 
                   window_seconds: float = 300) -> dict:
        """Get contextual window around current time"""
        pass
        
    def backup(self, label: str = None) -> str:
        """Create backup of database"""
        pass
        
    def restore(self, backup_id: str) -> bool:
        """Restore from backup"""
        pass
        
    def get_status(self) -> dict:
        """Get store status and statistics"""
        pass
```

### Implementation Considerations

1. **SQLite Optimization:**
   - WAL mode for concurrent access
   - Proper indexing for performance
   - Connection pooling for thread safety

2. **Data Integrity:**
   - Transaction-based updates
   - Automatic backups
   - Data validation before storage

3. **Performance Optimization:**
   - Query result caching
   - Efficient JSON serialization
   - Prepared statements for frequent queries

4. **Storage Management:**
   - Size limitation and rotation
   - Cleanup of old data
   - Disk space monitoring

### Example Implementation Request

```
claude "Implement the TimelineGenerator class for the DataStore module that creates chronological event sequences from stored events.

Requirements:
- Must extract events from SQLite database efficiently
- Must organize events chronologically with proper timestamps
- Should handle event relationships and causality
- Must support filtering by event type and metadata
- Should implement caching for performance

Interface:
```python
class TimelineGenerator:
    def __init__(self, database_connection: Any, config: dict = None):
        """Initialize with database connection"""
        pass
        
    def generate_timeline(self,
                         start_time: float = None,
                         end_time: float = None,
                         event_types: List[str] = None,
                         filters: dict = None) -> List[dict]:
        """
        Generate timeline of events
        
        Args:
            start_time: Timeline start time (epoch seconds)
            end_time: Timeline end time (epoch seconds)
            event_types: Filter by event types
            filters: Additional metadata filters
            
        Returns:
            Chronological list of events with relationships
        """
        pass
        
    def generate_summary(self, timeline: List[dict]) -> dict:
        """Generate timeline summary with key events"""
        pass
        
    def identify_relationships(self, timeline: List[dict]) -> List[dict]:
        """Identify relationships between events in timeline"""
        pass
        
    def export_timeline(self, timeline: List[dict], format: str = 'json') -> str:
        """Export timeline in specified format"""
        pass
```

Implement using efficient SQL queries with proper indexes, parameter binding for security, and proper handling of timestamp conversions. Include relationship detection between medical events (e.g., injury → treatment → outcome)."
```

---

## 📊 System Integration [MODULE]

### Overview

The System module coordinates all components, manages initialization sequence, handles cross-component communication, and provides unified interfaces for external interaction.

### Key Components

- **System Manager:** Orchestrates component lifecycle
- **Event Bus:** Facilitates component communication
- **Configuration Manager:** Handles system-wide settings
- **Display Integration:** Coordinates visual outputs
- **Error Handler:** Manages system-wide errors

### Interface Contract

```python
class TCCCSystem:
    def initialize(self, config: dict, mock_modules: List[str] = None) -> bool:
        """Initialize system with configuration"""
        pass
        
    def start_audio_capture(self, source_id: str = None) -> bool:
        """Start audio capture from specified source"""
        pass
        
    def process_event(self, event_data: dict) -> dict:
        """Process an external event"""
        pass
        
    def generate_reports(self, report_types: List[str]) -> List[dict]:
        """Generate reports of specified types"""
        pass
        
    def get_status(self) -> dict:
        """Get system status"""
        pass
        
    def start(self) -> bool:
        """Start system operation"""
        pass
        
    def stop(self) -> bool:
        """Stop system operation"""
        pass
        
    def shutdown(self) -> bool:
        """Shutdown system and release resources"""
        pass
```

### Implementation Considerations

1. **Initialization Sequence:**
   - Dependency-based initialization order
   - Component health validation
   - Graceful handling of component failures

2. **Event Handling:**
   - Standardized event schema
   - Event routing and transformation
   - Event persistence and logging

3. **Error Management:**
   - Error isolation and containment
   - Error propagation with appropriate handling
   - Recovery strategies for different error types

4. **Resource Coordination:**
   - Cross-component resource allocation
   - Priority-based access to shared resources
   - Memory budget distribution

### Example Implementation Request

```
claude "Implement the EventBus class for the System Integration module that facilitates communication between components.

Requirements:
- Must support publish-subscribe pattern for event distribution
- Must maintain event schema validation
- Should route events based on type and content
- Must handle event delivery failures gracefully
- Should implement delivery guarantees (at-least-once)

Interface:
```python
class EventBus:
    def __init__(self, config: dict = None):
        """Initialize event bus with configuration"""
        pass
        
    def register_subscriber(self, 
                           component_id: str, 
                           event_types: List[str], 
                           callback: Callable) -> str:
        """
        Register subscriber for event types
        
        Args:
            component_id: Subscriber component ID
            event_types: List of event types to subscribe to
            callback: Function to call when event received
            
        Returns:
            Subscription ID
        """
        pass
        
    def unregister_subscriber(self, subscription_id: str) -> bool:
        """Unregister subscriber"""
        pass
        
    def publish_event(self, event: dict, delivery_options: dict = None) -> bool:
        """
        Publish event to subscribers
        
        Args:
            event: Event data to publish
            delivery_options: Delivery options
            
        Returns:
            Success flag
        """
        pass
        
    def validate_event_schema(self, event: dict, schema_type: str = None) -> bool:
        """Validate event against schema"""
        pass
        
    def get_status(self) -> dict:
        """Get event bus status and statistics"""
        pass
```

Implement with thread-safe operation, efficient event routing, and proper error handling for delivery failures. Include robust schema validation for different event types defined in the system."
```

---

## 🧪 Testing and Verification [WORKFLOW]

### Testing Approach

The TCCC.ai project uses a comprehensive testing strategy:

1. **Unit Testing:** Component-level testing
2. **Integration Testing:** Cross-component interaction testing
3. **System Testing:** End-to-end functionality verification
4. **Performance Testing:** Resource utilization and benchmarking
5. **Hardware Testing:** Validation on target Jetson hardware

### Verification Scripts

```bash
# Run all verification scripts
./run_all_verifications.sh

# Verify specific components
python verification_script_audio_pipeline.py
python verification_script_stt_engine.py
python verification_script_llm_analysis.py
python verification_script_document_library.py
python verification_script_system.py

# Verify async/sync compatibility
python verification_script_async_modules.py

# Test specific hardware interactions
python verification_script_jetson_optimizer.py
```

### Test-Driven Development

When implementing new features:

1. Write verification script first defining expected behavior
2. Implement feature to pass verification tests
3. Run specific verification script to validate
4. Run all verifications to ensure no regressions

### Performance Benchmarking

When optimizing performance:

1. Establish baseline performance metrics
2. Implement optimizations with specific targets
3. Measure improvements against baseline
4. Validate on target hardware

### Example Testing Request

```
claude "Create a verification script for the StreamBuffer implementation in the AudioPipeline.

Requirements:
- Must test thread safety with multiple producers/consumers
- Must verify correct handling of buffer overflow conditions
- Should test timeout functionality
- Must include both normal and error conditions
- Should include performance benchmarking

The script should thoroughly verify the StreamBuffer implementation against its interface contract and performance requirements. Include setup and teardown functions, proper assertions, and comprehensive test coverage."
```

---

## 🚀 Optimization Techniques [REFERENCE]

### Memory Optimization

1. **Model Quantization:**
   - INT8/INT4 quantization for neural networks
   - Mixed precision where appropriate
   - Custom quantization for critical operations

2. **Buffer Management:**
   - Fixed-size pre-allocated buffers
   - Memory pooling for frequent allocations/deallocations
   - Explicit garbage collection triggers

3. **Lazy Loading:**
   - Load resources only when needed
   - Unload resources when idle
   - Progressive loading of large models

### CPU Optimization

1. **Vectorization:**
   - Numpy vectorized operations
   - SIMD instructions where possible
   - Batch processing for efficiency

2. **Concurrency:**
   - Task-based parallelism
   - Thread pool management
   - Proper synchronization with minimal locking

3. **Algorithm Selection:**
   - Time-space tradeoffs appropriate for constraints
   - O(n) or better algorithms for critical paths
   - Approximation algorithms where appropriate

### GPU Optimization

1. **TensorRT Acceleration:**
   - Model conversion to TensorRT format
   - FP16 precision for GPU operations
   - Kernel fusion for multiple operations

2. **CUDA Optimizations:**
   - Custom CUDA kernels for specific operations
   - Stream management for concurrent execution
   - Memory transfers minimization

3. **Resource Sharing:**
   - GPU memory management
   - Computation scheduling
   - Power/thermal management

### Example Optimization Request

```
claude "Optimize the following audio processing function for better performance on the Jetson Orin Nano:

```python
def process_audio_frame(frame):
    # Apply FFT
    fft_result = np.fft.rfft(frame)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # Apply spectral enhancement
    enhanced_magnitude = np.zeros_like(magnitude)
    for i in range(len(magnitude)):
        if i < len(magnitude) // 4:  # Low frequencies
            enhanced_magnitude[i] = magnitude[i] * 1.2
        elif i < len(magnitude) // 2:  # Mid frequencies
            enhanced_magnitude[i] = magnitude[i] * 1.5
        else:  # High frequencies
            enhanced_magnitude[i] = magnitude[i] * 0.8
    
    # Apply noise reduction
    noise_floor = np.mean(magnitude[-100:]) * 2
    mask = enhanced_magnitude > noise_floor
    enhanced_magnitude = enhanced_magnitude * mask
    
    # Reconstruct signal
    enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_frame = np.fft.irfft(enhanced_fft)
    
    return enhanced_frame
```

Optimize this function for better performance on the Jetson Orin Nano. Consider vectorization, GPU acceleration if appropriate, and algorithmic improvements. The function should maintain the same functionality while using less CPU and executing faster."
```

---

## 🔍 Troubleshooting Guide [REFERENCE]

### Common Error Patterns

1. **Resource Exhaustion:**
   - Memory leaks in long-running processes
   - GPU memory fragmentation
   - File handle exhaustion

2. **Concurrency Issues:**
   - Deadlocks between components
   - Race conditions in shared state
   - Thread synchronization failures

3. **Hardware Interaction:**
   - Microphone initialization failures
   - Display communication errors
   - Thermal throttling

4. **Model Execution:**
   - CUDA errors in model inference
   - Quantization artifacts
   - TensorRT compatibility issues

### Diagnostic Approaches

1. **Systematic Isolation:**
   - Component-by-component testing
   - Mock interfaces for isolation
   - Controlled resource environment

2. **Logging Analysis:**
   - Detailed logging with timestamps
   - Component-specific log levels
   - Performance metrics logging

3. **Resource Monitoring:**
   - Memory profiling
   - CPU/GPU utilization tracking
   - Thermal monitoring

### Example Troubleshooting Request

```
claude "Debug this error occurring in the STT Engine:

```
CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
Device GPU memory allocation failed during model initialization
```

This error occurs when initializing the STT Engine with the medium Whisper model. The system has 8GB of memory, and the GPU should be able to handle this model size. 

Provide:
1. A systematic debugging approach to identify the root cause
2. Potential solutions for this specific error
3. Prevention strategies to avoid similar issues in the future

Consider memory fragmentation, concurrent model loading, and Jetson-specific CUDA constraints."
```

---

## 📇 Terminal Snippet Library [TERMINAL SNIPPETS]

### 1. Implement Module Function

```
claude "Implement the [function_name] function for the [module_name] module that [purpose].

Requirements:
- Must follow interface: [interface_details]
- Handle errors: [error_conditions]
- Performance target: [performance_requirements]

Context:
- This function is called by: [calling_context]
- Interacts with: [related_components]

Example inputs/outputs:
[examples]"
```

### 2. Debug Module Issue

```
claude "Debug the following error in the [module_name] module:

```
[error_message]
```

The error occurs when [reproduction_steps].
Relevant code:
```python
[relevant_code]
```

Please:
1. Identify the root cause
2. Explain the issue
3. Provide a corrected implementation
4. Suggest how to prevent similar issues"
```

### 3. Optimize Performance

```
claude "Optimize the following [module_name] function for Jetson Orin Nano performance:

```python
[function_code]
```

Current performance metrics:
- Execution time: [current_time]
- Memory usage: [current_memory]
- CPU/GPU utilization: [current_utilization]

Target metrics:
- Execution time: [target_time]
- Memory usage: [target_memory]
- CPU/GPU utilization: [target_utilization]

Constraints:
- Must maintain same interface and functionality
- Must handle same error conditions
- Consider hardware-specific optimizations"
```

### 4. Generate Unit Tests

```
claude "Create comprehensive unit tests for the following [module_name] function:

```python
[function_code]
```

Test requirements:
- Test normal operation paths
- Test all error conditions
- Test edge cases: [edge_cases]
- Include performance assertions where relevant
- Use pytest framework with standard fixtures

Include setup/teardown requirements and any mocking needed."
```

### 5. Architecture Review (CRITICAL)

```
claude "Perform an urgent architecture review for the [component_name] component.

We're experiencing these critical issues:
[issues_description]

Relevant architecture:
[architecture_description]

Specifically analyze:
1. Interface design problems
2. Dependency issues or circular references
3. Resource contention or race conditions
4. State management flaws
5. Error propagation weaknesses

Provide specific recommendations with examples."
```

### 6. Module Integration

```
claude "Create integration code between the [module1] and [module2] modules.

Module 1 Interface:
```python
[module1_interface]
```

Module 2 Interface:
```python
[module2_interface]
```

Integration requirements:
- Handle async communication between modules
- Manage resource sharing: [resource_details]
- Implement proper error propagation
- Include logging at integration points
- Consider performance impact of integration

Provide both the integration code and example usage."
```

---

## 📝 Documentation Generation [WORKFLOW]

### Documentation Standards

1. **Inline Documentation:**
   - Docstrings for all public methods
   - Type hints for parameters and returns
   - Examples for complex functions
   - Performance considerations documented

2. **Module Documentation:**
   - Purpose and responsibility
   - Dependencies and requirements
   - Usage examples
   - Configuration options

3. **Architecture Documentation:**
   - Component relationships
   - Data flow descriptions
   - Resource management approach
   - Initialization sequences

### Example Documentation Request

```
claude "Generate comprehensive documentation for the StreamBuffer class in AudioPipeline.

Include:
1. Class overview and purpose
2. Method documentation with parameters and return values
3. Usage examples for common scenarios
4. Thread safety considerations
5. Performance characteristics
6. Error handling approach

Format the documentation using proper docstring format with Google style, including type hints and examples."
```

---

## 🔧 CloudCode.md Maintenance Guidelines [META]

### Version Management

This document should be versioned with the project codebase. Use git commit messages with the prefix "CloudCode:" when updating this file.

### Update Process

1. When adding new modules:
   - Create a new module section with the [MODULE] tag
   - Follow the established template structure
   - Include interface contracts, implementation considerations, and examples

2. When updating existing modules:
   - Update interface contracts to reflect changes
   - Add new implementation considerations
   - Update examples to show current patterns

3. When adding workflows:
   - Create new workflow sections with the [WORKFLOW] tag
   - Include process description, steps, and examples
   - Reference relevant modules and interfaces

### Optimization for Claude

1. **Section Modularity:**
   - Keep sections self-contained for selective inclusion
   - Use consistent headings for navigation
   - Tag sections with their category ([CORE], [MODULE], etc.)

2. **Context Efficiency:**
   - Use concise language and avoid redundancy
   - Prioritize interface details and examples
   - Include only essential implementation guidance

3. **Progressive Detail:**
   - Start with high-level information
   - Provide details progressively
   - Use collapsible sections for detailed explanations

### Example Update Request

```
claude "Update the STT Engine section of CloudCode.md to include the new dynamic_model_scaling feature.

The feature allows automatic scaling of model size based on available GPU memory and should be documented with:
1. Interface changes to support this feature
2. Implementation considerations
3. Example usage
4. Integration with other components

Follow the existing format and style of the document, maintaining context efficiency."
```