# Event Schema Implementation

This document outlines the implementation of the standardized event schema for inter-module communication in the TCCC.ai system.

## Overview

The event schema provides a standardized way for components to communicate with each other through well-defined events. This approach addresses the problem of tight coupling between modules, making the system more maintainable, testable, and extensible.

## Key Components

### 1. Event Schema (`utils/event_schema.py`)

The event schema defines the structure and types of events used in the system:

- **BaseEvent**: The base class for all events, providing common functionality like serialization/deserialization.
- **Specialized Events**:
  - `AudioSegmentEvent`: For audio data from the AudioPipeline
  - `TranscriptionEvent`: For transcription results from the STT Engine
  - `ProcessedTextEvent`: For processed text data from the ProcessingCore
  - `LLMAnalysisEvent`: For analysis results from the LLM Analysis module
  - `ErrorEvent`: For error reporting from any component
  - `SystemStatusEvent`: For system status updates

### 2. Event Bus (`utils/event_bus.py`)

The event bus provides the message passing infrastructure:

- **EventBus**: Implements the pub-sub pattern for event delivery
- **EventSubscription**: Manages subscriptions to specific event types
- **Asynchronous Delivery**: Events are delivered asynchronously to avoid blocking
- **Thread Safety**: Uses locks to ensure thread-safe operation

### 3. Component Integration

The following components have been updated to use the event schema:

- **AudioPipeline**: 
  - Emits `AudioSegmentEvent` when audio is processed
  - Emits `ErrorEvent` when errors occur

- **STTEngine**: 
  - Subscribes to `AudioSegmentEvent` from AudioPipeline
  - Emits `TranscriptionEvent` with transcription results
  - Emits `ErrorEvent` when errors occur

- **ProcessingCore**:
  - Subscribes to `TranscriptionEvent` from STTEngine
  - Emits `ProcessedTextEvent` with extracted information
  - Emits `ErrorEvent` and `SystemStatusEvent`

- **LLMAnalysis**:
  - Subscribes to both `TranscriptionEvent` and `ProcessedTextEvent`
  - Emits `LLMAnalysisEvent` with medical analysis results
  - Emits `ErrorEvent` when errors occur

## Integration Flow

The event-based architecture enables the following flow:

1. **Audio Processing**:
   ```
   AudioPipeline -> AudioSegmentEvent -> STTEngine
   ```

2. **Transcription**:
   ```
   STTEngine -> TranscriptionEvent -> ProcessingCore/LLMAnalysis
   ```

3. **Text Processing**:
   ```
   ProcessingCore -> ProcessedTextEvent -> LLMAnalysis
   ```

4. **Medical Analysis**:
   ```
   LLMAnalysis -> LLMAnalysisEvent -> (Subscribers)
   ```

5. **Error Handling**:
   ```
   Any Component -> ErrorEvent -> (Error Subscribers)
   ```

## Benefits

1. **Loose Coupling**: Components only need to know about event types, not about other components
2. **Flexibility**: Components can be replaced or modified without affecting others
3. **Testability**: Components can be tested in isolation with mocked events
4. **Extensibility**: New components can be added by subscribing to relevant events
5. **Error Handling**: Standardized error reporting across all components
6. **Debugging**: Event flow can be monitored and logged for debugging

## Next Steps

1. **Event Persistence**: Add persistence for events to support debugging and audit
2. **Event Validation**: Add schema validation to ensure event correctness
3. **Event Monitoring**: Add visualization tools for event flow monitoring
4. **Back-pressure**: Implement back-pressure mechanisms for resource management
5. **Prioritization**: Add event prioritization for critical system events
6. **Extend to UI and External Systems**: Support event communication with UI and external systems

## Verification

The implementation includes a verification script (`verification_script_event_schema.py`) that tests:

1. Event bus functionality
2. Audio event transmission
3. Transcription event handling
4. Processing core event handling
5. LLM analysis event handling
6. Error event handling
7. End-to-end event flow

Run the verification script with:
```bash
./verification_script_event_schema.py
```