#!/usr/bin/env python3
"""
Verification script for event schema implementation.

This script verifies that the event schema implementation works correctly
across all modules (AudioPipeline, STTEngine, ProcessingCore, LLMAnalysis).

Note: This script complements other integration tests:
- verification_script_audio_stt_integration.py (audio pipeline to STT integration)
- verification_script_system_enhanced.py (full system integration)
"""

import os
import sys
import asyncio
import time
import threading
import json
from typing import Dict, List, Any, Optional
import logging
import uuid
import numpy as np
import queue

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import TCCC modules
from src.tccc.utils.event_schema import (
    BaseEvent, EventType, ErrorSeverity, AudioSegmentEvent, 
    TranscriptionEvent, ProcessedTextEvent, LLMAnalysisEvent, 
    ErrorEvent, create_event
)
from src.tccc.utils.event_bus import get_event_bus, EventBus
from src.tccc.audio_pipeline.audio_pipeline import AudioPipeline
from src.tccc.stt_engine.stt_engine import STTEngine
from src.tccc.processing_core.processing_core import ProcessingCore
from src.tccc.llm_analysis.llm_analysis import LLMAnalysis
from src.tccc.utils.config import Config
from src.tccc.utils.logging import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)


class EventLogger:
    """Logs all events received for verification."""
    
    def __init__(self):
        """Initialize the event logger."""
        self.events: Dict[str, List[BaseEvent]] = {
            EventType.AUDIO_SEGMENT.value: [],
            EventType.TRANSCRIPTION.value: [],
            EventType.PROCESSED_TEXT.value: [],
            EventType.LLM_ANALYSIS.value: [],
            EventType.ERROR.value: [],
            EventType.SYSTEM_STATUS.value: [],
            "other": []
        }
        self.event_bus = get_event_bus()
        self.event_lock = threading.RLock()
        self.event_received = threading.Event()
        
    def subscribe(self):
        """Subscribe to all events."""
        return self.event_bus.subscribe(
            subscriber="event_logger",
            event_types=["*"],  # Subscribe to all events
            callback=self.handle_event
        )
        
    def handle_event(self, event: BaseEvent):
        """Handle received events."""
        with self.event_lock:
            event_type = event.type
            if event_type in self.events:
                self.events[event_type].append(event)
            else:
                self.events["other"].append(event)
            
            # Signal that an event was received
            self.event_received.set()
        
        # Log the event
        logger.info(f"Received event: {event.type} from {event.source}")
        
    def wait_for_event(self, event_type: str, timeout: float = 5.0) -> bool:
        """
        Wait for an event of the specified type.
        
        Args:
            event_type: Type of event to wait for
            timeout: Timeout in seconds
            
        Returns:
            True if event was received, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.event_lock:
                if self.events.get(event_type, []):
                    return True
            
            # Wait for any event with timeout
            self.event_received.wait(0.1)
            self.event_received.clear()
            
        return False
        
    def get_events(self, event_type: str) -> List[BaseEvent]:
        """Get all events of the specified type."""
        with self.event_lock:
            return self.events.get(event_type, [])[:]  # Return a copy
            
    def clear_events(self):
        """Clear all events."""
        with self.event_lock:
            for key in self.events:
                self.events[key] = []
            self.event_received.clear()


async def test_event_bus():
    """Test the event bus functionality."""
    logger.info("Testing event bus...")
    
    # Create event bus
    event_bus = get_event_bus()
    
    # Set up a test event queue
    event_queue = queue.Queue()
    
    # Create a subscriber
    def handle_test_event(event: BaseEvent):
        event_queue.put(event)
    
    # Subscribe to test events
    event_bus.subscribe(
        subscriber="test_subscriber",
        event_types=["test_event"],
        callback=handle_test_event
    )
    
    # Create and publish a test event
    test_event = create_event(
        event_type="test_event",
        source="test",
        data={"message": "This is a test"}
    )
    
    success = event_bus.publish(test_event)
    
    # Wait a bit for async delivery
    await asyncio.sleep(0.5)
    
    # Check if event was received
    try:
        received_event = event_queue.get(block=False)
        logger.info(f"Test event received: {received_event.data}")
        return True
    except queue.Empty:
        logger.error("Test event not received")
        return False


async def test_audio_events(event_logger: EventLogger):
    """Test audio segment events."""
    logger.info("Testing audio segment events...")
    
    # Create a minimal config
    config = {
        "audio": {
            "device": -1,  # No real device needed
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024
        }
    }
    
    # Initialize audio pipeline
    audio_pipeline = AudioPipeline()
    success = await audio_pipeline.initialize(config)
    
    if not success:
        logger.error("Failed to initialize audio pipeline")
        return False
        
    # Create a dummy audio segment
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    
    # Manually emit an audio segment event
    audio_pipeline._emit_audio_segment_event(
        audio_data=dummy_audio,
        sample_rate=16000,
        format_type="PCM16",
        channels=1,
        duration_ms=1000,
        is_speech=True
    )
    
    # Check if event was received
    success = event_logger.wait_for_event(EventType.AUDIO_SEGMENT.value, timeout=2.0)
    
    if success:
        audio_events = event_logger.get_events(EventType.AUDIO_SEGMENT.value)
        if audio_events:
            logger.info(f"Audio segment event received: duration_ms={audio_events[0].data.get('duration_ms')}")
            return True
    
    logger.error("Audio segment event not received")
    return False


async def test_transcription_events(event_logger: EventLogger):
    """Test transcription events."""
    logger.info("Testing transcription events...")
    
    # Create a minimal config
    config = {
        "stt_engine": {
            "engine": "mock",
            "model": "mock-model",
            "language": "en"
        }
    }
    
    # Initialize STT engine
    stt_engine = STTEngine()
    success = await stt_engine.initialize(config)
    
    if not success:
        logger.error("Failed to initialize STT engine")
        return False
    
    # Create a dummy audio segment
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    
    # Create an audio segment event
    audio_event = AudioSegmentEvent(
        source="audio_pipeline",
        audio_data=dummy_audio,
        sample_rate=16000,
        format_type="PCM16",
        channels=1,
        duration_ms=1000,
        is_speech=True,
        session_id=str(uuid.uuid4())
    )
    
    # Clear previous events
    event_logger.clear_events()
    
    # Process the audio event
    stt_engine._handle_audio_event(audio_event)
    
    # Check if transcription event was received
    success = event_logger.wait_for_event(EventType.TRANSCRIPTION.value, timeout=2.0)
    
    if success:
        transcription_events = event_logger.get_events(EventType.TRANSCRIPTION.value)
        if transcription_events:
            logger.info(f"Transcription event received: {transcription_events[0].data.get('text')}")
            return True
    
    logger.error("Transcription event not received")
    return False


async def test_processing_core_events(event_logger: EventLogger):
    """Test processing core events."""
    logger.info("Testing processing core events...")
    
    # Create a minimal config
    config = {
        "general": {
            "max_concurrent_tasks": 2
        },
        "event_handling": {
            "enabled": True
        }
    }
    
    # Initialize processing core
    processing_core = ProcessingCore()
    success = await processing_core.initialize(config)
    
    if not success:
        logger.error("Failed to initialize processing core")
        return False
    
    # Create a transcription event
    transcription_event = TranscriptionEvent(
        source="stt_engine",
        text="Patient has shortness of breath and chest pain",
        segments=[{"text": "Patient has shortness of breath and chest pain", "start": 0, "end": 5}],
        language="en",
        confidence=0.9,
        session_id=str(uuid.uuid4())
    )
    
    # Clear previous events
    event_logger.clear_events()
    
    # Process the transcription event
    processing_core._handle_transcription_event(transcription_event)
    
    # Check if processed text event was received (may take a bit longer due to async processing)
    success = event_logger.wait_for_event(EventType.PROCESSED_TEXT.value, timeout=5.0)
    
    if success:
        processed_events = event_logger.get_events(EventType.PROCESSED_TEXT.value)
        if processed_events:
            logger.info(f"Processed text event received: {processed_events[0].data.get('text')}")
            return True
    
    logger.error("Processed text event not received")
    return False


async def test_llm_analysis_events(event_logger: EventLogger):
    """Test LLM analysis events."""
    logger.info("Testing LLM analysis events...")
    
    # Create a minimal config
    config = {
        "model": {
            "primary": {
                "provider": "local",
                "name": "phi-2-mock"
            }
        },
        "event_handling": {
            "enabled": True
        }
    }
    
    # Initialize LLM analysis
    llm_analysis = LLMAnalysis()
    success = llm_analysis.initialize(config)
    
    if not success:
        logger.error("Failed to initialize LLM analysis")
        return False
    
    # Create a processed text event
    processed_text_event = ProcessedTextEvent(
        source="processing_core",
        text="Patient has shortness of breath and chest pain",
        entities=[
            {"text": "shortness of breath", "entity_type": "symptom", "start": 12, "end": 31},
            {"text": "chest pain", "entity_type": "symptom", "start": 36, "end": 46}
        ],
        intent={"intents": [], "primary": "unknown"},
        session_id=str(uuid.uuid4())
    )
    
    # Clear previous events
    event_logger.clear_events()
    
    # Process the processed text event
    llm_analysis._handle_transcription_event(processed_text_event)
    
    # Check if LLM analysis event was received (may take a bit longer)
    success = event_logger.wait_for_event(EventType.LLM_ANALYSIS.value, timeout=10.0)
    
    if success:
        analysis_events = event_logger.get_events(EventType.LLM_ANALYSIS.value)
        if analysis_events:
            logger.info(f"LLM analysis event received: {analysis_events[0].data.get('summary')}")
            return True
    
    logger.error("LLM analysis event not received")
    return False


async def test_error_events(event_logger: EventLogger):
    """Test error events."""
    logger.info("Testing error events...")
    
    # Create components
    audio_pipeline = AudioPipeline()
    stt_engine = STTEngine()
    processing_core = ProcessingCore()
    llm_analysis = LLMAnalysis()
    
    # Initialize with minimal configs
    await audio_pipeline.initialize({"audio": {"device": -1}})
    await stt_engine.initialize({"stt_engine": {"engine": "mock"}})
    await processing_core.initialize({"general": {}})
    llm_analysis.initialize({"model": {"primary": {"provider": "local"}}})
    
    # Clear previous events
    event_logger.clear_events()
    
    # Emit error events from each component
    audio_pipeline._emit_error_event("AUDIO_TEST", "Test audio error", ErrorSeverity.INFO)
    stt_engine._emit_error_event("STT_TEST", "Test STT error", ErrorSeverity.WARNING)
    processing_core._emit_error_event("PROC_TEST", "Test processing error", ErrorSeverity.ERROR)
    llm_analysis._emit_error_event("LLM_TEST", "Test LLM error", ErrorSeverity.CRITICAL)
    
    # Wait for all error events
    success = event_logger.wait_for_event(EventType.ERROR.value, timeout=2.0)
    
    if success:
        error_events = event_logger.get_events(EventType.ERROR.value)
        if len(error_events) >= 4:
            logger.info(f"Error events received from multiple components: {len(error_events)}")
            for event in error_events:
                logger.info(f"  Error from {event.source}: {event.data.get('message')}")
            return True
    
    logger.error("Not all error events were received")
    return False


async def test_end_to_end_flow(event_logger: EventLogger):
    """Test end-to-end event flow through all components."""
    logger.info("Testing end-to-end event flow...")
    
    # Create components with minimal configs
    audio_pipeline = AudioPipeline()
    stt_engine = STTEngine()
    processing_core = ProcessingCore()
    llm_analysis = LLMAnalysis()
    
    # Initialize components
    await audio_pipeline.initialize({"audio": {"device": -1}})
    await stt_engine.initialize({"stt_engine": {"engine": "mock"}})
    await processing_core.initialize({"general": {}})
    llm_analysis.initialize({"model": {"primary": {"provider": "local"}}})
    
    # Create a session ID for correlation
    session_id = str(uuid.uuid4())
    
    # Clear previous events
    event_logger.clear_events()
    
    # Create a dummy audio segment
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    
    # Start the flow with an audio segment event
    audio_pipeline._emit_audio_segment_event(
        audio_data=dummy_audio,
        sample_rate=16000,
        format_type="PCM16",
        channels=1,
        duration_ms=1000,
        is_speech=True,
        session_id=session_id
    )
    
    # Wait for events from all components (give it more time for the whole chain)
    await asyncio.sleep(15.0)
    
    # Check each event type
    audio_events = event_logger.get_events(EventType.AUDIO_SEGMENT.value)
    transcription_events = event_logger.get_events(EventType.TRANSCRIPTION.value)
    processed_events = event_logger.get_events(EventType.PROCESSED_TEXT.value)
    analysis_events = event_logger.get_events(EventType.LLM_ANALYSIS.value)
    
    # Log results
    logger.info(f"End-to-end event flow results:")
    logger.info(f"  Audio events: {len(audio_events)}")
    logger.info(f"  Transcription events: {len(transcription_events)}")
    logger.info(f"  Processed text events: {len(processed_events)}")
    logger.info(f"  LLM analysis events: {len(analysis_events)}")
    
    # Check if we got at least one event of each type
    success = (
        len(audio_events) > 0 and
        len(transcription_events) > 0 and
        len(processed_events) > 0 and
        len(analysis_events) > 0
    )
    
    if success:
        logger.info("End-to-end event flow test successful")
        return True
    else:
        logger.error("End-to-end event flow test failed")
        return False


async def run_tests():
    """Run all event schema tests."""
    logger.info("Starting event schema verification...")
    
    # Create event logger
    event_logger = EventLogger()
    event_logger.subscribe()
    
    # Run tests
    test_results = {
        "event_bus": await test_event_bus(),
        "audio_events": await test_audio_events(event_logger),
        "transcription_events": await test_transcription_events(event_logger),
        "processing_events": await test_processing_core_events(event_logger),
        "llm_analysis_events": await test_llm_analysis_events(event_logger),
        "error_events": await test_error_events(event_logger),
        "end_to_end_flow": await test_end_to_end_flow(event_logger)
    }
    
    # Print results
    logger.info("\nEvent Schema Verification Results:")
    all_passed = True
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    # Final result
    if all_passed:
        logger.info("\nAll event schema tests PASSED")
        return 0
    else:
        logger.error("\nSome event schema tests FAILED")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        sys.exit(1)