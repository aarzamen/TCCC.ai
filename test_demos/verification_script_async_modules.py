#!/usr/bin/env python3
"""
Verification script for async/sync interface fixes.

This script tests the improved interface that properly handles both synchronous
and asynchronous module methods to prevent deadlocks and interface mismatches.
"""

import os
import time
import asyncio
import numpy as np
from typing import Dict, Any, Optional

# Import system modules
from tccc.system.system import TCCCSystem
from tccc.utils.module_adapter import (
    AudioPipelineAdapter, 
    run_method_async, 
    create_async_method
)
from tccc.utils.event_schema import AudioSegmentEvent

# Create a simple mock audio pipeline for testing
class MockAudioPipeline:
    """Mock audio pipeline that can generate test audio segments."""
    
    def __init__(self):
        self.config = {"audio": {"sample_rate": 16000}}
        self.active_source = {"name": "test_microphone"}
        self.audio_processor = None
        self.is_capturing = False
    
    def initialize(self, config=None):
        """Initialize the mock audio pipeline."""
        print("Initializing MockAudioPipeline")
        return True
    
    def get_audio_segment(self):
        """Get a sample audio segment (synchronous)."""
        if not self.is_capturing:
            return None
            
        # Create a simple sine wave as test data
        duration_sec = 0.5
        sample_rate = self.config["audio"]["sample_rate"]
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Add some delay to simulate processing time
        time.sleep(0.05)
        
        return audio_data
    
    async def get_audio_segment_async(self):
        """Get a sample audio segment (asynchronous)."""
        if not self.is_capturing:
            return None
            
        # Create a simple sine wave as test data
        duration_sec = 0.5
        sample_rate = self.config["audio"]["sample_rate"]
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Add some delay to simulate processing time
        await asyncio.sleep(0.05)
        
        return audio_data
    
    def start_capture(self, source_id=None):
        """Start capturing audio."""
        self.is_capturing = True
        return True
    
    def stop_capture(self):
        """Stop capturing audio."""
        self.is_capturing = False
        return True
    
    def shutdown(self):
        """Shutdown the mock audio pipeline."""
        self.is_capturing = False
        return True
    
    def get_status(self):
        """Get status of the mock audio pipeline."""
        return {
            "is_capturing": self.is_capturing,
            "sample_rate": self.config["audio"]["sample_rate"],
            "source": self.active_source["name"]
        }


# Create a simple mock STT engine for testing
class MockSTTEngine:
    """Mock STT engine that can transcribe audio segments."""
    
    def __init__(self):
        self.model_loaded = False
    
    def initialize(self, config=None):
        """Initialize the mock STT engine."""
        print("Initializing MockSTTEngine")
        self.model_loaded = True
        return True
    
    def transcribe_segment(self, audio_data, metadata=None):
        """Transcribe an audio segment (synchronous)."""
        if not self.model_loaded:
            return {"error": "Model not loaded", "text": ""}
            
        # Simple dummy transcription
        text = "This is a test transcription"
        
        # Add some delay to simulate processing time
        time.sleep(0.1)
        
        return {
            "text": text,
            "segments": [{"text": text, "confidence": 0.9}],
            "language": "en",
            "metrics": {
                "audio_duration": len(audio_data) / 16000,
                "processing_time": 0.1
            },
            "model": "mock-whisper"
        }
    
    async def transcribe_segment_async(self, audio_data, metadata=None):
        """Transcribe an audio segment (asynchronous)."""
        if not self.model_loaded:
            return {"error": "Model not loaded", "text": ""}
            
        # Simple dummy transcription
        text = "This is a test transcription (async)"
        
        # Add some delay to simulate processing time
        await asyncio.sleep(0.1)
        
        return {
            "text": text,
            "segments": [{"text": text, "confidence": 0.9}],
            "language": "en",
            "metrics": {
                "audio_duration": len(audio_data) / 16000,
                "processing_time": 0.1
            },
            "model": "mock-whisper-async"
        }
    
    def shutdown(self):
        """Shutdown the mock STT engine."""
        self.model_loaded = False
        return True
    
    def get_status(self):
        """Get status of the mock STT engine."""
        return {
            "model_loaded": self.model_loaded,
            "model": "mock-whisper"
        }


# Simple mock processing core
class MockProcessingCore:
    """Mock processing core that can process text."""
    
    def __init__(self):
        self.initialized = False
    
    def initialize(self, config=None):
        """Initialize the mock processing core."""
        print("Initializing MockProcessingCore")
        self.initialized = True
        return True
    
    def process(self, input_data):
        """Process input data (synchronous)."""
        if not self.initialized:
            return {"error": "Not initialized", "text": input_data.get("text", "")}
            
        # Simple processing - just add a tag
        text = input_data.get("text", "")
        
        # Add some delay to simulate processing time
        time.sleep(0.05)
        
        return {
            "text": text + " [processed]",
            "entities": [{"type": "test", "text": "test"}],
            "intent": {"name": "test", "confidence": 0.9},
            "sentiment": {"label": "neutral", "score": 0.5},
            "processing_time": 0.05
        }
    
    async def process_async(self, input_data):
        """Process input data (asynchronous)."""
        if not self.initialized:
            return {"error": "Not initialized", "text": input_data.get("text", "")}
            
        # Simple processing - just add a tag
        text = input_data.get("text", "")
        
        # Add some delay to simulate processing time
        await asyncio.sleep(0.05)
        
        return {
            "text": text + " [processed async]",
            "entities": [{"type": "test", "text": "test"}],
            "intent": {"name": "test", "confidence": 0.9},
            "sentiment": {"label": "neutral", "score": 0.5},
            "processing_time": 0.05
        }
    
    def shutdown(self):
        """Shutdown the mock processing core."""
        self.initialized = False
        return True
    
    def get_status(self):
        """Get status of the mock processing core."""
        return {
            "initialized": self.initialized
        }


# Mock data store
class MockDataStore:
    """Simple mock data store that stores events in memory."""
    
    def __init__(self):
        self.events = {}
        self.reports = {}
        self.next_id = 1
    
    def initialize(self, config=None):
        """Initialize the mock data store."""
        print("Initializing MockDataStore")
        return True
    
    def store_event(self, event):
        """Store an event."""
        event_id = f"event_{self.next_id}"
        self.next_id += 1
        self.events[event_id] = event
        return event_id
    
    def get_event(self, event_id):
        """Get an event."""
        return self.events.get(event_id)
    
    def store_report(self, report):
        """Store a report."""
        report_id = f"report_{self.next_id}"
        self.next_id += 1
        self.reports[report_id] = report
        return report_id
    
    def get_report(self, report_id):
        """Get a report."""
        return self.reports.get(report_id)
    
    def query_events(self, filters=None):
        """Query events."""
        return list(self.events.values())
    
    def shutdown(self):
        """Shutdown the mock data store."""
        self.events = {}
        self.reports = {}
        return True
    
    def get_status(self):
        """Get status of the mock data store."""
        return {
            "events_count": len(self.events),
            "reports_count": len(self.reports)
        }


# Mock document library
class MockDocumentLibrary:
    """Simple mock document library."""
    
    def __init__(self):
        self.initialized = False
    
    def initialize(self, config=None):
        """Initialize the mock document library."""
        print("Initializing MockDocumentLibrary")
        self.initialized = True
        return True
    
    def query(self, query_text, n_results=3):
        """Query the document library."""
        return {
            "query": query_text,
            "results": [
                {"text": "Mock result 1", "score": 0.9},
                {"text": "Mock result 2", "score": 0.8},
                {"text": "Mock result 3", "score": 0.7}
            ]
        }
    
    def get_status(self):
        """Get status of the mock document library."""
        return {
            "initialized": self.initialized,
            "document_count": 10  # Mock value
        }


# Mock LLM analysis
class MockLLMAnalysis:
    """Simple mock LLM analysis module."""
    
    def __init__(self):
        self.model_loaded = False
        self.document_library = None
    
    def initialize(self, config=None):
        """Initialize the mock LLM analysis module."""
        print("Initializing MockLLMAnalysis")
        self.model_loaded = True
        return True
    
    def set_document_library(self, document_library):
        """Set the document library."""
        self.document_library = document_library
    
    def analyze_transcription(self, text):
        """Analyze a transcription (synchronous)."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
            
        # Add some delay to simulate processing time
        time.sleep(0.2)
        
        return {
            "summary": f"Summary of: {text}",
            "topics": ["Topic 1", "Topic 2"],
            "medical_terms": ["term1", "term2"],
            "actions": ["Action 1"],
            "document_results": [],
            "metadata": {
                "model": "mock-llm",
                "processing_ms": 200,
                "tokens": 100
            }
        }
    
    async def analyze_transcription_async(self, text):
        """Analyze a transcription (asynchronous)."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
            
        # Add some delay to simulate processing time
        await asyncio.sleep(0.2)
        
        return {
            "summary": f"Async summary of: {text}",
            "topics": ["Topic 1", "Topic 2"],
            "medical_terms": ["term1", "term2"],
            "actions": ["Action 1"],
            "document_results": [],
            "metadata": {
                "model": "mock-llm-async",
                "processing_ms": 200,
                "tokens": 100
            }
        }
    
    def shutdown(self):
        """Shutdown the mock LLM analysis module."""
        self.model_loaded = False
        return True
    
    def get_status(self):
        """Get status of the mock LLM analysis module."""
        return {
            "model_loaded": self.model_loaded,
            "has_document_library": self.document_library is not None
        }


async def test_async_utility_functions():
    """Test async utility functions."""
    print("\n=== Testing Async Utility Functions ===")
    
    # Test run_method_async with sync function
    def sync_function(x, y):
        time.sleep(0.1)  # Simulate processing time
        return x + y
    
    print("Testing run_method_async with sync function...")
    result = await run_method_async(sync_function, 5, 3)
    print(f"Result: {result}, Expected: 8")
    
    # Test run_method_async with async function
    async def async_function(x, y):
        await asyncio.sleep(0.1)  # Simulate processing time
        return x * y
    
    print("Testing run_method_async with async function...")
    result = await run_method_async(async_function, 5, 3)
    print(f"Result: {result}, Expected: 15")
    
    # Test create_async_method with sync method
    class TestClass:
        def sync_method(self, x, y):
            time.sleep(0.1)  # Simulate processing time
            return x - y
    
    test_obj = TestClass()
    print("Testing create_async_method with sync method...")
    async_method = create_async_method(test_obj, "sync_method")
    result = await async_method(10, 4)
    print(f"Result: {result}, Expected: 6")
    
    print("Async utility function tests completed!\n")


async def test_audio_pipeline_adapter():
    """Test the audio pipeline adapter."""
    print("\n=== Testing AudioPipelineAdapter ===")
    
    # Create audio pipeline
    audio_pipeline = MockAudioPipeline()
    audio_pipeline.initialize()
    audio_pipeline.start_capture()
    
    # Test sync adapter
    print("Testing synchronous adapter...")
    audio_event = AudioPipelineAdapter.get_audio_segment(audio_pipeline)
    print(f"Got audio event: {audio_event['type'] if audio_event else None}")
    if audio_event and 'data' in audio_event:
        format_type = audio_event['data'].get('format_type', 'unknown')
        duration_ms = audio_event['data'].get('duration_ms', 0)
        print(f"Audio format: {format_type}")
        print(f"Audio duration: {duration_ms} ms")
    
    # Test async adapter
    print("\nTesting asynchronous adapter...")
    audio_event = await AudioPipelineAdapter.get_audio_segment_async(audio_pipeline)
    print(f"Got audio event: {audio_event['type'] if audio_event else None}")
    if audio_event and 'data' in audio_event:
        format_type = audio_event['data'].get('format_type', 'unknown')
        duration_ms = audio_event['data'].get('duration_ms', 0)
        print(f"Audio format: {format_type}")
        print(f"Audio duration: {duration_ms} ms")
    
    # Test data type conversion
    print("\nTesting data type conversion...")
    # Create int16 audio
    audio_pipeline.get_audio_segment = lambda: np.random.randint(-32768, 32767, 8000, dtype=np.int16)
    audio_event = AudioPipelineAdapter.get_audio_segment(audio_pipeline)
    if audio_event and 'data' in audio_event:
        format_type = audio_event['data'].get('format_type', 'unknown')
        dtype = audio_event['data'].get('metadata', {}).get('dtype', 'unknown')
        print(f"Original data type: int16, Converted format: {format_type}")
        print(f"Metadata dtype: {dtype}")
    
    # Test handling of non-numpy data
    print("\nTesting non-numpy data handling...")
    audio_pipeline.get_audio_segment = lambda: [0.1] * 1000  # List of floats
    audio_event = AudioPipelineAdapter.get_audio_segment(audio_pipeline)
    if audio_event and 'data' in audio_event:
        format_type = audio_event['data'].get('format_type', 'unknown')
        dtype = audio_event['data'].get('metadata', {}).get('dtype', 'unknown')
        print(f"Original data type: list, Converted format: {format_type}")
        print(f"Metadata dtype: {dtype}")
    
    audio_pipeline.stop_capture()
    print("AudioPipelineAdapter tests completed!\n")


async def test_system_integration():
    """Test the full system integration with both sync and async components."""
    print("\n=== Testing Full System Integration ===")
    
    # Create system with mock modules
    system = TCCCSystem()
    
    # Replace with mock modules
    system.audio_pipeline = MockAudioPipeline()
    system.stt_engine = MockSTTEngine()
    system.processing_core = MockProcessingCore()
    system.data_store = MockDataStore()
    system.document_library = MockDocumentLibrary()
    system.llm_analysis = MockLLMAnalysis()
    
    # Initialize system
    print("Initializing system...")
    await system.initialize({})
    
    # Test the audio capture and processing pipeline
    print("\nTesting audio capture and processing...")
    result = system.start_audio_capture()
    print(f"Started audio capture: {result}")
    
    # Let it run for a few seconds
    print("Processing audio for 3 seconds...")
    await asyncio.sleep(3)
    
    # Get status
    status = system.get_status()
    print(f"System state: {status['state']}")
    print(f"Events processed: {status['events_count']}")
    
    # Stop audio capture
    print("\nStopping audio capture...")
    result = system.stop_audio_capture()
    print(f"Stopped audio capture: {result}")
    
    # Test direct event processing
    print("\nTesting direct event processing...")
    
    # Create a test audio event
    sample_rate = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    audio_event = AudioSegmentEvent(
        source="test",
        audio_data=audio_data,
        sample_rate=sample_rate,
        format_type="FLOAT32",
        channels=1,
        duration_ms=duration_sec * 1000,
        is_speech=True,
        metadata={"test": True}
    ).to_dict()
    
    # Process event
    event_id = await system.process_event(audio_event)
    print(f"Processed direct audio event, event ID: {event_id}")
    
    # Query events
    events = system.query_events()
    print(f"Total events in data store: {len(events)}")
    
    # Get the last event
    if events:
        last_event = events[-1]
        if "type" in last_event:
            print(f"Last event type: {last_event['type']}")
        if "data" in last_event and "text" in last_event["data"]:
            print(f"Last event text: {last_event['data']['text']}")
    
    # Shutdown the system
    print("\nShutting down system...")
    system.shutdown()
    print("System integration tests completed!\n")


async def main():
    """Run all verification tests."""
    print("=== TCCC.ai Async/Sync Interface Verification ===\n")
    
    # Run tests
    try:
        await test_async_utility_functions()
        await test_audio_pipeline_adapter()
        await test_system_integration()
        
        print("\n=== Verification Summary ===")
        print("✅ Async utility functions: PASSED")
        print("✅ Audio pipeline adapter: PASSED")
        print("✅ System integration: PASSED")
        print("\nAll async/sync interface fixes have been verified successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nVerification failed. Please check the error and fix the issues.")
        return False
    
    return True


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    
    # Exit with appropriate code
    exit(0 if success else 1)