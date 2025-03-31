#!/usr/bin/env python3
"""
TCCC.ai System Event Flow Test

Tests the event flow through the system using the new event schema.
This is a simple script to verify that events can pass through the system
with the updated architecture.
"""

import os
import sys
import time
import asyncio
import logging
import json
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemEventFlowTest")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import system components
from tccc.system.system import TCCCSystem, SystemState
from tccc.utils.event_schema import (
    BaseEvent, TranscriptionEvent, ProcessedTextEvent, 
    EventType
)


async def test_system_initialization():
    """Test system initialization."""
    logger.info("=== Testing System Initialization ===")
    
    # Create system
    system = TCCCSystem()
    
    # Initialize with minimal configuration
    config = {
        "data_store": {"storage_path": "data/test_event_flow"},
        "stt_engine": {"model": {"type": "whisper", "size": "medium"}},
        "audio_pipeline": {
            "audio": {"sample_rate": 16000, "channels": 1, "format": "int16"},
            "io": {"input_sources": [{"name": "test", "type": "file", "path": "test_data/test_speech.wav"}]}
        }
    }
    
    # Initialize system
    result = await system.initialize(config)
    
    # Check result
    if result:
        logger.info("✓ System initialized successfully")
    else:
        logger.error("✗ System initialization failed")
        return None
    
    return system


async def test_event_processing(system: TCCCSystem):
    """Test event processing with different event types."""
    if not system:
        logger.error("System not provided, skipping event processing test")
        return False
    
    logger.info("=== Testing Event Processing ===")
    
    # Test with different event types
    await test_transcription_event(system)
    await test_processed_text_event(system)
    
    return True


async def test_transcription_event(system: TCCCSystem):
    """Test processing a transcription event."""
    logger.info("Testing Transcription Event Processing")
    
    # Create a test transcription event
    event = TranscriptionEvent(
        source="test_script",
        text="Patient has tension pneumothorax and needs immediate decompression",
        segments=[{
            "text": "Patient has tension pneumothorax and needs immediate decompression",
            "start_time": 0.0,
            "end_time": 5.0,
            "confidence": 0.95
        }],
        language="en",
        confidence=0.95,
        metadata={"test": True}
    ).to_dict()
    
    # Process event
    start_time = time.time()
    result = await system.process_event(event)
    elapsed = time.time() - start_time
    
    # Check result
    if result:
        logger.info(f"✓ Transcription event processed successfully (event ID: {result})")
        logger.info(f"  Processing time: {elapsed:.3f} seconds")
    else:
        logger.error("✗ Transcription event processing failed")
    
    # Give some time for async processing to complete
    await asyncio.sleep(0.5)
    
    return result is not None


async def test_processed_text_event(system: TCCCSystem):
    """Test processing a processed text event."""
    logger.info("Testing Processed Text Event Processing")
    
    # Create a test processed text event
    event = ProcessedTextEvent(
        source="test_script",
        text="Patient has tension pneumothorax and needs immediate decompression",
        entities=[{
            "text": "tension pneumothorax",
            "type": "MEDICAL_CONDITION",
            "start": 12,
            "end": 31,
            "confidence": 0.98
        }],
        intent={
            "name": "report_medical_condition",
            "confidence": 0.87,
            "slots": {"condition": "tension pneumothorax"}
        },
        sentiment={"label": "urgent", "score": 0.85},
        metadata={"test": True}
    ).to_dict()
    
    # Process event
    start_time = time.time()
    result = await system.process_event(event)
    elapsed = time.time() - start_time
    
    # Check result
    if result:
        logger.info(f"✓ Processed text event processed successfully (event ID: {result})")
        logger.info(f"  Processing time: {elapsed:.3f} seconds")
    else:
        logger.error("✗ Processed text event processing failed")
    
    # Give some time for async processing to complete
    await asyncio.sleep(0.5)
    
    return result is not None


async def test_system_status(system: TCCCSystem):
    """Test system status."""
    if not system:
        logger.error("System not provided, skipping status test")
        return False
    
    logger.info("=== Testing System Status ===")
    
    # Get system status
    status = system.get_status()
    
    # Check status
    if status and "state" in status:
        logger.info(f"✓ System state: {status['state']}")
        
        # Print module statuses
        if "modules" in status:
            logger.info("Module statuses:")
            for module_name, module_status in status["modules"].items():
                status_str = "✓" if module_status.get("initialized", False) else "✗"
                logger.info(f"  {status_str} {module_name}")
        
        return True
    else:
        logger.error("✗ Failed to get system status")
        return False


async def test_audio_processing(system: TCCCSystem):
    """Test audio processing thread."""
    if not system:
        logger.error("System not provided, skipping audio processing test")
        return False
    
    logger.info("=== Testing Audio Processing ===")
    
    # Start audio capture
    result = system.start_audio_capture("test")
    
    if not result:
        logger.error("✗ Failed to start audio capture")
        return False
    
    logger.info("✓ Started audio capture")
    logger.info("Letting audio processing run for 5 seconds...")
    
    # Let it run for a few seconds
    await asyncio.sleep(5)
    
    # Stop audio capture
    stop_result = system.stop_audio_capture()
    
    if stop_result:
        logger.info("✓ Stopped audio capture")
    else:
        logger.error("✗ Failed to stop audio capture")
    
    # Check if events were processed
    events = system.events
    logger.info(f"Processed {len(events)} events during audio capture")
    
    return len(events) > 0


async def main():
    """Main test function."""
    logger.info("TCCC.ai System Event Flow Test")
    logger.info("==============================")
    
    try:
        # Test system initialization
        system = await test_system_initialization()
        if not system:
            logger.error("Initialization failed, exiting")
            return 1
        
        # Test system status
        status_result = await test_system_status(system)
        
        # Test event processing
        event_result = await test_event_processing(system)
        
        # Test audio processing
        audio_result = await test_audio_processing(system)
        
        # Shutdown
        logger.info("Shutting down system...")
        system.shutdown()
        
        # Print summary
        logger.info("==============================")
        logger.info("Test Summary")
        logger.info("==============================")
        logger.info(f"System Initialization: {'✓ PASS' if system else '✗ FAIL'}")
        logger.info(f"System Status: {'✓ PASS' if status_result else '✗ FAIL'}")
        logger.info(f"Event Processing: {'✓ PASS' if event_result else '✗ FAIL'}")
        logger.info(f"Audio Processing: {'✓ PASS' if audio_result else '✗ FAIL'}")
        
        # Overall result
        overall = all([system, status_result, event_result, audio_result])
        logger.info("==============================")
        logger.info(f"Overall Result: {'✓ PASS' if overall else '✗ FAIL'}")
        
        return 0 if overall else 1
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    # Set up async event loop
    loop = asyncio.get_event_loop()
    try:
        exit_code = loop.run_until_complete(main())
        sys.exit(exit_code)
    finally:
        loop.close()