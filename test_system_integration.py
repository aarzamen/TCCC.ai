#!/usr/bin/env python3
"""
Test script for TCCC.ai System Integration.

This script tests basic system integration functionality to ensure
all components are working together correctly.
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import system components
from tccc.system.system import TCCCSystem, SystemState

async def run_integration_test():
    """Run a basic system integration test."""
    print("Starting system integration test...")
    
    # Create system instance with empty config
    system = TCCCSystem()
    
    # Initialize with empty config and all mock modules
    mock_modules = [
        "processing_core", 
        "data_store", 
        "document_library",
        "audio_pipeline", 
        "stt_engine", 
        "llm_analysis"
    ]
    await system.initialize({}, mock_modules=mock_modules)
    
    if not system.initialized:
        print("ERROR: System initialization failed")
        return False
    
    print("System initialized successfully. Testing event processing...")
    
    # Create a test event
    test_event = {
        "type": "test_event",
        "text": "This is a test event for system integration",
        "timestamp": 1709682904.5,
        "metadata": {"test": True}
    }
    
    # Process event
    event_id = system.process_event(test_event)
    
    if not event_id:
        print("ERROR: Event processing failed")
        return False
    
    print(f"Event processed successfully, ID: {event_id}")
    
    # Check if we can retrieve the event
    event = system.query_events()[0]
    if not event:
        print("ERROR: Unable to retrieve event")
        return False
    
    print("System status:")
    print(json.dumps(system.get_status(), indent=2, default=str))
    
    # Test shutdown
    shutdown_success = system.shutdown()
    if not shutdown_success:
        print("ERROR: System shutdown failed")
        return False
    
    print("System integration test passed!")
    return True

async def main():
    """Main entry point."""
    success = await run_integration_test()
    return 0 if success else 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)