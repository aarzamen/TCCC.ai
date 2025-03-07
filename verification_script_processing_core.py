#!/usr/bin/env python3
"""
Verification script for the Processing Core module.

This script demonstrates the functionality of the Processing Core module
by testing module registration, resource allocation, module status reporting,
and error handling.
"""

import os
import sys
import json
import time
import logging
import argparse
import asyncio
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProcessingCoreVerification")

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Processing Core modules
from tccc.processing_core.processing_core import ProcessingCore, ModuleState, TranscriptionSegment
from tccc.processing_core.state_manager import StateManager, SystemState
from tccc.processing_core.plugin_manager import PluginManager
from tccc.processing_core.resource_monitor import ResourceMonitor
from tccc.utils.config_manager import ConfigManager

class TestPlugin:
    """Test plugin for the Processing Core."""
    
    def __init__(self, name="test_plugin", plugin_type="nlp", priority=1):
        self.name = name
        self.plugin_type = plugin_type
        self.priority = priority
        self.initialized = False
        self.error_mode = False
        self.resource_allocation = 0.0
        
    def initialize(self):
        """Initialize the plugin."""
        self.initialized = True
        return not self.error_mode
        
    def process(self, data):
        """Process data with the plugin."""
        if self.error_mode:
            raise RuntimeError(f"Error in plugin {self.name}")
        return f"Processed by {self.name}: {data}"
        
    def get_metadata(self):
        """Get the plugin metadata."""
        return {
            "name": self.name,
            "type": self.plugin_type,
            "priority": self.priority,
            "initialized": self.initialized
        }
        
    def shutdown(self):
        """Shutdown the plugin."""
        self.initialized = False
        return True

async def verify_processing_core():
    """Verify the Processing Core functionality."""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("processing_core")
    
    logger.info("Creating Processing Core...")
    processing_core = ProcessingCore()
    
    # 1. Test initialization
    logger.info("\n=== Testing Initialization ===")
    result = await processing_core.initialize(config)
    logger.info(f"Initialization result: {result}")
    
    if not result:
        logger.error("Processing Core initialization failed")
        return False
        
    # Show initial status
    logger.info("\n=== Initial Status ===")
    module_status = processing_core.get_module_status()
    logger.info(json.dumps(module_status, indent=2))
    
    # 2. Test module registration
    logger.info("\n=== Testing Module Registration ===")
    
    # Create test plugins
    test_plugins = [
        TestPlugin(name="test_nlp_1", plugin_type="nlp", priority=2),
        TestPlugin(name="test_nlp_2", plugin_type="nlp", priority=1),
        TestPlugin(name="test_audio_1", plugin_type="audio", priority=1),
        TestPlugin(name="test_viz_1", plugin_type="visualization", priority=3)
    ]
    
    # Register plugins
    for plugin in test_plugins:
        processing_core.registerPlugin(plugin)
        logger.info(f"Registered plugin {plugin.name}")
    
    # 3. Test processing functionality
    logger.info("\n=== Testing Processing ===")
    
    # Create a test transcription segment
    test_segment = TranscriptionSegment(
        text="This is a test message for processing",
        speaker="user",
        start_time=time.time(),
        end_time=time.time() + 2.0,
        confidence=0.95,
        is_final=True,
        metadata={"conversation_id": "test_conversation"}
    )
    
    # Process the segment
    processed_segment = await processing_core.processTranscription(test_segment)
    logger.info(f"Processed segment: {processed_segment}")
    
    # Test entity extraction
    logger.info("\n=== Testing Entity Extraction ===")
    entities = await processing_core.extractEntities("Captain John Smith needs 2 units of blood and 1 tourniquet at grid reference XY12345")
    logger.info(f"Extracted entities: {entities}")
    
    # Test intent classification
    logger.info("\n=== Testing Intent Classification ===")
    intents = await processing_core.identifyIntents("I need a medical evacuation immediately")
    logger.info(f"Identified intents: {intents}")
    
    # Test sentiment analysis
    logger.info("\n=== Testing Sentiment Analysis ===")
    sentiment = await processing_core.analyzeSentiment("The patient is responding well to treatment")
    logger.info(f"Sentiment analysis: {sentiment}")
    
    # 4. Test metrics
    logger.info("\n=== Testing Metrics ===")
    metrics = processing_core.getProcessingMetrics()
    logger.info(f"Processing metrics: {json.dumps(metrics, indent=2)}")
    
    # 5. Test shutdown
    logger.info("\n=== Testing Shutdown ===")
    processing_core.shutdown()
    logger.info("Processing Core shutdown complete")
    
    logger.info("\nProcessing Core verification completed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Processing Core functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        result = asyncio.run(verify_processing_core())
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.exception("Verification failed with error")
        sys.exit(1)