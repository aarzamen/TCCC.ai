#!/usr/bin/env python3
"""
TCCC System Integration Test

This script validates the integration between all TCCC components
after the implementation of the standardized event schema.
"""

import os
import sys
import time
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure the src directory is in the path
sys.path.insert(0, str(Path(__file__).parent))

# Import system components
from src.tccc.system.system import TCCCSystem
from src.tccc.utils.event_schema import (
    AudioSegmentEvent, 
    TranscriptionEvent,
    ProcessedTextEvent, 
    LLMAnalysisEvent,
    ErrorEvent,
    EventType
)
from src.tccc.utils.logging import get_logger, set_log_level

# Set up logging
logger = get_logger("system_integration_test")
set_log_level(logger, logging.INFO)

class SystemIntegrationTest:
    """Test class for validating system integration."""
    
    def __init__(self, mock_mode: bool = False):
        """Initialize the test environment."""
        self.system = None
        self.mock_mode = mock_mode
        self.events_processed = 0
        self.test_session_id = f"test-{int(time.time())}"
        self.config = self._load_config()
        
        # Set mock mode if requested
        if mock_mode:
            logger.info("Running in MOCK mode")
            os.environ["TCCC_USE_MOCK_STT"] = "1"
            os.environ["TCCC_USE_MOCK_LLM"] = "1"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the system configuration."""
        config = {
            "system": {
                "mock_mode": self.mock_mode,
                "debug": True,
                "log_level": "DEBUG"
            },
            "audio_pipeline": {
                "use_mock": self.mock_mode,
                "sample_rate": 16000,
                "channels": 1
            },
            "stt_engine": {
                "use_mock": self.mock_mode,
                "model": {
                    "type": "mock",
                    "size": "tiny-en"
                } if self.mock_mode else "whisper-tiny"
            },
            "llm_analysis": {
                "use_mock": self.mock_mode,
                "model": {
                    "type": "mock",
                    "name": "mock-med-gpt" 
                } if self.mock_mode else "phi-tiny"
            },
            "document_library": {
                "use_cache": True,
                "cache_size": 100,
                "storage": "memory" if self.mock_mode else "disk"
            },
            "processing_core": {
                "use_mock": self.mock_mode
            },
            "data_store": {
                "use_mock": self.mock_mode,
                "storage": "memory" if self.mock_mode else "disk"
            }
        }
        
        logger.info(f"Loaded test configuration: {json.dumps(config, indent=2)}")
        return config
    
    async def initialize_system(self) -> bool:
        """Initialize the TCCC system."""
        try:
            self.system = TCCCSystem()
            init_result = await self.system.initialize(self.config)
            
            if not init_result:
                logger.error("System initialization failed")
                return False
            
            logger.info("System initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    async def test_audio_to_stt(self) -> bool:
        """Test the integration between AudioPipeline and STT Engine."""
        try:
            logger.info("=== Testing Audio Pipeline → STT Engine integration ===")
            
            # Create a simulated audio event
            audio_data = bytes([0] * 32000)  # 1 second of silence at 16kHz, 2 bytes per sample
            
            # Create the audio segment event
            audio_event = AudioSegmentEvent(
                source="test_audio_pipeline",
                audio_data=audio_data,
                sample_rate=16000,
                format_type="PCM16",
                channels=1,
                duration_ms=1000,
                is_speech=True,
                start_time=time.time(),
                metadata={},
                session_id=self.test_session_id,
                sequence=1
            )
            
            # Process the event
            result = self.system.process_event(audio_event.to_dict())
            self.events_processed += 1
            
            # Verify the result
            if not result:
                logger.error("Failed to process audio event")
                return False
            
            # In mock mode, we expect an immediate transcription result
            if self.mock_mode:
                # Wait a moment for processing
                await asyncio.sleep(0.5)
                
                # Check for a transcription event in the system events
                transcription_events = self._get_events_by_type(EventType.TRANSCRIPTION)
                if not transcription_events:
                    logger.error("No transcription events generated")
                    return False
                
                logger.info(f"Generated transcription: {transcription_events[-1].get('text', '')}")
            
            logger.info("Audio Pipeline → STT Engine integration passed")
            return True
        
        except Exception as e:
            logger.error(f"Error testing audio to STT integration: {e}")
            return False
    
    async def test_stt_to_processing(self) -> bool:
        """Test the integration between STT Engine and Processing Core."""
        try:
            logger.info("=== Testing STT Engine → Processing Core integration ===")
            
            # Create a simulated transcription event
            transcription_event = TranscriptionEvent(
                source="test_stt_engine",
                text="Patient has a tension pneumothorax and needs immediate chest decompression",
                segments=[{"text": "Patient has a tension pneumothorax", "start": 0.0, "end": 2.0}],
                language="en",
                confidence=0.95,
                is_partial=False,
                metadata={},
                session_id=self.test_session_id,
                sequence=2
            )
            
            # Process the event
            result = self.system.process_event(transcription_event.to_dict())
            self.events_processed += 1
            
            # Verify the result
            if not result:
                logger.error("Failed to process transcription event")
                return False
            
            # Check for a processed text event in the system events
            if self.mock_mode:
                # Wait a moment for processing
                await asyncio.sleep(0.5)
                
                processed_events = self._get_events_by_type(EventType.PROCESSED_TEXT)
                if not processed_events:
                    logger.error("No processed text events generated")
                    return False
                
                logger.info(f"Generated processed text event with entities: {processed_events[-1].get('entities', [])}")
            
            logger.info("STT Engine → Processing Core integration passed")
            return True
        
        except Exception as e:
            logger.error(f"Error testing STT to processing integration: {e}")
            return False
    
    async def test_processing_to_llm(self) -> bool:
        """Test the integration between Processing Core and LLM Analysis."""
        try:
            logger.info("=== Testing Processing Core → LLM Analysis integration ===")
            
            # Create a simulated processed text event
            processed_event = ProcessedTextEvent(
                source="test_processing_core",
                text="Patient has a tension pneumothorax and needs immediate chest decompression",
                entities=[{"type": "medical_condition", "value": "tension pneumothorax"}],
                intent={"type": "medical_emergency", "confidence": 0.92},
                metadata={},
                session_id=self.test_session_id,
                sequence=3
            )
            
            # Process the event
            result = self.system.process_event(processed_event.to_dict())
            self.events_processed += 1
            
            # Verify the result
            if not result:
                logger.error("Failed to process text event")
                return False
            
            # Check for an LLM analysis event in the system events
            if self.mock_mode:
                # Wait a moment for processing
                await asyncio.sleep(0.5)
                
                llm_events = self._get_events_by_type(EventType.LLM_ANALYSIS)
                if not llm_events:
                    logger.error("No LLM analysis events generated")
                    return False
                
                logger.info(f"Generated LLM analysis with topics: {llm_events[-1].get('topics', [])}")
            
            logger.info("Processing Core → LLM Analysis integration passed")
            return True
        
        except Exception as e:
            logger.error(f"Error testing processing to LLM integration: {e}")
            return False
    
    async def test_llm_to_document(self) -> bool:
        """Test the integration between LLM Analysis and Document Library."""
        try:
            logger.info("=== Testing LLM Analysis → Document Library integration ===")
            
            # Create a simulated LLM analysis event with queries
            llm_event = LLMAnalysisEvent(
                source="test_llm_analysis",
                summary="Patient presenting with tension pneumothorax",
                topics=["tension pneumothorax", "chest decompression"],
                medical_terms=[{"term": "tension pneumothorax", "definition": "A life-threatening condition"}],
                actions=[{"action": "chest decompression", "priority": "high"}],
                document_results=[{"query": "treatment for tension pneumothorax", "results": []}],
                metadata={},
                session_id=self.test_session_id,
                sequence=4
            )
            
            # Process the event
            result = self.system.process_event(llm_event.to_dict())
            self.events_processed += 1
            
            # Verify the result
            if not result:
                logger.error("Failed to process LLM event")
                return False
            
            # Check for document results in the system events
            if self.mock_mode:
                # Wait a moment for processing
                await asyncio.sleep(0.5)
                
                # The updated LLM event should now have document_results field
                updated_llm_events = self._get_events_by_type(EventType.LLM_ANALYSIS)
                if not updated_llm_events:
                    logger.error("No updated LLM analysis events with document results")
                    return False
                
                latest_event = updated_llm_events[-1]
                if "document_results" not in latest_event:
                    logger.error("No document_results field in LLM analysis event")
                    return False
                
                logger.info(f"Generated document results: {latest_event.get('document_results', [])}")
            
            logger.info("LLM Analysis → Document Library integration passed")
            return True
        
        except Exception as e:
            logger.error(f"Error testing LLM to document integration: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling across components."""
        try:
            logger.info("=== Testing error handling integration ===")
            
            # Create a simulated error event
            error_event = ErrorEvent(
                source="test_component",
                error_code="test_error",
                message="Test error message",
                severity="error",
                component="test",
                recoverable=True,
                metadata={},
                session_id=self.test_session_id,
                sequence=5
            )
            
            # Process the event
            result = self.system.process_event(error_event.to_dict())
            self.events_processed += 1
            
            # Verify the result
            if not result:
                logger.error("Failed to process error event")
                return False
            
            logger.info("Error handling integration passed")
            return True
        
        except Exception as e:
            logger.error(f"Error testing error handling: {e}")
            return False
    
    async def test_end_to_end(self) -> bool:
        """Test the complete end-to-end flow."""
        try:
            logger.info("=== Testing end-to-end integration ===")
            
            # Create a simulated audio event
            audio_data = bytes([0] * 32000)  # 1 second of silence at 16kHz, 2 bytes per sample
            
            # Create the audio segment event
            audio_event = AudioSegmentEvent(
                source="test_audio_pipeline",
                audio_data=audio_data,
                sample_rate=16000,
                format_type="PCM16",
                channels=1,
                duration_ms=1000,
                is_speech=True,
                start_time=time.time(),
                metadata={},
                session_id=self.test_session_id,
                sequence=6
            )
            
            # Process the event and wait for the complete flow
            start_time = time.time()
            timeout = 10  # seconds
            
            # Process the initial event
            result = self.system.process_event(audio_event.to_dict())
            self.events_processed += 1
            
            if not result:
                logger.error("Failed to process audio event in end-to-end test")
                return False
            
            # In mock mode, we expect the full flow to complete quickly
            if self.mock_mode:
                # Wait for all events to be processed
                max_waiting_time = 5  # seconds
                start_time = time.time()
                
                while time.time() - start_time < max_waiting_time:
                    # Check if we have all expected event types
                    transcription_events = self._get_events_by_type(EventType.TRANSCRIPTION)
                    processed_events = self._get_events_by_type(EventType.PROCESSED_TEXT)
                    llm_events = self._get_events_by_type(EventType.LLM_ANALYSIS)
                    
                    # If we have events of all types, we're done
                    if transcription_events and processed_events and llm_events:
                        for event_list, name in [
                            (transcription_events, "Transcription"),
                            (processed_events, "Processed Text"),
                            (llm_events, "LLM Analysis")
                        ]:
                            logger.info(f"{name} Events: {len(event_list)}")
                        
                        # Check if the last LLM event has document results
                        if llm_events and "document_results" in llm_events[-1]:
                            logger.info("Complete event flow verified - all components processed events")
                            return True
                    
                    # Sleep a bit before checking again
                    await asyncio.sleep(0.1)
                
                logger.error("End-to-end test timed out waiting for all event types")
                return False
            
            logger.info("End-to-end integration test completed")
            return True
        
        except Exception as e:
            logger.error(f"Error testing end-to-end integration: {e}")
            return False
    
    def _get_events_by_type(self, event_type: EventType) -> List[Dict[str, Any]]:
        """Get events from the system by their type."""
        if not hasattr(self.system, "events") or not self.system.events:
            return []
        
        return [e for e in self.system.events if e.get("type") == event_type.value]
    
    async def run_all_tests(self) -> bool:
        """Run all integration tests."""
        try:
            # Initialize the system
            if not await self.initialize_system():
                return False
            
            # Run the tests
            tests = [
                self.test_audio_to_stt,
                self.test_stt_to_processing,
                self.test_processing_to_llm,
                self.test_llm_to_document,
                self.test_error_handling,
                self.test_end_to_end
            ]
            
            results = []
            for test in tests:
                result = await test()
                results.append(result)
            
            # Report results
            logger.info("=== Integration Test Results ===")
            for i, test in enumerate(tests):
                logger.info(f"{test.__name__}: {'PASSED' if results[i] else 'FAILED'}")
            
            all_passed = all(results)
            logger.info(f"Integration Tests: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
            logger.info(f"Total events processed: {self.events_processed}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Error running integration tests: {e}")
            return False


async def main():
    """Run the system integration test."""
    parser = argparse.ArgumentParser(description="TCCC System Integration Test")
    parser.add_argument("--mock", action="store_true", help="Run with mock components")
    args = parser.parse_args()
    
    # Create and run the test
    test = SystemIntegrationTest(mock_mode=args.mock)
    result = await test.run_all_tests()
    
    # Return the result as an exit code
    return 0 if result else 1


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)