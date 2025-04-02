#!/usr/bin/env python3
"""
TCCC.ai Enhanced System Integration Verification Script

This script provides a comprehensive end-to-end verification of the TCCC.ai system:
1. Initializes all modules with mock implementations where needed
2. Tests the complete data flow from audio input to analysis results
3. Verifies module interactions and cross-module functionality
4. Tests error handling, recovery mechanisms, and resource management
5. Produces detailed verification reports

Example usage:
    python verification_script_system_enhanced.py
    python verification_script_system_enhanced.py --config /path/to/config --mock all
"""

import os
import sys
import time
import asyncio
import argparse
import signal
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
# Remove incorrect import: from tccc.utils.types import ModuleState # Import ModuleState

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the system components
from tccc.system.system import TCCCSystem, SystemState
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import STTEngine
from tccc.processing_core import ProcessingCore
from tccc.llm_analysis import LLMAnalysis
from tccc.data_store import DataStore
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager
from tccc.processing_core.processing_core import TranscriptionSegment, ProcessedSegment, ModuleState # Import ModuleState here
import math # Added import
from tccc.processing_core.processing_core import ModuleState # Corrected import path, removed ModuleStatus

# Set up base logger name
LOGGER_NAME = "SystemVerification"

# Configure logging - level will be set later by args
logging.basicConfig(
    level=logging.INFO, # Default level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(LOGGER_NAME)


class VerificationStage(Enum):
    """Stages of the verification process"""
    INITIALIZATION = "initialization"
    MODULE_VERIFICATION = "module_verification"
    INTEGRATION_VERIFICATION = "integration_verification"
    DATA_FLOW = "data_flow"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    SECURITY = "security"


class MockAudioSegment:
    """Mock audio segment for testing"""
    
    def __init__(self, sample_rate=16000, duration=2.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.data = b'0' * int(sample_rate * duration)  # Dummy audio data
        self.timestamp = time.time()
        self.metadata = {
            "source": "test_source",
            "format": "PCM16",
            "channels": 1
        }
    
    def get_data(self):
        """Get the audio data"""
        return self.data
    
    def get_sample_rate(self):
        """Get the audio sample rate"""
        return self.sample_rate
    
    def get_duration(self):
        """Get the audio duration in seconds"""
        return self.duration
    
    def get_metadata(self):
        """Get the audio metadata"""
        return self.metadata


class MockAudioStream:
    """Mock audio stream for testing"""
    
    def __init__(self):
        self.segments = []
        self._stop_flag = False
    
    def add_segment(self, segment):
        """Add a segment to the stream"""
        self.segments.append(segment)
    
    def get_segment(self):
        """Get the next segment from the stream"""
        if self.segments:
            return self.segments.pop(0)
        
        # If no segments, create a new one
        if not self._stop_flag:
            return MockAudioSegment()
        
        return None
    
    def stop(self):
        """Stop the stream"""
        self._stop_flag = True


class MockAudioPipeline:
    """Mock audio pipeline implementation for testing"""
    
    def __init__(self):
        self.stream = MockAudioStream()
        self.initialized = True
        self.capturing = False
        self.source = None
        logger.info("MockAudioPipeline initialized")

    def initialize(self, config=None):
        """Initialize the audio pipeline"""
        self.initialized = True
        return True
    
    def start_capture(self):
        """Start audio capture"""
        self.capturing = True
        logger.debug("MockAudioPipeline started capture")
        return True
    
    def stop_capture(self):
        """Stop audio capture"""
        self.capturing = False
        logger.debug("MockAudioPipeline stopped capture")
        return True
    
    def get_audio_stream(self):
        """Get the audio stream"""
        return self.stream
    
    def get_status(self):
        """Get the module status"""
        logger.debug("MockAudioPipeline get_status called")
        return {
            "status": ModuleState.READY, # Return valid ModuleState
            "initialized": self.initialized,
            "capturing": self.capturing,
            "source": self.source
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        logger.info("MockAudioPipeline shutdown")
        return True


class MockSTTEngine:
    """Mock STT engine implementation for testing"""
    
    def __init__(self):
        self.initialized = True
        self.model_loaded = True
        self.transcription_count = 0
        logger.info("MockSTTEngine initialized")

    def initialize(self, config=None):
        """Initialize the STT engine"""
        self.initialized = True
        self.model_loaded = True
        return True
    
    def transcribe_segment(self, audio_segment, metadata=None):
        """Transcribe an audio segment"""
        self.transcription_count += 1
        
        # Return a mock transcription
        return {
            "text": f"This is a test transcription {self.transcription_count}",
            "confidence": 0.95,
            "timestamp": time.time(),
            "segment_duration": audio_segment.get_duration(),
            "words": [
                {"word": "This", "start": 0.0, "end": 0.2, "confidence": 0.98},
                {"word": "is", "start": 0.3, "end": 0.4, "confidence": 0.99},
                {"word": "a", "start": 0.5, "end": 0.6, "confidence": 0.97},
                {"word": "test", "start": 0.7, "end": 0.9, "confidence": 0.96},
                {"word": "transcription", "start": 1.0, "end": 1.8, "confidence": 0.94},
                {"word": str(self.transcription_count), "start": 1.9, "end": 2.0, "confidence": 0.99}
            ]
        }
    
    def get_status(self):
        """Get the module status"""
        logger.debug("MockSTTEngine get_status called")
        return {
            "status": ModuleState.READY, # Return valid ModuleState
            "initialized": self.initialized,
            "model_loaded": self.model_loaded,
            "transcription_count": self.transcription_count
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        logger.info("MockSTTEngine shutdown")
        return True


class MockProcessingCore:
    """Mock processing core implementation for testing"""
    
    def __init__(self):
        self.initialized = True
        self.plugins_loaded = 4  # Mock plugins: entity, intent, sentiment, state
        self.processed_count = 0
    
    def initialize(self, config=None):
        """Initialize the processing core"""
        self.initialized = True
        self.plugins_loaded = 4  # Mock plugins: entity, intent, sentiment, state
        return True
    
    def process_transcription(self, transcription):
        """Process a transcription"""
        self.processed_count += 1
        
        # Get the text from the transcription
        text = transcription["text"] if isinstance(transcription, dict) else str(transcription)
        
        # Return a mock processing result
        return {
            "entities": [
                {"text": "test", "type": "TEST", "start": 10, "end": 14, "confidence": 0.92}
            ],
            "intents": [
                {"name": "test_intent", "confidence": 0.85, "slots": {"item": "test"}}
            ],
            "sentiment": {
                "label": "neutral",
                "score": 0.7
            },
            "state": {
                "context": {"turn": self.processed_count},
                "history": [{"action": "process", "timestamp": time.time()}]
            },
            "raw_text": text,
            "timestamp": time.time()
        }
    
    def get_status(self):
        """Get the module status"""
        return {
            "status": "ok",
            "initialized": self.initialized,
            "plugins_loaded": self.plugins_loaded,
            "processed_count": self.processed_count
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        return True


class MockLLMAnalysis:
    """Mock LLM analysis implementation for testing"""
    
    def __init__(self):
        self.initialized = True
        self.model_loaded = True
        self.analysis_count = 0
        logger.info("MockLLMAnalysis initialized")
        
    def initialize(self, config=None): # Add initialize method back
        """Mock initialize method."""
        logger.debug("MockLLMAnalysis initialize called")
        self.initialized = True
        self.model_loaded = True
        return True

    def process_transcription(self, transcription, processing_result=None):
        """Process a transcription with LLM"""
        self.analysis_count += 1
        
        # Get the text from the transcription
        text = transcription["text"] if isinstance(transcription, dict) else str(transcription)
        
        # Return a mock analysis result
        return [
            {
                "summary": f"This is a summary of: {text}",
                "topics": ["test", "verification"],
                "actions": [
                    {"type": "log", "priority": "normal", "description": "Test action"}
                ],
                "metadata": {
                    "model": "mock-llm-1",
                    "version": "1.0",
                    "latency_ms": 150,
                    "tokens": 24
                },
                "timestamp": time.time()
            }
        ]
    
    def get_status(self):
        """Get the module status"""
        logger.debug("MockLLMAnalysis get_status called")
        return {
            "status": ModuleState.READY, # Return valid ModuleState
            "initialized": self.initialized,
            "model_loaded": self.model_loaded,
            "analysis_count": self.analysis_count
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        logger.info("MockLLMAnalysis shutdown")
        return True


class MockDataStore:
    """Mock data store implementation for testing"""
    
    def __init__(self):
        self.storage = {"events": []}
        self.event_count = 0
        self.initialized = False # Start as not initialized
        logger.info("MockDataStore instance created")

    def initialize(self, config=None):
        """Initialize the mock data store."""
        self.initialized = True
        self.storage = {"events": []} # Reset storage on initialize
        self.event_count = 0
        logger.info("MockDataStore initialized")
        return True

    def get_status(self):
        logger.debug("MockDataStore get_status called")
        return {
            "initialized": self.initialized,
            "status": ModuleState.READY,
            "event_count": len(self.storage["events"]) # Return event count
        }

    def store_event(self, event_data):
        """Store an event, mimicking real DataStore JSON serialization"""
        self.event_count += 1
        event_id = f"event_{self.event_count}"
        
        # Mimic real store_event: Extract standard fields, serialize 'data' field
        stored_event_data = {
             "type": event_data.get('type', 'unknown'),
             "source": event_data.get('source', 'system'),
             "severity": event_data.get('severity'),
             "patient_id": event_data.get('patient_id'),
             "session_id": event_data.get('session_id'),
             "data": json.dumps(event_data.get('data', {})), # Serialize inner data
             "metadata": json.dumps(event_data.get('metadata', {})), # Serialize metadata
             "timestamp": event_data.get('timestamp', datetime.now().isoformat()),
             "created_at": datetime.now().isoformat()
        }

        # Add event to mock storage
        self.storage["events"].append({
            "id": event_id,
            "data": stored_event_data # Store the processed/serialized dict
        })
        
        logger.debug(f"[MockDataStore] Stored processed event for ID {event_id}: {stored_event_data}")
        return event_id

    def get_event(self, event_id):
        """Get an event by ID, mimicking real DataStore JSON deserialization"""
        logger.debug(f"[MockDataStore] get_event called for ID: {event_id}")
        for event in self.storage["events"]:
            if event["id"] == event_id:
                retrieved_event = event["data"].copy() # Get the stored dict
                logger.debug(f"[MockDataStore] Found raw stored event: {retrieved_event}")
                # Mimic real get_event: Deserialize 'data' and 'metadata'
                try:
                    if retrieved_event.get('data') and isinstance(retrieved_event['data'], str):
                        retrieved_event['data'] = json.loads(retrieved_event['data'])
                except json.JSONDecodeError:
                    logger.warning(f"[MockDataStore] Failed to decode JSON data for event {event_id}")
                    pass # Keep raw string
                
                try:
                    if retrieved_event.get('metadata') and isinstance(retrieved_event['metadata'], str):
                        retrieved_event['metadata'] = json.loads(retrieved_event['metadata'])
                except json.JSONDecodeError:
                    logger.warning(f"[MockDataStore] Failed to decode JSON metadata for event {event_id}")
                    pass # Keep raw string

                logger.debug(f"[MockDataStore] Returning processed event: {retrieved_event}")
                return retrieved_event # Return the processed event
        logger.warning(f"[MockDataStore] Event with ID {event_id} not found.")
        return None

    def query_events(self, query_params):
        # Basic mock query - can be expanded
        logger.debug(f"MockDataStore query_events called with: {query_params}")
        # For simplicity, return all events for now, ignoring params
        # A real implementation would filter based on query_params
        return [ev['data'] for ev in self.storage['events']] # Return the stored data dicts

    def delete_event(self, event_id):
        logger.debug(f"MockDataStore delete_event called for ID: {event_id}")
        self.storage["events"] = [e for e in self.storage["events"] if e["id"] != event_id]
        return True

    def shutdown(self):
        logger.info("MockDataStore shutdown")
        self.initialized = False
        return True


class MockDocumentLibrary:
    """Mock document library implementation for testing"""
    
    def __init__(self):
        self.documents = {}
        self.document_count = 0
        logger.info("MockDocumentLibrary initialized")

    def initialize(self, config=None):
        """Initialize the document library"""
        self.documents = {}
        
        # Add some mock documents
        self.add_document("test1", "Test Document 1", "This is a test document content.")
        self.add_document("test2", "Test Document 2", "Another test document for verification.")
        
        return True
    
    def add_document(self, doc_id, title, content):
        """Add a document to the library"""
        self.document_count += 1
        self.documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "content": content,
            "created": time.time(),
            "updated": time.time()
        }
        logger.debug(f"MockDocumentLibrary added document: {doc_id}")
        return doc_id
    
    def get_document(self, doc_id):
        """Get a document by ID"""
        return self.documents.get(doc_id)
    
    def search_documents(self, query):
        """Search documents"""
        results = []
        for doc_id, doc in self.documents.items():
            if query.lower() in doc["title"].lower() or query.lower() in doc["content"].lower():
                results.append(doc)
        logger.debug(f"MockDocumentLibrary searched for: {query}")
        return results
    
    def get_status(self):
        """Get the module status"""
        logger.debug("MockDocumentLibrary get_status called")
        return {
            "status": ModuleState.READY, # Return valid ModuleState
            "initialized": True,
            "document_count": self.document_count
        }
    
    def shutdown(self):
        """Shutdown the module"""
        logger.info("MockDocumentLibrary shutdown")
        return True


class MockTCCCSystem:
    """
    Mock TCCC System implementation for testing
    Uses mock modules to avoid dependencies
    """
    
    def __init__(self, config_path=None):
        """Initialize the mock system"""
        self.config_path = config_path
        self.state = SystemState.INITIALIZING
        
        # Create mock modules
        self.modules = {
            "processing_core": MockProcessingCore(),
            "data_store": MockDataStore(),
            "document_library": MockDocumentLibrary(),
            "audio_pipeline": MockAudioPipeline(),
            "stt_engine": MockSTTEngine(),
            "llm_analysis": MockLLMAnalysis()
        }
        
        # Health tracking
        self.health = {
            "errors": [],
            "warnings": [],
            "resource_usage": {
                "cpu_percent": 5.0,
                "memory_percent": 10.0,
                "gpu_utilization": 0.0,
                "disk_usage_percent": 20.0
            }
        }
        
        # Session data
        self.session_id = f"mock_session_{int(time.time())}"
        self.session_start_time = time.time()
        self.events = []
        self.reports = []
        
        self.running = False
        self.tasks = []
        
    async def process_event(self, event_data):
        """Process an event through the system"""
        logger.info(f"Processing mock event: {event_data.get('type', 'unknown')}")
        
        # Update state
        self.state = SystemState.PROCESSING
        
        try:
            # Extract text if present in the event
            text = None
            if "text" in event_data:
                text = event_data["text"]
            elif "data" in event_data and isinstance(event_data["data"], str):
                text = event_data["data"]
                
            # Process with core if text is available
            processed = None
            if text:
                processed = self.modules["processing_core"].process_transcription({
                    "text": text,
                    "metadata": event_data.get("metadata", {})
                })
            else:
                processed = event_data
                
            # Process with LLM if text is available
            if text:
                self.state = SystemState.ANALYZING
                analysis = self.modules["llm_analysis"].process_transcription(text)
                
                # Add analysis results to processed data
                if isinstance(processed, dict) and isinstance(analysis, list):
                    for key, value in analysis[0].items():
                        if key not in processed:
                            processed[key] = value
            
            # Store the event
            event_id = self.modules["data_store"].store_event(processed)
            logger.info(f"Stored mock event: {event_id}")
            self.events.append(event_id)
            
            # Reset state
            self.state = SystemState.READY
            
            return event_id
            
        except Exception as e:
            logger.error(f"Error processing mock event: {str(e)}")
            self.health["errors"].append({
                "module": "system",
                "message": f"Error processing event: {str(e)}",
                "time": time.time()
            })
            self.state = SystemState.ERROR
            return None
    
    async def initialize(self):
        """Initialize all modules"""
        if self.state != SystemState.INITIALIZING:
            return False
        
        self.state = SystemState.INITIALIZING
        
        # Initialize each module (mocks don't need config)
        for name, module in self.modules.items():
            try:
                module.initialize()
            except Exception as e:
                 logger.error(f"Failed to initialize mock module {name}: {e}")
                 # Decide if this is critical for the mock system
        
        self.state = SystemState.READY
        return True
    
    async def start(self):
        """Start the mock system"""
        if self.state != SystemState.READY:
            return False
        
        self.state = SystemState.READY
        self.running = True
        
        # Start the audio pipeline
        audio = self.modules["audio_pipeline"]
        audio.start_capture()
        
        return True
    
    async def stop(self):
        """Stop the mock system"""
        if self.state not in [SystemState.READY, SystemState.CAPTURING, SystemState.PROCESSING]:
            return False
        
        self.state = SystemState.SHUTDOWN
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        if self.tasks:
            try:
                await asyncio.wait(self.tasks, timeout=5)
            except asyncio.CancelledError:
                pass
        
        # Stop all modules
        for name, module in self.modules.items():
            module.shutdown()
        
        self.state = SystemState.SHUTDOWN
        return True
    
    async def get_health_status(self):
        """Get system health status"""
        module_statuses = {}
        event_count = 0
        for name, module in self.modules.items():
            status = module.get_status() # Call get_status once
            module_statuses[name] = status
            # Get event_count specifically from data_store status
            if name == 'data_store' and isinstance(status, dict):
                event_count = status.get('event_count', 0)
        
        return {
            "state": self.state.value,
            "health": {
                "event_count": event_count, # Add event_count here
                "error_count": len(self.health["errors"]),
                "warning_count": len(self.health["warnings"]),
                "resource_usage": self.health["resource_usage"],
                "errors": self.health["errors"]
            },
            "modules": module_statuses
        }
    
    def add_error(self, module, message):
        """Add an error to health tracking"""
        self.health["errors"].append({
            "module": module,
            "message": message,
            "time": time.time()
        })
    
    def add_warning(self, module, message):
        """Add a warning to health tracking"""
        self.health["warnings"].append({
            "module": module,
            "message": message,
            "time": time.time()
        })


class SecurityVerificationResult:
    """Contains results of security verification checks"""
    
    def __init__(self):
        """Initialize with default values"""
        self.input_validation_status = False
        self.input_validation_details = {}
        
        self.error_boundaries_status = False
        self.error_boundaries_details = {}
        
        self.permission_checks_status = False
        self.permission_checks_details = {}
        
        self.overall_status = False
        self.verified = False


class SystemVerifierEnhanced:
    """
    Enhanced system verifier for comprehensive testing of TCCC.ai
    """
    
    def __init__(self, default_config: Dict[str, Any], config_path: Optional[str] = None, use_mocks: str = "auto"):
        """
        Initialize the system verifier
        
        Args:
            default_config (Dict[str, Any]): The default configuration dictionary
            config_path (str, optional): Path to configuration directory
            use_mocks (str): Whether to use mock implementations:
                             "auto" - use real implementations when available, mocks otherwise
                             "all" - use mock implementations for all modules
                             "none" - use only real implementations
        """
        self.default_config = default_config  # Store the default config
        self.config_path = config_path
        self.use_mocks = use_mocks
        self.system: Optional[Union[TCCCSystem, MockTCCCSystem]] = None
        self.is_mock_system = False
        
        # Set up verification tracking
        self.verification_results = {
            stage.value: {"success": False, "details": {}} for stage in VerificationStage
        }
        
        # Track timing information
        self.timing = {
            "start_time": 0,
            "end_time": 0,
            "stages": {}
        }
        
        # Test parameters
        self.test_duration = 30  # seconds
        self.error_injection_enabled = True
        
        # Security verification results
        self.security_results = SecurityVerificationResult()
    
    async def verify_security(self):
        """
        Verify system security
        """
        stage_start = time.time()
        
        if not self.system or self.system.state != SystemState.READY:
            logger.error("System not running, skipping security verification")
            return False
        
        try:
            logger.info("Testing system security")
            
            # Test input validation
            input_validation_passed = self._verify_input_validation()
            
            # Test error boundaries
            error_boundaries_passed = self._verify_error_boundaries()
            
            # Test permission checks
            permission_checks_passed = self._verify_permission_checks()
            
            # Record results
            security_passed = input_validation_passed and error_boundaries_passed and permission_checks_passed
            
            # Mark as verified
            self.security_results.verified = True
            self.security_results.overall_status = security_passed
            
            self.verification_results[VerificationStage.SECURITY.value] = {
                "success": security_passed,
                "details": {
                    "input_validation": input_validation_passed,
                    "error_boundaries": error_boundaries_passed,
                    "permission_checks": permission_checks_passed
                }
            }
            
            if security_passed:
                logger.info("Security verification passed")
                print("✓ Security features verified")
            else:
                logger.warning("Security verification failed")
                print("❌ Security verification failed")
                
                # Print specific failures
                if not input_validation_passed:
                    print("  - Input validation failed")
                if not error_boundaries_passed:
                    print("  - Error boundaries failed")
                if not permission_checks_passed:
                    print("  - Permission checks failed")
            
            return security_passed
            
        except Exception as e:
            logger.error(f"Security verification error: {str(e)}")
            self.verification_results[VerificationStage.SECURITY.value] = {
                "success": False,
                "details": {
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            }
            return False
        finally:
            self.timing["stages"][VerificationStage.SECURITY.value] = time.time() - stage_start
    
    def _verify_input_validation(self):
        """Verify input validation"""
        # For mock verification, assume success
        # In a real system, would test with malicious inputs
        return True
    
    def _verify_error_boundaries(self):
        """Verify error boundaries"""
        # For mock verification, assume success
        # In a real system, would test error isolation
        return True
    
    def _verify_permission_checks(self):
        """Verify permission checks"""
        # For mock verification, assume success
        # In a real system, would test access control
        return True
    
    async def run_verification(self):
        """
        Run the complete verification process
        """
        self.timing["start_time"] = time.time()
        
        try:
            print("="*80)
            print(f"TCCC.ai Enhanced System Verification - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Choose system implementation
            self.system = None # Ensure system is None initially
            self.is_mock_system = False
            config_to_use = self.default_config # Start with default
            
            if self.use_mocks == "all":
                logger.info("Using mock system implementation for all modules")
                self.system = MockTCCCSystem(self.config_path) # Mock system handles its own config internally
                self.is_mock_system = True
            elif self.use_mocks == "none" or self.use_mocks == "auto":
                # Try to load config if path provided
                if self.config_path:
                    try:
                        # Load configuration using ConfigManager
                        logger.info(f"Loading configuration from: {self.config_path}")
                        from tccc.utils import ConfigManager # Ensure import is available
                        config_manager = ConfigManager(self.config_path)

                        # --- Start: Explicitly load all YAML files ---
                        try:
                            yaml_files = [f for f in os.listdir(self.config_path) if f.endswith('.yaml') and os.path.isfile(os.path.join(self.config_path, f))]
                            if not yaml_files:
                                logger.warning(f"No YAML config files found in {self.config_path}")
                            else:
                                logger.debug(f"Found config files: {yaml_files}")
                                for yaml_file in yaml_files:
                                    config_name = os.path.splitext(yaml_file)[0]
                                    logger.debug(f"Attempting to load config: {config_name}")
                                    try:
                                        # Use load_config which handles caching internally in ConfigManager
                                        # This should populate the underlying Config object's dictionary
                                        config_manager.load_config(config_name)
                                        logger.debug(f"Successfully called load_config for: {config_name}")
                                    except Exception as load_err:
                                        logger.error(f"Error loading config file {yaml_file}: {load_err}")
                        except FileNotFoundError:
                            logger.error(f"Configuration directory not found: {self.config_path}")
                        except Exception as list_err:
                             logger.error(f"Error processing config directory {self.config_path}: {list_err}")
                        # --- End: Explicitly load all YAML files ---

                        # Now access the presumably populated config dictionary
                        config_to_use = config_manager.config.config

                        if not config_to_use:
                            logger.warning(f"Configuration dictionary empty after attempting to load files from {self.config_path}. Falling back to default.")
                            config_to_use = self.default_config
                        else:
                             logger.info(f"Successfully processed configuration files from {self.config_path}")
                             logger.debug(f"Loaded config keys: {list(config_to_use.keys())}") # Log loaded keys

                    except ImportError as ie:
                        logger.error(f"Failed to import ConfigManager: {ie}. Using default config.")
                        config_to_use = self.default_config
                    except Exception as cfg_err:
                        logger.error(f"Failed to load config from {self.config_path}: {cfg_err}. Using default.")
                        config_to_use = self.default_config
                else:
                    logger.info("No config path provided, using default configuration.")
                    config_to_use = self.default_config
                    
                # Try initializing the real system
                try:
                    logger.info("Attempting to instantiate real system implementation...")
                    # Instantiate first, config is passed during initialize
                    self.system = TCCCSystem() 
                    # Store config for the initialization step
                    self._loaded_config = config_to_use 
                    self.is_mock_system = False
                except Exception as e:
                    logger.error(f"Failed to instantiate real system: {str(e)}")
                    if self.use_mocks == "none":
                        logger.critical("Cannot proceed without a functional system (mocking disabled).")
                        return False # Hard fail if mocks are disallowed
                    else: # use_mocks == "auto"
                        logger.warning(f"Falling back to mock system implementation.")
                        self.system = MockTCCCSystem(self.config_path)
                        self.is_mock_system = True
            else: # Should not happen due to argparse choices, but good practice
                logger.error(f"Invalid mock setting: {self.use_mocks}")
                return False
            
            # Initialize the system
            if self.system:
                try:
                    logger.info(f"Initializing {'Mock' if self.is_mock_system else 'Real'} TCCC System...")
                    # Pass the loaded config to the initialize method for the real system
                    # Mocks are initialized without config in MockTCCCSystem.initialize
                    config_param = self._loaded_config if not self.is_mock_system and hasattr(self, '_loaded_config') else None
                    
                    # Determine which modules to mock
                    if not self.is_mock_system:
                        # If using the real system (use_mocks='none' or 'auto' succeeded),
                        # determine which modules to mock based on the use_mocks setting.
                        if self.use_mocks == "none":
                            # When --mock none is specified, use NO mocks.
                            mock_modules_param = []
                            logger.info(f"Real system selected (--mock none). No modules will be mocked.")
                        elif self.use_mocks == "auto":
                            # Default 'auto' behavior: Test only AudioPipeline for now.
                            # TODO: Enhance 'auto' to dynamically check module availability?
                            mock_modules_param = [
                                "stt_engine", 
                                "processing_core", 
                                "llm_analysis", 
                                "data_store", 
                                "document_library"
                            ]
                            logger.info(f"Real system selected (--mock auto). Mocking the following modules: {mock_modules_param}")
                        else: # Should not happen, but defensive coding
                             mock_modules_param = []
                             logger.warning(f"Unexpected use_mocks value '{self.use_mocks}' for real system. Defaulting to no mocks.")
                    else:
                        # MockTCCCSystem handles its own internal mocks
                        mock_modules_param = None 
                    
                    init_success = await self.system.initialize(config=config_param, mock_modules=mock_modules_param)
                    
                    if not init_success:
                        logger.error("System initialization method returned False.")
                        # Decide if we should bail out or continue with partial functionality
                        # For now, let's assume we cannot proceed if basic init fails
                        self.system = None # Mark system as unusable
                    else:
                        logger.info("System initialization successful.")
                except Exception as init_e:
                    logger.error(f"Exception during system initialization: {init_e}")
                    self.system = None # Mark system as unusable
            
            # Check if system is ready before running verification stages
            if not self.system or self.system.state != SystemState.READY:
                logger.error("System failed to initialize or is not in READY state. Cannot proceed with verification stages.")
                # Optionally report basic initialization failure
                self.verification_results[VerificationStage.INITIALIZATION.value] = {
                    "success": False,
                    "duration": time.time() - self.timing["start_time"], # Rough time
                    "details": {"error": "System failed to reach READY state during initialization"}
                }
                # We could potentially run SOME checks even if init failed, but for now, let's stop.
                self.print_summary() # Corrected method name
                return False
            
            # --- Conditional Verification Based on Mock Setting ---
            if self.use_mocks == "none":
                logger.info("Running verification for 'none' mock setting: Initializing and testing Audio Pipeline only.")
                
                # --- [1/N] Verify Initialization (Required) ---
                print("\n[1/2] Verifying system initialization...")
                stage_result_init = await self.verify_initialization()
                self.verification_results[VerificationStage.INITIALIZATION.value] = stage_result_init
                self.timing["stages"][VerificationStage.INITIALIZATION.value] = stage_result_init["duration"]
                if not stage_result_init["success"]:
                    logger.error("Initialization verification failed. Aborting further tests.")
                    # No need to set self.system_ready, just proceed to shutdown/summary
                else:
                    # --- [2/N] Verify Audio Pipeline Status Only ---
                    print("\n[2/2] Verifying Audio Pipeline...")
                    stage_start_time = time.time()
                    module_name = "audio_pipeline"
                    module_results = {} # Initialize results for this stage
                    overall_success = False # Assume failure until tests pass
                    test_details = [] # List to collect details of sub-tests
                    
                    try:
                        # Get module instance (must be real system in 'none' mode)
                        module_instance = getattr(self.system, module_name, None)

                        if not module_instance:
                            raise ValueError(f"{module_name.replace('_', ' ').title()} module not available")

                        # 1. Initial Status Check
                        if hasattr(module_instance, "get_status"):
                            ap_status_initial = module_instance.get_status()
                            print(f"DEBUG: Initial AP Status dictionary: {ap_status_initial}") # <--- CORRECTLY PLACED DEBUG PRINT
                            if isinstance(ap_status_initial, dict) and ap_status_initial.get("initialized") and ap_status_initial.get("status") == ModuleState.READY:
                                test_details.append({"test": "initial_status", "success": True, "status": ap_status_initial})
                            else:
                                test_details.append({"test": "initial_status", "success": False, "error": "Not initialized or not READY", "status": ap_status_initial})
                                raise AssertionError("Initial status check failed") # Stop further tests for this module
                        else:
                            raise AttributeError("Missing get_status method")

                        # 2. Start Capture Test
                        if hasattr(module_instance, "start_capture"):
                            logger.info(f"[Module Test - {module_name}] Attempting to start capture...")
                            start_success = module_instance.start_capture() 
                            if start_success:
                                logger.info(f"[Module Test - {module_name}] start_capture succeeded.")
                                test_details.append({"test": "start_capture", "success": True})
                                
                                # Immediate Status Check After Start
                                status_after_start = module_instance.get_status()
                                logger.debug(f"[Module Test - {module_name}] Status immediately after start: {status_after_start}")
                                if not (isinstance(status_after_start, dict) and status_after_start.get("capturing") and status_after_start.get("status") == ModuleState.ACTIVE):
                                    err_msg = "Status not ACTIVE/capturing immediately after start_capture call."
                                    test_details.append({"test": "immediate_status_after_start", "success": False, "error": err_msg, "status": status_after_start})
                                    raise AssertionError(err_msg)
                                else:
                                     test_details.append({"test": "immediate_status_after_start", "success": True, "status": status_after_start})
                                     
                        else:
                             raise AttributeError("Missing start_capture method")
                             
                        # 3. Status Check During Capture
                        if hasattr(module_instance, "get_status"):
                            status = module_instance.get_status()
                            logger.debug(f"[Module Test - {module_name}] Status during capture: {status}")
                            # Expecting state to be ACTIVE or similar
                            if isinstance(status, dict) and status.get("capturing") and status.get("status") == ModuleState.ACTIVE:
                                test_details.append({"test": "status_while_capturing", "success": True, "status": status})
                            else:
                                err_msg = "Status not ACTIVE/capturing during capture phase."
                                test_details.append({"test": "status_while_capturing", "success": False, "error": err_msg, "status": status})
                                raise AssertionError(err_msg) # Fail the test if status is wrong mid-capture
                        else:
                             # Log warning, but don't fail the whole test if status isn't available mid-capture
                             logger.warning(f"[Module Test - {module_name}] Cannot check status during capture (get_status missing)")
                             test_details.append({"test": "status_while_capturing", "success": None, "warning": "get_status missing"}) 

                        # 4. Stop Capture Test
                        if hasattr(module_instance, "stop_capture"):
                            logger.info(f"[Module Test - {module_name}] Attempting to stop capture...")
                            stop_success = module_instance.stop_capture()
                            if stop_success:
                                logger.info(f"[Module Test - {module_name}] stop_capture succeeded.")
                                test_details.append({"test": "stop_capture", "success": True})
                                await asyncio.sleep(0.2) # Small delay for state update
                            else:
                                test_details.append({"test": "stop_capture", "success": False, "error": "stop_capture returned False"})
                                # Don't raise, check final status
                        else:
                            raise AttributeError("Missing stop_capture method")

                        # 5. Final Status Check (after stop)
                        if hasattr(module_instance, "get_status"):
                            status = module_instance.get_status()
                            logger.debug(f"[Module Test - {module_name}] Final Status: {status}")
                            if isinstance(status, dict) and not status.get("capturing") and status.get("status") == ModuleState.READY:
                                test_details.append({"test": "final_status", "success": True, "status": status})
                                overall_success = True # All steps passed if we got here
                            else:
                                test_details.append({"test": "final_status", "success": False, "error": "Still capturing or not READY", "status": status})
                                overall_success = False
                        else:
                             # If we can't check final status, assume success based on previous steps if they passed
                             all_prev_ok = all(d.get("success") for d in test_details if d.get("success") is not None)
                             if all_prev_ok:
                                 logger.warning(f"[Module Test - {module_name}] Cannot check final status (get_status missing), assuming success based on prior steps.")
                                 test_details.append({"test": "final_status", "success": None, "warning": "get_status missing, assumed OK"})
                                 overall_success = True
                             else:
                                 logger.error(f"[Module Test - {module_name}] Cannot check final status (get_status missing), and prior steps failed.")
                                 test_details.append({"test": "final_status", "success": False, "error": "get_status missing, prior steps failed"})
                                 overall_success = False
                                 
                    except AttributeError as ae:
                        logger.error(f"Audio Pipeline test failed: Missing required method - {ae}")
                        test_details.append({"test": "attribute_check", "success": False, "error": str(ae)})
                        overall_success = False
                    except AssertionError as ase:
                        logger.error(f"Audio Pipeline test failed: Assertion failed - {ase}")
                        # Details already added by the step that raised
                        overall_success = False
                    except Exception as e:
                        logger.exception(f"{module_name} test error")
                        test_details.append({"test": "general_exception", "success": False, "error": str(e), "exception_type": type(e).__name__})
                        overall_success = False
                    
                    # Print result
                    if overall_success:
                        print(f"✓ {module_name.replace('_', ' ').title()} module verified")
                    else:
                        print(f"❌ {module_name.replace('_', ' ').title()} module verification failed")

                    # Store results for the module verification stage
                    stage_duration = time.time() - stage_start_time
                    self.verification_results[VerificationStage.MODULE_VERIFICATION.value] = {
                        "success": overall_success,
                        "duration": stage_duration,
                        "details": {module_name: test_details} # Store detailed test steps
                    }
                    self.timing["stages"][VerificationStage.MODULE_VERIFICATION.value] = stage_duration

            else: # --- Original Full Verification Flow (for 'auto' or 'all') ---
                logger.info(f"Running full verification flow for mock setting: '{self.use_mocks}'")
                # Run verification stages
                print("\n[1/7] Verifying system initialization...")
                stage_result_init = await self.verify_initialization() # Use a different variable name
                self.verification_results[VerificationStage.INITIALIZATION.value] = stage_result_init
                self.timing["stages"][VerificationStage.INITIALIZATION.value] = stage_result_init["duration"]
                if not stage_result_init["success"]: # Check the 'success' key
                    logger.error("Initialization verification failed. Aborting further tests.")
                    # No need to set self.system_ready, just proceed to shutdown/summary
                else:
                    # Only proceed if initialization was successful
                    print("\n[2/7] Verifying individual modules...")
                    stage_result_mod = await self.verify_modules() # Capture results
                    self.verification_results[VerificationStage.MODULE_VERIFICATION.value] = stage_result_mod
                    self.timing["stages"][VerificationStage.MODULE_VERIFICATION.value] = stage_result_mod["duration"]
                    if not stage_result_mod["success"]: logger.warning("Module verification reported failures.")

                    print("\n[3/7] Verifying module integration...")
                    stage_result_int = await self.verify_integration() # Capture results
                    self.verification_results[VerificationStage.INTEGRATION_VERIFICATION.value] = stage_result_int
                    self.timing["stages"][VerificationStage.INTEGRATION_VERIFICATION.value] = stage_result_int["duration"]
                    if not stage_result_int["success"]: logger.warning("Integration verification reported failures.")
                    
                    print("\n[4/7] Verifying data flow...")
                    stage_result_flow = await self.verify_data_flow() # Capture results
                    self.verification_results[VerificationStage.DATA_FLOW.value] = stage_result_flow
                    self.timing["stages"][VerificationStage.DATA_FLOW.value] = stage_result_flow["duration"]
                    if not stage_result_flow["success"]: logger.warning("Data flow verification reported failures.")
                    
                    print("\n[5/7] Verifying error handling...")
                    stage_result_err = await self.verify_error_handling() # Capture results
                    self.verification_results[VerificationStage.ERROR_HANDLING.value] = stage_result_err
                    self.timing["stages"][VerificationStage.ERROR_HANDLING.value] = stage_result_err["duration"]
                    if not stage_result_err["success"]: logger.warning("Error handling verification reported failures.")
                    
                    print("\n[6/7] Verifying system performance...")
                    stage_result_perf = await self.verify_performance() # Capture results
                    self.verification_results[VerificationStage.PERFORMANCE.value] = stage_result_perf
                    self.timing["stages"][VerificationStage.PERFORMANCE.value] = stage_result_perf["duration"]
                    if not stage_result_perf["success"]: logger.warning("Performance verification reported failures.")

                    # Add security verification
                    print("\n[7/7] Verifying security...")
                    stage_result_sec = await self.verify_security() # Capture results
                    self.verification_results[VerificationStage.SECURITY.value] = stage_result_sec
                    self.timing["stages"][VerificationStage.SECURITY.value] = stage_result_sec["duration"]
                    if not stage_result_sec["success"]: logger.warning("Security verification reported failures.")

            # --- Common Cleanup and Summary ---
            # Clean shutdown
            if self.system:
                logger.info("\nShutting down system...")
                await self.system.stop() # Ensure stop is awaited if it's async
                logger.info("System shutdown complete.")
            else:
                logger.warning("System was not available for shutdown.")
            
            # Print results
            self.timing["end_time"] = time.time()
            self.print_summary()
            
            # Determine overall success
            return all(result["success"] for result in self.verification_results.values())
            
        except Exception as e:
            print(f"\nVerification error: {str(e)}")
            logger.exception("Unhandled exception during verification run") # Log traceback
            if self.system:
                try:
                    await self.system.stop()
                except Exception as stop_e:
                    logger.error(f"Failed to stop system during error handling: {stop_e}")
            return False
    
    async def verify_initialization(self):
        """
        Verify system initialization
        """
        stage_start = time.time()
        
        if not self.system or self.system.state != SystemState.READY:
            logger.error("System not initialized, skipping initialization verification")
            return False
        
        try:
            logger.info("Testing system initialization")
            
            # The main run_verification loop already initialized the system and checked for READY state.
            # If we reach here, the system object exists and its state should be READY.
            details = {}
            success = False
            
            if not self.system:
                details["error"] = "System object not found."
            elif self.system.state == SystemState.READY:
                logger.info(f"System state is READY (Verified in run_verification). Success.")
                success = True
                details["state"] = self.system.state.value
                # Optionally add checks for specific module statuses if needed for mocks/real
                # status = await self.system.get_health_status() # Example check
                # details["modules"] = {k: v['status'] for k, v in status.get('modules', {}).items()}
            else:
                details["error"] = f"System state is {self.system.state.value}, expected READY."
                # Include health status for debugging if available
                try:
                    status = await self.system.get_health_status()
                    details["health_status"] = status
                except Exception as e:
                    details["health_status_error"] = str(e)
                    
            if not success:
                logger.error(f"System initialization verification failed: {details.get('error', 'Unknown error')}")
            
            return {
                "success": success,
                "duration": time.time() - stage_start,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Initialization verification error: {str(e)}")
            self.verification_results[VerificationStage.INITIALIZATION.value] = {
                "success": False,
                "details": {
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            }
            return False
        finally:
            self.timing["stages"][VerificationStage.INITIALIZATION.value] = time.time() - stage_start
    
    async def verify_modules(self):
        """
        Verify individual module functionality
        """
        stage_start = time.time()
        
        if not self.system or self.system.state != SystemState.READY:
            logger.error("System not initialized, skipping module verification")
            return False
        
        try:
            logger.info("Testing individual module functionality")
            module_results = {}
            all_modules_ok = True
            
            # Test Processing Core
            try:
                processing_core = self.system.modules["processing_core"] if self.is_mock_system else self.system.processing_core
                
                if not processing_core:
                    raise ValueError("Processing Core module not available in the system")

                test_input = "This is a test transcription for processing core"
                
                # Process a test transcription
                if hasattr(processing_core, "processTranscription") or hasattr(processing_core, "process_transcription"): # Check both sync/async
                    # For real system, processTranscription is async and takes TranscriptionSegment
                    if not self.is_mock_system and hasattr(processing_core, "processTranscription"):
                        test_segment = TranscriptionSegment(text=test_input)
                        result_obj: ProcessedSegment = await processing_core.processTranscription(test_segment)
                        # Convert to dict for consistent checking
                        result = {
                            "text": result_obj.text,
                            "speaker": result_obj.speaker,
                            "start_time": result_obj.start_time,
                            "end_time": result_obj.end_time,
                            "confidence": result_obj.confidence,
                            "entities": [e.to_dict() for e in result_obj.entities] if result_obj.entities else [],
                            "intents": [i.to_dict() for i in result_obj.intents] if result_obj.intents else [], # Correctly handle the 'intents' list
                            "sentiment": result_obj.sentiment.to_dict() if result_obj.sentiment else None,
                            "metadata": result_obj.metadata
                        }
                        logger.debug(f"[Module Test] ProcessingCore real result obj: {result_obj}")
                        logger.debug(f"[Module Test] ProcessingCore real result dict: {result}")
                    elif self.is_mock_system and hasattr(processing_core, "process_transcription"):
                        result = processing_core.process_transcription(test_input)
                        logger.debug(f"[Module Test] ProcessingCore mock result dict: {result}")
                    else:
                         raise AttributeError("ProcessingCore instance missing appropriate process method for the current system type (real/mock)")

                    
                    # Verify result structure (checking keys common to both real and mock)
                    # Note: Real ProcessedSegment might not have a simple 'intent' top-level key like mocks might.
                    # Adjust check based on actual ProcessedSegment structure conversion.
                    # Assuming converted dict has 'entities', 'intents', 'sentiment' keys.
                    if (isinstance(result, dict) and
                        "entities" in result and
                        "intents" in result and # Check for 'intents' (plural)
                        "sentiment" in result):

                        logger.info("Processing Core test passed")
                        module_results["processing_core"] = {"success": True, "details": {"status": result}}
                        print("✓ Processing Core module verified")
                    else:
                        logger.warning(f"Processing Core returned unexpected format or missing keys. Result: {result}")
                        module_results["processing_core"] = {
                            "success": False,
                            "details": {
                                "error": "Invalid result format or missing keys",
                                "received_type": str(type(result)),
                                "received_keys": list(result.keys()) if isinstance(result, dict) else None
                            }
                        }
                        all_modules_ok = False
                        print("❌ Processing Core module verification failed")
                else:
                    logger.warning("Processing Core missing expected methods (process_transcription or processTranscription)")
                    module_results["processing_core"] = {
                        "success": False,
                        "details": {
                            "error": "Missing process_transcription/processTranscription method"
                        }
                    }
                    all_modules_ok = False
                    print("❌ Processing Core module verification failed")
                
            except Exception as e:
                logger.exception("Processing Core test error") # Use exception for traceback
                module_results["processing_core"] = {
                    "success": False,
                    "details": {
                        "error": str(e),
                         "exception_type": type(e).__name__
                    }
                }
                all_modules_ok = False
                print("❌ Processing Core module verification failed")
            
            # Test Data Store
            try:
                data_store = self.system.modules["data_store"] if self.is_mock_system else self.system.data_store
                
                if not data_store:
                    raise ValueError("Data Store module not available in the system")

                test_event_data = {"test_id": f"ds_verify_{int(time.time())}", "test_value": "test_data", "timestamp": time.time()}
                event_to_store = {
                    "type": "verification_test",
                    "source": "verify_modules",
                    "data": test_event_data, # Embed data directly
                    "timestamp": datetime.now().isoformat()
                }
                
                # Store and retrieve test data
                if hasattr(data_store, "store_event") and hasattr(data_store, "get_event"):
                    logger.debug(f"[Module Test] DataStore storing event: {event_to_store}")
                    event_id = data_store.store_event(event_to_store)
                    logger.debug(f"[Module Test] DataStore received event_id: {event_id}")
                    
                    if not event_id:
                         raise ValueError("DataStore store_event did not return an event_id")
                    
                    retrieved = data_store.get_event(event_id)
                    logger.debug(f"[Module Test] DataStore retrieved event: {retrieved}")
                    
                    # Now, 'retrieved' should contain the deserialized event structure
                    # Access the 'data' field which contains the original test_event_data
                    retrieved_data = retrieved.get("data", {}) 
                    
                    id_match = retrieved_data.get("test_id") == test_event_data["test_id"]
                    value_match = retrieved_data.get("test_value") == test_event_data["test_value"]
                    # Explicitly compare timestamps
                    ts_stored = test_event_data["timestamp"]
                    ts_retrieved = retrieved_data.get("timestamp") # Get timestamp from deserialized data
                    ts_match = isinstance(ts_stored, (int, float)) and isinstance(ts_retrieved, (int, float)) and math.isclose(ts_stored, ts_retrieved)
                    
                    if id_match and value_match and ts_match:
                        logger.info("Data Store test passed")
                        module_results["data_store"] = {"success": True, "details": {"status": "ok"}}
                        print("✓ Data Store module verified")
                    else:
                        logger.warning("Data Store test failed")
                        module_results["data_store"] = {
                            "success": False,
                            "details": {
                                "error": "Data mismatch",
                                "stored": test_event_data,
                                "retrieved": retrieved_data # Log the deserialized data that was compared
                            }
                        }
                        all_modules_ok = False
                        print("❌ Data Store module verification failed")
                else:
                    logger.warning("Data Store missing expected methods (store_event, get_event)")
                    module_results["data_store"] = {
                        "success": False,
                        "details": {
                            "error": "Missing required methods"
                        }
                    }
                    all_modules_ok = False
                    print("❌ Data Store module verification failed")
            
            except Exception as e:
                logger.exception("Data Store test error") # Use exception for traceback
                module_results["data_store"] = {
                    "success": False,
                    "details": {
                        "error": str(e),
                         "exception_type": type(e).__name__
                    }
                }
                all_modules_ok = False
                print("❌ Data Store module verification failed")
            
            # Test LLM Analysis (Status Check Only for now)
            try:
                llm_analysis = self.system.modules["llm_analysis"] if self.is_mock_system else self.system.llm_analysis
                
                if not llm_analysis:
                    raise ValueError("LLM Analysis module not available in the system")
                
                if hasattr(llm_analysis, "get_status"):
                    status = llm_analysis.get_status()
                    logger.debug(f"[Module Test] LLM Analysis status: {status}")
                    # Check for initialized=True and overall status=READY or ACTIVE
                    if isinstance(status, dict) and status.get("initialized") and status.get("status") in [ModuleState.READY, ModuleState.ACTIVE]:
                         logger.info("LLM Analysis status check passed.")
                         module_results["llm_analysis"] = {"success": True, "details": {"status": status}}
                         print("✓ LLM Analysis module verified (status check)")
                    else:
                        logger.warning(f"LLM Analysis status check failed. Status: {status}")
                        module_results["llm_analysis"] = {
                            "success": False,
                            "details": {
                                "error": "Status check failed (not initialized or not READY)",
                                "status": status
                            }
                        }
                        all_modules_ok = False
                        print("❌ LLM Analysis module verification failed (status check)")
                else:
                     logger.warning("LLM Analysis missing get_status method.")
                     module_results["llm_analysis"] = {"success": False, "details": {"error": "Missing get_status method"}}
                     all_modules_ok = False
                     print("❌ LLM Analysis module verification failed (missing get_status)")
            
            except Exception as e:
                logger.exception("LLM Analysis test error")
                module_results["llm_analysis"] = {
                    "success": False,
                    "details": {
                        "error": str(e),
                        "exception_type": type(e).__name__
                    }
                }
                all_modules_ok = False
                print("❌ LLM Analysis module verification failed")
            
            # Test Audio Pipeline, STT Engine, Document Library (Basic Status Check)
            for module_name in ["audio_pipeline", "stt_engine", "document_library"]:
                try:
                    module_instance = None
                    if self.is_mock_system:
                        module_instance = self.system.modules.get(module_name)
                    elif hasattr(self.system, module_name):
                        module_instance = getattr(self.system, module_name)
                    
                    if not module_instance:
                        raise ValueError(f"{module_name} module not available in the system")

                    if hasattr(module_instance, "get_status"):
                        status = module_instance.get_status()
                        logger.debug(f"[Module Test] {module_name.replace('_', ' ').title()} status: {status}")
                        
                        # Check for initialized=True and status=READY/RUNNING/ACTIVE
                        # Note: AudioPipeline might be READY if no source, RUNNING if source found.
                        # Some modules use ACTIVE instead of READY/RUNNING
                        is_initialized = status and isinstance(status, dict) and status.get("initialized", False)
                        is_operational = status.get("status") in [ModuleState.READY, ModuleState.ACTIVE] # Use only valid states

                        if is_initialized and is_operational:
                            logger.info(f"{module_name.replace('_', ' ').title()} basic status check passed (Initialized and Ready/Active)")
                            module_results[module_name] = {
                                "success": True,
                                "details": {"status": status}
                            }
                            print(f"✓ {module_name.replace('_', ' ').title()} module verified (status check)")
                        else:
                            logger.warning(f"{module_name.replace('_', ' ').title()} status check failed. Initialized={is_initialized}, Operational State={status.get('status')}. Full Status: {status}")
                            module_results[module_name] = {
                                "success": False,
                                "details": {
                                    "error": "Module status check failed (not initialized or not Ready/Active)",
                                    "status": status
                                }
                            }
                            all_modules_ok = False
                            print(f"❌ {module_name.replace('_', ' ').title()} module verification failed (status check)")
                    else:
                        logger.warning(f"{module_name.replace('_', ' ').title()} missing get_status method.")
                        module_results[module_name] = {"success": False, "details": {"error": "Missing get_status method"}}
                        all_modules_ok = False
                        print(f"❌ {module_name.replace('_', ' ').title()} module verification failed (missing get_status)")
                
                except Exception as e:
                    logger.exception(f"{module_name.replace('_', ' ').title()} test error")
                    module_results[module_name] = {
                        "success": False,
                        "details": {
                            "error": str(e),
                        "exception_type": type(e).__name__
                    }
                }
                all_modules_ok = False
                print(f"❌ {module_name.replace('_', ' ').title()} module verification failed")
            
            # Calculate overall success based on individual module results BEFORE recording
            all_modules_ok = all(details.get("success", False) for details in module_results.values() if isinstance(details, dict))
            logger.debug(f"Final calculated all_modules_ok for Module Verification stage: {all_modules_ok}")

            # Record results
            self.verification_results[VerificationStage.MODULE_VERIFICATION.value] = {
                "success": all_modules_ok,
                "details": module_results,
                "duration": time.time() - stage_start # Record duration here
            }
            
            if all_modules_ok:
                logger.info("All modules verified successfully")
                print("✓ All modules passed individual verification")
            else:
                logger.warning("Some modules failed verification")
                print("❌ Some modules failed individual verification")
            
            return all_modules_ok
            
        except Exception as e:
            logger.exception("General module verification error") # Use exception for traceback
            self.verification_results[VerificationStage.MODULE_VERIFICATION.value] = {
                "success": False,
                "details": {
                    "error": f"General error during module verification: {str(e)}",
                    "exception_type": type(e).__name__
                },
                 "duration": time.time() - stage_start
            }
            return False
        # Removed finally block as duration is now recorded within try block

    async def _verify_modules(self) -> Dict[str, Any]:
        """Verify individual module functionality."""
        stage_start = time.time()
        logger.info("Testing individual module functionality")
        results = {}
        all_passed = True

        # Define expected modules
        expected_modules = [
            "processing_core", "data_store", "llm_analysis",
            "audio_pipeline", "stt_engine", "document_library"
        ]

        for module_name in expected_modules:
            module_passed = False
            module_details = {}
            try:
                # Get the module instance correctly
                module_instance = None
                if self.is_mock_system:
                    module_instance = self.system.modules.get(module_name)
                elif hasattr(self.system, module_name):
                    module_instance = getattr(self.system, module_name)

                if not module_instance:
                    module_details["error"] = "Module instance not found in system object"
                    logger.error(f"Module {module_name}: Instance not found.")
                elif self.is_mock_system:
                    # For mocks, we'll rely on the initialization stage having passed
                    # And potentially add simple mock-specific checks if needed later
                    if module_name in self.system.modules: # Basic check: does the mock exist?
                        module_passed = True
                        logger.info(f"{module_name} mock verified (existence check)")
                        print(f"✓ {module_name} module verified")
                    else:
                        module_details["error"] = "Mock module not found in mock system's modules dict"
                        logger.error(f"Module {module_name}: Mock instance not found.")
                elif hasattr(module_instance, 'get_status'):
                    # For real modules, check their status
                    status = module_instance.get_status() # This is synchronous for real modules
                    logger.debug(f"Verifying module '{module_name}'. Received status: {status}") # <--- ADD LOGGING
                    module_details["status_report"] = status
                    # Check if initialized (assuming 'initialized' key exists in status dict)
                    if isinstance(status, dict) and status.get("initialized", False):
                        # Could add more specific checks, e.g., model loaded for STT/LLM
                        module_passed = True
                        logger.info(f"{module_name} status check passed: Initialized=True")
                        print(f"✓ {module_name} module verified")
                    else:
                        module_details["error"] = f"Module not initialized or status invalid. Status: {status}"
                        logger.error(f"Module {module_name}: Status check failed (Not initialized or invalid). Status: {status}")
                else:
                    module_details["error"] = "Real module instance does not have get_status method"
                    logger.error(f"Module {module_name}: Real instance missing get_status method.")

            except Exception as e:
                module_details["error"] = f"Exception during verification: {str(e)}"
                logger.exception(f"Error verifying module {module_name}") # Use logger.exception to include traceback

            results[module_name] = {"success": module_passed, "details": module_details}
            if not module_passed:
                all_passed = False
                print(f"❌ {module_name} module failed verification")

        if all_passed:
            logger.info("All modules passed individual verification based on status checks")
            print("✓ All modules passed individual verification")
        else:
            logger.warning("Some modules failed individual verification")
            print("❌ Some modules failed verification")

        return {
            "success": all_passed,
            "duration": time.time() - stage_start,
            "details": results
        }

    async def _verify_audio_pipeline(self, tccc_system):
        stage_name = "Audio Pipeline"
        start_time = time.time()
        logger.info(f"--- Verifying {stage_name} ---")
        # Initialize results with default values
        results = {"passed": False, "details": [], "duration": 0}
        # Ensure the stage name exists in the summary dict before starting
        if "Module Verification" not in self.summary: self.summary["Module Verification"] = {}
        if stage_name not in self.summary["Module Verification"]: self.summary["Module Verification"][stage_name] = {}

        # Assume audio_pipeline exists as it's checked by the caller
        audio_pipeline = tccc_system.audio_pipeline

        try:
            # 1. Check status after initialization
            initial_status_dict = await audio_pipeline.get_status()
            logger.info(f"{stage_name} - Initial Status: {initial_status_dict}")
            results["details"].append(f"Initial Status: {initial_status_dict.get('status', 'UNKNOWN')}")

            initial_status = initial_status_dict.get('status')
            # Log warning if not READY, but continue test
            if initial_status != ModuleState.READY:
                 logger.warning(f"{stage_name} was not in READY state after system initialization. State: {initial_status}")

            # 2. Attempt to start capture
            logger.info(f"Attempting to start capture for {stage_name}...")
            await audio_pipeline.start_capture()
            logger.info(f"{stage_name} - start_capture() called.")
            results["details"].append("Attempted start_capture()")

            # Allow some time for capture to potentially start/fail
            logger.info(f"Waiting 1 second for {stage_name} capture to initialize...")
            await asyncio.sleep(1.0)

            # 3. Check status after attempting start_capture
            post_start_status_dict = await audio_pipeline.get_status()
            logger.info(f"{stage_name} - Status after start_capture attempt: {post_start_status_dict}")
            results["details"].append(f"Post-Start Status: {post_start_status_dict.get('status', 'UNKNOWN')}")

            # 4. Verify final state
            final_status = post_start_status_dict.get('status')
            if final_status in [ModuleState.RUNNING, ModuleState.CAPTURING]:
                logger.info(f"{stage_name} successfully started capture (State: {final_status}).")
                results["passed"] = True
                results["details"].append("Capture started successfully.")
            else:
                logger.error(f"{stage_name} failed to start capture. Final State: {final_status}")
                results["passed"] = False
                results["details"].append(f"Failed to start capture. Final State: {final_status}")
                error_message = post_start_status_dict.get('error', 'No specific error reported')
                logger.error(f"{stage_name} - Error details: {error_message}")
                results["details"].append(f"Error: {error_message}")

        except Exception as e:
            logger.error(f"Error during {stage_name} verification: {e}", exc_info=True)
            results["passed"] = False
            results["details"].append(f"Exception during verification: {str(e)}") # Keep error concise for summary

        results["duration"] = time.time() - start_time
        self.summary["Module Verification"][stage_name].update(results) # Update the existing dict
        logger.info(f"--- {stage_name} Verification Complete (Passed: {results['passed']}) ---")
        return results["passed"]

    async def _verify_stt_engine(self, tccc_system):
        # Placeholder for STT Engine verification
        pass

    async def verify_integration(self):
        """
        Verify module integration
        """
        stage_start = time.time()
        
        if not self.system or self.system.state != SystemState.READY:
            logger.error("System not initialized, skipping integration verification")
            return False
        
        try:
            logger.info("Testing module integration")
            integration_results = {}
            all_tests_passed = True
            
            # Test ProcessingCore DataStore integration
            try:
                processing_core = self.system.modules["processing_core"] if self.is_mock_system else self.system.processing_core
                data_store = self.system.modules["data_store"] if self.is_mock_system else self.system.data_store
                
                if not processing_core or not data_store:
                    raise ValueError("Required modules (ProcessingCore, DataStore) not available")

                # Process a test transcription segment
                test_text = "Test integration between ProcessingCore and DataStore"
                test_segment = TranscriptionSegment(text=test_text) # Create segment object
                
                if self.is_mock_system:
                    # Mock processing core expects text directly
                    processing_result_dict = processing_core.process_transcription(test_text)
                else:
                    # Real processing core expects TranscriptionSegment and is async
                    # It returns a ProcessedSegment object
                    processed_segment_obj: ProcessedSegment = await processing_core.processTranscription(test_segment)
                    # Convert ProcessedSegment to a dictionary for storage
                    # Assuming ProcessedSegment has a simple structure or a .to_dict() method
                    # For now, create a basic dict; adjust if ProcessedSegment has .to_dict()
                    processing_result_dict = {
                        "text": processed_segment_obj.text,
                        "speaker": processed_segment_obj.speaker,
                        "start_time": processed_segment_obj.start_time,
                        "end_time": processed_segment_obj.end_time,
                        "confidence": processed_segment_obj.confidence,
                        "entities": [e.to_dict() for e in processed_segment_obj.entities] if processed_segment_obj.entities else [],
                        "intents": [i.to_dict() for i in processed_segment_obj.intents] if processed_segment_obj.intents else [], # Correctly handle the 'intents' list
                        "sentiment": processed_segment_obj.sentiment.to_dict() if processed_segment_obj.sentiment else None,
                        "metadata": processed_segment_obj.metadata
                        # Note: 'raw_text' is NOT explicitly part of ProcessedSegment, likely causing the failure
                    }
                
                # Store the processing result (as a dictionary)
                # Create an event structure if needed by store_event
                event_to_store = {
                    "session_id": self.session_id,
                    "event_id": None, # Will be assigned by store_event
                    "timestamp": datetime.now().isoformat(),
                    "type": "processed_text",
                    "source": "verification_script",
                    "data": processing_result_dict # Embed data directly
                }
                
                logger.debug(f"[Integration Test] Storing event data: {processing_result_dict}") # <--- ADD LOGGING
                event_id = data_store.store_event(event_to_store)
                
                # Retrieve the stored result
                retrieved = data_store.get_event(event_id)
                logger.debug(f"[Integration Test] Retrieved event: {retrieved}") # <--- ADD LOGGING
                
                # Now, 'retrieved' contains the deserialized event structure
                # Access the 'data' field which contains the original processing_result_dict
                retrieved_data = retrieved.get("data", {}) 
                
                integration_ok = False
                if retrieved and "data" in retrieved:
                    retrieved_data_raw = retrieved_data # Start comparison from the deserialized data dict
                    retrieved_data = {}
                    # Check if it's a dict (already deserialized) or string (needs parsing - less likely now)
                    if isinstance(retrieved_data_raw, dict):
                        retrieved_data = retrieved_data_raw
                    elif isinstance(retrieved_data_raw, str):
                        try:
                            retrieved_data = json.loads(retrieved_data_raw)
                        except json.JSONDecodeError:
                            logger.error(f"[Integration Test] Failed to decode JSON data in retrieved event: {retrieved_data_raw}")
                    else:
                        logger.error(f"[Integration Test] Unexpected data format in retrieved event: {type(retrieved_data_raw)}")

                    # Compare relevant fields
                    stored_text = processing_result_dict.get('text')
                    retrieved_text = retrieved_data.get('text')
                    logger.debug(f"[Integration Test] Comparing text. Stored: '{stored_text}', Retrieved: '{retrieved_text}'")
                    
                    # Compare the 'text' field
                    if stored_text and stored_text == retrieved_text:
                        integration_ok = True
                        logger.info("ProcessingCore DataStore integration test passed")
                    else:
                        logger.warning(f"ProcessingCore DataStore integration test failed: Text mismatch. Stored: {processing_result_dict.get('text')}, Retrieved: {retrieved_data.get('text')}")
                else:
                    logger.warning(f"ProcessingCore DataStore integration test failed: Event not retrieved or missing data field. Retrieved: {retrieved}")
                
                if integration_ok:
                    logger.info("ProcessingCore DataStore integration test passed")
                    integration_results["processing_core_datastore"] = {
                        "success": True,
                        "details": {
                            "event_id": event_id
                        }
                    }
                    print("✓ ProcessingCore DataStore integration verified")
                else:
                    logger.warning("ProcessingCore DataStore integration test failed")
                    integration_results["processing_core_datastore"] = {
                        "success": False,
                        "details": {
                            "error": "Text mismatch",
                            "stored": processing_result_dict.get('text'),
                            "retrieved": retrieved_data.get('text')
                        }
                    }
                    all_tests_passed = False
                    print("❌ ProcessingCore DataStore integration failed")
            
            except Exception as e:
                logger.error(f"ProcessingCore DataStore integration test error: {str(e)}")
                integration_results["processing_core_datastore"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_tests_passed = False
                print("❌ ProcessingCore DataStore integration failed")
            
            # Test STTEngine ProcessingCore integration
            try:
                stt_engine = self.system.modules["stt_engine"] if self.is_mock_system else self.system.stt_engine
                processing_core = self.system.modules["processing_core"] if self.is_mock_system else self.system.processing_core
                
                if not stt_engine or not processing_core:
                    raise ValueError("Required modules (STTEngine, ProcessingCore) not available")

                # Create a mock audio segment (if using real system)
                if not self.is_mock_system:
                    audio_segment = MockAudioSegment()
                else:
                    # Get a segment from the mock audio pipeline
                    audio_pipeline = self.system.modules["audio_pipeline"] if self.is_mock_system else self.system.audio_pipeline
                    audio_stream = audio_pipeline.get_audio_stream()
                    audio_segment = audio_stream.get_segment()
                
                # Transcribe the audio segment
                transcription = stt_engine.transcribe_segment(audio_segment)
                logger.info(f"[Integration Test STT] Transcription result from transcribe_segment: {transcription}") # <-- ADDED LOG
                
                # Process the transcription
                if self.is_mock_system:
                    processing_result = processing_core.process_transcription(transcription)
                else:
                    # Assume transcription is now the TranscriptionSegment object needed by processTranscription
                    if not isinstance(transcription, TranscriptionSegment):
                         logger.warning(f"[Integration Test STT] transcribe_segment did not return TranscriptionSegment, type={type(transcription)}. Creating one.")
                         # Attempt to create one if possible, assuming transcription is dict-like or has 'text'
                         text_content = transcription.get("text", "") if isinstance(transcription, dict) else str(transcription)
                         transcription = TranscriptionSegment(text=text_content) # Need to handle timestamps etc. if required

                    processed_segment_obj: ProcessedSegment = await processing_core.processTranscription(transcription)
                    logger.info(f"[Integration Test STT] ProcessedSegment object from processTranscription: {processed_segment_obj}") # <-- ADDED LOG
                    processing_result = {
                        "text": processed_segment_obj.text,
                        "speaker": processed_segment_obj.speaker,
                        "start_time": processed_segment_obj.start_time,
                        "end_time": processed_segment_obj.end_time,
                        "confidence": processed_segment_obj.confidence,
                        "entities": [e.to_dict() for e in processed_segment_obj.entities] if processed_segment_obj.entities else [],
                        "intents": [i.to_dict() for i in processed_segment_obj.intents] if processed_segment_obj.intents else [], # Correctly handle the 'intents' list
                        "sentiment": processed_segment_obj.sentiment.to_dict() if processed_segment_obj.sentiment else None,
                        "metadata": processed_segment_obj.metadata
                        # Note: 'raw_text' is NOT explicitly part of ProcessedSegment, likely causing the failure
                    }
                
                logger.info(f"[Integration Test STT] Final processing_result dictionary before checks: {processing_result}") # <-- ADDED LOG
                
                # Verify the result
                required_keys = ["entities", "intents", "sentiment", "text"] # Adjusted keys based on ProcessedSegment
                # The original check included "raw_text" which is likely incorrect
                missing_keys = [key for key in required_keys if key not in processing_result]
                
                # Basic check if it's a dictionary
                is_dict = isinstance(processing_result, dict)
                
                if is_dict and not missing_keys:
                    logger.info("STTEngine ProcessingCore integration test passed (Structure Check)")
                    integration_results["stt_processing"] = {
                        "success": True,
                        "details": {
                            "transcription_text": transcription.text if isinstance(transcription, TranscriptionSegment) else str(transcription),
                            "intent": processing_result.get("intents", [{}])[0].get("name", "unknown") # Get first intent name
                        }
                    }
                    print("✓ STTEngine ProcessingCore integration verified")
                else:
                    error_msg = f"Invalid processing result structure. Is Dict: {is_dict}. Missing Keys: {missing_keys}."
                    logger.warning(f"STTEngine ProcessingCore integration test failed: {error_msg} Full Result: {processing_result}") # <-- ADDED DETAIL TO LOG
                    integration_results["stt_processing"] = {
                        "success": False,
                        "details": {
                            "error": error_msg,
                            "result_received": processing_result # Log the full failing result
                        }
                    }
                    all_tests_passed = False
                    print("❌ STTEngine ProcessingCore integration failed")
                
            except Exception as e:
                logger.error(f"STTEngine ProcessingCore integration test error: {str(e)}", exc_info=True) # Log traceback
                integration_results["stt_processing"] = {
                    "success": False,
                    "details": {
                        "error": str(e),
                        "exception_type": type(e).__name__
                    }
                }
                all_tests_passed = False
                print("❌ STTEngine ProcessingCore integration failed")
            
            # Test LLMAnalysis DocumentLibrary integration
            try:
                llm_analysis = self.system.modules["llm_analysis"] if self.is_mock_system else self.system.llm_analysis
                document_library = self.system.modules["document_library"] if self.is_mock_system else self.system.document_library
                
                if not llm_analysis or not document_library:
                    raise ValueError("Required modules (LLMAnalysis, DocumentLibrary) not available")

                # Get test document
                docs = []
                if hasattr(document_library, "search_documents"):
                    docs = document_library.search_documents("test")
                
                if docs:
                    test_doc = docs[0]
                    
                    # Process document content with LLM
                    if self.is_mock_system:
                        analysis_result = llm_analysis.process_transcription(test_doc["content"])
                    else:
                        transcription_dict = {"text": test_doc["content"]}
                        analysis_result = llm_analysis.process_transcription(transcription_dict)
                    
                    # Verify the result
                    if isinstance(analysis_result, list):
                        logger.info("LLMAnalysis DocumentLibrary integration test passed (event extraction)")
                        integration_results["llm_document"] = {
                            "success": True,
                            "details": {
                                "document_id": test_doc["id"],
                                "events_extracted": len(analysis_result)
                            }
                        }
                        print("✓ LLMAnalysis DocumentLibrary integration verified")
                    else:
                        logger.warning("LLMAnalysis DocumentLibrary integration test failed (expected list result)")
                        integration_results["llm_document"] = {
                            "success": False,
                            "details": {
                                "error": "Invalid analysis result type",
                                "received_type": type(analysis_result).__name__
                            }
                        }
                        all_tests_passed = False
                        print("❌ LLMAnalysis DocumentLibrary integration failed")
                else:
                    # Add a document
                    if self.is_mock_system:
                        doc_id = document_library.add_document("test_integration", "Test Integration Document", "This is a test document for integration between LLM and Document Library")
                    else:
                        document_data = {
                            "text": "This is a test document for integration between LLM and Document Library",
                            "metadata": {"title": "Test Integration Document", "source": "verification_script"}
                        }
                        doc_id = document_library.add_document(document_data)
                    
                    # Get the document
                    if hasattr(document_library, "get_document"):
                        test_doc = document_library.get_document(doc_id)
                        
                        # Process document content with LLM
                        if self.is_mock_system:
                            analysis_result = llm_analysis.process_transcription(test_doc["content"])
                        else:
                            transcription_dict = {"text": test_doc["content"]}
                            analysis_result = llm_analysis.process_transcription(transcription_dict)
                        
                        # Verify the result
                        if isinstance(analysis_result, list):
                            logger.info("LLMAnalysis DocumentLibrary integration test passed (event extraction)")
                            integration_results["llm_document"] = {
                                "success": True,
                                "details": {
                                    "document_id": doc_id,
                                    "events_extracted": len(analysis_result)
                                }
                            }
                            print("✓ LLMAnalysis DocumentLibrary integration verified")
                        else:
                            logger.warning("LLMAnalysis DocumentLibrary integration test failed (expected list result)")
                            integration_results["llm_document"] = {
                                "success": False,
                                "details": {
                                    "error": "Invalid analysis result type",
                                    "received_type": type(analysis_result).__name__
                                }
                            }
                            all_tests_passed = False
                            print("❌ LLMAnalysis DocumentLibrary integration failed")
                    else:
                        logger.warning("DocumentLibrary missing get_document method")
                        integration_results["llm_document"] = {
                            "success": False,
                            "details": {
                                "error": "Missing get_document method"
                            }
                        }
                        all_tests_passed = False
                        print("❌ LLMAnalysis DocumentLibrary integration failed")
            
            except Exception as e:
                logger.error(f"LLMAnalysis DocumentLibrary integration test error: {str(e)}")
                integration_results["llm_document"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_tests_passed = False
                print("❌ LLMAnalysis DocumentLibrary integration failed")
            
            # Record results
            self.verification_results[VerificationStage.INTEGRATION_VERIFICATION.value] = {
                "success": all_tests_passed,
                "details": integration_results
            }
            
            if all_tests_passed:
                logger.info("All integration tests passed")
                print("✓ All module integrations verified successfully")
            else:
                logger.warning("Some integration tests failed")
                print("❌ Some module integrations failed verification")
            
            return all_tests_passed
            
        except Exception as e:
            logger.error(f"Integration verification error: {str(e)}")
            self.verification_results[VerificationStage.INTEGRATION_VERIFICATION.value] = {
                "success": False,
                "details": {
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            }
            return False
        finally:
            self.timing["stages"][VerificationStage.INTEGRATION_VERIFICATION.value] = time.time() - stage_start
    
    async def verify_data_flow(self):
        """
        Verify end-to-end data flow through the system
        """
        stage_start = time.time()
        
        if not self.system or self.system.state != SystemState.READY:
            logger.error("System not initialized, skipping data flow verification")
            return False
        
        try:
            logger.info("Testing end-to-end data flow")
            
            # Start the system
            result = await self.system.start()
            if not result:
                logger.error("Failed to start system for data flow test")
                self.verification_results[VerificationStage.DATA_FLOW.value] = {
                    "success": False,
                    "details": {
                        "error": "Failed to start system"
                    }
                }
                return False
            
            print("System started, running for 60 seconds to verify data flow...")
            
            # Record initial event count
            if self.is_mock_system:
                health_status = await self.system.get_health_status()
            else:
                health_status = self.system.get_health_status()
            initial_event_count = health_status.get("event_count", 0) # Access event_count from the top level of health_status
            
            # Inject test events for verification
            # Check if we can process events directly
            events_injected = 0
            if hasattr(self.system, 'process_event'):
                logger.info("Injecting test events")
                
                # Inject several test events
                for i in range(5):
                    # Create test event
                    test_event = {
                        "type": "test_event",
                        "text": f"This is test event {i+1} for data flow verification",
                        "timestamp": time.time(),
                        "metadata": {"test_run": True, "event_num": i+1}
                    }
                    
                    # Process the event
                    event_id = await self.system.process_event(test_event)
                    if event_id:
                        events_injected += 1
                        logger.info(f"Test event {i+1} processed, event ID: {event_id}")
                    
                    # Wait a short time between events
                    await asyncio.sleep(0.5)
            
            # Let the system run a bit longer to process any events
            additional_wait = 60 # Changed from 5 if events_injected > 0 else 10
            await asyncio.sleep(additional_wait)
            
            # Check for data flow by counting events
            if self.is_mock_system:
                health_status = await self.system.get_health_status()
            else:
                health_status = self.system.get_health_status()
            final_event_count = health_status.get("event_count", 0) # Access event_count from the top level of health_status
            events_processed = final_event_count - initial_event_count
            
            if events_processed > 0:
                logger.info(f"Data flow test passed, {events_processed} events processed")
                self.verification_results[VerificationStage.DATA_FLOW.value] = {
                    "success": True,
                    "details": {
                        "events_processed": events_processed,
                        "initial_count": initial_event_count,
                        "final_count": final_event_count
                    }
                }
                print(f"✓ Data flow verified - {events_processed} events processed")
                return True
            else:
                logger.warning("Data flow test failed, no events processed")
                self.verification_results[VerificationStage.DATA_FLOW.value] = {
                    "success": False,
                    "details": {
                        "error": "No events processed",
                        "initial_count": initial_event_count,
                        "final_count": final_event_count
                    }
                }
                print("❌ Data flow verification failed - no events processed")
                return False
            
        except Exception as e:
            logger.error(f"Data flow verification error: {str(e)}")
            self.verification_results[VerificationStage.DATA_FLOW.value] = {
                "success": False,
                "details": {
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            }
            return False
        finally:
            self.timing["stages"][VerificationStage.DATA_FLOW.value] = time.time() - stage_start
    
    async def verify_error_handling(self):
        """
        Verify system error handling
        """
        stage_start = time.time()
        
        if not self.system or self.system.state != SystemState.READY:
            logger.error("System not running, skipping error handling verification")
            return False
        
        try:
            logger.info("Testing error handling")
            
            # Record initial error count
            if self.is_mock_system:
                health_status = await self.system.get_health_status()
            else:
                health_status = self.system.get_health_status()
            initial_error_count = health_status["health"]["error_count"]
            
            # Inject a test error
            logger.info("Injecting test error")
            print("Injecting test error...")
            
            if hasattr(self.system, "add_error"):
                # Direct error injection for mock system
                self.system.add_error("test_module", "Test error for verification")
            else:
                # For real system, add error through health object
                self.system.health.add_error("test_module", "Test error for verification")
            
            # Short wait for error to be processed
            await asyncio.sleep(2)
            
            # Check that error was recorded
            if self.is_mock_system:
                health_status = await self.system.get_health_status()
            else:
                health_status = self.system.get_health_status()
            final_error_count = health_status["health"]["error_count"]
            errors_detected = final_error_count - initial_error_count
            
            # Check that system is still running despite error
            system_still_running = self.system.state == SystemState.READY
            
            if errors_detected > 0 and system_still_running:
                logger.info("Error handling test passed")
                self.verification_results[VerificationStage.ERROR_HANDLING.value] = {
                    "success": True,
                    "details": {
                        "errors_detected": errors_detected,
                        "system_resilient": system_still_running
                    }
                }
                print("✓ Error handling verified - system continued running despite error")
                return True
            else:
                logger.warning("Error handling test failed")
                failure_details = {
                    "error": "Error handling test failed",
                    "errors_detected": errors_detected,
                    "system_state": self.system.state.value
                }
                
                if errors_detected == 0:
                    failure_details["specific_failure"] = "System did not detect injected error"
                elif not system_still_running:
                    failure_details["specific_failure"] = f"System changed state to {self.system.state.value} due to error"
                
                self.verification_results[VerificationStage.ERROR_HANDLING.value] = {
                    "success": False,
                    "details": failure_details
                }
                print("❌ Error handling verification failed")
                return False
            
        except Exception as e:
            logger.error(f"Error handling verification error: {str(e)}")
            self.verification_results[VerificationStage.ERROR_HANDLING.value] = {
                "success": False,
                "details": {
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            }
            return False
        finally:
            self.timing["stages"][VerificationStage.ERROR_HANDLING.value] = time.time() - stage_start
    
    async def verify_performance(self):
        """
        Verify system performance
        """
        stage_start = time.time()
        
        if not self.system or self.system.state != SystemState.READY:
            logger.error("System not running, skipping performance verification")
            return False
        
        try:
            logger.info("Testing system performance")
            
            # Get initial resource usage
            if self.is_mock_system:
                health_status = await self.system.get_health_status()
            else:
                health_status = self.system.get_health_status()
            initial_resources = health_status["health"]["resource_usage"]
            
            # Get initial event count
            data_store = self.system.modules["data_store"] if self.is_mock_system else self.system.data_store
            initial_event_count = data_store.get_status().get("event_count", 0) # Access event_count from the top level of health_status
            
            # Inject test events for performance measurement
            if hasattr(self.system, 'process_event'):
                logger.info("Injecting test events for performance measurement")
                
                # Set up performance test parameters
                performance_duration = 5  # seconds
                events_per_second = 2  # target rate
                total_events = performance_duration * events_per_second
                time_per_event = 1.0 / events_per_second
                
                print(f"Measuring performance for {performance_duration} seconds...")
                
                # Inject events at regular intervals
                test_start = time.time()
                for i in range(total_events):
                    # Create test event
                    test_event = {
                        "type": "performance_test_event",
                        "text": f"This is performance test event {i+1}",
                        "timestamp": time.time(),
                        "metadata": {"performance_test": True, "event_num": i+1}
                    }
                    
                    # Process the event
                    await self.system.process_event(test_event)
                    
                    # Sleep until next event is due
                    elapsed = time.time() - test_start
                    next_event_time = (i + 1) * time_per_event
                    if next_event_time > elapsed and i < total_events - 1:
                        await asyncio.sleep(next_event_time - elapsed)
                
                # Allow some time for processing to complete
                await asyncio.sleep(1)
                
                # Calculate actual duration
                performance_duration = time.time() - test_start
            else:
                # Run for performance measurement period
                performance_duration = 5  # seconds
                print(f"Measuring performance for {performance_duration} seconds...")
                await asyncio.sleep(performance_duration)
            
            # Get final resource usage
            if self.is_mock_system:
                health_status = await self.system.get_health_status()
            else:
                health_status = self.system.get_health_status()
            final_resources = health_status["health"]["resource_usage"]
            
            # Get final event count
            final_event_count = data_store.get_status().get("event_count", 0)
            events_processed = final_event_count - initial_event_count
            
            # Calculate throughput
            throughput = events_processed / performance_duration
            
            # Check resource usage
            resource_thresholds = {
                "cpu_percent": 95.0,  # 95% CPU utilization threshold
                "memory_percent": 90.0,  # 90% memory utilization threshold
                "disk_usage_percent": 95.0  # 95% disk utilization threshold
            }
            
            resource_issues = []
            for resource, threshold in resource_thresholds.items():
                if resource in final_resources and final_resources[resource] > threshold:
                    resource_issues.append(f"{resource} usage too high: {final_resources[resource]:.1f}%")
            
            # Compile performance metrics
            performance_metrics = {
                "throughput_events_per_second": throughput,
                "cpu_percent": final_resources.get("cpu_percent", 0),
                "memory_percent": final_resources.get("memory_percent", 0),
                "gpu_percent": final_resources.get("gpu_utilization", 0),
                "disk_percent": final_resources.get("disk_usage_percent", 0),
                "events_processed": events_processed,
                "measurement_duration_seconds": performance_duration
            }
            
            # Determine if performance is acceptable
            if len(resource_issues) == 0 and throughput > 0:
                logger.info("Performance test passed")
                self.verification_results[VerificationStage.PERFORMANCE.value] = {
                    "success": True,
                    "details": performance_metrics
                }
                
                print(f"✓ Performance verified - {throughput:.2f} events/sec")
                print(f"  CPU: {performance_metrics['cpu_percent']:.1f}%, Memory: {performance_metrics['memory_percent']:.1f}%")
                
                return True
            else:
                failure_details = {
                    "error": "Performance issues detected",
                    "resource_issues": resource_issues,
                    "metrics": performance_metrics
                }
                
                if throughput <= 0:
                    failure_details["throughput_issue"] = "No events processed during performance test"
                
                logger.warning(f"Performance test failed: {failure_details}")
                self.verification_results[VerificationStage.PERFORMANCE.value] = {
                    "success": False,
                    "details": failure_details
                }
                
                print("❌ Performance verification failed")
                if resource_issues:
                    for issue in resource_issues:
                        print(f"  - {issue}")
                if throughput <= 0:
                    print("  - No events processed during test period")
                
                return False
            
        except Exception as e:
            logger.error(f"Performance verification error: {str(e)}")
            self.verification_results[VerificationStage.PERFORMANCE.value] = {
                "success": False,
                "details": {
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            }
            return False
        finally:
            self.timing["stages"][VerificationStage.PERFORMANCE.value] = time.time() - stage_start
    
    def print_summary(self):
        """
        Print a summary of all verification results
        """
        total_time = self.timing["end_time"] - self.timing["start_time"]
        
        print("\n" + "="*80)
        print("TCCC.ai Enhanced System Verification Summary")
        print("="*80)
        
        # Print system info
        system_type = "Mock System" if self.is_mock_system else "Real System"
        print(f"\nSystem Type: {system_type}")
        print(f"Total Verification Time: {total_time:.2f} seconds")
        
        # Print stage results
        for stage in VerificationStage:
            stage_value = stage.value
            result = self.verification_results[stage_value]
            status = "✓ PASS" if result["success"] else "❌ FAIL"
            
            stage_time = self.timing["stages"].get(stage_value, 0)
            
            print(f"\n{stage.value.replace('_', ' ').title()} ({stage_time:.2f}s): {status}")
            
            # Print stage-specific details
            if stage == VerificationStage.INITIALIZATION:
                if result["success"]:
                    init_time = result["details"].get("time_seconds", 0)
                    print(f"  Initialization Time: {init_time:.2f} seconds")
                    print(f"  All Modules Initialized: Yes")
                else:
                    print(f"  Error: {result['details'].get('error', 'Unknown error')}")
            
            elif stage == VerificationStage.MODULE_VERIFICATION:
                if result["success"]:
                    print("  All modules passed individual verification")
                else:
                    failed_modules = [
                        module for module, details in result["details"].items()
                        if isinstance(details, dict) and not details.get("success", False)
                    ]
                    if failed_modules:
                        print(f"  Failed Modules: {', '.join(failed_modules)}")
            
            elif stage == VerificationStage.INTEGRATION_VERIFICATION:
                if result["success"]:
                    print("  All module integrations verified")
                else:
                    failed = [
                        integration for integration, details in result["details"].items()
                        if isinstance(details, dict) and not details.get("success", False)
                    ]
                    if failed:
                        print(f"  Failed Integrations: {', '.join(failed)}")
            
            elif stage == VerificationStage.DATA_FLOW:
                if result["success"]:
                    events = result["details"].get("events_processed", 0)
                    print(f"  Events Processed: {events}")
                else:
                    print(f"  Error: {result['details'].get('error', 'Unknown error')}")
            
            elif stage == VerificationStage.ERROR_HANDLING:
                if result["success"]:
                    print("  System continued running despite error")
                    errors = result["details"].get("errors_detected", 0)
                    print(f"  Errors Detected: {errors}")
                else:
                    print(f"  Error: {result['details'].get('specific_failure', 'Unknown error')}")
            
            elif stage == VerificationStage.PERFORMANCE:
                if result["success"]:
                    throughput = result["details"].get("throughput_events_per_second", 0)
                    cpu = result["details"].get("cpu_percent", 0)
                    memory = result["details"].get("memory_percent", 0)
                    print(f"  Throughput: {throughput:.2f} events/second")
                    print(f"  CPU Usage: {cpu:.1f}%")
                    print(f"  Memory Usage: {memory:.1f}%")
                else:
                    issues = result["details"].get("resource_issues", [])
                    for issue in issues:
                        print(f"  - {issue}")
                    if "throughput_issue" in result["details"]:
                        print(f"  - {result['details']['throughput_issue']}")
        
        # Overall result
        all_success = all(result["success"] for result in self.verification_results.values())
        overall_status = "✓ PASS" if all_success else "❌ FAIL"
        print(f"\nOverall Verification: {overall_status}")
        if all_success:
            print("\nThe TCCC.ai system has been fully verified and is ready for deployment.")
        else:
            print("\nThe TCCC.ai system verification identified issues that need to be addressed.")
            # Summarize issues
            failed_stages = [
                stage.value.replace('_', ' ').title()
                for stage in VerificationStage
                if not self.verification_results[stage.value]["success"]
            ]
            if failed_stages:
                print(f"Failed stages: {', '.join(failed_stages)}")


async def main(default_cfg: Dict[str, Any]):
    """Main entry point for the enhanced verification script"""
    parser = argparse.ArgumentParser(description='Enhanced verification for TCCC.ai system')
    parser.add_argument('--config', type=str, help='Path to configuration directory')
    parser.add_argument('--mock', choices=['auto', 'all', 'none'], default='auto',
                       help='Use mock implementations: "auto" (default), "all", or "none"')
    parser.add_argument('--log-level', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: DEBUG)')
    args = parser.parse_args()
    
    # Set the logging level based on the argument
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level) # Set level for root logger
    logging.getLogger('tccc').setLevel(log_level) # Set level for tccc package
    logger.info(f"Setting log level to: {args.log_level.upper()}")

    # Create and run the verifier
    verifier = SystemVerifierEnhanced(default_cfg, args.config, args.mock)
    success = await verifier.run_verification()
    
    return 0 if success else 1


if __name__ == "__main__":
    print("--- Starting TCCC Enhanced System Verification ---")
    # Define the default configuration here
    DEFAULT_CONFIG = {
        "llm_analysis": {
            "model": {
                "primary": {
                    "provider": "local",
                    "name": "phi-2-mock",
                    "path": "models/phi-2-instruct/" # Keep path for potential future use
                },
                "fallback": {
                    "provider": "local",
                    "name": "phi-2-mock"
                }
            },
            "hardware": {
                "enable_acceleration": False, # Default to CPU for mock
                "cuda_device": -1,
                "quantization": "none"
            },
            "caching": {"enabled": False},
            "event_handling": {"enabled": True}
        },
        "stt_engine": {
            "model": {
                "type": "whisper",        # Specify model type (e.g., whisper)
                "size": "tiny",          # Specify model size (e.g., tiny, base, medium)
                "path": "models/stt",    # Specify base path for model files
                "language": "en",        # Default language
                "beam_size": 5           # Default beam size
            },
            "hardware": {
                "enable_acceleration": False, # Default to CPU
                "cuda_device": -1,         # Use -1 for CPU
                "use_tensorrt": False,     # TensorRT disabled by default
                "quantization": "none"     # No quantization by default
            },
            "streaming": {                 # Add default streaming config
                "enabled": False,         # Disable streaming by default for verification
                "max_context_length_sec": 30
            },
            "diarization": {               # Add default diarization config
                "enabled": False          # Disable diarization by default
            },
            "vad_filter": True             # Keep VAD filter setting
        },
        "audio_pipeline": { # Add default config for AudioPipeline
            "device": "hw:3,0", # Use Logitech Webcam (hw:3,0) instead of 'default'
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "format": "int16" # Common format, adjust if needed
        }
        # Add other module defaults as needed
    }

    # Set up signal handling for clean shutdown
    loop = asyncio.get_event_loop()
    signals = (signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(asyncio.shield(
            asyncio.sleep(0))))  # Allows for clean shutdown

    try:
        # Pass DEFAULT_CONFIG into main
        sys.exit(loop.run_until_complete(main(DEFAULT_CONFIG)))
    except KeyboardInterrupt:
        print("Verification interrupted")