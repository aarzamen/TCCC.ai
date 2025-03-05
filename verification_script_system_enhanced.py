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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the system components
from src.tccc.system.system import TCCCSystem, SystemState
from src.tccc.audio_pipeline import AudioPipeline
from src.tccc.stt_engine import STTEngine
from src.tccc.processing_core import ProcessingCore
from src.tccc.llm_analysis import LLMAnalysis
from src.tccc.data_store import DataStore
from src.tccc.document_library import DocumentLibrary
from src.tccc.utils import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemVerification")


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
        self.initialized = False
        self.capturing = False
        self.source = None
    
    def initialize(self, config=None):
        """Initialize the audio pipeline"""
        self.initialized = True
        return True
    
    def start_capture(self, source="mock"):
        """Start audio capture"""
        self.capturing = True
        self.source = source
        return True
    
    def stop_capture(self):
        """Stop audio capture"""
        self.capturing = False
        self.stream.stop()
        return True
    
    def get_audio_stream(self):
        """Get the audio stream"""
        return self.stream
    
    def get_status(self):
        """Get the module status"""
        return {
            "status": "ok",
            "initialized": self.initialized,
            "capturing": self.capturing,
            "source": self.source
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.capturing = False
        self.initialized = False
        return True


class MockSTTEngine:
    """Mock STT engine implementation for testing"""
    
    def __init__(self):
        self.initialized = False
        self.model_loaded = False
        self.transcription_count = 0
    
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
        return {
            "status": "ok",
            "initialized": self.initialized,
            "model_loaded": self.model_loaded,
            "transcription_count": self.transcription_count
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        self.model_loaded = False
        return True


class MockProcessingCore:
    """Mock processing core implementation for testing"""
    
    def __init__(self):
        self.initialized = False
        self.plugins_loaded = 0
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
            "intent": {
                "name": "test_intent",
                "confidence": 0.85,
                "slots": {"item": "test"}
            },
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
        self.initialized = False
        self.model_loaded = False
        self.analysis_count = 0
    
    def initialize(self, config=None):
        """Initialize the LLM analysis"""
        self.initialized = True
        self.model_loaded = True
        return True
    
    def process_transcription(self, transcription, processing_result=None):
        """Process a transcription with LLM"""
        self.analysis_count += 1
        
        # Get the text from the transcription
        text = transcription["text"] if isinstance(transcription, dict) else str(transcription)
        
        # Return a mock analysis result
        return {
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
    
    def get_status(self):
        """Get the module status"""
        return {
            "status": "ok",
            "initialized": self.initialized,
            "model_loaded": self.model_loaded,
            "analysis_count": self.analysis_count
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        self.model_loaded = False
        return True


class MockDataStore:
    """Mock data store implementation for testing"""
    
    def __init__(self):
        self.initialized = False
        self.storage = {}
        self.event_count = 0
    
    def initialize(self, config=None):
        """Initialize the data store"""
        self.initialized = True
        self.storage = {
            "events": [],
            "sessions": {},
            "metrics": {}
        }
        return True
    
    def store_event(self, event_data):
        """Store an event"""
        self.event_count += 1
        event_id = f"event_{self.event_count}"
        
        # Add event to storage
        self.storage["events"].append({
            "id": event_id,
            "timestamp": time.time(),
            "data": event_data
        })
        
        return event_id
    
    def get_event(self, event_id):
        """Get an event by ID"""
        for event in self.storage["events"]:
            if event["id"] == event_id:
                return event
        return None
    
    def get_status(self):
        """Get the module status"""
        return {
            "status": "ok",
            "initialized": self.initialized,
            "event_count": self.event_count,
            "storage_size": sum(len(str(e)) for e in self.storage["events"])
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        self.storage = {}
        return True


class MockDocumentLibrary:
    """Mock document library implementation for testing"""
    
    def __init__(self):
        self.initialized = False
        self.documents = {}
        self.document_count = 0
    
    def initialize(self, config=None):
        """Initialize the document library"""
        self.initialized = True
        
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
        return results
    
    def get_status(self):
        """Get the module status"""
        return {
            "status": "ok",
            "initialized": self.initialized,
            "document_count": self.document_count
        }
    
    def shutdown(self):
        """Shutdown the module"""
        self.initialized = False
        self.documents = {}
        return True


class MockTCCCSystem:
    """
    Mock TCCC System implementation for testing
    Uses mock modules to avoid dependencies
    """
    
    def __init__(self, config_path=None):
        """Initialize the mock system"""
        self.config_path = config_path
        self.state = SystemState.UNINITIALIZED
        
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
        
        self.running = False
        self.tasks = []
    
    async def initialize(self):
        """Initialize all modules"""
        if self.state != SystemState.UNINITIALIZED:
            return False
        
        self.state = SystemState.INITIALIZING
        
        # Initialize modules
        config = {}
        if self.config_path:
            config_manager = ConfigManager(self.config_path)
            for module_name in self.modules:
                try:
                    module_config = config_manager.get_config(module_name, {})
                    config[module_name] = module_config
                except Exception:
                    config[module_name] = {}
        
        # Initialize each module
        for name, module in self.modules.items():
            module.initialize(config.get(name, {}))
        
        self.state = SystemState.READY
        return True
    
    async def start(self):
        """Start the mock system"""
        if self.state != SystemState.READY:
            return False
        
        self.state = SystemState.RUNNING
        self.running = True
        
        # Start the audio pipeline
        audio = self.modules["audio_pipeline"]
        audio.start_capture("mock_source")
        
        return True
    
    async def stop(self):
        """Stop the mock system"""
        if self.state not in [SystemState.RUNNING, SystemState.PAUSED]:
            return False
        
        self.state = SystemState.SHUTTING_DOWN
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
    
    def get_health_status(self):
        """Get system health status"""
        module_statuses = {}
        for name, module in self.modules.items():
            module_statuses[name] = module.get_status()
        
        return {
            "state": self.state.value,
            "health": {
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


class SystemVerifierEnhanced:
    """
    Enhanced system verifier for comprehensive testing of TCCC.ai
    """
    
    def __init__(self, config_path=None, use_mocks="auto"):
        """
        Initialize the system verifier
        
        Args:
            config_path (str, optional): Path to configuration directory
            use_mocks (str): Whether to use mock implementations:
                             "auto" - use real implementations when available, mocks otherwise
                             "all" - use mock implementations for all modules
                             "none" - use only real implementations
        """
        self.config_path = config_path
        self.use_mocks = use_mocks
        self.system = None
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
            if self.use_mocks == "all":
                logger.info("Using mock system implementation for all modules")
                self.system = MockTCCCSystem(self.config_path)
                self.is_mock_system = True
            else:
                try:
                    logger.info("Attempting to initialize real system implementation")
                    self.system = TCCCSystem(self.config_path)
                    self.is_mock_system = False
                except Exception as e:
                    if self.use_mocks == "none":
                        logger.error(f"Failed to initialize real system: {str(e)}")
                        return False
                    
                    logger.warning(f"Failed to initialize real system: {str(e)}. Falling back to mock implementation.")
                    self.system = MockTCCCSystem(self.config_path)
                    self.is_mock_system = True
            
            # Run verification stages
            print("\n[1/6] Verifying system initialization...")
            await self.verify_initialization()
            
            print("\n[2/6] Verifying individual modules...")
            await self.verify_modules()
            
            print("\n[3/6] Verifying module integration...")
            await self.verify_integration()
            
            print("\n[4/6] Verifying data flow...")
            await self.verify_data_flow()
            
            print("\n[5/6] Verifying error handling...")
            await self.verify_error_handling()
            
            print("\n[6/6] Verifying system performance...")
            await self.verify_performance()
            
            # Clean shutdown
            if self.system:
                print("\nShutting down system...")
                await self.system.stop()
            
            # Print results
            self.timing["end_time"] = time.time()
            self.print_summary()
            
            # Determine overall success
            return all(result["success"] for result in self.verification_results.values())
            
        except KeyboardInterrupt:
            print("\nVerification interrupted")
            if self.system:
                await self.system.stop()
            return False
        except Exception as e:
            print(f"\nVerification error: {str(e)}")
            if self.system:
                await self.system.stop()
            return False
    
    async def verify_initialization(self):
        """
        Verify system initialization
        """
        stage_start = time.time()
        
        try:
            logger.info("Testing system initialization")
            
            # Initialize the system
            result = await self.system.initialize()
            
            # Check if initialization succeeded
            if not result:
                logger.error("System initialization failed")
                self.verification_results[VerificationStage.INITIALIZATION.value] = {
                    "success": False,
                    "details": {
                        "error": "Initialization returned False"
                    }
                }
                return False
            
            # Check system state
            if self.system.state != SystemState.READY:
                logger.error(f"System initialized but in wrong state: {self.system.state}")
                self.verification_results[VerificationStage.INITIALIZATION.value] = {
                    "success": False,
                    "details": {
                        "error": f"System in incorrect state: {self.system.state}, expected: {SystemState.READY}"
                    }
                }
                return False
            
            # Check if all modules were initialized
            health_status = self.system.get_health_status()
            all_modules_ok = True
            module_statuses = {}
            
            for module_name, status in health_status["modules"].items():
                module_status = "ok"
                if isinstance(status, dict):
                    module_status = status.get("status", "unknown")
                    initialized = status.get("initialized", False)
                    if not initialized or module_status != "ok":
                        all_modules_ok = False
                        logger.error(f"Module {module_name} not properly initialized: {status}")
                
                module_statuses[module_name] = module_status
                logger.info(f"Module {module_name} status: {module_status}")
            
            if not all_modules_ok:
                self.verification_results[VerificationStage.INITIALIZATION.value] = {
                    "success": False,
                    "details": {
                        "error": "Not all modules initialized correctly",
                        "module_statuses": module_statuses
                    }
                }
                return False
            
            # Record success
            self.verification_results[VerificationStage.INITIALIZATION.value] = {
                "success": True,
                "details": {
                    "module_statuses": module_statuses,
                    "time_seconds": time.time() - stage_start
                }
            }
            
            logger.info("System initialization verified successfully")
            print("✓ System initialized with all modules reporting ready")
            return True
            
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
                processing_core = self.system.modules["processing_core"]
                test_input = "This is a test transcription for processing core"
                
                # Process a test transcription
                if hasattr(processing_core, "process_transcription"):
                    result = processing_core.process_transcription(test_input)
                    
                    # Verify result structure
                    if (isinstance(result, dict) and
                        "entities" in result and
                        "intent" in result and
                        "sentiment" in result):
                        
                        logger.info("Processing Core test passed")
                        module_results["processing_core"] = {
                            "success": True,
                            "details": {
                                "input": test_input,
                                "entities_count": len(result.get("entities", [])),
                                "intent": result.get("intent", {}).get("name", "unknown"),
                                "sentiment": result.get("sentiment", {}).get("label", "unknown")
                            }
                        }
                        print("✓ Processing Core module verified")
                    else:
                        logger.warning("Processing Core returned unexpected format")
                        module_results["processing_core"] = {
                            "success": False,
                            "details": {
                                "error": "Invalid result format",
                                "received": str(type(result))
                            }
                        }
                        all_modules_ok = False
                        print("❌ Processing Core module verification failed")
                else:
                    logger.warning("Processing Core missing expected methods")
                    module_results["processing_core"] = {
                        "success": False,
                        "details": {
                            "error": "Missing process_transcription method"
                        }
                    }
                    all_modules_ok = False
                    print("❌ Processing Core module verification failed")
                
            except Exception as e:
                logger.error(f"Processing Core test error: {str(e)}")
                module_results["processing_core"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_modules_ok = False
                print("❌ Processing Core module verification failed")
            
            # Test Data Store
            try:
                data_store = self.system.modules["data_store"]
                test_data = {"test_id": "123", "test_value": "test_data", "timestamp": time.time()}
                
                # Store and retrieve test data
                if hasattr(data_store, "store_event") and hasattr(data_store, "get_event"):
                    event_id = data_store.store_event(test_data)
                    retrieved = data_store.get_event(event_id)
                    
                    if retrieved and "data" in retrieved:
                        retrieved_data = retrieved["data"]
                        if (retrieved_data.get("test_id") == test_data["test_id"] and
                            retrieved_data.get("test_value") == test_data["test_value"]):
                            
                            logger.info("Data Store test passed")
                            module_results["data_store"] = {
                                "success": True,
                                "details": {
                                    "event_id": event_id,
                                    "data_matches": True
                                }
                            }
                            print("✓ Data Store module verified")
                        else:
                            logger.warning("Data Store retrieved different data than stored")
                            module_results["data_store"] = {
                                "success": False,
                                "details": {
                                    "error": "Data mismatch",
                                    "stored": test_data,
                                    "retrieved": retrieved_data
                                }
                            }
                            all_modules_ok = False
                            print("❌ Data Store module verification failed")
                    else:
                        logger.warning("Data Store failed to retrieve stored data")
                        module_results["data_store"] = {
                            "success": False,
                            "details": {
                                "error": "Failed to retrieve data",
                                "event_id": event_id,
                                "retrieved": retrieved
                            }
                        }
                        all_modules_ok = False
                        print("❌ Data Store module verification failed")
                else:
                    logger.warning("Data Store missing expected methods")
                    module_results["data_store"] = {
                        "success": False,
                        "details": {
                            "error": "Missing required methods"
                        }
                    }
                    all_modules_ok = False
                    print("❌ Data Store module verification failed")
                
            except Exception as e:
                logger.error(f"Data Store test error: {str(e)}")
                module_results["data_store"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_modules_ok = False
                print("❌ Data Store module verification failed")
            
            # Test LLM Analysis
            try:
                llm_analysis = self.system.modules["llm_analysis"]
                test_input = "Test transcription for LLM analysis"
                
                # Process a test transcription
                if hasattr(llm_analysis, "process_transcription"):
                    result = llm_analysis.process_transcription(test_input)
                    
                    # Verify result structure
                    if (isinstance(result, dict) and
                        "summary" in result):
                        
                        logger.info("LLM Analysis test passed")
                        module_results["llm_analysis"] = {
                            "success": True,
                            "details": {
                                "input": test_input,
                                "has_summary": "summary" in result,
                                "has_topics": "topics" in result
                            }
                        }
                        print("✓ LLM Analysis module verified")
                    else:
                        logger.warning("LLM Analysis returned unexpected format")
                        module_results["llm_analysis"] = {
                            "success": False,
                            "details": {
                                "error": "Invalid result format",
                                "received": str(type(result))
                            }
                        }
                        all_modules_ok = False
                        print("❌ LLM Analysis module verification failed")
                else:
                    logger.warning("LLM Analysis missing expected methods")
                    module_results["llm_analysis"] = {
                        "success": False,
                        "details": {
                            "error": "Missing process_transcription method"
                        }
                    }
                    all_modules_ok = False
                    print("❌ LLM Analysis module verification failed")
                
            except Exception as e:
                logger.error(f"LLM Analysis test error: {str(e)}")
                module_results["llm_analysis"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_modules_ok = False
                print("❌ LLM Analysis module verification failed")
            
            # Test Audio Pipeline & STT Engine (simplified tests)
            for module_name in ["audio_pipeline", "stt_engine", "document_library"]:
                try:
                    module = self.system.modules[module_name]
                    status = module.get_status()
                    
                    if status and (isinstance(status, dict) and status.get("initialized", False)):
                        logger.info(f"{module_name} basic test passed")
                        module_results[module_name] = {
                            "success": True,
                            "details": {
                                "status": status
                            }
                        }
                        print(f"✓ {module_name} module verified")
                    else:
                        logger.warning(f"{module_name} not properly initialized")
                        module_results[module_name] = {
                            "success": False,
                            "details": {
                                "error": "Module not properly initialized",
                                "status": status
                            }
                        }
                        all_modules_ok = False
                        print(f"❌ {module_name} module verification failed")
                
                except Exception as e:
                    logger.error(f"{module_name} test error: {str(e)}")
                    module_results[module_name] = {
                        "success": False,
                        "details": {
                            "error": str(e)
                        }
                    }
                    all_modules_ok = False
                    print(f"❌ {module_name} module verification failed")
            
            # Record results
            self.verification_results[VerificationStage.MODULE_VERIFICATION.value] = {
                "success": all_modules_ok,
                "details": module_results
            }
            
            if all_modules_ok:
                logger.info("All modules verified successfully")
                print("✓ All modules passed individual verification")
            else:
                logger.warning("Some modules failed verification")
                print("❌ Some modules failed individual verification")
            
            return all_modules_ok
            
        except Exception as e:
            logger.error(f"Module verification error: {str(e)}")
            self.verification_results[VerificationStage.MODULE_VERIFICATION.value] = {
                "success": False,
                "details": {
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            }
            return False
        finally:
            self.timing["stages"][VerificationStage.MODULE_VERIFICATION.value] = time.time() - stage_start
    
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
            
            # Test ProcessingCore ↔ DataStore integration
            try:
                processing_core = self.system.modules["processing_core"]
                data_store = self.system.modules["data_store"]
                
                # Process a test transcription
                test_input = "Test integration between ProcessingCore and DataStore"
                processing_result = processing_core.process_transcription(test_input)
                
                # Store the processing result
                event_id = data_store.store_event(processing_result)
                
                # Retrieve the stored result
                retrieved = data_store.get_event(event_id)
                
                if retrieved and "data" in retrieved:
                    retrieved_data = retrieved["data"]
                    
                    # Verify that key parts of the processing result are preserved
                    integration_ok = (
                        "entities" in retrieved_data and
                        "intent" in retrieved_data and
                        "sentiment" in retrieved_data
                    )
                    
                    if integration_ok:
                        logger.info("ProcessingCore ↔ DataStore integration test passed")
                        integration_results["processing_core_datastore"] = {
                            "success": True,
                            "details": {
                                "event_id": event_id
                            }
                        }
                        print("✓ ProcessingCore ↔ DataStore integration verified")
                    else:
                        logger.warning("ProcessingCore ↔ DataStore integration test failed")
                        integration_results["processing_core_datastore"] = {
                            "success": False,
                            "details": {
                                "error": "Data structure mismatch",
                                "missing_fields": [f for f in ["entities", "intent", "sentiment"] if f not in retrieved_data]
                            }
                        }
                        all_tests_passed = False
                        print("❌ ProcessingCore ↔ DataStore integration failed")
                else:
                    logger.warning("Failed to retrieve stored data in integration test")
                    integration_results["processing_core_datastore"] = {
                        "success": False,
                        "details": {
                            "error": "Failed to retrieve data",
                            "event_id": event_id
                        }
                    }
                    all_tests_passed = False
                    print("❌ ProcessingCore ↔ DataStore integration failed")
                
            except Exception as e:
                logger.error(f"ProcessingCore ↔ DataStore integration test error: {str(e)}")
                integration_results["processing_core_datastore"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_tests_passed = False
                print("❌ ProcessingCore ↔ DataStore integration failed")
            
            # Test STTEngine ↔ ProcessingCore integration
            try:
                stt_engine = self.system.modules["stt_engine"]
                processing_core = self.system.modules["processing_core"]
                
                # Create a mock audio segment (if using real system)
                if not self.is_mock_system:
                    audio_segment = MockAudioSegment()
                else:
                    # Get a segment from the mock audio pipeline
                    audio_pipeline = self.system.modules["audio_pipeline"]
                    audio_stream = audio_pipeline.get_audio_stream()
                    audio_segment = audio_stream.get_segment()
                
                # Transcribe the audio segment
                transcription = stt_engine.transcribe_segment(audio_segment)
                
                # Process the transcription
                processing_result = processing_core.process_transcription(transcription)
                
                # Verify the result
                if (isinstance(processing_result, dict) and
                    "entities" in processing_result and
                    "intent" in processing_result and
                    "sentiment" in processing_result and
                    "raw_text" in processing_result):
                    
                    logger.info("STTEngine ↔ ProcessingCore integration test passed")
                    integration_results["stt_processing"] = {
                        "success": True,
                        "details": {
                            "transcription_text": transcription["text"] if isinstance(transcription, dict) else str(transcription),
                            "intent": processing_result.get("intent", {}).get("name", "unknown")
                        }
                    }
                    print("✓ STTEngine ↔ ProcessingCore integration verified")
                else:
                    logger.warning("STTEngine ↔ ProcessingCore integration test failed")
                    integration_results["stt_processing"] = {
                        "success": False,
                        "details": {
                            "error": "Invalid processing result",
                            "missing_fields": [f for f in ["entities", "intent", "sentiment", "raw_text"] if f not in processing_result]
                        }
                    }
                    all_tests_passed = False
                    print("❌ STTEngine ↔ ProcessingCore integration failed")
                
            except Exception as e:
                logger.error(f"STTEngine ↔ ProcessingCore integration test error: {str(e)}")
                integration_results["stt_processing"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_tests_passed = False
                print("❌ STTEngine ↔ ProcessingCore integration failed")
            
            # Test LLMAnalysis ↔ DocumentLibrary integration
            try:
                llm_analysis = self.system.modules["llm_analysis"]
                document_library = self.system.modules["document_library"]
                
                # Get test document
                docs = []
                if hasattr(document_library, "search_documents"):
                    docs = document_library.search_documents("test")
                
                if docs:
                    test_doc = docs[0]
                    
                    # Process document content with LLM
                    analysis_result = llm_analysis.process_transcription(test_doc["content"])
                    
                    # Verify the result
                    if isinstance(analysis_result, dict) and "summary" in analysis_result:
                        logger.info("LLMAnalysis ↔ DocumentLibrary integration test passed")
                        integration_results["llm_document"] = {
                            "success": True,
                            "details": {
                                "document_title": test_doc["title"],
                                "has_summary": True
                            }
                        }
                        print("✓ LLMAnalysis ↔ DocumentLibrary integration verified")
                    else:
                        logger.warning("LLMAnalysis ↔ DocumentLibrary integration test failed")
                        integration_results["llm_document"] = {
                            "success": False,
                            "details": {
                                "error": "Invalid analysis result",
                                "missing_fields": ["summary" if "summary" not in analysis_result else None]
                            }
                        }
                        all_tests_passed = False
                        print("❌ LLMAnalysis ↔ DocumentLibrary integration failed")
                else:
                    # Add a document
                    doc_id = document_library.add_document(
                        "test_integration",
                        "Test Integration Document",
                        "This is a test document for integration between LLM and Document Library"
                    )
                    
                    # Get the document
                    if hasattr(document_library, "get_document"):
                        test_doc = document_library.get_document(doc_id)
                        
                        # Process document content with LLM
                        analysis_result = llm_analysis.process_transcription(test_doc["content"])
                        
                        # Verify the result
                        if isinstance(analysis_result, dict) and "summary" in analysis_result:
                            logger.info("LLMAnalysis ↔ DocumentLibrary integration test passed")
                            integration_results["llm_document"] = {
                                "success": True,
                                "details": {
                                    "document_title": test_doc["title"],
                                    "has_summary": True
                                }
                            }
                            print("✓ LLMAnalysis ↔ DocumentLibrary integration verified")
                        else:
                            logger.warning("LLMAnalysis ↔ DocumentLibrary integration test failed")
                            integration_results["llm_document"] = {
                                "success": False,
                                "details": {
                                    "error": "Invalid analysis result"
                                }
                            }
                            all_tests_passed = False
                            print("❌ LLMAnalysis ↔ DocumentLibrary integration failed")
                    else:
                        logger.warning("DocumentLibrary missing get_document method")
                        integration_results["llm_document"] = {
                            "success": False,
                            "details": {
                                "error": "Missing get_document method"
                            }
                        }
                        all_tests_passed = False
                        print("❌ LLMAnalysis ↔ DocumentLibrary integration failed")
                
            except Exception as e:
                logger.error(f"LLMAnalysis ↔ DocumentLibrary integration test error: {str(e)}")
                integration_results["llm_document"] = {
                    "success": False,
                    "details": {
                        "error": str(e)
                    }
                }
                all_tests_passed = False
                print("❌ LLMAnalysis ↔ DocumentLibrary integration failed")
            
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
            
            print("System started, running for 10 seconds to verify data flow...")
            
            # Record initial event count
            data_store = self.system.modules["data_store"]
            initial_event_count = data_store.get_status().get("event_count", 0)
            
            # Let the system run for a short time
            await asyncio.sleep(10)
            
            # Check for data flow by counting events
            final_event_count = data_store.get_status().get("event_count", 0)
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
        
        if not self.system or self.system.state != SystemState.RUNNING:
            logger.error("System not running, skipping error handling verification")
            return False
        
        try:
            logger.info("Testing error handling")
            
            # Record initial error count
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
            health_status = self.system.get_health_status()
            final_error_count = health_status["health"]["error_count"]
            errors_detected = final_error_count - initial_error_count
            
            # Check that system is still running despite error
            system_still_running = self.system.state == SystemState.RUNNING
            
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
        
        if not self.system or self.system.state != SystemState.RUNNING:
            logger.error("System not running, skipping performance verification")
            return False
        
        try:
            logger.info("Testing system performance")
            
            # Get initial resource usage
            health_status = self.system.get_health_status()
            initial_resources = health_status["health"]["resource_usage"]
            
            # Get initial event count
            data_store = self.system.modules["data_store"]
            initial_event_count = data_store.get_status().get("event_count", 0)
            
            # Run for performance measurement period
            performance_duration = 5  # seconds
            print(f"Measuring performance for {performance_duration} seconds...")
            await asyncio.sleep(performance_duration)
            
            # Get final resource usage
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


async def main():
    """Main entry point for the enhanced verification script"""
    parser = argparse.ArgumentParser(description='Enhanced verification for TCCC.ai system')
    parser.add_argument('--config', type=str, help='Path to configuration directory')
    parser.add_argument('--mock', choices=['auto', 'all', 'none'], default='auto',
                       help='Use mock implementations: "auto" (default), "all", or "none"')
    args = parser.parse_args()
    
    # Create and run the verifier
    verifier = SystemVerifierEnhanced(args.config, args.mock)
    success = await verifier.run_verification()
    
    return 0 if success else 1


if __name__ == "__main__":
    # Set up signal handling for clean shutdown
    loop = asyncio.get_event_loop()
    signals = (signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(asyncio.shield(
            asyncio.sleep(0))))  # Allows for clean shutdown
    
    try:
        sys.exit(loop.run_until_complete(main()))
    except KeyboardInterrupt:
        print("Verification interrupted")
        sys.exit(1)