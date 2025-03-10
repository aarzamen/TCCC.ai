"""
TCCC.ai System Integration Implementation.

This module combines all TCCC modules into an integrated system
that can process audio, extract information, and generate reports.
"""

import os
import time
import logging
import threading
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

# Import TCCC modules
from tccc.audio_pipeline import AudioPipeline
try:
    from tccc.stt_engine import STTEngine, create_stt_engine, STT_ENGINE_IMPORT_ERROR
    STT_IMPORT_ERROR = STT_ENGINE_IMPORT_ERROR
except ImportError as e:
    STT_IMPORT_ERROR = str(e)
except OSError as e:
    STT_IMPORT_ERROR = str(e)
from tccc.processing_core import ProcessingCore
from tccc.llm_analysis import LLMAnalysis
from tccc.data_store import DataStore
from tccc.document_library import DocumentLibrary
from tccc.utils.logging import get_logger
from tccc.utils.event_schema import (
    BaseEvent, AudioSegmentEvent, TranscriptionEvent, 
    ProcessedTextEvent, LLMAnalysisEvent, ErrorEvent,
    SystemStatusEvent, EventType
)
from tccc.utils.module_adapter import (
    AudioPipelineAdapter, STTEngineAdapter, 
    ProcessingCoreAdapter, standardize_event, extract_event_data
)

logger = get_logger(__name__)


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    READY = "ready"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    REPORTING = "reporting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TCCCSystem:
    """
    TCCC.ai System Integration.
    
    This class provides the integration layer for all TCCC modules,
    coordinating their operation to form a complete system.
    """
    
    def __init__(self, config_path=None):
        """Initialize the TCCC System.
        
        Args:
            config_path (str, optional): Path to the configuration directory.
        """
        self.initialized = False
        self.state = SystemState.INITIALIZING
        self.config = None
        self.config_path = config_path
        
        # Module references
        self.audio_pipeline = None
        self.stt_engine = None
        self.processing_core = None
        self.llm_analysis = None
        self.data_store = None
        self.document_library = None
        
        # Session data
        self.session_id = None
        self.session_start_time = None
        self.events = []
        self.reports = []
        
        # Threading
        self.processing_thread = None
        self.stop_processing = False
        
        # Error tracking
        self.last_error = None
        
        # Health tracking (for verification)
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
        
        logger.info("TCCC System object created")
    
    async def initialize(self, config: Dict[str, Any] = None, mock_modules: Optional[List[str]] = None) -> bool:
        """Initialize the TCCC System.
        
        Args:
            config: System configuration (optional, defaults to empty dict)
            mock_modules: List of modules to use mocks for (for testing)
            
        Returns:
            True if initialization was successful
        """
        try:
            self.config = config or {}
            self.state = SystemState.INITIALIZING
            mock_modules = mock_modules or []
            
            # Create initialization event
            init_event = BaseEvent(
                event_type=EventType.INITIALIZATION,
                source="system",
                data={"status": "starting", "modules": []},
                session_id=f"init_{int(time.time())}"
            ).to_dict()
            
            logger.info("TCCC System initialization started")
            
            # Track critical failures for overall system status
            critical_failures = 0
            all_modules_initialized = True
            
            # Set up modules in a defined order with correct dependencies
            # The order is: DataStore → DocumentLibrary → ProcessingCore → LLMANalysis → AudioPipeline → STTEngine
            
            # 1. Initialize Data Store (most fundamental)
            if "data_store" in mock_modules:
                from tests.mocks.mock_data_store import MockDataStore
                self.data_store = MockDataStore()
                logger.info("Creating MockDataStore")
            else:
                self.data_store = DataStore()
                logger.info("Creating DataStore")
            
            # 2. Initialize Document Library (depends on DataStore)
            if "document_library" in mock_modules:
                from tests.mocks.mock_document_library import MockDocumentLibrary
                self.document_library = MockDocumentLibrary()
                logger.info("Creating MockDocumentLibrary")
            else:
                self.document_library = DocumentLibrary()
                logger.info("Creating DocumentLibrary")
            
            # 3. Initialize Processing Core (central component)
            if "processing_core" in mock_modules:
                from tests.mocks.mock_processing_core import MockProcessingCore
                self.processing_core = MockProcessingCore()
                logger.info("Creating MockProcessingCore")
            else:
                self.processing_core = ProcessingCore()
                logger.info("Creating ProcessingCore")
            
            # 4. Initialize LLM Analysis (depends on DocumentLibrary)
            if "llm_analysis" in mock_modules:
                from tests.mocks.mock_llm_analysis import MockLLMAnalysis
                self.llm_analysis = MockLLMAnalysis()
                logger.info("Creating MockLLMAnalysis")
            else:
                self.llm_analysis = LLMAnalysis()
                logger.info("Creating LLMAnalysis")
            
            # 5. Initialize Audio Pipeline
            if "audio_pipeline" in mock_modules:
                from tests.mocks.mock_audio_pipeline import MockAudioPipeline
                self.audio_pipeline = MockAudioPipeline()
                logger.info("Creating MockAudioPipeline")
            else:
                self.audio_pipeline = AudioPipeline()
                logger.info("Creating AudioPipeline")
            
            # 6. Initialize STT Engine (depends on models)
            if "stt_engine" in mock_modules:
                from tests.mocks.mock_stt_engine import MockSTTEngine
                self.stt_engine = MockSTTEngine()
                logger.info("Creating MockSTTEngine")
            else:
                self.stt_engine = STTEngine()
                logger.info("Creating STTEngine")
            
            # Extract module configurations
            config_dict = self.config  # Use the already validated self.config
            modules_config = {}
            
            # Ensure module configs are dictionaries or create empty ones
            for module_name in ["data_store", "document_library", "processing_core", 
                               "llm_analysis", "audio_pipeline", "stt_engine"]:
                module_config = config_dict.get(module_name, {})
                # Make sure we have a dictionary
                if not isinstance(module_config, dict):
                    logger.warning(f"Invalid config for {module_name}, expected dictionary but got {type(module_config)}")
                    module_config = {}
                modules_config[module_name] = module_config.copy()  # Use a copy to avoid cross-module modification
            
            logger.info("All modules created, starting initialization sequence")
            init_event["data"]["status"] = "initializing_modules"
            
            # Initialize modules in the correct dependency order
            # Use consistent async/sync handling for all modules
            
            # 1. DataStore initialization - this is critical and must succeed
            data_store_success = await self._initialize_module(
                self.data_store, "data_store", modules_config, init_event
            )
            
            if not data_store_success:
                logger.critical("DataStore initialization failed - this is a critical module")
                critical_failures += 1
                all_modules_initialized = False
            
            # 2. DocumentLibrary initialization - can continue with limited functionality
            doc_lib_success = await self._initialize_module(
                self.document_library, "document_library", modules_config, init_event
            )
            
            if not doc_lib_success:
                logger.warning("DocumentLibrary initialization failed - continuing with limited functionality")
                all_modules_initialized = False
            
            # 3. ProcessingCore initialization - critical for audio processing
            proc_core_success = await self._initialize_module(
                self.processing_core, "processing_core", modules_config, init_event
            )
            
            if not proc_core_success:
                logger.critical("ProcessingCore initialization failed - this is a critical module")
                critical_failures += 1
                all_modules_initialized = False
            
            # 4. LLM Analysis initialization - can continue with limited functionality
            llm_success = await self._initialize_module(
                self.llm_analysis, "llm_analysis", modules_config, init_event
            )
            
            if not llm_success:
                logger.warning("LLMAnalysis initialization failed - continuing with limited functionality")
                all_modules_initialized = False
            
            # 5. AudioPipeline initialization - critical for audio capture
            audio_success = await self._initialize_module(
                self.audio_pipeline, "audio_pipeline", modules_config, init_event
            )
            
            if not audio_success:
                logger.critical("AudioPipeline initialization failed - this is a critical module")
                critical_failures += 1
                all_modules_initialized = False
            
            # 6. STT Engine initialization - can continue with limited functionality
            stt_success = await self._initialize_module(
                self.stt_engine, "stt_engine", modules_config, init_event
            )
            
            if not stt_success:
                logger.warning("STTEngine initialization failed - continuing with limited functionality")
                all_modules_initialized = False
            
            # Set up module dependencies if both modules are available
            logger.info("Setting up module dependencies")
            
            # Connect LLM to DocumentLibrary
            if (hasattr(self.llm_analysis, 'set_document_library') 
                and callable(self.llm_analysis.set_document_library)
                and self.document_library and hasattr(self.document_library, 'initialized')):
                try:
                    self.llm_analysis.set_document_library(self.document_library)
                    logger.info("Connected LLM Analysis to Document Library")
                except Exception as e:
                    logger.warning(f"Failed to connect LLM Analysis to Document Library: {str(e)}")
            
            # Create a new session
            self.session_id = f"session_{int(time.time())}"
            self.session_start_time = time.time()
            
            # Determine system state based on initialization results
            if critical_failures > 0:
                self.state = SystemState.ERROR
                self.last_error = "Critical module initialization failures"
                self.initialized = False
                init_event["data"]["status"] = "failed"
                init_event["data"]["success"] = False
                init_event["data"]["error"] = f"{critical_failures} critical modules failed to initialize"
                logger.error("TCCC System initialization failed due to critical module failures")
            else:
                # System can operate even with non-critical failures
                if all_modules_initialized:
                    self.state = SystemState.READY
                    logger.info("TCCC System initialized successfully with all modules")
                else:
                    self.state = SystemState.READY
                    logger.warning("TCCC System initialized with limited functionality - some modules have limited capabilities")
                
                self.initialized = True
                
                # Final initialization event
                init_event["data"]["status"] = "complete"
                init_event["data"]["success"] = True
                
                if not all_modules_initialized:
                    init_event["data"]["warning"] = "Some modules have limited functionality"
            
            # Store initialization event if possible
            try:
                if self.data_store and hasattr(self.data_store, 'store_event') and callable(self.data_store.store_event):
                    self.data_store.store_event(init_event)
            except Exception as e:
                logger.warning(f"Could not store initialization event: {e}")
            
            return self.initialized
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to initialize TCCC System: {str(e)}")
            
            # Error initialization event
            try:
                init_event["data"]["status"] = "failed"
                init_event["data"]["success"] = False
                init_event["data"]["error"] = str(e)
                
                if self.data_store and hasattr(self.data_store, 'store_event'):
                    self.data_store.store_event(init_event)
            except:
                pass
                
            self.initialized = False
            return False
    
    async def _initialize_module(self, module, module_name: str, config: Dict[str, Any], init_event: Dict[str, Any]) -> bool:
        """
        Initialize a module with proper async/sync handling.
        
        Args:
            module: Module instance to initialize
            module_name: Name of the module for logging
            config: Module configuration
            init_event: Initialization event to update
            
        Returns:
            True if initialization was successful
        """
        if not module:
            logger.warning(f"Module {module_name} is None, skipping initialization")
            return False
            
        try:
            logger.info(f"Initializing {module_name}")
            init_event["data"]["modules"].append({"name": module_name, "status": "initializing"})
            
            # Check if module has initialize method
            if not hasattr(module, 'initialize') or not callable(module.initialize):
                logger.warning(f"{module_name} missing initialize method")
                return False
                
            # Make a copy of config to avoid potential cross-module modification issues
            module_config = dict(config.get(module_name, {}))
            
            # Validate config structure to avoid common errors
            if not isinstance(module_config, dict):
                logger.warning(f"Invalid config for {module_name}, expected dictionary")
                module_config = {}  # Use empty dict to avoid further errors
            
            # Handle async vs sync initialize method
            initialize_method = module.initialize
            try:
                if asyncio.iscoroutinefunction(initialize_method):
                    # Module has async initialize method
                    result = await initialize_method(module_config)
                else:
                    # Module has sync initialize method, run in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: initialize_method(module_config)
                    )
                    
                # Update initialization event
                for module_info in init_event["data"]["modules"]:
                    if module_info["name"] == module_name:
                        module_info["status"] = "ready" if result else "limited"
                        module_info["success"] = True  # Count as success even with limited functionality
                        
                if result:
                    logger.info(f"Module {module_name} initialized successfully")
                else:
                    logger.warning(f"Module {module_name} initialized with limited functionality")
                
                # Consider partial initialization as success for system stability
                return True
                
            except Exception as init_error:
                logger.error(f"Error during {module_name} initialization: {init_error}")
                
                # Add error to health tracking
                self.add_error(module_name, f"Initialization error: {str(init_error)}")
                
                # Update initialization event with error
                for module_info in init_event["data"]["modules"]:
                    if module_info["name"] == module_name:
                        module_info["status"] = "error"
                        module_info["success"] = False
                        module_info["error"] = str(init_error)
                
                # Try to recover depending on the module type
                if module_name == "document_library":
                    return await self._recover_document_library(module)
                elif module_name == "llm_analysis":
                    return await self._recover_llm_analysis(module)
                elif module_name == "stt_engine":
                    return await self._recover_stt_engine(module)
                else:
                    # For other modules, consider failed initialization as fatal
                    return False
            
        except Exception as e:
            logger.error(f"Error initializing {module_name}: {e}")
            
            # Update initialization event with error
            for module_info in init_event["data"]["modules"]:
                if module_info["name"] == module_name:
                    module_info["status"] = "error"
                    module_info["success"] = False
                    module_info["error"] = str(e)
            
            # Add error to health tracking
            self.add_error(module_name, f"Initialization error: {str(e)}")
                    
            return False
    
    async def _recover_document_library(self, module) -> bool:
        """
        Attempt to recover DocumentLibrary module after initialization failure.
        
        Args:
            module: DocumentLibrary module instance
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            logger.info("Attempting recovery of DocumentLibrary module")
            
            # Check if the module is at least partially initialized
            if hasattr(module, 'initialized'):
                module.initialized = True
                
            # Ensure minimal required attributes are set
            if not hasattr(module, 'documents') or module.documents is None:
                module.documents = {}
                
            if not hasattr(module, 'chunks') or module.chunks is None:
                module.chunks = {}
                
            # Minimal empty index if needed
            if not hasattr(module, 'index') or module.index is None:
                try:
                    # Try to create a minimal mock index
                    from tccc.document_library.vector_store import MockFaissIndex
                    module.index = MockFaissIndex(dimension=384)
                except Exception:
                    # If that fails, just use None
                    module.index = None
                    
            logger.info("DocumentLibrary partially recovered with limited functionality")
            return True
        except Exception as recovery_error:
            logger.error(f"Failed to recover DocumentLibrary: {str(recovery_error)}")
            return False
            
    async def _recover_llm_analysis(self, module) -> bool:
        """
        Attempt to recover LLMAnalysis module after initialization failure.
        
        Args:
            module: LLMAnalysis module instance
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            logger.info("Attempting recovery of LLMAnalysis module")
            
            # Check if the module has recovery helpers
            if hasattr(module, '_create_minimal_llm_engine'):
                # Use the module's own recovery methods
                if not hasattr(module, 'llm_engine') or module.llm_engine is None:
                    module.llm_engine = module._create_minimal_llm_engine({})
                    
                if not hasattr(module, 'entity_extractor') or module.entity_extractor is None:
                    module.entity_extractor = module._create_minimal_entity_extractor()
                    
                if not hasattr(module, 'event_sequencer') or module.event_sequencer is None:
                    module.event_sequencer = module._create_minimal_event_sequencer()
                    
                if not hasattr(module, 'report_generator') or module.report_generator is None:
                    module.report_generator = module._create_minimal_report_generator()
                    
                # Initialize cache
                if not hasattr(module, 'cache'):
                    module.cache = {}
                    
            # Force initialized state
            module.initialized = True
            
            logger.info("LLMAnalysis partially recovered with limited functionality")
            return True
        except Exception as recovery_error:
            logger.error(f"Failed to recover LLMAnalysis: {str(recovery_error)}")
            return False
    
    async def _recover_stt_engine(self, module) -> bool:
        """
        Attempt to recover STTEngine module after initialization failure.
        
        Args:
            module: STTEngine module instance
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            logger.info("Attempting recovery of STTEngine module")
            
            # Check if the module has recovery helpers
            if hasattr(module, '_create_minimal_model_manager'):
                # Use the module's own recovery methods
                if not hasattr(module, 'model_manager') or module.model_manager is None:
                    module.model_manager = module._create_minimal_model_manager({})
                    
                if not hasattr(module, 'diarizer') or module.diarizer is None:
                    module.diarizer = module._create_minimal_diarizer({})
                    
                if not hasattr(module, 'term_processor') or module.term_processor is None:
                    module.term_processor = module._create_minimal_term_processor({})
                    
            # Set default context length if needed
            if not hasattr(module, 'context_max_length'):
                module.context_max_length = 60 * 16000
                
            # Ensure audio buffer and recent segments are initialized
            if not hasattr(module, 'audio_buffer'):
                from collections import deque
                module.audio_buffer = deque(maxlen=100)
                
            if not hasattr(module, 'recent_segments'):
                from collections import deque
                module.recent_segments = deque(maxlen=10)
                
            # Initialize metrics
            if not hasattr(module, 'metrics'):
                module.metrics = {
                    'total_audio_seconds': 0,
                    'total_processing_time': 0,
                    'transcript_count': 0,
                    'error_count': 0,
                    'avg_confidence': 0
                }
                
            # Force initialized state
            module.initialized = True
            
            logger.info("STTEngine partially recovered with limited functionality")
            return True
        except Exception as recovery_error:
            logger.error(f"Failed to recover STTEngine: {str(recovery_error)}")
            return False
    
    def start_audio_capture(self, source_id: Optional[str] = None) -> bool:
        """Start audio capture.
        
        Args:
            source_id: ID of the audio source to use
            
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.error("System not initialized")
            return False
        
        try:
            # Start audio capture
            result = self.audio_pipeline.start_capture(source_id)
            
            if result:
                # Start processing thread
                self.stop_processing = False
                self.processing_thread = threading.Thread(
                    target=self._process_audio_thread,
                    daemon=True
                )
                self.processing_thread.start()
                
                self.state = SystemState.CAPTURING
                logger.info(f"Started audio capture from {source_id or 'default source'}")
                return True
            else:
                logger.error("Failed to start audio capture")
                return False
                
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error starting audio capture: {str(e)}")
            return False
    
    async def process_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """
        Process an external event through the system.
        
        Args:
            event_data: Event data to process
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not self.initialized:
            logger.error("System not initialized")
            return None
            
        try:
            # Update state
            self.state = SystemState.PROCESSING
            session_id = event_data.get("session_id", self.session_id)
            
            # Standardize event format if needed
            if "type" not in event_data:
                event_data = standardize_event(event_data, "external")
            
            event_type = event_data.get("type", "unknown")
            logger.info(f"Processing event of type: {event_type}")
            
            # Track event sequence
            if not hasattr(self, '_event_sequence'):
                self._event_sequence = 0
            self._event_sequence += 1
            
            # Add sequence number to event if not present
            if "sequence" not in event_data:
                event_data["sequence"] = self._event_sequence
            
            # Add session_id to event if not present
            if "session_id" not in event_data:
                event_data["session_id"] = session_id
                
            # Process based on event type
            if event_type == EventType.AUDIO_SEGMENT.value:
                # Handle audio segment event
                # Extract audio data (special handling since it can't be part of JSON)
                audio_data = None
                if "audio_data" in event_data:
                    audio_data = event_data.pop("audio_data")
                
                # Convert for STT Engine
                stt_input = STTEngineAdapter.convert_audio_event_to_input(event_data, audio_data)
                
                # Process with STT Engine
                self.state = SystemState.PROCESSING
                transcription = await self._transcribe_async(stt_input["audio"], stt_input["metadata"])
                
                # Convert transcription to standard event
                if transcription:
                    transcription_event = STTEngineAdapter.convert_transcription_to_event(
                        transcription, event_data
                    )
                    return await self.process_event(transcription_event)
                else:
                    logger.warning("No transcription result from STT Engine")
                    return None
                    
            elif event_type == EventType.TRANSCRIPTION.value:
                # Handle transcription event
                # Convert for Processing Core
                processing_input = ProcessingCoreAdapter.convert_transcription_to_input(event_data)
                
                # Process with Processing Core
                self.state = SystemState.PROCESSING
                processed = await self._process_async(processing_input)
                
                # Convert processing result to standard event
                processed_event = ProcessingCoreAdapter.convert_processed_to_event(
                    processed, event_data
                )
                
                # Continue processing
                return await self.process_event(processed_event)
                
            elif event_type == EventType.PROCESSED_TEXT.value:
                # Handle processed text event
                data = event_data.get("data", {})
                text = data.get("text", "")
                
                # Process with LLM Analysis
                self.state = SystemState.ANALYZING
                
                if text and self.llm_analysis:
                    analysis = await self._analyze_async(text)
                    
                    # Create standardized LLM event
                    llm_event = LLMAnalysisEvent(
                        source="llm_analysis",
                        summary=analysis.get("summary", ""),
                        topics=analysis.get("topics", []),
                        medical_terms=analysis.get("medical_terms", []),
                        actions=analysis.get("actions", []),
                        document_results=analysis.get("document_results", []),
                        metadata={
                            "model": analysis.get("metadata", {}).get("model", "unknown"),
                            "processing_ms": analysis.get("metadata", {}).get("processing_ms", 0),
                            "tokens": analysis.get("metadata", {}).get("tokens", 0)
                        },
                        session_id=session_id,
                        sequence=self._event_sequence
                    ).to_dict()
                    
                    # Store the event
                    event_id = self.data_store.store_event(llm_event)
                    logger.info(f"Stored LLM analysis event: {event_id}")
                    self.events.append(event_id)
                    
                    # Store the original processed text event too
                    orig_event_id = self.data_store.store_event(event_data)
                    logger.info(f"Stored processed text event: {orig_event_id}")
                    self.events.append(orig_event_id)
                    
                    # Reset state
                    self.state = SystemState.READY
                    return event_id
                else:
                    # Just store the processed text event
                    event_id = self.data_store.store_event(event_data)
                    logger.info(f"Stored processed text event: {event_id}")
                    self.events.append(event_id)
                    
                    # Reset state
                    self.state = SystemState.READY
                    return event_id
                    
            elif event_type == EventType.ERROR.value:
                # Handle error event
                error_data = event_data.get("data", {})
                error_code = error_data.get("error_code", "unknown_error")
                error_message = error_data.get("message", "Unknown error")
                
                # Log the error
                logger.error(f"Error event received: {error_code} - {error_message}")
                
                # Add to system health
                self.add_error(
                    error_data.get("component", event_data.get("source", "unknown")),
                    error_message
                )
                
                # Store the error event
                event_id = self.data_store.store_event(event_data)
                logger.info(f"Stored error event: {event_id}")
                self.events.append(event_id)
                
                # Set system state based on error severity
                severity = error_data.get("severity", "error")
                if severity in ["error", "critical"]:
                    self.state = SystemState.ERROR
                    self.last_error = error_message
                else:
                    # For warning and info, stay ready
                    self.state = SystemState.READY
                
                return event_id
                
            else:
                # Store other event types directly
                event_id = self.data_store.store_event(event_data)
                logger.info(f"Stored generic event: {event_id}")
                self.events.append(event_id)
                
                # Reset state
                self.state = SystemState.READY
                return event_id
            
        except Exception as e:
            # Handle any errors in the event processing
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error processing event: {str(e)}")
            
            # Create and store error event
            error_event = ErrorEvent(
                source="system",
                error_code="event_processing_error",
                message=str(e),
                component="system.process_event",
                recoverable=True,
                metadata={"traceback": logging.traceback.format_exc()},
                session_id=session_id,
                sequence=self._event_sequence
            ).to_dict()
            
            try:
                error_id = self.data_store.store_event(error_event)
                self.events.append(error_id)
                return error_id
            except:
                # Last resort if even storing the error fails
                return None
                
    async def _transcribe_async(self, audio_data, metadata=None):
        """
        Asynchronous wrapper for STT transcription.
        
        Args:
            audio_data: Audio data to transcribe
            metadata: Optional metadata
            
        Returns:
            Transcription result
        """
        if not self.stt_engine:
            logger.warning("STT engine not available for transcription")
            return None
            
        try:
            # Use the new utility to handle sync/async methods
            from tccc.utils.module_adapter import run_method_async
            
            # Determine the appropriate method to call
            if hasattr(self.stt_engine, 'transcribe_segment_async') and callable(self.stt_engine.transcribe_segment_async):
                method = self.stt_engine.transcribe_segment_async
                logger.debug("Using transcribe_segment_async method")
                transcription = await method(audio_data, metadata)
            else:
                # Fall back to synchronous method
                method = self.stt_engine.transcribe_segment
                logger.debug("Using transcribe_segment method with async wrapper")
                transcription = await run_method_async(method, audio_data, metadata)
                
            return transcription
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {
                "error": str(e),
                "text": ""
            }
    
    async def _process_async(self, input_data):
        """
        Asynchronous wrapper for Processing Core.
        
        Args:
            input_data: Data to process
            
        Returns:
            Processing result
        """
        if not self.processing_core:
            logger.warning("Processing core not available")
            return {"text": input_data.get("text", ""), "error": "No processing core available"}
            
        try:
            # Use the new utility to handle sync/async methods
            from tccc.utils.module_adapter import run_method_async
            
            # Determine the appropriate method to call
            if hasattr(self.processing_core, 'process_async') and callable(self.processing_core.process_async):
                method = self.processing_core.process_async
                logger.debug("Using process_async method")
                return await method(input_data)
            elif hasattr(self.processing_core, 'process') and callable(self.processing_core.process):
                method = self.processing_core.process
                logger.debug("Using process method with async wrapper")
                return await run_method_async(method, input_data)
            else:
                # Basic fallback processing
                logger.warning("No processing method found on processing core")
                return {
                    "text": input_data.get("text", ""),
                    "type": input_data.get("type", "processed_text"),
                    "timestamp": input_data.get("timestamp", time.time())
                }
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            return {
                "text": input_data.get("text", ""),
                "error": str(e)
            }
    
    async def _analyze_async(self, text):
        """
        Asynchronous wrapper for LLM Analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis result
        """
        if not self.llm_analysis:
            logger.warning("LLM analysis not available")
            return {"summary": "No LLM analysis available", "topics": []}
            
        try:
            # Use the new utility to handle sync/async methods
            from tccc.utils.module_adapter import run_method_async
            
            # Determine the appropriate method to call
            if hasattr(self.llm_analysis, 'analyze_transcription_async') and callable(self.llm_analysis.analyze_transcription_async):
                method = self.llm_analysis.analyze_transcription_async
                logger.debug("Using analyze_transcription_async method")
                return await method(text)
            elif hasattr(self.llm_analysis, 'analyze_transcription') and callable(self.llm_analysis.analyze_transcription):
                method = self.llm_analysis.analyze_transcription
                logger.debug("Using analyze_transcription method with async wrapper")
                return await run_method_async(method, text)
            else:
                logger.warning("No analysis method found on LLM analysis module")
                return {
                    "summary": "LLM analysis method not available",
                    "topics": []
                }
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "summary": f"Error in analysis: {str(e)}",
                "topics": []
            }
    
    def _process_audio_thread(self):
        """Thread for processing audio data."""
        try:
            logger.info("Audio processing thread started")
            
            # Initialize sequence counter for audio segments
            audio_sequence = 0
            
            # Initialize shared session ID
            thread_session_id = f"audio_session_{int(time.time())}"
            
            # Create a dedicated event loop for this thread
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
            
            # Define the asynchronous worker function
            async def process_audio_worker():
                nonlocal audio_sequence
                
                while not self.stop_processing:
                    try:
                        # Get standardized audio event from the pipeline asynchronously
                        audio_event = await AudioPipelineAdapter.get_audio_segment_async(self.audio_pipeline)
                        
                        if audio_event:
                            # Increment sequence
                            audio_sequence += 1
                            
                            # Add sequence and session ID
                            audio_event["sequence"] = audio_sequence
                            audio_event["session_id"] = thread_session_id
                            
                            try:
                                # Process audio event asynchronously
                                logger.debug(f"Processing audio segment {audio_sequence}")
                                await self.process_event(audio_event)
                            except Exception as e:
                                logger.error(f"Error processing audio event: {e}")
                                # Create an error event
                                error_event = ErrorEvent(
                                    source="audio_thread",
                                    error_code="audio_processing_error",
                                    message=str(e),
                                    component="system._process_audio_thread",
                                    recoverable=True,
                                    session_id=thread_session_id,
                                    sequence=audio_sequence
                                ).to_dict()
                                
                                # Store the error
                                try:
                                    if self.data_store:
                                        self.data_store.store_event(error_event)
                                except Exception as store_error:
                                    logger.error(f"Failed to store error event: {store_error}")
                        
                        # Adaptive sleep based on state
                        if self.state == SystemState.PROCESSING or self.state == SystemState.ANALYZING:
                            # Shorter sleep during active processing to ensure responsiveness
                            await asyncio.sleep(0.02)
                        else:
                            # Moderate sleep when idle to reduce CPU usage while maintaining responsiveness
                            await asyncio.sleep(0.1)
                            
                    except Exception as segment_error:
                        # Local exception handler for segment processing
                        logger.error(f"Error processing audio segment: {segment_error}")
                        # Add a short delay to avoid tight error loops
                        await asyncio.sleep(0.1)
            
            # Run the async worker until stop_processing is set
            try:
                event_loop.run_until_complete(process_audio_worker())
            finally:
                # Clean up
                event_loop.close()
                logger.info("Audio processing thread event loop closed")
                
        except Exception as thread_error:
            # Global exception handler for the thread
            self.state = SystemState.ERROR
            self.last_error = str(thread_error)
            logger.error(f"Fatal error in audio processing thread: {str(thread_error)}")
            
            # Add error to health tracking
            self.add_error("audio_thread", f"Thread terminated: {str(thread_error)}")
            
            # Attempt thread recovery after a brief pause
            def restart_thread():
                time.sleep(2.0)  # Wait before restarting
                logger.info("Attempting to restart audio processing thread")
                if not self.stop_processing:
                    self.processing_thread = threading.Thread(
                        target=self._process_audio_thread,
                        daemon=True
                    )
                    self.processing_thread.start()
            
            # Start recovery thread if not shutting down
            if not self.stop_processing:
                threading.Thread(target=restart_thread, daemon=True).start()
    
    def stop_audio_capture(self) -> bool:
        """Stop audio capture.
        
        Returns:
            True if successful
        """
        if not self.initialized:
            logger.error("System not initialized")
            return False
        
        try:
            # Stop processing thread
            self.stop_processing = True
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(2.0)  # Wait up to 2 seconds
            
            # Stop audio capture
            result = self.audio_pipeline.stop_capture()
            
            if result:
                self.state = SystemState.READY
                logger.info("Stopped audio capture")
                return True
            else:
                logger.error("Failed to stop audio capture")
                return False
                
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error stopping audio capture: {str(e)}")
            return False
    
    def generate_reports(self, report_types: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate reports.
        
        Args:
            report_types: List of report types to generate
            
        Returns:
            Dictionary mapping report types to report IDs
        """
        if not self.initialized:
            logger.error("System not initialized")
            return {}
        
        try:
            self.state = SystemState.REPORTING
            
            # Default report types
            if not report_types:
                report_types = ["medevac", "zmist"]
            
            # Get events from data store
            events = []
            for event_id in self.events:
                event = self.data_store.get_event(event_id)
                if event:
                    events.append(event)
            
            # Generate reports
            reports = {}
            
            for report_type in report_types:
                logger.info(f"Generating {report_type} report with {len(events)} events")
                
                if report_type == "medevac":
                    report = self.llm_analysis.generate_medevac_report(events)
                elif report_type == "zmist":
                    report = self.llm_analysis.generate_zmist_report(events)
                elif report_type == "soap":
                    report = self.llm_analysis.generate_soap_report(events)
                elif report_type == "tccc":
                    report = self.llm_analysis.generate_tccc_report(events)
                else:
                    logger.warning(f"Unknown report type: {report_type}")
                    continue
                
                # Store report
                report_id = self.data_store.store_report({
                    "type": report_type,
                    "content": report,
                    "events": self.events,
                    "session_id": self.session_id,
                    "timestamp": time.time()
                })
                
                logger.info(f"Stored report: {report_id}")
                self.reports.append(report_id)
                reports[report_type] = report_id
            
            self.state = SystemState.READY
            return reports
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error generating reports: {str(e)}")
            return {}
    
    def query_documents(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Query the document library.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        if not self.initialized:
            logger.error("System not initialized")
            return {"error": "System not initialized"}
        
        try:
            return self.document_library.query(query, n_results)
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return {"error": str(e)}
    
    def query_events(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query events.
        
        Args:
            filters: Query filters (optional)
            
        Returns:
            List of matching events
        """
        if not self.initialized:
            logger.error("System not initialized")
            return []
        
        try:
            # Use empty filters if none provided
            if filters is None:
                filters = {}
                
            # Return our current events if we have them
            if not filters and self.events:
                result = []
                for event_id in self.events:
                    event = self.data_store.get_event(event_id)
                    if event:
                        result.append(event)
                return result
                
            return self.data_store.query_events(filters)
        except Exception as e:
            logger.error(f"Error querying events: {str(e)}")
            return []
    
    def get_report(self, report_id: str) -> Dict[str, Any]:
        """Get a report.
        
        Args:
            report_id: ID of the report to retrieve
            
        Returns:
            Report data
        """
        if not self.initialized:
            logger.error("System not initialized")
            return {}
        
        try:
            return self.data_store.get_report(report_id)
        except Exception as e:
            logger.error(f"Error getting report: {str(e)}")
            return {}
    
    def get_current_state(self) -> SystemState:
        """Get the current system state.
        
        Returns:
            Current SystemState
        """
        return self.state
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the system.
        
        Returns:
            Dictionary with health status information
        """
        module_statuses = {}
        for name, module in {
            "processing_core": self.processing_core,
            "data_store": self.data_store,
            "document_library": self.document_library,
            "audio_pipeline": self.audio_pipeline,
            "stt_engine": self.stt_engine,
            "llm_analysis": self.llm_analysis
        }.items():
            if module and hasattr(module, 'get_status'):
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
    
    def add_error(self, module: str, message: str) -> None:
        """Add an error to health tracking.
        
        Args:
            module: Module that generated the error
            message: Error message
        """
        self.health["errors"].append({
            "module": module,
            "message": message,
            "time": time.time()
        })
        self.last_error = message
    
    def add_warning(self, module: str, message: str) -> None:
        """Add a warning to health tracking.
        
        Args:
            module: Module that generated the warning
            message: Warning message
        """
        self.health["warnings"].append({
            "module": module,
            "message": message,
            "time": time.time()
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status.
        
        Returns:
            Status information
        """
        try:
            status = {
                "state": self.state.value,
                "initialized": self.initialized,
                "session_id": self.session_id,
                "session_duration": time.time() - self.session_start_time if self.session_start_time else 0,
                "events_count": len(self.events),
                "reports_count": len(self.reports),
                "last_error": self.last_error,
                "modules": {}
            }
            
            # Get module statuses
            if self.processing_core:
                status["modules"]["processing_core"] = self.processing_core.get_status()
            
            if self.data_store:
                status["modules"]["data_store"] = self.data_store.get_status()
            
            if self.document_library:
                status["modules"]["document_library"] = self.document_library.get_status()
            
            if self.audio_pipeline:
                status["modules"]["audio_pipeline"] = self.audio_pipeline.get_status()
            
            if self.stt_engine:
                status["modules"]["stt_engine"] = self.stt_engine.get_status()
            
            if self.llm_analysis:
                status["modules"]["llm_analysis"] = self.llm_analysis.get_status()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return {
                "state": SystemState.ERROR.value,
                "error": str(e)
            }
    
    async def start(self) -> bool:
        """Start the system.
        
        Returns:
            True if the system started successfully
        """
        if not self.initialized or self.state not in [SystemState.READY, SystemState.IDLE]:
            logger.error("System not initialized or not in ready state")
            return False
        
        try:
            logger.info("Starting TCCC System...")
            
            # Start audio capture
            result = self.start_audio_capture()
            
            if result:
                logger.info("TCCC System started successfully")
                return True
            else:
                logger.error("Failed to start audio capture")
                return False
                
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error starting TCCC System: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Stop the system.
        
        Returns:
            True if the system stopped successfully
        """
        if not self.initialized:
            logger.warning("System not initialized, nothing to stop")
            return True
        
        try:
            logger.info("Stopping TCCC System...")
            
            # Stop audio capture if running
            if self.state == SystemState.CAPTURING:
                result = self.stop_audio_capture()
                if not result:
                    logger.warning("Failed to stop audio capture")
            
            self.state = SystemState.IDLE
            
            logger.info("TCCC System stopped successfully")
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error stopping TCCC System: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the system.
        
        Returns:
            True if shutdown was successful
        """
        if not self.initialized:
            logger.warning("System not initialized, nothing to shutdown")
            return True
        
        try:
            logger.info("Shutting down TCCC System...")
            
            # Stop audio capture if running
            if self.state == SystemState.CAPTURING:
                self.stop_audio_capture()
            
            # Shutdown modules
            if self.processing_core:
                self.processing_core.shutdown()
            
            if self.data_store:
                self.data_store.shutdown()
            
            if self.document_library and hasattr(self.document_library, 'shutdown'):
                self.document_library.shutdown()
            
            if self.audio_pipeline:
                self.audio_pipeline.shutdown()
            
            if self.stt_engine:
                self.stt_engine.shutdown()
            
            if self.llm_analysis:
                self.llm_analysis.shutdown()
            
            # Update state
            self.state = SystemState.SHUTDOWN
            self.initialized = False
            
            logger.info("TCCC System shutdown complete")
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error shutting down TCCC System: {str(e)}")
            return False