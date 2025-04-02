"""
TCCC.ai System Integration Implementation.

This module combines all TCCC modules into an integrated system
that can process audio, extract information, and generate reports.
"""

import os
import sys
import time
import asyncio
import threading
import argparse
import logging
import atexit
import traceback
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from enum import Enum
from pathlib import Path

# Ensure the src directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components from the TCCC project
from tccc.utils.config import Config
from tccc.utils.logging import get_logger, configure_logging
from tccc.utils.event_bus import EventBus, EventSubscription, get_event_bus
from tccc.utils.event_schema import BaseEvent, EventType, ErrorEvent, LLMAnalysisEvent, TranscriptionEvent
from tccc.utils.module_adapter import AudioPipelineAdapter, STTEngineAdapter, ProcessingCoreAdapter

# Import modules dynamically for flexibility
print("DEBUG: Importing ProcessingCore...")
try:
    from tccc.processing_core import ProcessingCore
    print("DEBUG: ProcessingCore imported.")
except ImportError as e:
    print(f"Warning: Could not import ProcessingCore - {e}")
    ProcessingCore = None
except Exception as e: # Catch other potential errors during import
    print(f"ERROR importing ProcessingCore: {e}\n{traceback.format_exc()}")
    ProcessingCore = None 
    sys.exit(1) # Exit explicitly on unexpected import error

print("DEBUG: Importing DataStore...")
try:
    from tccc.data_store import DataStore
    print("DEBUG: DataStore imported.")
except ImportError as e:
    print(f"Warning: Could not import DataStore - {e}")
    DataStore = None
except Exception as e:
    print(f"ERROR importing DataStore: {e}\n{traceback.format_exc()}")
    DataStore = None
    sys.exit(1)

print("DEBUG: Importing DocumentLibrary...")
try:
    from tccc.document_library import DocumentLibrary
    print("DEBUG: DocumentLibrary imported.")
except ImportError as e:
    print(f"Info: DocumentLibrary not available - {e}") 
    DocumentLibrary = None
except Exception as e:
    print(f"ERROR importing DocumentLibrary: {e}\n{traceback.format_exc()}")
    DocumentLibrary = None
    # Don't exit for DocLib, it's optional

print("DEBUG: Importing AudioPipeline...")
try:
    from tccc.audio_pipeline import AudioPipeline
    print("DEBUG: AudioPipeline imported.")
except ImportError as e:
    print(f"Warning: Could not import AudioPipeline - {e}")
    AudioPipeline = None
except Exception as e:
    print(f"ERROR importing AudioPipeline: {e}\n{traceback.format_exc()}")
    AudioPipeline = None
    sys.exit(1)

print("DEBUG: Importing STTEngine...")
try:
    from tccc.stt_engine import STTEngine
    print("DEBUG: STTEngine imported.")
except ImportError as e:
    print(f"Warning: Could not import STTEngine - {e}")
    STTEngine = None
except Exception as e:
    print(f"ERROR importing STTEngine: {e}\n{traceback.format_exc()}")
    STTEngine = None
    sys.exit(1)

print("DEBUG: Importing LLMAnalysis...")
try:
    from tccc.llm_analysis import LLMAnalysis
    print("DEBUG: LLMAnalysis imported.")
except ImportError as e:
    print(f"Warning: Could not import LLMAnalysis - {e}")
    LLMAnalysis = None
except Exception as e:
    print(f"ERROR importing LLMAnalysis: {e}\n{traceback.format_exc()}")
    LLMAnalysis = None
    sys.exit(1)

# Get logger instance (logging is configured in __main__.py)
logger = logging.getLogger(__name__)

# System State Enum
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
    
    def __init__(self, config_dir=None):
        """Initialize the TCCC System.
        
        Args:
            config_dir (str, optional): Path to the configuration directory.
        """
        logger.debug("TCCCSystem.__init__: Entering constructor")
        self.initialized = False
        self.state = SystemState.INITIALIZING
        self.config_dir = config_dir # Keep track of original request if needed
        self.config = {} # Initialize config, will be populated by initialize()
        self.config_file = None # Reset config file tracking
        logger.info("TCCC System object created")
        
        self.event_bus = EventBus()
        
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
        self._main_event_loop = None
        self._audio_sequence = 0
        self._thread_session_id = None
        
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

    async def initialize(self, config: Dict[str, Any] = None, mock_modules: Optional[List[str]] = None) -> bool:
        """Initialize the TCCC System.
        
        Args:
            config: System configuration (optional, defaults to empty dict)
            mock_modules: List of modules to use mocks for (for testing)
        
        Returns:
            True if initialization was successful
        """
        # Prevent re-initialization
        if self.initialized:
            logger.warning("System already initialized.")
            return True

        # Set the configuration provided by the caller (__main__.py)
        if config:
            self.config = config
            logger.info("TCCCSystem.initialize: Configuration assigned from argument.")
            # Optionally, try to find the original file path from the loaded config if needed later
            # self.config_file = self.config.get("loaded_files", [None])[0] # Example if load_config adds this
        else:
            logger.warning("TCCCSystem.initialize: No configuration provided.")
            # Cannot proceed without configuration
            self.state = SystemState.ERROR
            self.last_error = "Initialization called without providing configuration."
            return False

        # Initialize components
        logger.info("Initializing TCCC System components...")
        self.state = SystemState.INITIALIZING

        # Initialize modules
        init_events = {
            "audio_pipeline": {"future": asyncio.Future(), "initialized": False},
            "stt_engine": {"future": asyncio.Future(), "initialized": False},
            "processing_core": {"future": asyncio.Future(), "initialized": False},
            "llm_analysis": {"future": asyncio.Future(), "initialized": False},
            "data_store": {"future": asyncio.Future(), "initialized": False},
            "document_library": {"future": asyncio.Future(), "initialized": False}
        }

        # Create module instances (or mocks)
        if mock_modules is None:
            mock_modules = []
        
        if "audio_pipeline" in mock_modules:
            self.audio_pipeline = MockAudioPipeline(config=self.config.get("audio_pipeline", {}))
        else:
            self.audio_pipeline = AudioPipeline(config=self.config.get("audio_pipeline", {}))
        
        if "stt_engine" in mock_modules:
            self.stt_engine = MockSTTEngine(config=self.config.get("stt_engine", {}))
        else:
            self.stt_engine = STTEngine(config=self.config.get("stt_engine", {}))
        
        if "processing_core" in mock_modules:
            self.processing_core = MockProcessingCore(config=self.config.get("processing_core", {}))
        else:
            self.processing_core = ProcessingCore(config=self.config.get("processing_core", {}))
        
        if "llm_analysis" in mock_modules:
            self.llm_analysis = MockLLMAnalysis(config=self.config.get("llm_analysis", {}))
        else:
            self.llm_analysis = LLMAnalysis(config=self.config.get("llm_analysis", {}))
        
        # self.data_store = DataStore(config=self.config.get("data_store", {}))
        
        if "document_library" in mock_modules:
            self.document_library = MockDocumentLibrary(config=self.config.get("document_library", {}))
        else:
            self.document_library = DocumentLibrary(config=self.config.get("document_library", {}))

        # Initialize modules asynchronously
        module_initializations = [
            self._initialize_module(self.audio_pipeline, "AudioPipeline", self.config.get("audio_pipeline", {}), init_events["audio_pipeline"]),
            self._initialize_module(self.stt_engine, "STTEngine", self.config.get("stt_engine", {}), init_events["stt_engine"]),
            self._initialize_module(self.processing_core, "ProcessingCore", self.config.get("processing_core", {}), init_events["processing_core"]),
            self._initialize_module(self.llm_analysis, "LLMAnalysis", self.config.get("llm_analysis", {}), init_events["llm_analysis"]),
            # self._initialize_module(self.data_store, "DataStore", self.config.get("data_store", {}), init_events["data_store"]),
            self._initialize_module(self.document_library, "DocumentLibrary", self.config.get("document_library", {}), init_events["document_library"])
        ]

        # Wait for all modules to initialize
        await asyncio.gather(*module_initializations)

        # Check if all modules initialized successfully
        all_initialized = all(event["initialized"] for event in init_events.values())

        if all_initialized:
            logger.info("All modules initialized successfully.")
            self.state = SystemState.READY
            self.initialized = True
            self.event_bus.fire_event("system_ready", {})
            
            # Start the main event loop
            self._main_event_loop = asyncio.get_event_loop()
            
            return True
        else:
            logger.error("Failed to initialize all modules.")
            self.state = SystemState.ERROR
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
            init_event["initialized"] = True
            
            # Check if module has initialize method
            if not hasattr(module, 'initialize') or not callable(module.initialize):
                logger.warning(f"{module_name} missing initialize method")
                return False
                
            # Make a copy of config to avoid potential cross-module modification issues
            module_config = dict(config)
            
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
                    success_flag = result if isinstance(result, bool) else False
                    if not isinstance(result, bool):
                        logger.warning(f"Async initialize for {module_name} did not return a boolean. Assuming failure based on non-boolean return ({type(result)}).")
                else:
                    # Module has sync initialize method, run in executor
                    loop = asyncio.get_event_loop()
                    # Use the helper function within the lambda
                    success_flag, raw_result = await loop.run_in_executor(
                        None, 
                        lambda: _execute_sync_init(module_name, initialize_method, module_config)
                    )
                    
                # Update initialization event
                init_event["success"] = success_flag  # Store the boolean flag
                
                if success_flag:
                    logger.info(f"Module {module_name} initialized successfully")
                else:
                    logger.warning(f"Module {module_name} initialized with limited functionality")
                
                return success_flag # Return the boolean flag
                
            except Exception as init_error:
                logger.error(f"Error during {module_name} initialization: {init_error}")
                
                # Add error to health tracking
                self.add_error(module_name, f"Initialization error: {str(init_error)}")
                
                # Update initialization event with error
                init_event["success"] = False
                init_event["error"] = str(init_error)
                
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
            init_event["success"] = False
            init_event["error"] = str(e)
            
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
    
    def start_audio_capture(self) -> bool:
        """Start audio capture using the callback mechanism."""
        if not self.initialized:
            logger.error("System not initialized, cannot start audio capture.")
            return False
        if not self.audio_pipeline:
            logger.error("AudioPipeline not initialized, cannot start capture.")
            return False
        logger.debug("start_audio_capture: Checking if already capturing")
        if self.state == SystemState.CAPTURING:
             logger.warning("Audio capture already running.")
             return True # Already running is not an error

        logger.debug("start_audio_capture: Attempting to start audio capture")
        try:
            # Reset audio sequence and session ID for new capture session
            self._audio_sequence = 0
            self._thread_session_id = f"audio_session_{int(time.time())}"
            logger.info(f"Starting new audio session: {self._thread_session_id}")

            # Start audio capture with the callback
            # Ensure _process_audio_chunk is defined
            logger.debug("start_audio_capture: Calling audio_pipeline.start")
            logger.info("Attempting to start audio pipeline with callback...")
            result = self.audio_pipeline.start(self._process_audio_chunk)

            if result:
                self.state = SystemState.CAPTURING
                active_source_name = "unknown"
                if hasattr(self.audio_pipeline, 'get_active_source_name') and callable(self.audio_pipeline.get_active_source_name):
                     active_source_name = self.audio_pipeline.get_active_source_name() or 'default'
                logger.info(f"Audio capture started successfully using callback from '{active_source_name}'")
                return True
            else:
                logger.error("Failed to start audio capture: audio_pipeline.start returned False.")
                # Attempt to determine why it failed if possible
                status = self.audio_pipeline.get_status() if hasattr(self.audio_pipeline, 'get_status') else {}
                logger.error(f"Audio pipeline status: {status}")
                return False

        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.exception(f"Exception occurred while starting audio capture: {str(e)}") # Use exception logger
            return False

    def _process_audio_chunk(self, audio_data: 'np.ndarray'):
        """Callback function for AudioPipeline to process incoming audio chunks."""
        # logger.debug(f"Received audio chunk, size: {audio_data.shape}, dtype: {audio_data.dtype}") # DEBUG log
        if not self._main_event_loop or not self._main_event_loop.is_running():
            logger.error("Main event loop not available or not running. Cannot process audio chunk.")
            return

        try:
            # Increment sequence
            self._audio_sequence += 1
            current_sequence = self._audio_sequence # Capture current sequence for the event

            # Gather metadata (handle potential missing attributes/methods)
            metadata = {
                "timestamp": time.time(),
                "source_type": "unknown",
                "sample_rate": 16000, # Default
                "channels": 1,      # Default
                "dtype": "int16"     # Default
            }
            if self.audio_pipeline:
                 if hasattr(self.audio_pipeline, 'get_active_source_type') and callable(self.audio_pipeline.get_active_source_type):
                      metadata["source_type"] = self.audio_pipeline.get_active_source_type() or "unknown"
                 if hasattr(self.audio_pipeline, 'sample_rate'):
                      metadata["sample_rate"] = self.audio_pipeline.sample_rate
                 if hasattr(self.audio_pipeline, 'channels'):
                      metadata["channels"] = self.audio_pipeline.channels
                 if hasattr(self.audio_pipeline, 'dtype') and self.audio_pipeline.dtype:
                      # Store dtype as string representation
                      metadata["dtype"] = str(np.dtype(self.audio_pipeline.dtype))

            logger.debug(f"Active audio source: {self.audio_pipeline.source.name} ({self.audio_pipeline.source.type})")
            logger.debug(f"Active audio source: name={metadata.get('source_type')}, type={metadata.get('source_type')}")

            # Create audio event dictionary directly
            # The actual AudioSegmentEvent object creation might happen inside process_event or adapter
            event_payload = {
                "source": "audio_pipeline",
                "type": EventType.AUDIO_SEGMENT.value, # Use enum value for type
                "timestamp": metadata["timestamp"],
                "session_id": self._thread_session_id,
                "sequence": current_sequence,
                "data": {
                     "audio_data": audio_data, # Pass the raw numpy array
                     "metadata": metadata # Pass collected metadata
                }
            }
            
            # logger.debug(f"Scheduling audio event processing for sequence {current_sequence}") # DEBUG log
            # Schedule the async process_event coroutine on the main event loop
            future = asyncio.run_coroutine_threadsafe(self.process_event(event_payload), self._main_event_loop)

            # Optional: Add callback to future to log result/errors after execution
            def log_audio_event_result(f):
                 try:
                      result = f.result()
                      # logger.debug(f"Audio event (seq {current_sequence}) processing finished. Result: {result}") # DEBUG log
                 except Exception as e_future:
                      logger.error(f"Error processing audio event (seq {current_sequence}) in scheduled task: {e_future}")

            future.add_done_callback(log_audio_event_result)

        except Exception as e:
            # Log exception originating from within the callback itself
            logger.exception(f"Error in _process_audio_chunk callback (sequence {self._audio_sequence}): {e}")

    def stop_audio_capture(self) -> bool:
        """Stop audio capture."""
        if not self.initialized:
            logger.error("System not initialized, cannot stop audio capture.")
            return False
        if not self.audio_pipeline:
             logger.warning("AudioPipeline not initialized or already stopped. Nothing to stop.")
             # Return True as there's nothing active to stop from system's perspective
             if self.state == SystemState.CAPTURING: # If state was capturing, reset it
                  self.state = SystemState.READY
             return True
        if self.state != SystemState.CAPTURING:
             logger.warning(f"Audio capture is not running (state: {self.state}). Stop request ignored.")
             return True # Not running is not an error

        try:
            logger.info("Attempting to stop audio pipeline...")
            # Stop audio capture using the pipeline's stop method
            result = self.audio_pipeline.stop() # Use stop, not stop_capture

            if result:
                self.state = SystemState.READY
                logger.info("Audio capture stopped successfully via pipeline stop method.")
                return True
            else:
                # Maybe it was already stopped? Check status if possible
                was_running = True # Assume it was running if state was CAPTURING
                if hasattr(self.audio_pipeline, 'is_running'):
                     was_running = self.audio_pipeline.is_running
                
                if not was_running:
                     logger.warning("Audio pipeline stop method returned False, but pipeline was already not running.")
                     self.state = SystemState.READY # Ensure state is READY
                     return True # Consider this success
                else:
                     logger.error("Audio pipeline stop method returned False, and pipeline might still be running.")
                     # Don't change state from CAPTURING if stop failed and it thinks it's running
                     return False

        except Exception as e:
            self.state = SystemState.ERROR # An exception during stop is an error
            self.last_error = str(e)
            logger.exception(f"Exception occurred while stopping audio capture: {str(e)}") # Use exception logger
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
                # Data is now structured as event_data['data']['audio_data'] and event_data['data']['metadata']
                
                # Convert for STT Engine using the adapter
                # The adapter needs to handle the new event structure
                try:
                     logger.debug(f"Converting audio event (seq {event_data.get('sequence', 'N/A')}) for STT...")
                     stt_input = STTEngineAdapter.convert_audio_event_to_input(event_data)
                except Exception as adapter_error:
                     logger.exception(f"Error converting audio event using STTEngineAdapter: {adapter_error}")
                     # Create and store error event
                     error_event = ErrorEvent(
                         source="system", error_code="adapter_error", message=f"STT Adapter failed: {adapter_error}",
                         component="process_event.stt_adapter", recoverable=True, session_id=session_id, sequence=self._event_sequence
                     ).to_dict()
                     # await self._store_event_safe(error_event) # Helper needed
                     return None # Stop processing this event

                # Process with STT Engine
                self.state = SystemState.PROCESSING
                logger.debug(f"Sending audio (seq {event_data.get('sequence', 'N/A')}) to STT engine...")
                transcription = await self._transcribe_async(stt_input.get("audio"), stt_input.get("metadata")) # Use .get for safety
                
                # Convert transcription to standard event
                if transcription and transcription.get("text", ""): # Check if transcription is valid
                    logger.debug(f"STT result (seq {event_data.get('sequence', 'N/A')}): '{transcription.get('text', '')[:50]}...'")
                    try:
                         transcription_event = STTEngineAdapter.convert_transcription_to_event(
                              transcription, event_data # Pass original event for context
                         )
                    except Exception as adapter_error:
                         logger.exception(f"Error converting transcription using STTEngineAdapter: {adapter_error}")
                         # Create and store error event...
                         return None
                    # Recursively call process_event with the new transcription event
                    return await self.process_event(transcription_event)
                elif transcription and "error" in transcription:
                     logger.error(f"STT engine returned an error for sequence {event_data.get('sequence', 'N/A')}: {transcription['error']}")
                     # Create and store error event...
                     return None
                else:
                    # No transcription text, maybe just silence or VAD decided no speech
                    logger.debug(f"No transcription result from STT Engine for sequence {event_data.get('sequence', 'N/A')}")
                    # Reset state if no further processing happens for this chunk
                    self.state = SystemState.READY # Or CAPTURING if still running? Check logic
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
                    # event_id = self.data_store.store_event(llm_event)
                    logger.info(f"Stored LLM analysis event: N/A")
                    self.events.append("N/A")
                    
                    # Store the original processed text event too
                    # orig_event_id = self.data_store.store_event(event_data)
                    logger.info(f"Stored processed text event: N/A")
                    self.events.append("N/A")
                    
                    # Reset state
                    self.state = SystemState.READY
                    return "N/A"
                else:
                    # Just store the processed text event
                    # event_id = self.data_store.store_event(event_data)
                    logger.info(f"Stored processed text event: N/A")
                    self.events.append("N/A")
                    
                    # Reset state
                    self.state = SystemState.READY
                    return "N/A"
                    
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
                # event_id = self.data_store.store_event(event_data)
                logger.info(f"Stored error event: N/A")
                self.events.append("N/A")
                
                # Set system state based on error severity
                severity = error_data.get("severity", "error")
                if severity in ["error", "critical"]:
                    self.state = SystemState.ERROR
                    self.last_error = error_message
                else:
                    # For warning and info, stay ready
                    self.state = SystemState.READY
                
                return "N/A"
                
            else:
                # Store other event types directly
                # event_id = self.data_store.store_event(event_data)
                logger.info(f"Stored generic event: N/A")
                self.events.append("N/A")
                
                # Reset state
                self.state = SystemState.READY
                return "N/A"
            
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
                session_id=session_id,
                sequence=self._event_sequence
            ).to_dict()
            
            try:
                # error_id = self.data_store.store_event(error_event)
                logger.info(f"Stored error event: N/A")
                self.events.append("N/A")
                return "N/A"
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
                # event = self.data_store.get_event(event_id)
                event = None
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
                # report_id = self.data_store.store_report({
                #     "type": report_type,
                #     "content": report,
                #     "events": self.events,
                #     "session_id": self.session_id,
                #     "timestamp": time.time()
                # })
                logger.info(f"Stored report: N/A")
                self.reports.append("N/A")
                reports[report_type] = "N/A"
            
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
        """Query events from the data store based on filters."""
        logger.info(f"Querying events with filters: {filters}")
        # return self.data_store.query_events(filters)
        logger.warning("DataStore disabled for MVP, cannot query events.")
        return [] # Return empty list
 
    def get_tccc_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific TCCC report by its ID."""
        logger.info(f"Retrieving TCCC report with ID: {report_id}")
        # return self.data_store.get_report(report_id)
        logger.warning(f"DataStore disabled for MVP, cannot retrieve report ID: {report_id}")
        return None # Return None
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific event by its ID."""
        logger.info(f"Retrieving event with ID: {event_id}")
        # event = self.data_store.get_event(event_id)
        logger.warning(f"DataStore disabled for MVP, cannot retrieve event ID: {event_id}")
        return None # Return None
    
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
                # status["modules"]["data_store"] = self.data_store.get_status()
                logger.warning("DataStore disabled for MVP, cannot retrieve status.")
            
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
        logger.info("TCCCSystem.start: Starting the system...")
        if not self.initialized or self.state not in [SystemState.READY, SystemState.IDLE]:
            logger.error("System not initialized or not in ready state")
            return False
        
        try:
            logger.info("TCCC System starting...")
            self.state = SystemState.IDLE

            return True
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
            
            # Shutdown modules - Reordered for potentially safer shutdown
            # Shutdown consumers first
            if self.stt_engine:
                self.stt_engine.shutdown()

            if self.llm_analysis:
                self.llm_analysis.shutdown()

            if self.processing_core:
                self.processing_core.shutdown()
            
            # Shutdown producers/stores next
            if self.audio_pipeline:
                self.audio_pipeline.shutdown()

            if self.document_library and hasattr(self.document_library, 'shutdown'):
                self.document_library.shutdown()

            # if self.data_store:
            #     self.data_store.shutdown()
            
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

def _execute_sync_init(module_name: str, init_method: Callable, config: Dict[str, Any]) -> Tuple[bool, Optional[Any]]:
    """Helper to execute synchronous init methods safely."""
    print(f"DEBUG: _execute_sync_init: Executing for {module_name}") # Added print
    init_result = False
    raw_result = None
    try:
        print(f"DEBUG: _execute_sync_init: About to call {module_name}.initialize with config: {config}")
        # --- Add specific try-except around the call --- 
        try:
            print(f"DEBUG: _execute_sync_init: ABOUT TO CALL {module_name}.initialize({config})") # Kept print
            raw_result = init_method(config)
            print(f"DEBUG: _execute_sync_init: Returned from {module_name}.initialize call.") # Kept print
        except Exception as call_error:
            print(f"DEBUG: _execute_sync_init: EXCEPTION DURING {module_name}.initialize CALL: {call_error}\n{traceback.format_exc()}")
            raw_result = False # Assume failure if call itself fails
        # --- End specific try-except ---
        
        # Check result type and update success flag
        if isinstance(raw_result, bool):
            init_result = raw_result
        elif raw_result is None: # Consider None as success if no explicit bool needed
            init_result = True 
        else: # Unexpected return type, treat as failure
            logger.warning(f"Module {module_name} init returned unexpected type: {type(raw_result)}. Treating as failure.")
            init_result = False
                    
    except Exception as e:
        logger.error(f"Error initializing module {module_name}: {e}", exc_info=True)
        init_result = False
        # Optionally capture traceback
        # tb_str = traceback.format_exc()
        # logger.debug(f"Traceback for {module_name} init error:\n{tb_str}")

    print(f"DEBUG: Sync execution completed for {module_name}, raw result: {raw_result}") # Added print
    return init_result, raw_result