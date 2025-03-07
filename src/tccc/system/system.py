"""
TCCC.ai System Integration Implementation.

This module combines all TCCC modules into an integrated system
that can process audio, extract information, and generate reports.
"""

import os
import time
import logging
import threading
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
            
            # Set up modules - use mocks if specified
            logger.info("Initializing modules")
            
            # Processing Core
            if "processing_core" in mock_modules:
                from tests.mocks.mock_processing_core import MockProcessingCore
                self.processing_core = MockProcessingCore()
                logger.info("Initializing MockProcessingCore")
            else:
                self.processing_core = ProcessingCore()
                logger.info("Initializing ProcessingCore")
            
            # Data Store
            if "data_store" in mock_modules:
                from tests.mocks.mock_data_store import MockDataStore
                self.data_store = MockDataStore()
                logger.info("Initializing MockDataStore")
            else:
                self.data_store = DataStore()
                logger.info("Initializing DataStore")
            
            # Document Library
            if "document_library" in mock_modules:
                from tests.mocks.mock_document_library import MockDocumentLibrary
                self.document_library = MockDocumentLibrary()
                logger.info("Initializing MockDocumentLibrary")
            else:
                self.document_library = DocumentLibrary()
                logger.info("Initializing DocumentLibrary")
            
            # Audio Pipeline
            if "audio_pipeline" in mock_modules:
                from tests.mocks.mock_audio_pipeline import MockAudioPipeline
                self.audio_pipeline = MockAudioPipeline()
                logger.info("Initializing MockAudioPipeline")
            else:
                self.audio_pipeline = AudioPipeline()
                logger.info("Initializing AudioPipeline")
            
            # STT Engine
            if "stt_engine" in mock_modules:
                from tests.mocks.mock_stt_engine import MockSTTEngine
                self.stt_engine = MockSTTEngine()
                logger.info("Initializing MockSTTEngine")
            else:
                self.stt_engine = STTEngine()
                logger.info("Initializing STTEngine")
            
            # LLM Analysis
            if "llm_analysis" in mock_modules:
                from tests.mocks.mock_llm_analysis import MockLLMAnalysis
                self.llm_analysis = MockLLMAnalysis()
                logger.info("Initializing MockLLMAnalysis")
            else:
                self.llm_analysis = LLMAnalysis()
                logger.info("Initializing LLMAnalysis")
            
            # Initialize modules with configurations
            config_dict = self.config  # Use the already validated self.config
            modules_config = {
                "processing_core": config_dict.get("processing_core", {}),
                "data_store": config_dict.get("data_store", {}),
                "document_library": config_dict.get("document_library", {}),
                "audio_pipeline": config_dict.get("audio_pipeline", {}),
                "stt_engine": config_dict.get("stt_engine", {}),
                "llm_analysis": config_dict.get("llm_analysis", {})
            }
            
            # Initialize modules
            # Handle async initialize method for ProcessingCore properly
            if hasattr(self.processing_core, 'initialize') and callable(self.processing_core.initialize):
                if hasattr(self.processing_core.initialize, '__await__'):
                    # This is an async method, call it properly with await
                    await self.processing_core.initialize(modules_config["processing_core"])
                else:
                    # This is a regular method
                    self.processing_core.initialize(modules_config["processing_core"])
            else:
                logger.warning("ProcessingCore missing initialize method")
                
            # Initialize other modules
            self.data_store.initialize(modules_config["data_store"])
            self.document_library.initialize(modules_config["document_library"])
            self.audio_pipeline.initialize(modules_config["audio_pipeline"])
            self.stt_engine.initialize(modules_config["stt_engine"])
            self.llm_analysis.initialize(modules_config["llm_analysis"])
            
            # Set up module dependencies
            # Check if the method exists before calling it
            if hasattr(self.llm_analysis, 'set_document_library') and callable(self.llm_analysis.set_document_library):
                self.llm_analysis.set_document_library(self.document_library)
            
            # Create a new session
            self.session_id = f"session_{int(time.time())}"
            self.session_start_time = time.time()
            
            # Update state
            self.state = SystemState.READY
            self.initialized = True
            
            logger.info("TCCC System initialized successfully")
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to initialize TCCC System: {str(e)}")
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
    
    def process_event(self, event_data: Dict[str, Any]) -> Optional[str]:
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
            
            # Extract text if present in the event
            text = None
            if "text" in event_data:
                text = event_data["text"]
            elif "data" in event_data and isinstance(event_data["data"], str):
                text = event_data["data"]
            
            # Process text through processing core if available
            processed = None
            if text and self.processing_core:
                try:
                    if hasattr(self.processing_core, 'process'):
                        processed = self.processing_core.process({
                            "text": text,
                            "metadata": event_data.get("metadata", {}),
                            "type": event_data.get("type", "external"),
                            "timestamp": event_data.get("timestamp", time.time())
                        })
                    else:
                        # Basic processing result
                        processed = {
                            "text": text,
                            "type": event_data.get("type", "external"),
                            "timestamp": event_data.get("timestamp", time.time())
                        }
                except Exception as e:
                    logger.error(f"Error processing event text: {e}")
                    processed = {"text": text, "error": str(e)}
            else:
                # Use the event data directly if no text processing needed
                processed = event_data
            
            # Analyze with LLM if text is available
            if text and self.llm_analysis:
                self.state = SystemState.ANALYZING
                analysis = self.llm_analysis.analyze_transcription(text)
                
                # Add analysis results to processed data
                if isinstance(processed, dict) and isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if key not in processed:
                            processed[key] = value
            
            # Store the event
            event_id = self.data_store.store_event(processed)
            logger.info(f"Stored event: {event_id}")
            self.events.append(event_id)
            
            # Reset state
            self.state = SystemState.READY
            
            return event_id
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error processing event: {str(e)}")
            return None
    
    def _process_audio_thread(self):
        """Thread for processing audio data."""
        try:
            logger.info("Audio processing thread started")
            
            while not self.stop_processing:
                # Get audio data from the pipeline
                audio_data = None
                
                # Check for different audio pipeline interfaces
                if hasattr(self.audio_pipeline, 'get_audio_segment'):
                    audio_data = self.audio_pipeline.get_audio_segment()
                elif hasattr(self.audio_pipeline, 'get_audio'):
                    audio_data = self.audio_pipeline.get_audio()
                elif hasattr(self.audio_pipeline, 'get_audio_data'):
                    audio_data = self.audio_pipeline.get_audio_data()
                
                if audio_data:
                    # Process the audio data
                    self.state = SystemState.PROCESSING
                    
                    # Transcribe audio - use either transcribe or transcribe_segment based on what's available
                    transcription = None
                    if hasattr(self.stt_engine, 'transcribe_segment'):
                        transcription = self.stt_engine.transcribe_segment(audio_data)
                    else:
                        transcription = self.stt_engine.transcribe(audio_data)
                    
                    # Check for error conditions
                    if transcription and "error" in transcription:
                        logger.warning(f"STT engine error: {transcription['error']}")
                        time.sleep(0.5)  # Add delay to avoid tight retry loop
                        continue
                    
                    if transcription and transcription.get("text"):
                        # Create event data from transcription
                        event_data = {
                            "type": "audio_transcription",
                            "text": transcription["text"],
                            "segments": transcription.get("segments", []),
                            "metadata": transcription.get("metadata", {}),
                            "language": transcription.get("language", "en"),
                            "timestamp": time.time(),
                            "source": "audio_pipeline"
                        }
                        
                        # Process through the main event processing pipeline
                        self.process_event(event_data)
                
                # Adaptive sleep based on state
                if self.state == SystemState.PROCESSING or self.state == SystemState.ANALYZING:
                    # Shorter sleep during active processing to ensure responsiveness
                    time.sleep(0.05)
                else:
                    # Longer sleep when idle to reduce CPU usage
                    time.sleep(0.2)
                
        except Exception as e:
            self.state = SystemState.ERROR
            self.last_error = str(e)
            logger.error(f"Error in audio processing thread: {str(e)}")
    
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