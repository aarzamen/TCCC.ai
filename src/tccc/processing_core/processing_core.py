"""
Processing Core module for TCCC.ai system.

This module implements the main Processing Core component of the TCCC.ai system.
Includes module registration system with dependency tracking.
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Type
from dataclasses import dataclass
from enum import Enum, auto
import networkx as nx

from tccc.utils.logging import get_logger
from tccc.utils.config import Config
from tccc.processing_core.entity_extractor import EntityExtractor, Entity
from tccc.processing_core.intent_classifier import IntentClassifier, Intent
from tccc.processing_core.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysis
from tccc.processing_core.resource_monitor import ResourceMonitor
from tccc.processing_core.plugin_manager import PluginManager
from tccc.processing_core.state_manager import StateManager

logger = get_logger(__name__)


class ModuleState(Enum):
    """Operational states for modules in the ProcessingCore."""
    UNINITIALIZED = auto()  # Module is registered but not initialized
    INITIALIZING = auto()   # Module is in the process of initializing
    READY = auto()          # Module is initialized and ready for use
    ACTIVE = auto()         # Module is actively processing data
    ERROR = auto()          # Module encountered an error
    STANDBY = auto()        # Module is initialized but temporarily inactive
    SHUTDOWN = auto()       # Module has been shut down


@dataclass
class ModuleInfo:
    """Information about a registered module."""
    name: str
    module_type: str
    instance: Any
    dependencies: List[str]
    state: ModuleState = ModuleState.UNINITIALIZED
    state_message: Optional[str] = None
    last_state_change: Optional[float] = None
    metrics: Dict[str, Any] = None
    config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metrics is None:
            self.metrics = {}
        self.last_state_change = time.time()
    
    def update_state(self, state: ModuleState, message: Optional[str] = None) -> None:
        """
        Update the module state.
        
        Args:
            state: The new state.
            message: Optional message about the state change.
        """
        self.state = state
        self.state_message = message
        self.last_state_change = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the module info.
        """
        return {
            "name": self.name,
            "type": self.module_type,
            "state": self.state.name,
            "state_message": self.state_message,
            "last_state_change": self.last_state_change,
            "dependencies": self.dependencies,
            "metrics": self.metrics
        }


@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed text."""
    text: str
    speaker: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = 1.0
    is_final: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessedSegment:
    """Represents a processed transcription segment with analysis results."""
    text: str
    speaker: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = 1.0
    entities: Optional[List[Entity]] = None
    intents: Optional[List[Intent]] = None
    sentiment: Optional[SentimentAnalysis] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationContext:
    """Represents the context of a conversation."""
    id: str
    segments: List[ProcessedSegment]
    metadata: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    last_updated: Optional[float] = None


@dataclass
class Summary:
    """Represents a conversation summary."""
    text: str
    highlights: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    generated_at: Optional[float] = None


class ProcessingMetrics:
    """
    Tracks processing metrics for the Processing Core.
    """
    
    def __init__(self):
        """Initialize processing metrics."""
        # Processing times
        self.total_processing_time = 0.0
        self.entity_extraction_time = 0.0
        self.intent_classification_time = 0.0
        self.sentiment_analysis_time = 0.0
        self.summarization_time = 0.0
        self.plugin_processing_time = 0.0
        
        # Counts
        self.segments_processed = 0
        self.entities_extracted = 0
        self.intents_identified = 0
        self.sentiment_analyzed = 0
        self.summaries_generated = 0
        self.errors_encountered = 0
        
        # Performance tracking
        self.last_reset_time = time.time()
        self.processing_times: List[float] = []
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()
    
    def record_processing_time(self, processing_time: float):
        """
        Record a processing time.
        
        Args:
            processing_time: The processing time in seconds.
        """
        self.total_processing_time += processing_time
        self.processing_times.append(processing_time)
        # Keep only the last 100 processing times
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def get_average_processing_time(self) -> float:
        """
        Get the average processing time.
        
        Returns:
            The average processing time in seconds.
        """
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.
        
        Returns:
            A dictionary of metrics.
        """
        uptime = time.time() - self.last_reset_time
        avg_processing_time = self.get_average_processing_time()
        
        return {
            "uptime_seconds": uptime,
            "segments_processed": self.segments_processed,
            "entities_extracted": self.entities_extracted,
            "intents_identified": self.intents_identified,
            "sentiment_analyzed": self.sentiment_analyzed,
            "summaries_generated": self.summaries_generated,
            "errors_encountered": self.errors_encountered,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "entity_extraction_time": self.entity_extraction_time,
            "intent_classification_time": self.intent_classification_time,
            "sentiment_analysis_time": self.sentiment_analysis_time,
            "summarization_time": self.summarization_time,
            "plugin_processing_time": self.plugin_processing_time
        }


class ProcessingCore:
    """
    Main Processing Core implementation.
    
    Implements the ProcessingCoreInterface defined in the module_interfaces.md
    specification. Handles text processing, entity extraction, intent classification, 
    sentiment analysis, and conversation summarization. Includes a module registration
    system with dependency tracking.
    """
    
    def __init__(self):
        """Initialize the Processing Core."""
        self.initialized = False
        self.config: Optional[Dict[str, Any]] = None
        
        # Module registry and dependency graph
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph = nx.DiGraph()
        
        # Core components (preserved for backward compatibility)
        self.entity_extractor: Optional[EntityExtractor] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.plugin_manager: Optional[PluginManager] = None
        self.state_manager: Optional[StateManager] = None
        
        # Metrics
        self.metrics = ProcessingMetrics()
        
        # Active conversations
        self.conversations: Dict[str, ConversationContext] = {}
        
        # Locks for thread-safety
        self.processing_lock = threading.RLock()
        self.conversation_lock = threading.RLock()
        self.module_lock = threading.RLock()
        
        # Current operational state
        self.operational_state = ModuleState.UNINITIALIZED
    
    def register_module(self, name: str, module_type: str, instance: Any, 
                        dependencies: List[str] = None, config: Dict[str, Any] = None) -> bool:
        """
        Register a module with the ProcessingCore.
        
        Args:
            name: Unique name for the module.
            module_type: Type of module (e.g., "extractor", "classifier").
            instance: The module instance.
            dependencies: List of module names this module depends on.
            config: Optional configuration for the module.
            
        Returns:
            True if registration was successful, False otherwise.
        """
        with self.module_lock:
            # Check if module with this name already exists
            if name in self.modules:
                logger.warning(f"Module {name} is already registered")
                return False
            
            # Create module info
            module_info = ModuleInfo(
                name=name,
                module_type=module_type,
                instance=instance,
                dependencies=dependencies or [],
                config=config
            )
            
            # Add to registry
            self.modules[name] = module_info
            
            # Update dependency graph
            self.dependency_graph.add_node(name)
            for dep in module_info.dependencies:
                if dep in self.modules:
                    self.dependency_graph.add_edge(dep, name)
                else:
                    logger.warning(f"Module {name} depends on unregistered module {dep}")
            
            logger.info(f"Registered module: {name} of type {module_type}")
            return True
    
    def unregister_module(self, name: str) -> bool:
        """
        Unregister a module.
        
        Args:
            name: Name of the module to unregister.
            
        Returns:
            True if unregistered successfully, False otherwise.
        """
        with self.module_lock:
            if name not in self.modules:
                logger.warning(f"Module {name} is not registered")
                return False
            
            # Check if other modules depend on this one
            dependent_modules = []
            for module_name, module_info in self.modules.items():
                if name in module_info.dependencies:
                    dependent_modules.append(module_name)
            
            if dependent_modules:
                logger.error(f"Cannot unregister module {name} as it is required by: {', '.join(dependent_modules)}")
                return False
            
            # Update module state
            try:
                self.modules[name].update_state(ModuleState.SHUTDOWN, "Module unregistered")
            except Exception as e:
                logger.error(f"Error updating state for module {name}: {e}")
            
            # Remove from registry and dependency graph
            del self.modules[name]
            self.dependency_graph.remove_node(name)
            
            logger.info(f"Unregistered module: {name}")
            return True
    
    def get_module(self, name: str) -> Optional[Any]:
        """
        Get a module instance by name.
        
        Args:
            name: Name of the module.
            
        Returns:
            The module instance, or None if not found.
        """
        module_info = self.modules.get(name)
        return module_info.instance if module_info else None
    
    def get_module_info(self, name: str) -> Optional[ModuleInfo]:
        """
        Get information about a module.
        
        Args:
            name: Name of the module.
            
        Returns:
            ModuleInfo for the module, or None if not found.
        """
        return self.modules.get(name)
    
    def get_module_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all modules.
        
        Returns:
            Dictionary mapping module names to their status.
        """
        with self.module_lock:
            return {name: info.to_dict() for name, info in self.modules.items()}
    
    def get_initialization_order(self) -> List[str]:
        """
        Get the correct order to initialize modules based on dependencies.
        
        Returns:
            List of module names in initialization order.
        """
        try:
            # Use topological sort to determine initialization order
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            # Circular dependencies detected
            logger.error("Circular module dependencies detected")
            return list(self.modules.keys())
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the processing core with configuration.
        
        Args:
            config: Configuration for the processing core.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        if self.initialized:
            logger.warning("Processing Core is already initialized")
            return True
        
        try:
            self.config = config
            
            # Update operational state
            self.operational_state = ModuleState.INITIALIZING
            
            # Initialize resource monitor first
            resource_config = config.get("resource_management", {})
            self.resource_monitor = ResourceMonitor(resource_config)
            self.register_module(
                name="resource_monitor",
                module_type="monitor",
                instance=self.resource_monitor,
                config=resource_config
            )
            self.modules["resource_monitor"].update_state(ModuleState.INITIALIZING)
            self.resource_monitor.start()
            self.modules["resource_monitor"].update_state(ModuleState.ACTIVE)
            
            # Initialize state manager
            state_config = config.get("state_management", {})
            self.state_manager = StateManager(state_config)
            self.register_module(
                name="state_manager",
                module_type="state",
                instance=self.state_manager,
                config=state_config
            )
            self.modules["state_manager"].update_state(ModuleState.READY)
            
            # Initialize entity extractor
            entity_config = config.get("entity_extraction", {})
            self.entity_extractor = EntityExtractor(entity_config)
            self.register_module(
                name="entity_extractor",
                module_type="extractor",
                instance=self.entity_extractor,
                dependencies=["resource_monitor"],
                config=entity_config
            )
            self.modules["entity_extractor"].update_state(ModuleState.READY)
            
            # Initialize intent classifier
            intent_config = config.get("intent_classification", {})
            self.intent_classifier = IntentClassifier(intent_config)
            self.register_module(
                name="intent_classifier",
                module_type="classifier",
                instance=self.intent_classifier,
                dependencies=["resource_monitor"],
                config=intent_config
            )
            self.modules["intent_classifier"].update_state(ModuleState.READY)
            
            # Initialize sentiment analyzer
            sentiment_config = config.get("sentiment_analysis", {})
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
            self.register_module(
                name="sentiment_analyzer",
                module_type="analyzer",
                instance=self.sentiment_analyzer,
                dependencies=["resource_monitor"],
                config=sentiment_config
            )
            self.modules["sentiment_analyzer"].update_state(ModuleState.READY)
            
            # Initialize plugin manager
            plugin_config = config.get("plugins", {})
            self.plugin_manager = PluginManager(plugin_config)
            self.register_module(
                name="plugin_manager",
                module_type="manager",
                instance=self.plugin_manager,
                dependencies=["resource_monitor", "state_manager"],
                config=plugin_config
            )
            self.modules["plugin_manager"].update_state(ModuleState.READY)
            
            # Register resource monitor callback to track resource usage
            if self.resource_monitor and self.state_manager:
                self.resource_monitor.register_callback(self._on_resource_update)
            
            # Configure dynamic resource allocation
            resource_config = config.get("resource_management", {})
            self.enable_dynamic_allocation = resource_config.get("enable_dynamic_allocation", True)
            self.cpu_high_threshold = resource_config.get("cpu_high_threshold", 80)
            self.cpu_low_threshold = resource_config.get("cpu_low_threshold", 20)
            self.memory_high_threshold = resource_config.get("memory_high_threshold", 80)
            self.memory_low_threshold = resource_config.get("memory_low_threshold", 20)
            self.max_concurrent_tasks = config.get("general", {}).get("max_concurrent_tasks", 2)
            self.min_concurrent_tasks = 1
            self.current_concurrent_tasks = self.max_concurrent_tasks
            
            self.initialized = True
            self.operational_state = ModuleState.READY
            logger.info("Processing Core initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Processing Core: {str(e)}")
            self.operational_state = ModuleState.ERROR
            return False
    
    def _on_resource_update(self, usage, *args, **kwargs):
        """
        Callback for resource usage updates.
        Handles dynamic resource allocation based on system load.
        
        Args:
            usage: Resource usage information.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if self.state_manager:
            # Update state with resource metrics
            self.state_manager.update_state({
                "resource_usage": usage.to_dict(),
                "resource_metrics": self.resource_monitor.get_resource_metrics()
            })
        
        # Handle dynamic resource allocation if enabled
        if hasattr(self, 'enable_dynamic_allocation') and self.enable_dynamic_allocation and self.initialized:
            try:
                self._adjust_resource_allocation(usage)
            except Exception as e:
                logger.error(f"Error in dynamic resource allocation: {str(e)}")
    
    def _adjust_resource_allocation(self, usage):
        """
        Adjust resource allocation based on current system load.
        Scales concurrent tasks and processing modes.
        
        Args:
            usage: Current resource usage information.
        """
        # Check CPU usage
        cpu_usage = usage.cpu_usage
        memory_usage = usage.memory_usage
        
        old_concurrent_tasks = self.current_concurrent_tasks
        
        # Adjust concurrent tasks based on load
        if cpu_usage > self.cpu_high_threshold or memory_usage > self.memory_high_threshold:
            # High load - reduce concurrent tasks
            self.current_concurrent_tasks = max(self.min_concurrent_tasks, self.current_concurrent_tasks - 1)
            
            # Put non-essential modules in standby if load is extremely high
            if cpu_usage > 90 or memory_usage > 90:
                self._adjust_module_states(ModuleState.STANDBY, ["plugin_manager"])
                
        elif cpu_usage < self.cpu_low_threshold and memory_usage < self.memory_low_threshold:
            # Low load - increase concurrent tasks up to maximum
            self.current_concurrent_tasks = min(self.max_concurrent_tasks, self.current_concurrent_tasks + 1)
            
            # Reactivate modules that were in standby
            self._adjust_module_states(ModuleState.READY, ["plugin_manager"])
        
        # Log changes if resource allocation changed
        if old_concurrent_tasks != self.current_concurrent_tasks:
            logger.info(f"Adjusted concurrent tasks: {old_concurrent_tasks} -> {self.current_concurrent_tasks} " +
                       f"(CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)")
            
            # Update state manager with new allocation
            if self.state_manager:
                self.state_manager.set_state_value(
                    "resource_allocation", 
                    {
                        "concurrent_tasks": self.current_concurrent_tasks,
                        "cpu_usage": cpu_usage,
                        "memory_usage": memory_usage,
                        "timestamp": time.time()
                    }
                )
    
    def _adjust_module_states(self, target_state: ModuleState, module_names: List[str]):
        """
        Adjust the operational state of specified modules.
        
        Args:
            target_state: The state to set modules to.
            module_names: List of module names to adjust.
        """
        with self.module_lock:
            for name in module_names:
                if name in self.modules:
                    module_info = self.modules[name]
                    current_state = module_info.state
                    
                    # Only change state if different
                    if current_state != target_state:
                        # For standby, make sure the module supports it
                        if target_state == ModuleState.STANDBY:
                            module_info.update_state(ModuleState.STANDBY, "Resource conservation mode")
                            logger.info(f"Module {name} placed in STANDBY mode due to high system load")
                        
                        # For ready, reactivate the module
                        elif target_state == ModuleState.READY and current_state == ModuleState.STANDBY:
                            module_info.update_state(ModuleState.READY, "Resumed from standby")
                            logger.info(f"Module {name} reactivated from STANDBY mode")
    
    async def processTranscription(self, segment: TranscriptionSegment) -> ProcessedSegment:
        """
        Process incoming transcription segments.
        Takes advantage of dynamic resource allocation to adjust processing.
        
        Args:
            segment: The transcription segment to process.
        
        Returns:
            A processed segment with analysis results.
        """
        if not self.initialized:
            raise RuntimeError("Processing Core is not initialized")
        
        # Update module state to ACTIVE during processing
        if "entity_extractor" in self.modules:
            self.modules["entity_extractor"].update_state(ModuleState.ACTIVE)
        if "intent_classifier" in self.modules:
            self.modules["intent_classifier"].update_state(ModuleState.ACTIVE)
        if "sentiment_analyzer" in self.modules:
            self.modules["sentiment_analyzer"].update_state(ModuleState.ACTIVE)
        
        start_time = time.time()
        
        try:
            # Create a processed segment with default values
            processed = ProcessedSegment(
                text=segment.text,
                speaker=segment.speaker,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                metadata=segment.metadata.copy() if segment.metadata else {}
            )
            
            # Create a semaphore to limit concurrent processing based on resource allocation
            semaphore = asyncio.Semaphore(self.current_concurrent_tasks)
            
            # Process the segment with parallel tasks, controlled by semaphore
            async def run_with_semaphore(coro):
                async with semaphore:
                    return await coro
            
            entity_task = run_with_semaphore(self.extractEntities(segment.text))
            intent_task = run_with_semaphore(self.identifyIntents(segment.text))
            sentiment_task = run_with_semaphore(self.analyzeSentiment(segment.text))
            
            # Gather results
            results = await asyncio.gather(
                entity_task,
                intent_task,
                sentiment_task,
                return_exceptions=True
            )
            
            # Extract results
            processed.entities = results[0] if not isinstance(results[0], Exception) else []
            processed.intents = results[1] if not isinstance(results[1], Exception) else []
            processed.sentiment = results[2] if not isinstance(results[2], Exception) else None
            
            # Handle any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in processing segment: {str(result)}")
                    self.metrics.errors_encountered += 1
                    
                    # Update module state to ERROR
                    if i == 0 and "entity_extractor" in self.modules:
                        self.modules["entity_extractor"].update_state(ModuleState.ERROR, str(result))
                    elif i == 1 and "intent_classifier" in self.modules:
                        self.modules["intent_classifier"].update_state(ModuleState.ERROR, str(result))
                    elif i == 2 and "sentiment_analyzer" in self.modules:
                        self.modules["sentiment_analyzer"].update_state(ModuleState.ERROR, str(result))
            
            # Apply plugins if manager is in READY or ACTIVE state
            if self.plugin_manager and "plugin_manager" in self.modules:
                module_state = self.modules["plugin_manager"].state
                if module_state in (ModuleState.READY, ModuleState.ACTIVE):
                    # Convert to dictionary for plugin processing
                    segment_dict = {
                        "text": processed.text,
                        "speaker": processed.speaker,
                        "start_time": processed.start_time,
                        "end_time": processed.end_time,
                        "confidence": processed.confidence,
                        "entities": [e.to_dict() for e in processed.entities] if processed.entities else [],
                        "intents": [i.to_dict() for i in processed.intents] if processed.intents else [],
                        "sentiment": processed.sentiment.to_dict() if processed.sentiment else None,
                        "metadata": processed.metadata or {}
                    }
                    
                    # Update module state
                    self.modules["plugin_manager"].update_state(ModuleState.ACTIVE)
                    
                    # Process with plugins
                    plugin_start = time.time()
                    processed_dict = self.plugin_manager.process_data(segment_dict)
                    self.metrics.plugin_processing_time += time.time() - plugin_start
                    
                    # Update processed segment with plugin results
                    processed.metadata = processed_dict.get("metadata", {})
                    # Other fields could be updated here if needed
                    
                    # Return to READY state
                    self.modules["plugin_manager"].update_state(ModuleState.READY)
            
            # Record metrics
            self.metrics.segments_processed += 1
            self.metrics.entities_extracted += len(processed.entities) if processed.entities else 0
            self.metrics.intents_identified += len(processed.intents) if processed.intents else 0
            self.metrics.sentiment_analyzed += 1 if processed.sentiment else 0
            
            processing_time = time.time() - start_time
            self.metrics.record_processing_time(processing_time)
            
            # Add segment to conversation context if conversation ID is provided
            if segment.metadata and "conversation_id" in segment.metadata:
                conversation_id = segment.metadata["conversation_id"]
                self._add_segment_to_conversation(conversation_id, processed)
            
            # Update module states back to READY
            if "entity_extractor" in self.modules and self.modules["entity_extractor"].state == ModuleState.ACTIVE:
                self.modules["entity_extractor"].update_state(ModuleState.READY)
            if "intent_classifier" in self.modules and self.modules["intent_classifier"].state == ModuleState.ACTIVE:
                self.modules["intent_classifier"].update_state(ModuleState.READY)
            if "sentiment_analyzer" in self.modules and self.modules["sentiment_analyzer"].state == ModuleState.ACTIVE:
                self.modules["sentiment_analyzer"].update_state(ModuleState.READY)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing transcription segment: {str(e)}")
            self.metrics.errors_encountered += 1
            
            # Update core operational state in case of critical error
            if self.operational_state == ModuleState.ACTIVE:
                self.operational_state = ModuleState.ERROR
                logger.critical(f"Processing Core entered ERROR state: {str(e)}")
                
                # Attempt recovery
                if self.state_manager:
                    try:
                        # Record error in state manager
                        self.state_manager.set_state_value("last_error", {
                            "message": str(e),
                            "timestamp": time.time(),
                            "segment_id": segment.metadata.get("id") if segment.metadata else None
                        })
                        
                        # Return to READY state after error recording
                        self.operational_state = ModuleState.READY
                        logger.info("Processing Core recovered from ERROR state")
                    except Exception as recovery_error:
                        logger.error(f"Failed to recover from error: {str(recovery_error)}")
            
            # Return a minimal processed segment on error
            return ProcessedSegment(
                text=segment.text,
                speaker=segment.speaker,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
                metadata={"error": str(e)}
            )
    
    def _add_segment_to_conversation(self, conversation_id: str, segment: ProcessedSegment):
        """
        Add a processed segment to a conversation context.
        
        Args:
            conversation_id: The ID of the conversation.
            segment: The processed segment to add.
        """
        with self.conversation_lock:
            # Get or create conversation context
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = ConversationContext(
                    id=conversation_id,
                    segments=[],
                    metadata={"created_at": time.time()}
                )
            
            # Add segment to conversation
            conversation = self.conversations[conversation_id]
            conversation.segments.append(segment)
            conversation.last_updated = time.time()
            
            # Update state if needed
            if self.state_manager:
                self.state_manager.set_state_value(
                    f"conversation_{conversation_id}",
                    {
                        "id": conversation_id,
                        "num_segments": len(conversation.segments),
                        "last_updated": conversation.last_updated
                    }
                )
    
    async def extractEntities(self, text: str) -> List[Entity]:
        """
        Extract key entities from text.
        
        Args:
            text: The text to extract entities from.
        
        Returns:
            A list of extracted entities.
        """
        if not self.initialized:
            raise RuntimeError("Processing Core is not initialized")
        
        start_time = time.time()
        
        try:
            entities = await self.entity_extractor.extract_entities(text)
            self.metrics.entity_extraction_time += time.time() - start_time
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            self.metrics.errors_encountered += 1
            return []
    
    async def identifyIntents(self, statement: str) -> List[Intent]:
        """
        Identify intents in customer statements.
        
        Args:
            statement: The statement to analyze.
        
        Returns:
            A list of identified intents.
        """
        if not self.initialized:
            raise RuntimeError("Processing Core is not initialized")
        
        start_time = time.time()
        
        try:
            intents = await self.intent_classifier.identify_intents(statement)
            self.metrics.intent_classification_time += time.time() - start_time
            return intents
        except Exception as e:
            logger.error(f"Error identifying intents: {str(e)}")
            self.metrics.errors_encountered += 1
            return []
    
    async def analyzeSentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment in text.
        
        Args:
            text: The text to analyze.
        
        Returns:
            A SentimentAnalysis object with sentiment information.
        """
        if not self.initialized:
            raise RuntimeError("Processing Core is not initialized")
        
        start_time = time.time()
        
        try:
            sentiment = await self.sentiment_analyzer.analyze_sentiment(text)
            self.metrics.sentiment_analysis_time += time.time() - start_time
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            self.metrics.errors_encountered += 1
            return SentimentAnalysis("neutral", 0.5)
    
    async def generateSummary(self, conversation: ConversationContext) -> Summary:
        """
        Generate real-time summaries of conversation.
        
        Args:
            conversation: The conversation context to summarize.
        
        Returns:
            A Summary object with the generated summary.
        """
        if not self.initialized:
            raise RuntimeError("Processing Core is not initialized")
        
        start_time = time.time()
        
        try:
            # For a complete implementation, this would use a summarization model
            # Here we're just concatenating a simple summary
            
            # Extract key information from conversation
            all_text = "\n".join([s.text for s in conversation.segments])
            
            # Count entities and intents
            all_entities = []
            all_intents = []
            sentiment_scores = []
            
            for segment in conversation.segments:
                if segment.entities:
                    all_entities.extend(segment.entities)
                if segment.intents:
                    all_intents.extend(segment.intents)
                if segment.sentiment:
                    sentiment_scores.append(segment.sentiment.score)
            
            # Generate a simple summary
            summary_text = f"Conversation with {len(conversation.segments)} exchanges."
            
            if all_entities:
                entity_types = set(e.entity_type for e in all_entities)
                summary_text += f" Mentions {len(all_entities)} entities of types: {', '.join(entity_types)}."
            
            if all_intents:
                intent_types = set(i.intent_type for i in all_intents)
                summary_text += f" Expresses intents: {', '.join(intent_types)}."
            
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_desc = "positive" if avg_sentiment > 0.6 else "neutral" if avg_sentiment > 0.4 else "negative"
                summary_text += f" Overall sentiment is {sentiment_desc}."
            
            # Create highlights
            highlights = []
            for segment in conversation.segments:
                if segment.intents and any(i.confidence > 0.8 for i in segment.intents):
                    highlights.append(segment.text)
            
            # Create summary object
            summary = Summary(
                text=summary_text,
                highlights=highlights[:3] if highlights else None,  # Just keep a few highlights
                metadata={
                    "conversation_id": conversation.id,
                    "segment_count": len(conversation.segments),
                    "entity_count": len(all_entities),
                    "intent_count": len(all_intents)
                },
                generated_at=time.time()
            )
            
            # Update conversation with summary
            with self.conversation_lock:
                if conversation.id in self.conversations:
                    self.conversations[conversation.id].summary = summary_text
            
            self.metrics.summaries_generated += 1
            self.metrics.summarization_time += time.time() - start_time
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            self.metrics.errors_encountered += 1
            
            # Return a minimal summary on error
            return Summary(
                text=f"Error generating summary: {str(e)}",
                generated_at=time.time()
            )
    
    def registerPlugin(self, plugin) -> None:
        """
        Register custom processing plugins.
        
        Args:
            plugin: The plugin to register.
        """
        if not self.initialized:
            raise RuntimeError("Processing Core is not initialized")
        
        if not self.plugin_manager:
            logger.error("Plugin manager is not initialized")
            return
        
        try:
            # If plugin is a name, register by name
            if isinstance(plugin, str):
                success = self.plugin_manager.register_plugin(plugin)
                if success:
                    logger.info(f"Registered plugin: {plugin}")
                else:
                    logger.error(f"Failed to register plugin: {plugin}")
            else:
                # Register plugin instance by adding to available plugins
                plugin_name = getattr(plugin, "name", plugin.__class__.__name__.lower())
                
                # Add to plugin manager if has the right interface
                if (hasattr(plugin, "initialize") and 
                    hasattr(plugin, "process") and 
                    hasattr(plugin, "get_metadata") and
                    hasattr(plugin, "name")):
                    
                    # TODO: This is a simplified approach, in a real system
                    # we would need to add the plugin to the plugin manager's available plugins
                    
                    self.plugin_manager.available_plugins[plugin_name] = plugin.__class__
                    success = self.plugin_manager.register_plugin(plugin_name)
                    
                    if success:
                        logger.info(f"Registered plugin: {plugin_name}")
                    else:
                        logger.error(f"Failed to register plugin: {plugin_name}")
                else:
                    logger.error(f"Invalid plugin interface: {plugin}")
        
        except Exception as e:
            logger.error(f"Error registering plugin: {str(e)}")
    
    def getProcessingMetrics(self) -> Dict[str, Any]:
        """
        Get processing metrics and performance data.
        
        Returns:
            A dictionary with processing metrics.
        """
        metrics = self.metrics.get_metrics()
        
        # Add resource metrics if available
        if self.resource_monitor:
            resource_metrics = self.resource_monitor.get_resource_metrics()
            metrics["resources"] = resource_metrics
        
        # Add plugin metrics if available
        if self.plugin_manager:
            metrics["plugins"] = {
                "active_plugins": self.plugin_manager.get_active_plugins(),
                "available_plugins": self.plugin_manager.get_available_plugins()
            }
        
        return metrics
    
    def shutdown(self):
        """
        Shutdown the processing core with graceful module termination.
        Modules are shut down in reverse dependency order to ensure proper cleanup.
        """
        logger.info("Shutting down Processing Core")
        self.operational_state = ModuleState.SHUTDOWN
        
        # Get modules in reverse dependency order
        try:
            shutdown_order = list(reversed(self.get_initialization_order()))
        except Exception as e:
            logger.error(f"Error determining shutdown order: {e}")
            shutdown_order = list(self.modules.keys())
        
        # Shutdown each module in order
        with self.module_lock:
            for module_name in shutdown_order:
                if module_name not in self.modules:
                    continue
                
                module_info = self.modules[module_name]
                logger.info(f"Shutting down module: {module_name}")
                
                try:
                    # Update module state
                    module_info.update_state(ModuleState.SHUTDOWN, "Shutting down")
                    
                    # Call shutdown method if available
                    if hasattr(module_info.instance, 'shutdown'):
                        module_info.instance.shutdown()
                    elif hasattr(module_info.instance, 'stop'):
                        module_info.instance.stop()
                    
                    logger.info(f"Module {module_name} shutdown complete")
                except Exception as e:
                    logger.error(f"Error shutting down module {module_name}: {str(e)}")
        
        # For backward compatibility, ensure these are definitely stopped
        if self.resource_monitor:
            try:
                self.resource_monitor.stop()
            except Exception as e:
                logger.error(f"Error stopping resource monitor: {str(e)}")
        
        if self.state_manager:
            try:
                self.state_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping state manager: {str(e)}")
        
        self.initialized = False
        logger.info("Processing Core shutdown complete")