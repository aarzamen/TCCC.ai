"""
Processing Core module for TCCC.ai system.

This module implements the main Processing Core component of the TCCC.ai system.
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass

from tccc.utils.logging import get_logger
from tccc.utils.config import Config
from tccc.processing_core.entity_extractor import EntityExtractor, Entity
from tccc.processing_core.intent_classifier import IntentClassifier, Intent
from tccc.processing_core.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysis
from tccc.processing_core.resource_monitor import ResourceMonitor
from tccc.processing_core.plugin_manager import PluginManager
from tccc.processing_core.state_manager import StateManager

logger = get_logger(__name__)


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
    sentiment analysis, and conversation summarization.
    """
    
    def __init__(self):
        """Initialize the Processing Core."""
        self.initialized = False
        self.config: Optional[Dict[str, Any]] = None
        
        # Components
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
            
            # Initialize the resource monitor first
            resource_config = config.get("resource_management", {})
            self.resource_monitor = ResourceMonitor(resource_config)
            self.resource_monitor.start()
            
            # Initialize state manager
            state_config = config.get("state_management", {})
            self.state_manager = StateManager(state_config)
            
            # Initialize entity extractor
            entity_config = config.get("entity_extraction", {})
            self.entity_extractor = EntityExtractor(entity_config)
            
            # Initialize intent classifier
            intent_config = config.get("intent_classification", {})
            self.intent_classifier = IntentClassifier(intent_config)
            
            # Initialize sentiment analyzer
            sentiment_config = config.get("sentiment_analysis", {})
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
            
            # Initialize plugin manager
            plugin_config = config.get("plugins", {})
            self.plugin_manager = PluginManager(plugin_config)
            
            # Register resource monitor callback to track resource usage
            if self.resource_monitor and self.state_manager:
                self.resource_monitor.register_callback(self._on_resource_update)
            
            self.initialized = True
            logger.info("Processing Core initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Processing Core: {str(e)}")
            return False
    
    def _on_resource_update(self, usage, *args, **kwargs):
        """
        Callback for resource usage updates.
        
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
    
    async def processTranscription(self, segment: TranscriptionSegment) -> ProcessedSegment:
        """
        Process incoming transcription segments.
        
        Args:
            segment: The transcription segment to process.
        
        Returns:
            A processed segment with analysis results.
        """
        if not self.initialized:
            raise RuntimeError("Processing Core is not initialized")
        
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
            
            # Process the segment with parallel tasks
            entity_task = self.extractEntities(segment.text)
            intent_task = self.identifyIntents(segment.text)
            sentiment_task = self.analyzeSentiment(segment.text)
            
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
            
            # Apply any plugins
            if self.plugin_manager:
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
                
                plugin_start = time.time()
                processed_dict = self.plugin_manager.process_data(segment_dict)
                self.metrics.plugin_processing_time += time.time() - plugin_start
                
                # Update processed segment with plugin results
                processed.metadata = processed_dict.get("metadata", {})
                # Other fields could be updated here if needed
            
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
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing transcription segment: {str(e)}")
            self.metrics.errors_encountered += 1
            
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
        Shutdown the processing core.
        """
        logger.info("Shutting down Processing Core")
        
        # Stop the resource monitor
        if self.resource_monitor:
            self.resource_monitor.stop()
        
        # Stop the state manager
        if self.state_manager:
            self.state_manager.stop()
        
        self.initialized = False
        logger.info("Processing Core shutdown complete")