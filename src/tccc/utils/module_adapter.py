"""
Module Adapter for TCCC.ai system.

This module provides adapter functions to convert between different module interfaces
and ensure compatibility with the standardized event schema.
"""

import time
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Coroutine

from tccc.utils.event_schema import (
    BaseEvent, AudioSegmentEvent, TranscriptionEvent, 
    ProcessedTextEvent, ErrorEvent, EventType
)
from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class AudioPipelineAdapter:
    """Adapter for Audio Pipeline module."""
    
    @staticmethod
    async def get_audio_segment_async(audio_pipeline) -> Optional[Dict[str, Any]]:
        """
        Asynchronously get audio segment from audio pipeline.
        
        Args:
            audio_pipeline: Audio pipeline instance
            
        Returns:
            Audio segment data in standardized format, or None if no data available
        """
        try:
            # Determine if the audio pipeline has async methods
            if hasattr(audio_pipeline, 'get_audio_segment_async') and callable(audio_pipeline.get_audio_segment_async):
                audio_data = await audio_pipeline.get_audio_segment_async()
                logger.debug("Got audio from get_audio_segment_async()")
            elif hasattr(audio_pipeline, 'get_audio_async') and callable(audio_pipeline.get_audio_async):
                audio_data = await audio_pipeline.get_audio_async()
                logger.debug("Got audio from get_audio_async()")
            else:
                # Fall back to synchronous methods using a thread pool
                audio_data = await AudioPipelineAdapter._get_audio_sync_in_thread(audio_pipeline)
            
            return AudioPipelineAdapter._process_audio_data(audio_pipeline, audio_data)
                
        except Exception as e:
            logger.error(f"Error getting audio segment asynchronously: {e}")
            return None
    
    @staticmethod
    def get_audio_segment(audio_pipeline) -> Optional[Dict[str, Any]]:
        """
        Synchronously get audio segment from audio pipeline.
        
        Args:
            audio_pipeline: Audio pipeline instance
            
        Returns:
            Audio segment data in standardized format, or None if no data available
        """
        try:
            # Get audio data using synchronous methods
            audio_data = AudioPipelineAdapter._get_audio_sync(audio_pipeline)
            return AudioPipelineAdapter._process_audio_data(audio_pipeline, audio_data)
                
        except Exception as e:
            logger.error(f"Error getting audio segment: {e}")
            return None
    
    @staticmethod
    async def _get_audio_sync_in_thread(audio_pipeline) -> Optional[np.ndarray]:
        """
        Get audio data using synchronous methods in a thread pool.
        
        Args:
            audio_pipeline: Audio pipeline instance
            
        Returns:
            Audio data as numpy array or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: AudioPipelineAdapter._get_audio_sync(audio_pipeline)
        )
    
    @staticmethod
    def _get_audio_sync(audio_pipeline) -> Optional[np.ndarray]:
        """
        Get audio data using synchronous methods.
        
        Args:
            audio_pipeline: Audio pipeline instance
            
        Returns:
            Audio data as numpy array or None
        """
        audio_data = None
        
        # Try different methods to get audio data
        if hasattr(audio_pipeline, 'get_audio_segment') and callable(audio_pipeline.get_audio_segment):
            audio_data = audio_pipeline.get_audio_segment()
            logger.debug("Got audio from get_audio_segment()")
        elif hasattr(audio_pipeline, 'get_audio') and callable(audio_pipeline.get_audio):
            audio_data = audio_pipeline.get_audio()
            logger.debug("Got audio from get_audio()")
        elif hasattr(audio_pipeline, 'get_audio_data') and callable(audio_pipeline.get_audio_data):
            audio_data = audio_pipeline.get_audio_data()
            logger.debug("Got audio from get_audio_data()")
        
        return audio_data
    
    @staticmethod
    def _process_audio_data(audio_pipeline, audio_data) -> Optional[Dict[str, Any]]:
        """
        Process and validate audio data into standardized event format.
        
        Args:
            audio_pipeline: Audio pipeline instance
            audio_data: Audio data to process
            
        Returns:
            Standardized event dictionary or None if data is invalid
        """
        # Validate audio data
        if audio_data is None:
            return None
            
        # Check for empty data
        if hasattr(audio_data, 'size') and audio_data.size == 0:
            return None
            
        # Convert to numpy array if it's not already
        if not isinstance(audio_data, np.ndarray):
            try:
                audio_data = np.array(audio_data, dtype=np.float32)
                logger.debug("Converted audio data to numpy array")
            except Exception as e:
                logger.error(f"Failed to convert audio data to numpy array: {e}")
                return None
        
        # Ensure audio data is in float32 format
        if audio_data.dtype != np.float32:
            try:
                # Convert int16 PCM to float32 normalized to [-1.0, 1.0]
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                # For other formats, just convert to float32
                else:
                    audio_data = audio_data.astype(np.float32)
                logger.debug(f"Converted audio data from {audio_data.dtype} to float32")
            except Exception as e:
                logger.error(f"Failed to convert audio data to float32: {e}")
                # Continue with original format if conversion fails
                
        # Extract audio properties
        sample_rate = 16000  # Default
        if hasattr(audio_pipeline, 'config') and audio_pipeline.config:
            if 'audio' in audio_pipeline.config:
                sample_rate = audio_pipeline.config['audio'].get('sample_rate', 16000)
        
        # Determine if speech is present
        is_speech = False
        if hasattr(audio_pipeline, 'audio_processor') and audio_pipeline.audio_processor:
            # Audio processor might have VAD results
            is_speech = getattr(audio_pipeline.audio_processor, 'is_speech', False)
        
        # Get audio duration
        if hasattr(audio_data, '__len__'):
            duration_ms = len(audio_data) / sample_rate * 1000
        elif hasattr(audio_data, 'shape') and len(audio_data.shape) > 0:
            duration_ms = audio_data.shape[0] / sample_rate * 1000
        else:
            duration_ms = 0
            logger.warning("Could not determine audio duration")
        
        # Get audio format
        format_type = "PCM16"  # Default format
        if hasattr(audio_data, 'dtype'):
            if audio_data.dtype == np.float32:
                format_type = "FLOAT32"
            elif audio_data.dtype == np.int16:
                format_type = "PCM16"
        
        # Get number of channels
        channels = 1  # Default to mono
        if hasattr(audio_data, 'shape') and len(audio_data.shape) > 1:
            channels = audio_data.shape[1]
        
        # Create standardized event
        event = AudioSegmentEvent(
            source="audio_pipeline",
            audio_data=audio_data,
            sample_rate=sample_rate,
            format_type=format_type,
            channels=channels,
            duration_ms=duration_ms,
            is_speech=is_speech,
            start_time=time.time(),
            metadata={
                "source_device": getattr(audio_pipeline, 'active_source', {}).get('name', 'unknown') 
                if hasattr(audio_pipeline, 'active_source') else "unknown",
                "dtype": str(audio_data.dtype) if hasattr(audio_data, 'dtype') else "unknown",
                "shape": str(audio_data.shape) if hasattr(audio_data, 'shape') else "unknown"
            }
        )
        
        return event.to_dict()


class STTEngineAdapter:
    """Adapter for STT Engine module."""
    
    @staticmethod
    def convert_audio_event_to_input(event: Dict[str, Any], audio_data: Any) -> Dict[str, Any]:
        """
        Convert audio event to STT input format.
        
        Args:
            event: Audio event data
            audio_data: Raw audio data
            
        Returns:
            STT input dictionary
        """
        return {
            # Include original audio data
            "audio": audio_data,
            
            # Add metadata from event
            "metadata": {
                "source": event.get("source", "unknown"),
                "timestamp": event.get("timestamp", time.time()),
                "session_id": event.get("session_id", "unknown"),
                "sample_rate": event.get("data", {}).get("sample_rate", 16000),
                "format": event.get("data", {}).get("format", "PCM16"),
                "duration_ms": event.get("data", {}).get("duration_ms", 0),
                "is_speech": event.get("data", {}).get("is_speech", False)
            }
        }
    
    @staticmethod
    def convert_transcription_to_event(transcription: Dict[str, Any], source_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert STT transcription to standard event format.
        
        Args:
            transcription: STT transcription result
            source_event: Original audio event if available
            
        Returns:
            Standardized transcription event
        """
        try:
            # Extract session_id from source event if available
            session_id = None
            if source_event:
                session_id = source_event.get("session_id")
            
            # Create event from transcription
            event = TranscriptionEvent(
                source="stt_engine",
                text=transcription.get("text", ""),
                segments=transcription.get("segments", []),
                language=transcription.get("language", "en"),
                confidence=next(
                    (s.get("confidence", 0.0) for s in transcription.get("segments", []) if "confidence" in s), 
                    0.0
                ),
                is_partial=transcription.get("is_partial", False),
                metadata={
                    "audio_duration_ms": transcription.get("metrics", {}).get("audio_duration", 0) * 1000 
                    if "metrics" in transcription else 0,
                    "processing_ms": transcription.get("metrics", {}).get("processing_time", 0) * 1000 
                    if "metrics" in transcription else 0,
                    "model": transcription.get("model", transcription.get("metadata", {}).get("model", "unknown")) 
                    if "model" in transcription or "metadata" in transcription else "unknown"
                },
                session_id=session_id
            )
            
            return event.to_dict()
            
        except Exception as e:
            logger.error(f"Error converting transcription to event: {e}")
            
            # Return error event
            error_event = ErrorEvent(
                source="module_adapter",
                error_code="transcription_conversion_error",
                message=f"Failed to convert transcription: {str(e)}",
                component="STTEngineAdapter",
                recoverable=True,
                session_id=source_event.get("session_id") if source_event else None
            )
            
            return error_event.to_dict()


class ProcessingCoreAdapter:
    """Adapter for Processing Core module."""
    
    @staticmethod
    def convert_transcription_to_input(event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert transcription event to Processing Core input format.
        
        Args:
            event: Transcription event data
            
        Returns:
            Processing Core input dictionary
        """
        return {
            "text": event.get("data", {}).get("text", ""),
            "metadata": {
                "source": event.get("source", "unknown"),
                "timestamp": event.get("timestamp", time.time()),
                "session_id": event.get("session_id", "unknown"),
                "language": event.get("data", {}).get("language", "en"),
                "confidence": event.get("data", {}).get("confidence", 0.0),
                "is_partial": event.get("data", {}).get("is_partial", False)
            }
        }
    
    @staticmethod
    def convert_processed_to_event(processed: Dict[str, Any], source_event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert Processing Core result to standard event format.
        
        Args:
            processed: Processing Core result
            source_event: Original transcription event if available
            
        Returns:
            Standardized processed text event
        """
        try:
            # Extract session_id from source event if available
            session_id = None
            if source_event:
                session_id = source_event.get("session_id")
            
            # Extract required fields
            text = processed.get("text", processed.get("raw_text", ""))
            entities = processed.get("entities", [])
            intent = processed.get("intent", {})
            sentiment = processed.get("sentiment")
            
            # Create event
            event = ProcessedTextEvent(
                source="processing_core",
                text=text,
                entities=entities,
                intent=intent,
                sentiment=sentiment,
                metadata={"processing_ms": processed.get("processing_time", 0) * 1000},
                session_id=session_id
            )
            
            return event.to_dict()
            
        except Exception as e:
            logger.error(f"Error converting processed result to event: {e}")
            
            # Return error event
            error_event = ErrorEvent(
                source="module_adapter",
                error_code="processed_conversion_error",
                message=f"Failed to convert processed result: {str(e)}",
                component="ProcessingCoreAdapter",
                recoverable=True,
                session_id=source_event.get("session_id") if source_event else None
            )
            
            return error_event.to_dict()


def standardize_event(data: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    """
    Convert arbitrary data to a standardized event format.
    
    Args:
        data: Data to convert
        source: Source component
        
    Returns:
        Standardized event dictionary
    """
    event_type = None
    
    # Try to determine event type from data
    if "type" in data:
        event_type = data["type"]
    elif "text" in data and "segments" in data:
        event_type = EventType.TRANSCRIPTION.value
    elif "text" in data and "entities" in data:
        event_type = EventType.PROCESSED_TEXT.value
    elif "audio" in data or hasattr(data, 'dtype'):
        event_type = EventType.AUDIO_SEGMENT.value
    elif "error" in data or "error_code" in data:
        event_type = EventType.ERROR.value
    else:
        # Default event type
        event_type = "generic_event"
    
    # Create base event
    base_event = BaseEvent(
        event_type=event_type,
        source=source,
        data=data,
        metadata={},
        session_id=data.get("session_id"),
        sequence=data.get("sequence"),
        timestamp=data.get("timestamp", time.time())
    )
    
    return base_event.to_dict()


def extract_event_data(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract usable data from an event.
    
    Args:
        event: Event dictionary
        
    Returns:
        Dictionary with extracted data
    """
    # Start with the data field if it exists
    result = event.get("data", {}).copy() if isinstance(event.get("data"), dict) else {}
    
    # Add fields from the event root if they're not in data
    for key, value in event.items():
        if key not in ["data", "metadata", "type", "source", "timestamp", "session_id", "sequence"]:
            if key not in result:
                result[key] = value
    
    # Add metadata
    if "metadata" in event and isinstance(event["metadata"], dict):
        result["metadata"] = event["metadata"]
    
    # Ensure some common fields are available
    if "text" not in result and "text" in event:
        result["text"] = event["text"]
    
    if "timestamp" not in result and "timestamp" in event:
        result["timestamp"] = event["timestamp"]
    
    return result


async def run_method_async(method: Callable, *args, **kwargs) -> Any:
    """
    Run a method asynchronously, handling both sync and async methods.
    
    Args:
        method: Method to run
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
        
    Returns:
        Result from the method
    """
    if asyncio.iscoroutinefunction(method):
        # Method is already async, just await it
        return await method(*args, **kwargs)
    else:
        # Method is sync, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: method(*args, **kwargs)
        )


def create_async_method(obj: Any, method_name: str) -> Callable:
    """
    Create an async wrapper for a method that might be sync or async.
    
    Args:
        obj: Object containing the method
        method_name: Name of the method to wrap
        
    Returns:
        Async function that wraps the method
    """
    if not hasattr(obj, method_name):
        async def missing_method(*args, **kwargs):
            logger.warning(f"Method {method_name} not found on {obj.__class__.__name__}")
            return None
        return missing_method
        
    method = getattr(obj, method_name)
    
    if asyncio.iscoroutinefunction(method):
        # Already async, just return it
        return method
    else:
        # Create async wrapper
        async def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: method(*args, **kwargs)
            )
        return async_wrapper