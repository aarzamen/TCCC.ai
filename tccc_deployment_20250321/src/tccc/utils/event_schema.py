"""
TCCC Event Schema Implementation

This module defines the standardized event schema for TCCC system components.
All inter-component communication should use these event structures.
"""

import time
import json
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Union, TypedDict


class EventType(str, Enum):
    """Standard event types for TCCC system."""
    AUDIO_SEGMENT = "audio_segment"
    TRANSCRIPTION = "transcription"
    PROCESSED_TEXT = "processed_text"
    LLM_ANALYSIS = "llm_analysis"
    DOCUMENT_QUERY = "document_query"
    DOCUMENT_RESULTS = "document_results"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    COMMAND = "command"
    INITIALIZATION = "initialization"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventMetadata(TypedDict, total=False):
    """Optional metadata for events."""
    source_device: str
    processing_ms: int
    model: str
    context: str
    uptime_sec: float
    device_info: Dict[str, Any]
    traceback: str


class BaseEvent:
    """Base event class for all TCCC events."""
    
    def __init__(
        self, 
        event_type: Union[EventType, str],
        source: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sequence: Optional[int] = None,
        timestamp: Optional[float] = None
    ):
        """
        Initialize a standard event.
        
        Args:
            event_type: Type of event
            source: Source component
            data: Event data
            metadata: Optional metadata
            session_id: Session identifier
            sequence: Event sequence number
            timestamp: Event timestamp (defaults to current time)
        """
        self.type = event_type if isinstance(event_type, str) else event_type.value
        self.source = source
        self.timestamp = timestamp or time.time()
        self.session_id = session_id or str(uuid.uuid4())
        self.sequence = sequence
        self.data = data
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        event_dict = {
            "type": self.type,
            "timestamp": self.timestamp,
            "source": self.source,
            "session_id": self.session_id,
            "data": self.data
        }
        
        if self.sequence is not None:
            event_dict["sequence"] = self.sequence
            
        if self.metadata:
            event_dict["metadata"] = self.metadata
            
        return event_dict
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'BaseEvent':
        """Create event from dictionary."""
        return cls(
            event_type=event_dict["type"],
            source=event_dict["source"],
            data=event_dict["data"],
            metadata=event_dict.get("metadata"),
            session_id=event_dict.get("session_id"),
            sequence=event_dict.get("sequence"),
            timestamp=event_dict.get("timestamp")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEvent':
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


class AudioSegmentEvent(BaseEvent):
    """Event for audio segment data."""
    
    def __init__(
        self,
        source: str,
        audio_data: Any,
        sample_rate: int,
        format_type: str,
        channels: int,
        duration_ms: float,
        is_speech: bool,
        start_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sequence: Optional[int] = None
    ):
        """
        Initialize audio segment event.
        
        Args:
            source: Source component
            audio_data: Audio data (numpy array, bytes, or list)
            sample_rate: Sample rate in Hz
            format_type: Audio format (e.g., PCM16)
            channels: Number of channels
            duration_ms: Duration in milliseconds
            is_speech: VAD result
            start_time: Start time of segment
            metadata: Optional metadata
            session_id: Session identifier
            sequence: Event sequence number
        """
        # Validate inputs
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}. Must be a positive integer.")
            
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"Invalid channels: {channels}. Must be a positive integer.")
            
        if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
            raise ValueError(f"Invalid duration: {duration_ms}. Must be a non-negative number.")
        
        # Store the actual audio data first, before processing it for serialization
        self.audio_data = audio_data
        
        # Identify the audio data type and create appropriate metadata
        if hasattr(audio_data, 'shape') and hasattr(audio_data, 'dtype'):
            # Likely a numpy array
            audio_info = {
                "shape": list(audio_data.shape),
                "dtype": str(audio_data.dtype),
                "format": "numpy"
            }
            
            # For reconstruction, store a serializable representation
            self._serialized_data = None
            
        elif isinstance(audio_data, bytes):
            # Byte array
            audio_info = {
                "length": len(audio_data),
                "format": "bytes"
            }
            
        elif isinstance(audio_data, list):
            # List representation (e.g., list of samples)
            audio_info = {
                "length": len(audio_data),
                "format": "list"
            }
            
        else:
            # Unknown format
            audio_info = {
                "type": str(type(audio_data)),
                "length": len(audio_data) if hasattr(audio_data, "__len__") else "unknown",
                "format": "unknown"
            }
        
        data = {
            "audio_info": audio_info,
            "sample_rate": sample_rate,
            "format": format_type,
            "channels": channels,
            "duration_ms": duration_ms,
            "is_speech": is_speech,
            "start_time": start_time or time.time()
        }
        
        super().__init__(
            event_type=EventType.AUDIO_SEGMENT,
            source=source,
            data=data,
            metadata=metadata,
            session_id=session_id,
            sequence=sequence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary with special handling for audio data."""
        # Start with the base dictionary representation
        event_dict = super().to_dict()
        
        # Create a serializable representation of audio data if not already done
        try:
            if hasattr(self.audio_data, 'shape') and hasattr(self.audio_data, 'dtype'):
                # For numpy arrays, we encode to base64 for transmission
                import numpy as np
                import base64
                import io
                
                # Numpy's save function writes to a file-like object
                buffer = io.BytesIO()
                np.save(buffer, self.audio_data)
                buffer.seek(0)
                
                # Add the base64-encoded representation to the metadata
                b64_data = base64.b64encode(buffer.read()).decode('utf-8')
                event_dict["data"]["audio_data_b64"] = b64_data
                
            elif isinstance(self.audio_data, bytes):
                # For byte arrays, we also encode to base64
                import base64
                b64_data = base64.b64encode(self.audio_data).decode('utf-8')
                event_dict["data"]["audio_data_b64"] = b64_data
                
            elif isinstance(self.audio_data, list) and len(self.audio_data) < 10000:
                # For small lists, directly include the data
                event_dict["data"]["audio_data_list"] = self.audio_data
            
            # For other types or very large lists, skip adding the data to the dict
            # as it might cause serialization issues
        except Exception as e:
            # If serialization fails, just log the error and continue without the audio data
            event_dict["data"]["audio_serialization_error"] = str(e)
        
        return event_dict
    
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'AudioSegmentEvent':
        """Create event from dictionary with special handling for audio data."""
        # Extract basic data
        data = event_dict.get("data", {})
        audio_info = data.get("audio_info", {})
        
        # Try to recover audio data from serialized format
        audio_data = None
        
        try:
            if "audio_data_b64" in data:
                # Check if this is numpy data or raw bytes
                if audio_info.get("format") == "numpy":
                    # Reconstruct numpy array
                    import numpy as np
                    import base64
                    import io
                    
                    # Decode and load
                    binary_data = base64.b64decode(data["audio_data_b64"])
                    buffer = io.BytesIO(binary_data)
                    audio_data = np.load(buffer, allow_pickle=True)
                else:
                    # Just decode as bytes
                    import base64
                    audio_data = base64.b64decode(data["audio_data_b64"])
            
            elif "audio_data_list" in data:
                # Direct list representation
                audio_data = data["audio_data_list"]
            
            # If we couldn't recover audio data, create a placeholder
            if audio_data is None:
                if audio_info.get("format") == "numpy" and "shape" in audio_info and "dtype" in audio_info:
                    # Create an empty numpy array with the right shape and dtype
                    import numpy as np
                    shape = audio_info["shape"]
                    dtype_str = audio_info["dtype"]
                    
                    # Handle common numpy dtype strings
                    if dtype_str == "float32":
                        dtype = np.float32
                    elif dtype_str == "float64":
                        dtype = np.float64
                    elif dtype_str == "int16":
                        dtype = np.int16
                    elif dtype_str == "int32":
                        dtype = np.int32
                    else:
                        # Default to float32 if we can't parse the dtype
                        dtype = np.float32
                    
                    # Create zeros array with the right shape and dtype
                    audio_data = np.zeros(shape, dtype=dtype)
                else:
                    # For other formats, create an empty bytes object
                    audio_data = b''
        except Exception as e:
            # If reconstruction fails, use an empty array/bytes
            import numpy as np
            audio_data = np.array([], dtype=np.float32)
            # Add error info to metadata
            if "metadata" not in event_dict:
                event_dict["metadata"] = {}
            event_dict["metadata"]["audio_reconstruction_error"] = str(e)
        
        # Create the event instance
        event = cls(
            source=event_dict.get("source", "unknown"),
            audio_data=audio_data,
            sample_rate=data.get("sample_rate", 16000),
            format_type=data.get("format", "PCM16"),
            channels=data.get("channels", 1),
            duration_ms=data.get("duration_ms", 0),
            is_speech=data.get("is_speech", False),
            start_time=data.get("start_time"),
            metadata=event_dict.get("metadata"),
            session_id=event_dict.get("session_id"),
            sequence=event_dict.get("sequence")
        )
        
        return event


class TranscriptionEvent(BaseEvent):
    """Event for transcription data."""
    
    def __init__(
        self,
        source: str,
        text: str,
        segments: List[Dict[str, Any]],
        language: str,
        confidence: float,
        is_partial: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sequence: Optional[int] = None
    ):
        """
        Initialize transcription event.
        
        Args:
            source: Source component
            text: Transcribed text
            segments: Detailed segments
            language: Detected language
            confidence: Overall confidence
            is_partial: Whether this is a partial result
            metadata: Optional metadata
            session_id: Session identifier
            sequence: Event sequence number
        """
        data = {
            "text": text,
            "segments": segments,
            "language": language,
            "confidence": confidence,
            "is_partial": is_partial
        }
        
        super().__init__(
            event_type=EventType.TRANSCRIPTION,
            source=source,
            data=data,
            metadata=metadata,
            session_id=session_id,
            sequence=sequence
        )


class ProcessedTextEvent(BaseEvent):
    """Event for processed text data."""
    
    def __init__(
        self,
        source: str,
        text: str,
        entities: List[Dict[str, Any]],
        intent: Dict[str, Any],
        sentiment: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sequence: Optional[int] = None
    ):
        """
        Initialize processed text event.
        
        Args:
            source: Source component
            text: Original text
            entities: Extracted entities
            intent: Detected intent
            sentiment: Optional sentiment analysis
            metadata: Optional metadata
            session_id: Session identifier
            sequence: Event sequence number
        """
        data = {
            "text": text,
            "entities": entities,
            "intent": intent
        }
        
        if sentiment is not None:
            data["sentiment"] = sentiment
        
        super().__init__(
            event_type=EventType.PROCESSED_TEXT,
            source=source,
            data=data,
            metadata=metadata,
            session_id=session_id,
            sequence=sequence
        )


class LLMAnalysisEvent(BaseEvent):
    """Event for LLM analysis data."""
    
    def __init__(
        self,
        source: str,
        summary: str,
        topics: List[str],
        medical_terms: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        document_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sequence: Optional[int] = None
    ):
        """
        Initialize LLM analysis event.
        
        Args:
            source: Source component
            summary: Text summary
            topics: Extracted topics
            medical_terms: Medical terminology
            actions: Recommended actions
            document_results: Optional document results
            metadata: Optional metadata
            session_id: Session identifier
            sequence: Event sequence number
        """
        data = {
            "summary": summary,
            "topics": topics,
            "medical_terms": medical_terms,
            "actions": actions
        }
        
        if document_results is not None:
            data["document_results"] = document_results
        
        super().__init__(
            event_type=EventType.LLM_ANALYSIS,
            source=source,
            data=data,
            metadata=metadata,
            session_id=session_id,
            sequence=sequence
        )


class ErrorEvent(BaseEvent):
    """Event for error reporting."""
    
    def __init__(
        self,
        source: str,
        error_code: str,
        message: str,
        severity: Union[ErrorSeverity, str] = ErrorSeverity.ERROR,
        component: Optional[str] = None,
        recoverable: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sequence: Optional[int] = None
    ):
        """
        Initialize error event.
        
        Args:
            source: Source component
            error_code: Error code
            message: Error message
            severity: Error severity
            component: Specific component
            recoverable: Whether error is recoverable
            metadata: Optional metadata
            session_id: Session identifier
            sequence: Event sequence number
        """
        severity_value = severity if isinstance(severity, str) else severity.value
        
        data = {
            "error_code": error_code,
            "message": message,
            "severity": severity_value,
            "recoverable": recoverable
        }
        
        if component is not None:
            data["component"] = component
        
        super().__init__(
            event_type=EventType.ERROR,
            source=source,
            data=data,
            metadata=metadata,
            session_id=session_id,
            sequence=sequence
        )


class SystemStatusEvent(BaseEvent):
    """Event for system status reporting."""
    
    def __init__(
        self,
        source: str,
        state: str,
        components: Dict[str, Dict[str, Any]],
        resources: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        sequence: Optional[int] = None
    ):
        """
        Initialize system status event.
        
        Args:
            source: Source component
            state: System state
            components: Component status
            resources: Resource usage
            metadata: Optional metadata
            session_id: Session identifier
            sequence: Event sequence number
        """
        data = {
            "state": state,
            "components": components,
            "resources": resources
        }
        
        super().__init__(
            event_type=EventType.SYSTEM_STATUS,
            source=source,
            data=data,
            metadata=metadata,
            session_id=session_id,
            sequence=sequence
        )


def create_event(event_type: Union[EventType, str], **kwargs) -> BaseEvent:
    """
    Factory function to create an event of the specified type.
    
    Args:
        event_type: Type of event to create
        **kwargs: Arguments for the event
        
    Returns:
        A BaseEvent or derived class instance
    """
    type_str = event_type if isinstance(event_type, str) else event_type.value
    
    event_classes = {
        EventType.AUDIO_SEGMENT.value: AudioSegmentEvent,
        EventType.TRANSCRIPTION.value: TranscriptionEvent,
        EventType.PROCESSED_TEXT.value: ProcessedTextEvent,
        EventType.LLM_ANALYSIS.value: LLMAnalysisEvent,
        EventType.ERROR.value: ErrorEvent,
        EventType.SYSTEM_STATUS.value: SystemStatusEvent
    }
    
    event_class = event_classes.get(type_str, BaseEvent)
    
    if event_class == BaseEvent:
        # For types without a specific class, use BaseEvent
        return BaseEvent(
            event_type=type_str,
            source=kwargs.get("source", "unknown"),
            data=kwargs.get("data", {}),
            metadata=kwargs.get("metadata"),
            session_id=kwargs.get("session_id"),
            sequence=kwargs.get("sequence"),
            timestamp=kwargs.get("timestamp")
        )
    else:
        # Let the specific class handle its initialization
        return event_class(**kwargs)