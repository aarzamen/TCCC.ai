"""
Voice Activity Detection (VAD) Manager for TCCC.ai system.

This module provides a centralized way to handle VAD functionality across
different components of the system, ensuring thread safety and consistent
interface for voice detection results.
"""

import os
import time
import numpy as np
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass

from tccc.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import webrtcvad, but provide fallback if not available
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logger.warning("webrtcvad not installed. Using energy-based VAD only.")

# Try to import torch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class VADResult:
    """Voice Activity Detection result data structure."""
    is_speech: bool
    confidence: float = 0.0
    energy_level: float = 0.0
    frame_count: int = 1
    source_component: str = "vad_manager"
    timestamp: float = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


class VADMode(Enum):
    """Voice Activity Detection mode enumeration."""
    STANDARD = 0
    AGGRESSIVE = 1
    VERY_AGGRESSIVE = 2
    BATTLEFIELD = 3
    CUSTOM = 4


class VADManager:
    """
    Manages Voice Activity Detection across system components.
    Provides thread-safe VAD instances and consistent interface
    for speech detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize VAD Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default VAD configuration
        vad_config = self.config.get('vad', {})
        self.vad_enabled = vad_config.get('enabled', True)
        self.vad_mode = vad_config.get('mode', VADMode.STANDARD.value)
        self.vad_sensitivity = vad_config.get('sensitivity', 2)
        self.vad_frame_duration_ms = vad_config.get('frame_duration_ms', 30)
        self.vad_min_speech_duration_ms = vad_config.get('min_speech_duration_ms', 100)
        self.vad_padding_duration_ms = vad_config.get('padding_duration_ms', 300)
        self.vad_energy_threshold = vad_config.get('energy_threshold', 0.005)
        self.vad_holdover_frames = vad_config.get('holdover_frames', 10)
        
        # Collection of VAD instances per component
        self.vad_instances = {}
        self.vad_locks = {}
        
        # Last N speech detection results for context
        self.result_history = []
        self.max_history = 20
        self.history_lock = threading.Lock()
        
        # State for adaptive energy threshold
        self.noise_floor = 0.01
        self.speech_frame_counter = 0
        self.silence_frame_counter = 0
        
        # Default sample rate
        self.sample_rate = self.config.get('audio', {}).get('sample_rate', 16000)
        
        # Battlefield-specific settings
        battlefield_config = vad_config.get('battlefield', {})
        self.battlefield_mode_enabled = battlefield_config.get('enabled', False)
        self.battlefield_threshold_adjustment = battlefield_config.get('threshold_adjustment', 1.5)
        self.battlefield_holdover_adjustment = battlefield_config.get('holdover_adjustment', 2.0)
        
        logger.info("VAD Manager initialized")
    
    def get_vad_instance(self, component_name: str) -> Any:
        """
        Get or create a dedicated VAD instance for a component.
        
        Args:
            component_name: Name of component requesting VAD
            
        Returns:
            VAD instance
        """
        # Check if instance already exists
        if component_name in self.vad_instances:
            return self.vad_instances[component_name]
        
        # Create new VAD instance
        try:
            if WEBRTCVAD_AVAILABLE:
                vad = webrtcvad.Vad(self.vad_sensitivity)
                logger.info(f"Created WebRTCVAD instance for {component_name} with sensitivity {self.vad_sensitivity}")
            else:
                # Create placeholder for energy-based VAD
                vad = None
                logger.info(f"Created energy-based VAD instance for {component_name}")
            
            # Store instance and create lock
            self.vad_instances[component_name] = vad
            self.vad_locks[component_name] = threading.Lock()
            
            return vad
            
        except Exception as e:
            logger.error(f"Failed to create VAD instance for {component_name}: {e}")
            return None
    
    def detect_speech(self, 
                      audio_data: np.ndarray,
                      component_name: str,
                      sample_rate: int = None) -> VADResult:
        """
        Perform speech detection on audio data.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            component_name: Name of component requesting detection
            sample_rate: Sample rate of audio (default: initialized value)
            
        Returns:
            VADResult with detection information
        """
        if not self.vad_enabled:
            # Default to assuming speech if VAD is disabled
            return VADResult(is_speech=True, confidence=1.0, source_component=component_name)
        
        # Use component sample rate if provided, else use default
        sample_rate = sample_rate or self.sample_rate
        
        # Get VAD instance for this component
        vad = self.get_vad_instance(component_name)
        
        # Get lock for this component's VAD
        vad_lock = self.vad_locks.get(component_name)
        if vad_lock is None:
            vad_lock = threading.Lock()
            self.vad_locks[component_name] = vad_lock
        
        # Combine multiple detection methods for better accuracy
        webrtc_speech = False
        energy_speech = False
        
        # Calculate energy level (common for all methods)
        if len(audio_data) == 0:
            rms_energy = 0.0
        else:
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Adaptive energy threshold based on component and mode
        adaptive_threshold = self._get_adaptive_threshold(component_name, rms_energy)
        
        # Energy-based detection (always performed)
        energy_speech = rms_energy > adaptive_threshold
        
        try:
            with vad_lock:
                # WebRTC VAD if available
                if WEBRTCVAD_AVAILABLE and vad is not None:
                    # Convert float audio to int16 for webrtcvad
                    audio_data_int16 = (audio_data * 32767).astype(np.int16)
                    
                    # Calculate frame duration in samples
                    frame_duration_samples = int(sample_rate * self.vad_frame_duration_ms / 1000)
                    
                    # Process audio in valid frame sizes for WebRTCVAD
                    webrtc_speech_frames = 0
                    total_frames = 0
                    
                    for i in range(0, len(audio_data_int16), frame_duration_samples):
                        frame = audio_data_int16[i:i+frame_duration_samples]
                        
                        # Ensure frame is the correct size
                        if len(frame) == frame_duration_samples:
                            total_frames += 1
                            frame_bytes = frame.tobytes()
                            
                            # Detect speech in frame
                            if vad.is_speech(frame_bytes, sample_rate):
                                webrtc_speech_frames += 1
                    
                    # Consider speech if enough frames detected
                    webrtc_speech = webrtc_speech_frames >= max(1, total_frames // 3)
                
        except Exception as e:
            logger.error(f"Error in WebRTC VAD processing for {component_name}: {e}")
            webrtc_speech = False
        
        # Update state tracking based on component
        # (each component has its own counters)
        self._update_component_state(component_name, energy_speech)
        
        # Combine detection methods (weighted decision)
        is_speech = False
        confidence = 0.0
        
        if WEBRTCVAD_AVAILABLE and vad is not None:
            # If WebRTC available, give it more weight
            if webrtc_speech and energy_speech:
                # Both methods agree - high confidence
                is_speech = True
                confidence = 0.9
            elif webrtc_speech:
                # Only WebRTC detected speech
                is_speech = True
                confidence = 0.7
            elif energy_speech:
                # Only energy detected speech
                is_speech = True
                confidence = 0.5
            else:
                # No speech detected
                is_speech = False
                confidence = 0.8  # High confidence in no speech
        else:
            # Energy-based only (less reliable)
            is_speech = energy_speech
            confidence = 0.6 if energy_speech else 0.5
        
        # Create result
        result = VADResult(
            is_speech=is_speech,
            confidence=confidence,
            energy_level=float(rms_energy),
            frame_count=1,
            source_component=component_name
        )
        
        # Add to history
        self._add_to_history(result)
        
        return result
    
    def _get_adaptive_threshold(self, component_name: str, current_energy: float) -> float:
        """
        Get component-specific adaptive energy threshold.
        
        Args:
            component_name: Name of component
            current_energy: Current energy level
            
        Returns:
            Adaptive threshold value
        """
        # Base threshold from config
        threshold = self.vad_energy_threshold
        
        # Apply battlefield mode if enabled
        if self.battlefield_mode_enabled:
            threshold *= self.battlefield_threshold_adjustment
        
        # Component-specific adjustments
        if "audio_pipeline" in component_name:
            # Audio pipeline needs more resilient detection
            threshold *= 0.9  # Slightly more sensitive
        elif "whisper" in component_name:
            # Whisper needs higher precision
            threshold *= 1.1  # Slightly less sensitive
        
        # Compare with noise floor
        adaptive_threshold = max(threshold, self.noise_floor * 3)
        
        # Update noise floor estimate during silence
        if current_energy < self.noise_floor * 1.5:
            # Slow adaptation
            self.noise_floor = self.noise_floor * 0.95 + current_energy * 0.05
        
        return adaptive_threshold
    
    def _update_component_state(self, component_name: str, is_speech: bool):
        """
        Update speech/silence state for specific component.
        
        Args:
            component_name: Name of component
            is_speech: Current speech detection state
        """
        # Initialize component counters if needed
        if not hasattr(self, f"{component_name}_speech_counter"):
            setattr(self, f"{component_name}_speech_counter", 0)
            setattr(self, f"{component_name}_silence_counter", 0)
        
        # Get component counters
        speech_counter = getattr(self, f"{component_name}_speech_counter")
        silence_counter = getattr(self, f"{component_name}_silence_counter")
        
        # Update counters
        if is_speech:
            speech_counter += 1
            silence_counter = 0
        else:
            silence_counter += 1
            speech_counter = 0
        
        # Store updated counters
        setattr(self, f"{component_name}_speech_counter", speech_counter)
        setattr(self, f"{component_name}_silence_counter", silence_counter)
    
    def _add_to_history(self, result: VADResult):
        """
        Add result to history with thread safety.
        
        Args:
            result: VAD result to add
        """
        with self.history_lock:
            self.result_history.append(result)
            if len(self.result_history) > self.max_history:
                self.result_history.pop(0)
    
    def get_speech_segments(self, audio_data: np.ndarray, component_name: str) -> List[Tuple[int, int]]:
        """
        Detect speech segments in longer audio.
        Returns start and end indices of each segment.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            component_name: Name of component requesting detection
            
        Returns:
            List of (start_index, end_index) tuples
        """
        if not self.vad_enabled or len(audio_data) == 0:
            # If VAD disabled, treat entire audio as speech
            return [(0, len(audio_data))]
        
        # Frame size in samples
        frame_size = int(self.sample_rate * self.vad_frame_duration_ms / 1000)
        
        # Process audio in frames
        frames = []
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i+frame_size]
            # Ensure frame is complete
            if len(frame) == frame_size:
                # Detect speech in frame
                result = self.detect_speech(frame, component_name)
                frames.append(result.is_speech)
        
        # Find segments
        segments = []
        in_speech = False
        start_idx = 0
        
        # Holdover count (to avoid cutting off speech too quickly)
        holdover = self.vad_holdover_frames
        if self.battlefield_mode_enabled:
            # Longer holdover for battlefield conditions
            holdover = int(holdover * self.battlefield_holdover_adjustment)
        
        holdover_counter = 0
        
        for i, is_speech in enumerate(frames):
            frame_start = i * frame_size
            
            if is_speech and not in_speech:
                # Speech start
                in_speech = True
                start_idx = frame_start
                holdover_counter = 0
            elif not is_speech and in_speech:
                # Potential speech end - use holdover
                holdover_counter += 1
                if holdover_counter >= holdover:
                    # End of speech after holdover
                    in_speech = False
                    end_idx = frame_start
                    segments.append((start_idx, end_idx))
                    holdover_counter = 0
            else:
                # Reset holdover if speech continues
                if in_speech and is_speech:
                    holdover_counter = 0
        
        # Handle final segment if still in speech
        if in_speech:
            segments.append((start_idx, len(audio_data)))
        
        # Ensure we have at least one segment
        if len(segments) == 0:
            # Return segment with highest energy if no clear speech
            if len(audio_data) > 0:
                # Find section with highest energy
                max_energy = 0
                max_energy_start = 0
                
                for i in range(0, len(audio_data), frame_size * 10):
                    end_idx = min(i + frame_size * 10, len(audio_data))
                    chunk = audio_data[i:end_idx]
                    energy = np.sqrt(np.mean(chunk ** 2))
                    
                    if energy > max_energy:
                        max_energy = energy
                        max_energy_start = i
                
                # Add segment with 1-second context around highest energy
                start_idx = max(0, max_energy_start - frame_size * 5)
                end_idx = min(len(audio_data), max_energy_start + frame_size * 15)
                segments.append((start_idx, end_idx))
            else:
                # Empty audio
                segments.append((0, 0))
        
        return segments
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Update VAD configuration.
        
        Args:
            config: New configuration dictionary
            
        Returns:
            Success status
        """
        try:
            vad_config = config.get('vad', {})
            
            # Update configuration
            if 'enabled' in vad_config:
                self.vad_enabled = vad_config['enabled']
            
            if 'sensitivity' in vad_config:
                new_sensitivity = int(vad_config['sensitivity'])
                self.vad_sensitivity = new_sensitivity
                
                # Update all VAD instances
                if WEBRTCVAD_AVAILABLE:
                    for component, vad in self.vad_instances.items():
                        if vad is not None:
                            with self.vad_locks[component]:
                                vad.set_mode(new_sensitivity)
            
            if 'energy_threshold' in vad_config:
                self.vad_energy_threshold = float(vad_config['energy_threshold'])
            
            if 'holdover_frames' in vad_config:
                self.vad_holdover_frames = int(vad_config['holdover_frames'])
            
            if 'frame_duration_ms' in vad_config:
                self.vad_frame_duration_ms = int(vad_config['frame_duration_ms'])
            
            # Update battlefield mode settings
            battlefield_config = vad_config.get('battlefield', {})
            if battlefield_config:
                if 'enabled' in battlefield_config:
                    self.battlefield_mode_enabled = battlefield_config['enabled']
                
                if 'threshold_adjustment' in battlefield_config:
                    self.battlefield_threshold_adjustment = float(battlefield_config['threshold_adjustment'])
                
                if 'holdover_adjustment' in battlefield_config:
                    self.battlefield_holdover_adjustment = float(battlefield_config['holdover_adjustment'])
            
            logger.info(f"Updated VAD configuration: sensitivity={self.vad_sensitivity}, "
                        f"enabled={self.vad_enabled}, battlefield_mode={self.battlefield_mode_enabled}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update VAD configuration: {e}")
            return False
    
    def set_mode(self, mode: Union[VADMode, int]) -> bool:
        """
        Set VAD detection mode.
        
        Args:
            mode: VAD mode (enum or int value)
            
        Returns:
            Success status
        """
        try:
            # Convert enum to value if needed
            if isinstance(mode, VADMode):
                mode_value = mode.value
            else:
                mode_value = int(mode)
            
            # Validate mode
            if mode_value not in [m.value for m in VADMode]:
                logger.error(f"Invalid VAD mode: {mode_value}")
                return False
            
            # Set mode
            self.vad_mode = mode_value
            
            # Apply mode settings
            if mode_value == VADMode.STANDARD.value:
                self.vad_sensitivity = 1
                self.vad_holdover_frames = 10
                self.battlefield_mode_enabled = False
            elif mode_value == VADMode.AGGRESSIVE.value:
                self.vad_sensitivity = 2
                self.vad_holdover_frames = 8
                self.battlefield_mode_enabled = False
            elif mode_value == VADMode.VERY_AGGRESSIVE.value:
                self.vad_sensitivity = 3
                self.vad_holdover_frames = 5
                self.battlefield_mode_enabled = False
            elif mode_value == VADMode.BATTLEFIELD.value:
                self.vad_sensitivity = 3
                self.vad_holdover_frames = 15
                self.battlefield_mode_enabled = True
                self.battlefield_threshold_adjustment = 1.5
                self.battlefield_holdover_adjustment = 2.0
            
            # Update all VAD instances with new sensitivity
            if WEBRTCVAD_AVAILABLE:
                for component, vad in self.vad_instances.items():
                    if vad is not None:
                        with self.vad_locks[component]:
                            vad.set_mode(self.vad_sensitivity)
            
            logger.info(f"Set VAD mode to {VADMode(mode_value).name if mode_value in [m.value for m in VADMode] else 'CUSTOM'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set VAD mode: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get VAD manager status.
        
        Returns:
            Status dictionary
        """
        # Count active instances
        active_instances = sum(1 for vad in self.vad_instances.values() if vad is not None)
        
        # Get average speech ratio from history
        speech_ratio = 0.0
        if self.result_history:
            speech_frames = sum(1 for result in self.result_history if result.is_speech)
            speech_ratio = speech_frames / len(self.result_history)
        
        status = {
            'enabled': self.vad_enabled,
            'mode': self.vad_mode,
            'mode_name': VADMode(self.vad_mode).name if self.vad_mode in [m.value for m in VADMode] else 'CUSTOM',
            'sensitivity': self.vad_sensitivity,
            'energy_threshold': self.vad_energy_threshold,
            'noise_floor': self.noise_floor,
            'active_components': len(self.vad_instances),
            'webrtc_instances': active_instances,
            'battlefield_mode': self.battlefield_mode_enabled,
            'speech_ratio': speech_ratio,
            'frame_duration_ms': self.vad_frame_duration_ms,
            'holdover_frames': self.vad_holdover_frames
        }
        
        # Add component-specific status
        component_status = {}
        for component in self.vad_instances:
            speech_counter = getattr(self, f"{component}_speech_counter", 0)
            silence_counter = getattr(self, f"{component}_silence_counter", 0)
            
            component_status[component] = {
                'speech_counter': speech_counter,
                'silence_counter': silence_counter,
                'is_active': speech_counter > 0 and speech_counter > silence_counter
            }
        
        status['components'] = component_status
        
        return status


# Singleton instance for shared access
_vad_manager_instance = None
_vad_manager_lock = threading.Lock()

def get_vad_manager(config: Dict[str, Any] = None) -> VADManager:
    """
    Get or create the singleton VAD manager instance.
    
    Args:
        config: Configuration dictionary (only used for first initialization)
        
    Returns:
        VADManager instance
    """
    global _vad_manager_instance
    
    with _vad_manager_lock:
        if _vad_manager_instance is None:
            _vad_manager_instance = VADManager(config)
            logger.info("Created new VAD Manager instance")
        
        return _vad_manager_instance