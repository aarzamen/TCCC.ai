"""
AudioPipeline implementation for TCCC.ai system.

This module provides real-time audio capture, processing, and streaming functionalities,
with support for noise reduction, audio enhancement, and voice activity detection.
"""

import os
import io
import time
import wave
import queue
import socket
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, BinaryIO, Tuple
from enum import Enum
from dataclasses import dataclass
import traceback # Added for debug printing

from tccc.utils.logging import get_logger
from tccc.utils.config import Config

logger = get_logger(__name__)

# Import ModuleState
from tccc.processing_core.processing_core import ModuleState


class AudioFormat(Enum):
    """Audio format enumeration."""
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"


class AudioSource:
    """
    Base class for audio input sources.
    Handles the capture of audio from different types of sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Audio source configuration
        """
        self.config = config
        self.name = config.get('name', 'unnamed_source')
        self.type = config.get('type', 'unknown')
        self.sample_rate = config.get('sample_rate', 16000)
        self.channels = config.get('channels', 1)
        self.format = config.get('format', 'int16')
        self.chunk_size = config.get('chunk_size', 1024)
        self.is_running = False
        self.thread = None
        self.data_callback = None
        
        # Format mapping
        self.format_map = {
            'int16': np.int16,
            'int32': np.int32,
            'float32': np.float32
        }
        
        self.dtype = self.format_map.get(self.format, np.int16)
    
    def start(self, data_callback: Callable[[np.ndarray], None]) -> bool:
        """
        Start audio capture.
        
        Args:
            data_callback: Callback function for captured audio data
            
        Returns:
            Success status
        """
        if self.is_running:
            logger.warning(f"Audio source '{self.name}' already running")
            return False
        
        self.data_callback = data_callback
        self.is_running = True
        
        self.thread = threading.Thread(
            target=self._capture_loop,
            name=f"AudioSource-{self.name}",
            daemon=True
        )
        self.thread.start()
        
        logger.info(f"Started audio source: {self.name}")
        return True
    
    def stop(self) -> bool:
        """
        Stop audio capture.
        
        Returns:
            Success status
        """
        if not self.is_running:
            logger.warning(f"Audio source '{self.name}' not running")
            return False
        
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                logger.warning(f"Audio source thread '{self.name}' did not terminate properly")
        
        logger.info(f"Stopped audio source: {self.name}")
        return True
    
    def _capture_loop(self):
        """
        Audio capture loop. Must be implemented by derived classes.
        """
        raise NotImplementedError("Audio source must implement _capture_loop")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the audio source.
        
        Returns:
            Dictionary with audio source information
        """
        return {
            'name': self.name,
            'type': self.type,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'format': self.format,
            'chunk_size': self.chunk_size,
            'is_running': self.is_running
        }


class MicrophoneSource(AudioSource):
    """Audio source for microphone capture."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize microphone source.
        
        Args:
            config: Microphone configuration
        """
        super().__init__(config)
        # Correctly read 'device_index' from config, default to None (system default)
        self.device_id = config.get('device_index', None) 
        
        # Import PyAudio here to avoid dependency for other sources
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.audio = pyaudio.PyAudio()
            logger.info(f"Initialized microphone source (device {self.device_id})")
        except ImportError:
            logger.error("PyAudio not installed. Microphone capture will not work.")
            self.audio = None
            raise  # Re-raise the ImportError to signal failure clearly
        except Exception as e: # Catch any other exception during initialization
            logger.exception(f"Failed to initialize PyAudio for microphone source (device {self.device_id}): {e}")
            self.audio = None
            raise  # Re-raise the exception to signal failure clearly
    
    def _capture_loop(self):
        """Microphone capture loop."""
        logger.info(f"Starting capture loop for microphone source (device {self.device_id})") # Add entry log
        if not self.audio:
            logger.error("Cannot start capture loop, PyAudio not initialized.")
            return
        
        try:
            stream = self.audio.open(
                format=self.audio.get_format_from_width(2),  # 16-bit
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                # Use self.device_id which now correctly holds the index or None
                input_device_index=self.device_id, 
                frames_per_buffer=self.chunk_size
            )
            
            logger.info(f"Started microphone capture (device {self.device_id})")
            
            while self.is_running:
                try:
                    # Read audio chunk
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(data, dtype=self.dtype)
                    
                    # Pass to callback
                    if self.data_callback:
                        self.data_callback(audio_data)
                        
                except Exception as e:
                    if self.is_running:  # Only log if we're supposed to be running
                        logger.error(f"Error capturing from microphone: {e}")
                        time.sleep(0.1)  # Avoid tight loop in case of repeated errors
            
            # Clean up
            stream.stop_stream()
            stream.close()
            logger.info("Microphone capture stopped")
            
        except Exception as e:
            logger.error(f"Failed to start microphone capture: {e}")
    
    def __del__(self):
        """Clean up PyAudio on deletion."""
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()


class NetworkSource(AudioSource):
    """Audio source for network streaming (e.g., VoIP)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize network source.
        
        Args:
            config: Network configuration
        """
        super().__init__(config)
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 5060)
        self.protocol = config.get('protocol', 'sip')
        self.socket = None
        self.buffer_size = 4096  # Network buffer size
        
        logger.info(f"Initialized network audio source ({self.protocol}://{self.host}:{self.port})")
    
    def _capture_loop(self):
        """Network capture loop."""
        try:
            # Create socket based on protocol
            if self.protocol.lower() == 'udp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.bind((self.host, self.port))
            else:  # Default to TCP
                # For SIP or other protocols, we would need a proper implementation
                # This is a simplified TCP socket example
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.bind((self.host, self.port))
                self.socket.listen(1)
                conn, addr = self.socket.accept()
                logger.info(f"Connection from {addr}")
            
            # Set timeout for non-blocking operation
            if self.socket:
                self.socket.settimeout(0.1)
            
            logger.info(f"Started network capture ({self.protocol}://{self.host}:{self.port})")
            
            # Main capture loop
            while self.is_running:
                try:
                    if self.protocol.lower() == 'udp':
                        data, addr = self.socket.recvfrom(self.buffer_size)
                    else:
                        data = conn.recv(self.buffer_size)
                    
                    if not data:
                        if self.protocol.lower() != 'udp':
                            # Connection closed
                            logger.info("Connection closed by peer")
                            break
                        continue
                    
                    # Process data
                    # Note: For real VoIP/SIP, we would need to extract RTP payload,
                    # handle protocol specifics, and decode audio
                    audio_data = np.frombuffer(data, dtype=self.dtype)
                    
                    # Pass to callback
                    if self.data_callback:
                        self.data_callback(audio_data)
                
                except socket.timeout:
                    # This is normal, just continue
                    continue
                except Exception as e:
                    if self.is_running:
                        logger.error(f"Error in network capture: {e}")
                        time.sleep(0.1)
            
            # Clean up
            if self.socket:
                if self.protocol.lower() != 'udp':
                    conn.close()
                self.socket.close()
                self.socket = None
            
            logger.info("Network capture stopped")
            
        except Exception as e:
            logger.error(f"Failed to start network capture: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None


class FileSource(AudioSource):
    """Audio source for file input (e.g., WAV files)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file source.
        
        Args:
            config: File configuration
        """
        super().__init__(config)
        self.file_path = config.get('path', '')
        self.loop = config.get('loop', False)
        self.playback_rate = config.get('playback_rate', 1.0)  # Speed factor
        
        # Validate file exists
        if not os.path.exists(self.file_path):
            logger.error(f"Audio file not found: {self.file_path}")
        else:
            logger.info(f"Initialized file audio source: {self.file_path}")
        
        print(f"DEBUG: END OF AudioPipeline.__init__ for {self}") # <--- Add print
    
    def _capture_loop(self):
        """File capture loop."""
        if not os.path.exists(self.file_path):
            logger.error(f"Audio file not found: {self.file_path}")
            return
        
        try:
            # Keep track of playback time for rate control
            playback_start = time.time()
            audio_position = 0  # Bytes read
            
            # Main playback loop
            while self.is_running:
                try:
                    # Open wave file for each loop iteration if looping
                    with wave.open(self.file_path, 'rb') as wf:
                        # Get file characteristics
                        file_sample_rate = wf.getframerate()
                        file_channels = wf.getnchannels()
                        file_sampwidth = wf.getsampwidth()
                        file_frames = wf.getnframes()
                        
                        # Log file details
                        logger.info(f"Playing audio file: {self.file_path}")
                        logger.info(f"File details: {file_sample_rate}Hz, {file_channels} channels, "
                                   f"{file_sampwidth*8} bits, {file_frames} frames")
                        
                        # Reset tracking for each loop
                        playback_start = time.time()
                        audio_position = 0
                        
                        # Read chunks and send to callback
                        while self.is_running:
                            # Read chunk of audio
                            data = wf.readframes(self.chunk_size)
                            
                            # If end of file
                            if not data:
                                if self.loop:
                                    break  # Will restart the file in the outer loop
                                else:
                                    logger.info("End of file reached")
                                    self.is_running = False
                                    break
                            
                            # Convert to numpy array
                            audio_data = np.frombuffer(data, dtype=self.dtype)
                            
                            # Pass to callback
                            if self.data_callback:
                                self.data_callback(audio_data)
                            
                            # Update position
                            audio_position += len(data)
                            
                            # Calculate expected playback time and actual time
                            expected_position = audio_position / (file_sample_rate * file_channels * file_sampwidth)
                            expected_time = expected_position / self.playback_rate
                            actual_time = time.time() - playback_start
                            
                            # Sleep to maintain correct playback rate
                            if actual_time < expected_time:
                                time.sleep(expected_time - actual_time)
                    
                    # If not looping, exit after file is done
                    if not self.loop:
                        break
                        
                except Exception as e:
                    logger.error(f"Error playing audio file: {e}")
                    time.sleep(0.5)
                    break
            
            logger.info("File playback stopped")
            
        except Exception as e:
            logger.error(f"Failed to start file playback: {e})")  # Added closing parenthesis


class StreamBuffer:
    """
    Manages a buffer of audio data for streaming to other components.
    Supports producer-consumer pattern with thread-safe queue.
    """
    
    def __init__(self, buffer_size: int = 10, timeout_ms: int = 100):
        """
        Initialize stream buffer.
        
        Args:
            buffer_size: Number of chunks to buffer
            timeout_ms: Timeout for blocking operations in milliseconds
        """
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.timeout = timeout_ms / 1000.0
        self.closed = False
    
    def write(self, data: np.ndarray) -> int:
        """
        Write data to the buffer.
        
        Args:
            data: Audio data to write
            
        Returns:
            Number of bytes written
        """
        if self.closed:
            return 0
            
        try:
            self.buffer.put(data, block=True, timeout=self.timeout)
            return len(data.tobytes())
        except queue.Full:
            return 0
    
    def read(self, size: int = -1) -> np.ndarray:
        """
        Read data from the buffer.
        
        Args:
            size: Number of bytes to read (unused, included for compatibility)
            
        Returns:
            Audio data or empty array if no data available
        """
        if self.closed:
            return np.array([], dtype=np.int16)
            
        try:
            return self.buffer.get(block=True, timeout=self.timeout)
        except queue.Empty:
            return np.array([], dtype=np.int16)
    
    def close(self):
        """Close the stream buffer."""
        self.closed = True
        
        # Clear the buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break


class AudioProcessor:
    """
    Processes audio data with enhanced noise reduction, multi-stage filtering, and 
    robust voice activity detection optimized for battlefield conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        print("DEBUG: Entering AudioProcessor.__init__") # Added print
        """
        Initialize audio processor.
        
        Args:
            config: Audio processing configuration
        """
        self.config = config
        
        # Extract audio format settings
        audio_config = config.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 1)
        self.format = audio_config.get('format', 'int16')
        self.chunk_size = audio_config.get('chunk_size', 1024)
        
        # Format mapping
        self.format_map = {
            'int16': (np.int16, 2**15),
            'int32': (np.int32, 2**31),
            'float32': (np.float32, 1.0)
        }
        
        # Get data type and normalization factor
        self.dtype, self.norm_factor = self.format_map.get(self.format, (np.int16, 2**15))
        
        # Noise reduction settings
        self.noise_reduction = config.get('noise_reduction', {})
        self.nr_enabled = self.noise_reduction.get('enabled', True)
        self.nr_strength = self.noise_reduction.get('strength', 0.7)
        self.nr_threshold_db = self.noise_reduction.get('threshold_db', -20)
        self.nr_smoothing = self.noise_reduction.get('smoothing', 0.05)
        
        # Battlefield-specific noise filtering (new)
        self.battlefield_filtering = config.get('battlefield_filtering', {})
        self.bf_enabled = self.battlefield_filtering.get('enabled', True)
        self.bf_gunshot_filter = self.battlefield_filtering.get('gunshot_filter', True)
        self.bf_explosion_filter = self.battlefield_filtering.get('explosion_filter', True)
        self.bf_vehicle_filter = self.battlefield_filtering.get('vehicle_filter', True)
        self.bf_wind_filter = self.battlefield_filtering.get('wind_filter', True)
        self.bf_threshold_db = self.battlefield_filtering.get('threshold_db', -15)
        
        # Enhancement settings
        self.enhancement = config.get('enhancement', {})
        self.enh_enabled = self.enhancement.get('enabled', True)
        self.enh_target_level_db = self.enhancement.get('target_level_db', -16)
        self.enh_compression = self.enhancement.get('compression', {})
        self.enh_threshold_db = self.enh_compression.get('threshold_db', -24)
        self.enh_ratio = self.enh_compression.get('ratio', 4.0)
        self.enh_attack_ms = self.enh_compression.get('attack_ms', 5)
        self.enh_release_ms = self.enh_compression.get('release_ms', 50)
        
        # Voice isolation (new)
        self.voice_isolation = config.get('voice_isolation', {})
        self.vi_enabled = self.voice_isolation.get('enabled', True)
        self.vi_strength = self.voice_isolation.get('strength', 0.8)
        self.vi_focus_width = self.voice_isolation.get('focus_width', 200)  # Hz
        self.vi_voice_boost_db = self.voice_isolation.get('voice_boost_db', 6)
        
        # Voice activity detection settings
        self.vad = config.get('vad', {})
        self.vad_enabled = self.vad.get('enabled', True)
        self.vad_sensitivity = self.vad.get('sensitivity', 2)
        self.vad_frame_duration_ms = self.vad.get('frame_duration_ms', 30)
        self.vad_min_speech_duration_ms = self.vad.get('min_speech_duration_ms', 100)
        self.vad_padding_duration_ms = self.vad.get('padding_duration_ms', 300)
        self.vad_holdover_frames = self.vad.get('holdover_frames', 10)  # Continue speech detection after gaps
        
        # Hardware acceleration settings
        self.hardware = config.get('hardware', {})
        self.hw_enable_acceleration = self.hardware.get('enable_acceleration', True)
        self.hw_cuda_device = self.hardware.get('cuda_device', 0)
        self.hw_use_tensorrt = self.hardware.get('use_tensorrt', True)
        
        print("DEBUG: AudioProcessor.__init__: Config parameters extracted") # Added print
        
        # State variables
        self.noise_profile = None  
        self.noise_floor = None  # For tracking ambient noise floor
        self.is_speech = False
        self.speech_buffer = []
        self.frame_count = 0
        self.speech_frame_counter = 0  # Count consecutive speech frames
        self.silence_frame_counter = 0  # Count consecutive silence frames
        
        # Voice frequency ranges (typical male/female ranges for battlefield)
        self.voice_freq_range = (85, 3500)  # Hz
        
        print("DEBUG: AudioProcessor.__init__: Before initialize_profiles()") # Added print
        # Initialize noise profiles
        self.initialize_profiles()
        print("DEBUG: AudioProcessor.__init__: After initialize_profiles()") # Added print
        
        # Initialize VAD if enabled
        if self.vad_enabled:
            print("DEBUG: AudioProcessor.__init__: Before initialize_vad()") # Added print
            self.initialize_vad()
            print("DEBUG: AudioProcessor.__init__: After initialize_vad()") # Added print
        else:
            print("DEBUG: AudioProcessor.__init__: VAD not enabled, skipping initialize_vad()") # Added print

        print("DEBUG: AudioProcessor.__init__: Before initialize_battlefield_filters()") # Added print
        # Initialize battlefield noise filters
        self.initialize_battlefield_filters()
        print("DEBUG: AudioProcessor.__init__: After initialize_battlefield_filters()") # Added print
        
        # Frequency bands for analysis in Hz
        self.freq_bands = {
            "sub_bass": (20, 60),      # Vehicle/explosion rumble 
            "bass": (60, 250),         # Low-frequency noise
            "low_mid": (250, 500),     # Lower voice components
            "mid": (500, 2000),        # Main voice frequencies
            "high_mid": (2000, 4000),  # Upper voice components 
            "high": (4000, 8000)       # Some consonants, high-frequency noise
        }
        
        print("DEBUG: AudioProcessor.__init__: Initialization complete") # Added print
        logger.info("Enhanced audio processor initialized for battlefield conditions")
    
    def initialize_profiles(self):
        """Initialize noise profiles for different processing stages."""
        # General noise profile (frequency domain)
        self.noise_profile = np.ones(self.chunk_size // 2 + 1) * 1e-6
        
        # Noise floor (time domain RMS)
        self.noise_floor = 0.01  # Initial estimate, will adapt
        
        # Frequency bands for analysis in Hz
        self.freq_bands = {
            "sub_bass": (20, 60),      # Vehicle/explosion rumble 
            "bass": (60, 250),         # Low-frequency noise
            "low_mid": (250, 500),     # Lower voice components
            "mid": (500, 2000),        # Main voice frequencies
            "high_mid": (2000, 4000),  # Upper voice components 
            "high": (4000, 8000)       # Some consonants, high-frequency noise
        }
        
        # Frequency-band specific profiles (for battlefield noise processing)
        self.band_profiles = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Converts frequency in Hz to FFT bins
            low_bin = max(0, int(low_freq * self.chunk_size / self.sample_rate))
            high_bin = min(self.chunk_size // 2, int(high_freq * self.chunk_size / self.sample_rate))
            band_size = high_bin - low_bin
            
            if band_size > 0:
                self.band_profiles[band_name] = {
                    "low_bin": low_bin,
                    "high_bin": high_bin,
                    "profile": np.ones(band_size) * 1e-6
                }
    
    def initialize_battlefield_filters(self):
        """Initialize battlefield-specific noise filters."""
        # For simple implementation, we use band-specific attenuation
        # In a complete implementation, this would use ML models to identify specific noise types
        
        # Gunshot: characterized by brief, high-intensity, broad spectrum noise
        self.gunshot_detector = {
            "rise_threshold": 20,  # dB sudden increase
            "duration_ms": 50,     # typical duration
            "cooldown_ms": 500     # min time between detections
        }
        
        # Explosion: characterized by low frequency rumble + broadband noise 
        self.explosion_detector = {
            "low_freq_threshold": 15,  # dB increase in low frequencies
            "duration_ms": 500,        # longer duration
            "cooldown_ms": 2000        # longer cooldown
        }
        
        # Vehicle noise: characterized by constant low-frequency components
        self.vehicle_detector = {
            "low_freq_threshold": 10,  # dB above ambient in low frequencies
            "stability_frames": 20     # Must be consistent over time
        }
        
        # Wind noise: characterized by rumble + frequency-dependent intensity shifts
        self.wind_detector = {
            "correlation_threshold": 0.7,  # High correlation between adjacent frames
            "low_variability": 2.0         # dB variation in low frequencies
        }
        
        # Detection state
        self.noise_detection_state = {
            "gunshot_active": False,
            "explosion_active": False,
            "vehicle_detected": False,
            "wind_detected": False,
            "last_gunshot_time": 0,
            "last_explosion_time": 0,
            "vehicle_stability_counter": 0,
            "previous_frame": None
        }
    
    def initialize_vad(self):
        """Initialize enhanced Voice Activity Detection using VAD Manager."""
        try:
            # Import VAD Manager
            from tccc.utils.vad_manager import get_vad_manager
            
            # Get VAD manager with our config
            vad_config = {
                'vad': self.vad,
                'audio': {
                    'sample_rate': self.sample_rate
                }
            }
            
            # Get VAD manager instance with "audio_processor" component name
            self.vad_manager = get_vad_manager(vad_config)
            logger.info("Using VAD Manager for speech detection")
            
            # Store original VAD parameters for compatibility
            self.vad_energy_threshold = self.vad.get('energy_threshold', 0.005)
            self.vad_speech_frames = 0
            self.vad_nonspeech_frames = 0
            self.vad_speech_detected = False
            self.vad_holdover_counter = 0
            self.speech_history = [False] * 10
            
            # Set battlefield mode if configured
            if self.bf_enabled:
                from tccc.utils.vad_manager import VADMode
                self.vad_manager.set_mode(VADMode.BATTLEFIELD)
                logger.info("Enabled battlefield mode for VAD")
            
            logger.info("Enhanced VAD initialized with VAD Manager")
            
        except ImportError as e:
            # Fall back to direct WebRTC VAD if VAD Manager not available
            logger.warning(f"VAD Manager not available ({e}). Falling back to direct WebRTC VAD.")
            try:
                # WebRTCVAD for primary detection
                import webrtcvad
                self.vad_processor = webrtcvad.Vad(self.vad_sensitivity)
                logger.info(f"Primary VAD initialized with sensitivity {self.vad_sensitivity}")
                
                # Enhanced secondary detection using energy and frequency analysis
                self.vad_energy_threshold = 0.005  # RMS energy threshold, will adapt
                self.vad_speech_frames = 0
                self.vad_nonspeech_frames = 0
                self.vad_speech_detected = False
                self.vad_holdover_counter = 0
                
                # For tracking consecutive speech/non-speech
                self.speech_history = [False] * 10
                
                logger.info("Enhanced VAD initialized with multi-factor detection")
            except ImportError:
                logger.warning("webrtcvad not installed. Falling back to energy-based VAD only.")
                self.vad_enabled = True  # Still enable VAD, just use energy-based
                self.vad_processor = None
                # Make sure energy threshold is initialized even when webrtcvad is not available
                self.vad_energy_threshold = 0.005  # RMS energy threshold, will adapt
                self.vad_speech_frames = 0
                self.vad_nonspeech_frames = 0
                self.vad_speech_detected = False
                self.vad_holdover_counter = 0
                self.speech_history = [False] * 10
    
    def process(self, audio_data: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process audio data with battlefield-optimized enhancements.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Tuple of (processed audio data, is_speech flag)
        """
        # Ensure correct data type
        if audio_data.dtype != self.dtype:
            audio_data = audio_data.astype(self.dtype)
        
        # Convert to float in range [-1, 1] for processing
        float_audio = audio_data.astype(np.float32) / self.norm_factor
        
        # Apply battlefield noise filtering first if enabled
        if self.bf_enabled:
            float_audio = self.apply_battlefield_filtering(float_audio)
        
        # Apply standard noise reduction
        if self.nr_enabled:
            float_audio = self.apply_noise_reduction(float_audio)
        
        # Apply voice isolation if enabled
        if self.vi_enabled:
            float_audio = self.apply_voice_isolation(float_audio)
        
        # Apply enhancement (compression/normalization)
        if self.enh_enabled:
            float_audio = self.apply_enhancement(float_audio)
        
        # Check for speech with enhanced VAD
        is_speech = False
        if self.vad_enabled:
            is_speech = self.enhanced_speech_detection(float_audio)
        
        # Convert back to original format
        processed_audio = (float_audio * self.norm_factor).astype(self.dtype)
        
        return processed_audio, is_speech
    
    def apply_battlefield_filtering(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply battlefield-specific noise filtering to handle gunshots, explosions, vehicles, etc.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Filtered audio data
        """
        try:
            # Convert to frequency domain
            # Ensure the audio chunk size is consistent for FFT
            # If the audio data length doesn't match expected chunk size, pad or truncate it
            expected_chunk_size = self.chunk_size
            if len(audio_data) != expected_chunk_size:
                logger.debug(f"Audio data size mismatch: expected {expected_chunk_size}, got {len(audio_data)}")
                if len(audio_data) < expected_chunk_size:
                    # Pad with zeros
                    padded_audio = np.zeros(expected_chunk_size, dtype=audio_data.dtype)
                    padded_audio[:len(audio_data)] = audio_data
                    audio_data = padded_audio
                else:
                    # Truncate
                    audio_data = audio_data[:expected_chunk_size]
            
            # Now perform FFT with consistent audio size
            fft_data = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            power = magnitude ** 2
            
            # Analyze frequency bands
            band_levels = {}
            for band_name, band_info in self.band_profiles.items():
                low_bin = band_info["low_bin"]
                high_bin = band_info["high_bin"]
                if high_bin > low_bin:
                    band_power = power[low_bin:high_bin]
                    band_levels[band_name] = np.mean(band_power)
            
            # Detect battlefield noise events
            current_time_ms = int(time.time() * 1000)
            
            # 1. Gunshot detection (fast, high-intensity transients)
            gunshot_detected = False
            if self.bf_gunshot_filter:
                energy_increase = np.mean(power) / (np.mean(self.noise_profile) + 1e-6)
                energy_db_increase = 10 * np.log10(energy_increase + 1e-6)
                
                cooldown_elapsed = (current_time_ms - self.noise_detection_state["last_gunshot_time"]) > self.gunshot_detector["cooldown_ms"]
                
                if energy_db_increase > self.gunshot_detector["rise_threshold"] and cooldown_elapsed:
                    gunshot_detected = True
                    self.noise_detection_state["gunshot_active"] = True
                    self.noise_detection_state["last_gunshot_time"] = current_time_ms
                    logger.debug("Gunshot-like noise detected")
            
            # 2. Explosion detection (longer, low-frequency dominant)
            explosion_detected = False
            if self.bf_explosion_filter:
                low_energy = band_levels.get("sub_bass", 0) + band_levels.get("bass", 0)
                low_energy_increase = low_energy / (np.mean(self.band_profiles["sub_bass"]["profile"]) +
                                                  np.mean(self.band_profiles["bass"]["profile"]) + 1e-6)
                low_energy_db_increase = 10 * np.log10(low_energy_increase + 1e-6)
                
                cooldown_elapsed = (current_time_ms - self.noise_detection_state["last_explosion_time"]) > self.explosion_detector["cooldown_ms"]
                
                if low_energy_db_increase > self.explosion_detector["low_freq_threshold"] and cooldown_elapsed:
                    explosion_detected = True
                    self.noise_detection_state["explosion_active"] = True
                    self.noise_detection_state["last_explosion_time"] = current_time_ms
                    logger.debug("Explosion-like noise detected")
            
            # 3. Vehicle detection (persistent low-frequency signature)
            if self.bf_vehicle_filter:
                low_freq_power = np.mean(power[:int(200 * self.chunk_size / self.sample_rate)])
                low_freq_increase = low_freq_power / (np.mean(self.noise_profile[:int(200 * self.chunk_size / self.sample_rate)]) + 1e-6)
                low_freq_db_increase = 10 * np.log10(low_freq_increase + 1e-6)
                
                if low_freq_db_increase > self.vehicle_detector["low_freq_threshold"]:
                    self.noise_detection_state["vehicle_stability_counter"] += 1
                    if self.noise_detection_state["vehicle_stability_counter"] >= self.vehicle_detector["stability_frames"]:
                        self.noise_detection_state["vehicle_detected"] = True
                else:
                    self.noise_detection_state["vehicle_stability_counter"] = max(0, self.noise_detection_state["vehicle_stability_counter"] - 1)
                    if self.noise_detection_state["vehicle_stability_counter"] == 0:
                        self.noise_detection_state["vehicle_detected"] = False
            
            # 4. Wind noise detection (correlated low frequency fluctuations)
            if self.bf_wind_filter and self.noise_detection_state["previous_frame"] is not None:
                prev_power = np.abs(np.fft.rfft(self.noise_detection_state["previous_frame"])) ** 2
                
                # Ensure arrays have compatible sizes for correlation
                min_length = min(len(prev_power), len(power))
                max_bin = int(min(500 * self.chunk_size / self.sample_rate, min_length))
                
                if min_length > 10:  # Ensure we have enough data for correlation
                    try:
                        # Make sure arrays are compatible for correlation calculation
                        prev_slice = prev_power[:max_bin]
                        curr_slice = power[:max_bin]
                        
                        # Check if slices have the same shape
                        if prev_slice.shape == curr_slice.shape and prev_slice.size > 0:
                            # Calculate correlation coefficient
                            corr_matrix = np.corrcoef(prev_slice, curr_slice)
                            # Check if the correlation matrix has the right shape
                            if corr_matrix.shape == (2, 2):
                                correlation = corr_matrix[0, 1]
                            else:
                                correlation = 0.0
                                logger.debug(f"Invalid correlation matrix shape: {corr_matrix.shape}")
                        else:
                            correlation = 0.0
                            logger.debug(f"Incompatible array shapes for correlation: {prev_slice.shape} vs {curr_slice.shape}")
                    except Exception as e:
                        logger.debug(f"Error calculating correlation: {e}")
                        correlation = 0.0
                    
                    # Variability in low frequencies, ensuring we stay within array bounds
                    low_freq_max = min(int(200 * self.chunk_size / self.sample_rate), len(power))
                    low_freq_var = np.std(power[:low_freq_max]) if low_freq_max > 0 else 0
                    
                    if correlation > self.wind_detector["correlation_threshold"] and low_freq_var > self.wind_detector["low_variability"]:
                        self.noise_detection_state["wind_detected"] = True
                    else:
                        self.noise_detection_state["wind_detected"] = False
            
            # Store current frame for next comparison
            self.noise_detection_state["previous_frame"] = audio_data.copy()
            
            # Apply filtering based on detected noise types
            if gunshot_detected or self.noise_detection_state["gunshot_active"]:
                # Attenuate high frequencies first (gunshot has strong high-frequency content)
                high_bin_start = int(2000 * self.chunk_size / self.sample_rate)
                # Make sure we don't create an empty array and check array bounds
                if high_bin_start < len(magnitude) and len(magnitude[high_bin_start:]) > 0:
                    # Create attenuation array that matches the size of the magnitude slice
                    attenuation = np.linspace(1.0, 0.1, len(magnitude[high_bin_start:]))
                    # Safe multiplication with shape checking
                    if attenuation.shape[0] == magnitude[high_bin_start:].shape[0]:
                        magnitude[high_bin_start:] *= attenuation
                    else:
                        logger.debug(f"Shape mismatch in gunshot filtering: attenuation shape {attenuation.shape}, magnitude slice shape {magnitude[high_bin_start:].shape}")
                else:
                    logger.debug(f"Skipping gunshot filtering: high_bin_start={high_bin_start}, magnitude length={len(magnitude)}")
                
                # Reset gunshot active flag after processing
                if gunshot_detected:
                    # Active for just this frame
                    self.noise_detection_state["gunshot_active"] = False
            
            if explosion_detected or self.noise_detection_state["explosion_active"]:
                # Apply lowpass filter to attenuate explosion rumble
                low_bin_end = int(300 * self.chunk_size / self.sample_rate)
                # Ensure we don't exceed array bounds
                if low_bin_end > 0 and low_bin_end <= len(magnitude):
                    # Create attenuation array that matches the size
                    attenuation = np.linspace(0.1, 1.0, low_bin_end)
                    # Safe multiplication with shape checking
                    if attenuation.shape[0] == magnitude[:low_bin_end].shape[0]:
                        magnitude[:low_bin_end] *= attenuation
                    else:
                        logger.debug(f"Shape mismatch in explosion filtering: attenuation shape {attenuation.shape}, magnitude slice shape {magnitude[:low_bin_end].shape}")
                else:
                    logger.debug(f"Skipping explosion filtering: low_bin_end={low_bin_end}, magnitude length={len(magnitude)}")
                
                # Reset explosion active flag after processing
                if explosion_detected:
                    # Active for just this frame
                    self.noise_detection_state["explosion_active"] = False
            
            if self.noise_detection_state["vehicle_detected"]:
                # Apply notch filtering for vehicle noise (focused on engine frequencies)
                engine_low = int(50 * self.chunk_size / self.sample_rate)
                engine_high = int(150 * self.chunk_size / self.sample_rate)
                # Ensure indices are valid and in range
                if engine_high > engine_low and engine_low >= 0 and engine_high <= len(magnitude):
                    # Safer approach - specify scalar value for multiplication
                    magnitude[engine_low:engine_high] = magnitude[engine_low:engine_high] * 0.3  # Stronger attenuation for engine frequencies
                else:
                    logger.debug(f"Skipping vehicle filtering: engine_low={engine_low}, engine_high={engine_high}, magnitude length={len(magnitude)}")
            
            if self.noise_detection_state["wind_detected"]:
                # Apply highpass filter to reduce wind rumble
                wind_cutoff = int(120 * self.chunk_size / self.sample_rate)
                # Ensure indices are valid
                if wind_cutoff > 0 and wind_cutoff <= len(magnitude):
                    # Safe scalar multiplication
                    magnitude[:wind_cutoff] = magnitude[:wind_cutoff] * 0.2
                else:
                    logger.debug(f"Skipping wind filtering: wind_cutoff={wind_cutoff}, magnitude length={len(magnitude)}")
            
            # Reconstruct signal with processed magnitude and original phase
            fft_filtered = magnitude * np.exp(1j * phase)
            filtered_audio = np.fft.irfft(fft_filtered)
            
            # Ensure same length as input
            filtered_audio = filtered_audio[:len(audio_data)]
            
            return filtered_audio
            
        except Exception as e:
            # Log error and return original audio if filtering fails
            logger.error(f"Error processing audio: {e}")
            logger.debug(f"Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            # Return original audio to ensure processing continues
            return audio_data
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply enhanced spectral subtraction noise reduction.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Noise-reduced audio data
        """
        # Ensure consistent audio length for FFT
        expected_chunk_size = self.chunk_size
        if len(audio_data) != expected_chunk_size:
            logger.debug(f"NR: Audio data size mismatch: expected {expected_chunk_size}, got {len(audio_data)}")
            if len(audio_data) < expected_chunk_size:
                # Pad with zeros
                padded_audio = np.zeros(expected_chunk_size, dtype=audio_data.dtype)
                padded_audio[:len(audio_data)] = audio_data
                audio_data = padded_audio
            else:
                # Truncate
                audio_data = audio_data[:expected_chunk_size]
        
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        
        # Get magnitude and phase
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Calculate power spectrum
        power = magnitude ** 2
        
        # Calculate frame energy
        frame_energy = np.mean(power)
        
        # Update noise profile when energy is low (likely silence)
        # Use a more sophisticated detection for silence compared to original
        is_silence = frame_energy < (self.noise_floor * 1.5)
        
        if is_silence or self.frame_count < 20:  # Use more frames for initial profile
            # Adaptive learning rate based on how stable the noise is
            if self.frame_count < 20:
                # Faster learning during initialization
                learn_rate = 0.1
            else:
                # Slower updates during regular operation
                learn_rate = self.nr_smoothing
                
            # Update noise profile
            self.noise_profile = self.noise_profile * (1 - learn_rate) + power * learn_rate
            
            # Also update noise floor estimate
            self.noise_floor = self.noise_floor * 0.95 + frame_energy * 0.05
            
            # Update band-specific profiles
            for band_name, band_info in self.band_profiles.items():
                low_bin = band_info["low_bin"]
                high_bin = band_info["high_bin"]
                if low_bin < high_bin:
                    band_power = power[low_bin:high_bin]
                    self.band_profiles[band_name]["profile"] = self.band_profiles[band_name]["profile"] * (1 - learn_rate) + band_power * learn_rate
        
        # Convert threshold from dB to power ratio
        threshold_power = 10 ** (self.nr_threshold_db / 10)
        
        # Apply enhanced spectral subtraction with oversubtraction factor
        # Oversubtraction: remove more noise in frequency bands where speech is less likely
        oversubtraction_factor = np.ones_like(power)
        
        # Voice frequency range (less aggressive in voice frequencies)
        voice_low_bin = int(self.voice_freq_range[0] * self.chunk_size / self.sample_rate)
        voice_high_bin = int(self.voice_freq_range[1] * self.chunk_size / self.sample_rate)
        
        # More aggressive in frequencies outside voice range
        oversubtraction_factor[:voice_low_bin] = 1.5  # More aggressive below voice
        oversubtraction_factor[voice_high_bin:] = 1.3  # More aggressive above voice
        
        # Calculate gain with oversubtraction
        gain = np.maximum(1 - self.nr_strength * oversubtraction_factor * self.noise_profile / (power + 1e-6), threshold_power)
        
        # Apply gain to magnitude
        magnitude_reduced = magnitude * gain
        
        # Reconstruct signal with original phase
        fft_reduced = magnitude_reduced * np.exp(1j * phase)
        
        # Convert back to time domain
        reduced_audio = np.fft.irfft(fft_reduced)
        
        # Ensure same length as input (FFT can change the length slightly)
        reduced_audio = reduced_audio[:len(audio_data)]
        
        # Increment frame counter
        self.frame_count += 1
        
        return reduced_audio
    
    def apply_voice_isolation(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply focused voice isolation to enhance speech in battlefield environments.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Voice-isolated audio data
        """
        # Ensure consistent audio length for FFT
        expected_chunk_size = self.chunk_size
        if len(audio_data) != expected_chunk_size:
            logger.debug(f"VI: Audio data size mismatch: expected {expected_chunk_size}, got {len(audio_data)}")
            if len(audio_data) < expected_chunk_size:
                # Pad with zeros
                padded_audio = np.zeros(expected_chunk_size, dtype=audio_data.dtype)
                padded_audio[:len(audio_data)] = audio_data
                audio_data = padded_audio
            else:
                # Truncate
                audio_data = audio_data[:expected_chunk_size]
                
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        
        # Get magnitude and phase
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Voice frequency range (Hz)
        voice_low = self.voice_freq_range[0]
        voice_high = self.voice_freq_range[1]
        
        # Convert to FFT bins
        voice_low_bin = int(voice_low * self.chunk_size / self.sample_rate)
        voice_high_bin = int(voice_high * self.chunk_size / self.sample_rate)
        
        # Create voice emphasis filter (bell curve centered on main speech frequencies)
        emphasis = np.ones_like(magnitude)
        
        # Strong attenuation below voice range
        emphasis[:voice_low_bin] = np.linspace(0.05, 0.3, voice_low_bin) if voice_low_bin > 0 else emphasis[:voice_low_bin]
        
        # Voice emphasis with focus on clarity (1000-3000 Hz for intelligibility)
        speech_center_bin = int(1800 * self.chunk_size / self.sample_rate)
        speech_width_bins = int(self.vi_focus_width * self.chunk_size / self.sample_rate)
        
        # Create bell curve for voice emphasis
        for i in range(voice_low_bin, voice_high_bin):
            # Distance from speech center (normalized)
            distance = abs(i - speech_center_bin) / max(1, speech_width_bins)
            # Gaussian-like emphasis (strongest at center, tapering at edges)
            if distance < 2.0:  # Within 2 standard deviations
                emphasis[i] = 1.0 + (self.vi_voice_boost_db / 20) * np.exp(-(distance ** 2))
            else:
                emphasis[i] = 1.0
        
        # Strong attenuation above voice range with smooth transition
        if voice_high_bin < len(emphasis):
            emphasis[voice_high_bin:] = np.linspace(0.3, 0.05, len(emphasis) - voice_high_bin)
        
        # Apply emphasis filter to magnitude
        magnitude_isolated = magnitude * emphasis
        
        # Reconstruct signal with modified magnitude and original phase
        fft_isolated = magnitude_isolated * np.exp(1j * phase)
        
        # Convert back to time domain
        isolated_audio = np.fft.irfft(fft_isolated)
        
        # Ensure same length as input
        isolated_audio = isolated_audio[:len(audio_data)]
        
        return isolated_audio
    
    def apply_enhancement(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply enhanced audio enhancement optimized for battlefield clarity.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Enhanced audio data
        """
        # Calculate current RMS level
        rms = np.sqrt(np.mean(audio_data ** 2)) + 1e-6
        current_db = 20 * np.log10(rms)
        
        # Calculate gain for normalization
        gain_db = self.enh_target_level_db - current_db
        
        # Apply adaptive compression based on signal characteristics
        # Different compression settings for speech vs. non-speech
        is_likely_speech = self.is_in_speech_range(audio_data)
        
        # Apply different compression strategies based on content
        if is_likely_speech:
            # More gentle compression for speech to preserve dynamics
            if current_db > self.enh_threshold_db:
                # Amount above threshold
                above_threshold = current_db - self.enh_threshold_db
                
                # Compressed amount
                compressed = above_threshold * (1 - 1/self.enh_ratio)
                
                # Adjust gain
                gain_db -= compressed
        else:
            # More aggressive compression for non-speech to control background noise
            if current_db > (self.enh_threshold_db - 3):  # Lower threshold for non-speech
                # Amount above threshold
                above_threshold = current_db - (self.enh_threshold_db - 3)
                
                # More aggressive compression ratio for non-speech
                compressed = above_threshold * (1 - 1/(self.enh_ratio * 1.5))
                
                # Adjust gain
                gain_db -= compressed
        
        # Convert gain from dB to linear
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        enhanced_audio = audio_data * gain_linear
        
        # Enhanced clipping protection with soft knee
        # Use hyperbolic tangent for smoother limiting
        enhanced_audio = np.tanh(enhanced_audio)
        
        return enhanced_audio
    
    def is_in_speech_range(self, audio_data: np.ndarray) -> bool:
        """
        Determine if audio data is likely to contain speech based on spectral characteristics.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            True if likely speech, False otherwise
        """
        # Ensure consistent audio length for FFT
        expected_chunk_size = self.chunk_size
        if len(audio_data) != expected_chunk_size:
            # Quietly fix the size
            if len(audio_data) < expected_chunk_size:
                # Pad with zeros
                padded_audio = np.zeros(expected_chunk_size, dtype=audio_data.dtype)
                padded_audio[:len(audio_data)] = audio_data
                audio_data = padded_audio
            else:
                # Truncate
                audio_data = audio_data[:expected_chunk_size]
                
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        power = np.abs(fft_data) ** 2
        
        # Get power in voice frequency range
        voice_low_bin = int(self.voice_freq_range[0] * self.chunk_size / self.sample_rate)
        voice_high_bin = int(self.voice_freq_range[1] * self.chunk_size / self.sample_rate)
        
        voice_power = np.sum(power[voice_low_bin:voice_high_bin])
        total_power = np.sum(power)
        
        # Check if significant portion of energy is in voice range
        voice_ratio = voice_power / (total_power + 1e-6)
        
        # Also check voice formant structure (simple approximation)
        # Real speech typically has peaks in formant regions
        has_formants = False
        if voice_high_bin > voice_low_bin + 10:  # Enough bins for analysis
            # Look for local peaks in voice range (simple formant detection)
            formant_regions = [
                (300, 1000),   # First formant region
                (1000, 2500)   # Second formant region
            ]
            
            peak_count = 0
            for low_f, high_f in formant_regions:
                low_bin = int(low_f * self.chunk_size / self.sample_rate)
                high_bin = int(high_f * self.chunk_size / self.sample_rate)
                
                if high_bin > low_bin:
                    # Check for at least one significant peak in this range
                    region_power = power[low_bin:high_bin]
                    if len(region_power) > 3:  # Need enough points to find peaks
                        # Find local maxima
                        peaks = []
                        for i in range(1, len(region_power)-1):
                            if region_power[i] > region_power[i-1] and region_power[i] > region_power[i+1]:
                                peaks.append(region_power[i])
                        
                        # If found peaks with significant energy
                        if peaks and max(peaks) > 3 * np.mean(region_power):
                            peak_count += 1
            
            has_formants = peak_count >= 1  # Need at least one formant region with peaks
        
        # Combine criteria
        return voice_ratio > 0.6 or has_formants
    
    def enhanced_speech_detection(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced speech detection using VAD Manager (or fallback methods if not available).
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            True if speech detected, False otherwise
        """
        # Use VAD Manager if available
        if hasattr(self, 'vad_manager'):
            # Get detection from VAD Manager
            vad_result = self.vad_manager.detect_speech(
                audio_data, 
                component_name="audio_processor"
            )
            
            # Use the result
            speech_detected = vad_result.is_speech
            
            # For compatibility with existing code, update the internal state
            if speech_detected:
                self.speech_frame_counter += 1
                self.silence_frame_counter = 0
                if self.speech_frame_counter >= 2:  # Need 2 consecutive frames for ON
                    self.vad_speech_detected = True
            else:
                self.silence_frame_counter += 1
                self.speech_frame_counter = 0
                
                # For OFF, use holdover to avoid choppy detection
                if self.silence_frame_counter >= self.vad_holdover_frames:
                    self.vad_speech_detected = False
            
            return self.vad_speech_detected
            
        # Fallback to original implementation if VAD Manager not available
        # Combine multiple detection methods for robustness
        webrtc_speech = False
        energy_speech = False
        spectral_speech = False
        
        # 1. WebRTC VAD (if available)
        if self.vad_processor:
            try:
                # Convert to format required by webrtcvad
                audio_data_int16 = (audio_data * 32767).astype(np.int16)
                
                # Calculate frame duration in samples
                frame_duration_samples = int(self.sample_rate * self.vad_frame_duration_ms / 1000)
                
                # Process in valid frame sizes
                webrtc_speech_frames = 0
                total_frames = 0
                
                # Process in 10ms, 20ms, or 30ms chunks as required by webrtcvad
                for i in range(0, len(audio_data_int16), frame_duration_samples):
                    frame = audio_data_int16[i:i+frame_duration_samples]
                    
                    # Ensure frame is the correct size
                    if len(frame) == frame_duration_samples:
                        total_frames += 1
                        frame_bytes = frame.tobytes()
                        if self.vad_processor.is_speech(frame_bytes, self.sample_rate):
                            webrtc_speech_frames += 1
                
                # Consider speech if enough frames are detected (more robust)
                webrtc_speech = webrtc_speech_frames >= max(1, total_frames // 3)
                
            except Exception as e:
                logger.error(f"WebRTC VAD processing error: {e}")
                webrtc_speech = False
        
        # 2. Energy-based detection (works in all conditions)
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Adapt threshold based on noise floor
        adaptive_threshold = max(self.vad_energy_threshold, self.noise_floor * 3)
        
        energy_speech = rms_energy > adaptive_threshold
        
        # Update adaptive energy threshold
        if energy_speech:
            self.vad_speech_frames += 1
            # Slowly raise threshold if we detect too much speech
            if self.vad_speech_frames > 100:
                self.vad_energy_threshold *= 1.01
        else:
            self.vad_nonspeech_frames += 1
            # Slowly lower threshold if we detect too little speech
            if self.vad_nonspeech_frames > 100:
                self.vad_energy_threshold *= 0.99
                self.vad_energy_threshold = max(0.001, self.vad_energy_threshold)  # Don't go too low
        
        # Reset counters periodically
        if self.vad_speech_frames + self.vad_nonspeech_frames > 200:
            self.vad_speech_frames = 0
            self.vad_nonspeech_frames = 0
        
        # 3. Spectral-based detection (frequency characteristics of speech)
        spectral_speech = self.is_in_speech_range(audio_data)
        
        # Combine detection methods (weighted approach)
        if self.vad_processor:
            # If WebRTC available, give it more weight
            speech_detected = webrtc_speech or (energy_speech and spectral_speech)
        else:
            # Without WebRTC, rely on energy and spectral
            speech_detected = energy_speech and spectral_speech
        
        # State tracking for robustness
        # Update history
        self.speech_history.pop(0)
        self.speech_history.append(speech_detected)
        
        # Decision with hysteresis (reduces false transitions)
        if speech_detected:
            self.speech_frame_counter += 1
            self.silence_frame_counter = 0
            if self.speech_frame_counter >= 2:  # Need 2 consecutive frames for ON
                self.vad_speech_detected = True
        else:
            self.silence_frame_counter += 1
            self.speech_frame_counter = 0
            
            # For OFF, use holdover to avoid choppy detection
            if self.silence_frame_counter >= self.vad_holdover_frames:
                self.vad_speech_detected = False
        
        return self.vad_speech_detected
    
    def detect_speech(self, audio_data: np.ndarray) -> bool:
        """
        Backwards compatible speech detection method.
        Now calls the enhanced version.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            True if speech detected, False otherwise
        """
        return self.enhanced_speech_detection(audio_data)


class AudioPipeline:
    """
    Main audio pipeline that coordinates capture, processing, and streaming.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None): 
        """Initialize the AudioPipeline instance."""
        print("DEBUG: ENTERED AudioPipeline.__init__")
        self.initialized = False
        self.status = ModuleState.INITIALIZING

        # Add detailed logging within __init__
        logger.debug("AudioPipeline.__init__: Entering")
        try:
            # Ensure self.config is always a Config object
            raw_config_dict = config or {}
            self.config = Config(raw_config_dict) # Wrap the dict
            self.system = None # Keep system placeholder
            logger.debug(f"AudioPipeline.__init__: Config object created from: {raw_config_dict}")
            logger.debug(f"AudioPipeline.__init__: Raw config check: {self.config.get_raw_config()}") # Now this should work

            # Get sample rate from config object
            # Assuming Config class allows accessing nested keys like this, or adjust if needed.
            # Since TCCCSystem passes the 'audio_pipeline' sub-dict, simple key access should work.
            self.sample_rate = self.config.get('sample_rate', 16000)
            logger.debug(f"AudioPipeline.__init__: Sample rate set to {self.sample_rate}")

            # Audio sources and processing
            self.sources = {} # Initialize sources dictionary
            self.default_input_source = None # Initialize default_input_source
            logger.debug(f"AudioPipeline.__init__: Sources initialized. Default: {self.default_input_source}")

            # Output streams (e.g., file writers, network streamers)
            self.output_streams = {}
            logger.debug("AudioPipeline.__init__: Output streams initialized")

            # Internal state
            self.is_running = False
            self.audio_lock = threading.Lock()
            self.active_source = None

            # Event Bus integration
            self.event_bus = None # Placeholder, might need proper injection
            logger.debug(f"AudioPipeline.__init__: Event bus obtained from system: {self.event_bus}")

            logger.info("AudioPipeline initialized successfully.")
            self.status = ModuleState.IDLE
        except Exception as e:
            logger.exception("AudioPipeline.__init__: Exception during initialization")
            self.status = ModuleState.ERROR
            raise # Re-raise the exception so TCCCSystem knows init failed
        finally:
            logger.debug("AudioPipeline.__init__: Exiting")
            
    
    def _initialize_sources(self):
        """Initialize audio sources based on configuration."""
        logger.debug("AudioPipeline._initialize_sources: Entering")
        io_config = self.config.get('io', {})
        
        if 'input_sources' not in io_config:
            logger.warning("AudioPipeline._initialize_sources: No input sources defined in configuration.")
            return

        sources_config = io_config.get('input_sources', [])
        
        if not isinstance(sources_config, list):
            logger.error(f"AudioPipeline._initialize_sources: Expected sources_config to be a list, but got {type(sources_config)}. Skipping.")
            return

        logger.debug(f"AudioPipeline._initialize_sources: Type of sources_config is {type(sources_config)}")
        
        logger.debug(f"AudioPipeline._initialize_sources: Found {len(sources_config)} sources in config")
        
        if not sources_config:
            logger.warning("AudioPipeline._initialize_sources: No input sources defined in configuration.")
            return

        logger.debug(f"AudioPipeline._initialize_sources: io_config = {io_config}") # Inspect io_config
        logger.debug(f"AudioPipeline._initialize_sources: sources_config = {sources_config}") # Inspect sources_config
        
        for source_config in sources_config: # Iterate through the list of source dicts
            if not isinstance(source_config, dict):
                logger.error(f"AudioPipeline._initialize_sources: Expected source_config to be a dict, but got {type(source_config)}. Skipping.")
                continue
                
            source_name = source_config.get('name')
            source_type = source_config.get('type')
            logger.debug(f"AudioPipeline._initialize_sources: Processing source '{source_name}' of type '{source_type}'")

            if not source_name or not source_type:
                logger.warning(f"AudioPipeline._initialize_sources: Skipping source with missing name or type: {source_config}")
                continue
                
            try:
                if source_type == 'pyaudio':
                    source = PyAudioSource(config=self.config, source_config=source_config, system=self.system)
                elif source_type == 'sounddevice':
                    source = SoundDeviceSource(config=self.config, source_config=source_config, system=self.system)
                elif source_type == 'file':
                    source = FileSource(config=self.config, source_config=source_config, system=self.system)
                else:
                    logger.warning(f"AudioPipeline._initialize_sources: Unsupported source type '{source_type}' for source '{source_name}'")
                    continue
                    
                self.sources[source_name] = source
                logger.info(f"AudioPipeline._initialize_sources: Initialized source '{source_name}' ({source_type})")
                
            except Exception as e:
                 logger.exception(f"AudioPipeline._initialize_sources: Failed to initialize source '{source_name}': {e}")

        # Set default source if defined
        default_input_name = io_config.get('default_input')
        if default_input_name and default_input_name in self.sources:
            self.active_source = self.sources[default_input_name]
            logger.info(f"AudioPipeline._initialize_sources: Set active source to '{default_input_name}'")
        elif self.sources:
            # Fallback to the first initialized source if default is not set or invalid
            first_source_name = list(self.sources.keys())[0]
            self.active_source = self.sources[first_source_name]
            logger.warning(f"AudioPipeline._initialize_sources: Default input '{default_input_name}' not found or invalid. Falling back to first available source: '{first_source_name}'")
        else:
             logger.error("AudioPipeline._initialize_sources: No audio sources were successfully initialized.")
             self.status = ModuleState.ERROR # Critical if no sources
             
        logger.debug("AudioPipeline._initialize_sources: Exiting")


    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize audio pipeline with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Success status
        """
        print("DEBUG: ENTERED AudioPipeline.initialize") # Added print BEFORE try
        self.active_source = None # Initialize active_source
        try:
            print(f"DEBUG: AudioPipeline.initialize received config: {config}") # Added print
            # Assign config early
            self.config = config
            
            # --- TEMPORARILY BYPASS AudioProcessor --- 
            print("DEBUG: AudioPipeline.initialize - SKIPPING AudioProcessor creation") # Added print
            # self.audio_processor = AudioProcessor(config) # <-- Commented out
            self.audio_processor = None # <-- Added placeholder
            # --- END TEMPORARY BYPASS --- 
            
            # Determine number of input channels based on config
            io_config = config.get('io', {})
            sources_config = io_config.get('input_sources', [])
            logger.info(f"Found Input Sources: {sources_config}") # Log the found input sources
            
            if not sources_config:
                logger.warning("No input sources defined in the 'io' section of the audio pipeline config!")
            
            # Mapping from config type string to AudioSource class
            source_class_map = {
                "sounddevice": MicrophoneSource, # Map sounddevice to MicrophoneSource for now
                "microphone": MicrophoneSource,
                "file": FileSource,
                # "network": NetworkSource, # Add if NetworkSource exists and is needed
            }
            
            for source_config in sources_config:
                if not isinstance(source_config, dict):
                    logger.error(f"AudioPipeline.initialize: Expected source_config to be a dict, but got {type(source_config)}. Skipping.")
                    continue
                
                source_type = source_config.get('type')
                source_name = source_config.get('name', f'source_{len(self.sources)}')
                
                SourceClass = source_class_map.get(source_type)
                source_instance = None
                
                if SourceClass:
                    try:
                        # Merge general audio config (like sample_rate, channels) with specific source config
                        # Source-specific config (like device_index, path) takes precedence
                        merged_config = {**audio_config, **source_config}
                        # Pass the merged config to the source class constructor
                        source_instance = SourceClass(merged_config) 
                        
                        if source_instance:
                            self.sources[source_name] = source_instance
                            logger.info(f"Successfully created audio source '{source_name}' of type '{source_type}'")
                        
                    except ImportError as ie:
                        logger.error(f"Failed to create source '{source_name}' ({source_type}): Missing dependency - {ie}. Please install required libraries (e.g., PyAudio).")
                        # Depending on policy, maybe raise, maybe continue without this source
                    except Exception as e:
                        logger.error(f"Failed to create source '{source_name}' of type '{source_type}': {e}")
                        # Optionally print traceback for detailed debugging
                        # print(f"DEBUG: EXCEPTION creating source {source_name}: {e}\n{traceback.format_exc()}")
                else:
                    logger.warning(f"Unknown or unmapped audio source type: '{source_type}' for source '{source_name}'")
            
            # Set default source
            default_source_name = io_config.get('default_input') # Get name from config
            print(f"DEBUG: Trying to set active source. Default name: '{default_source_name}', Exists in sources: {default_source_name in self.sources}, Source instance: {self.sources.get(default_source_name)}")
            if default_source_name and default_source_name in self.sources:
                self.active_source = self.sources[default_source_name]
            
            # Ensure an active source was actually set
            if not self.active_source:
                logger.error("Audio Pipeline initialization failed: No active audio source could be configured or initialized.")
                self.initialized = False # Mark as not initialized
                logger.debug("Returning False because no active source.")
                print("DEBUG: Returning False: No active source.") # DEBUG PRINT
                return False

            logger.debug("Active source was set. Setting self.initialized=True")
            print("DEBUG: Active source found. Setting self.initialized=True.") # DEBUG PRINT
            self.initialized = True
            logger.info("Audio Pipeline initialized successfully")
            logger.debug("Returning True from initialize method.")
            print("DEBUG: Returning True from initialize.") # DEBUG PRINT
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Audio Pipeline: {e}")
            print(f"DEBUG: EXCEPTION in AudioPipeline.initialize: {e}\n{traceback.format_exc()}") # DEBUG PRINT
            # --- END DEBUG PRINTS ---
            return False
    
    def is_running(self) -> bool:
        """Check if the audio pipeline is currently running."""
        return self.running and self.audio_stream is not None and self.audio_stream.active

    def shutdown(self):
        """Gracefully shut down the audio pipeline."""
        logger.info("Shutting down Audio Pipeline...")
        # Stop the active source first
        if hasattr(self, 'active_source') and self.active_source:
            logger.info(f"Stopping active audio source: {self.active_source.name}")
            try:
                self.active_source.stop()
            except Exception as e:
                logger.error(f"Error stopping active audio source {self.active_source.name}: {e}")
        
        # Also stop any other managed sources just in case
        if hasattr(self, 'sources'):
            for source_name, source_instance in self.sources.items():
                if source_instance != self.active_source and hasattr(source_instance, 'is_running') and source_instance.is_running:
                    logger.info(f"Stopping inactive but running source: {source_name}")
                    try:
                        source_instance.stop()
                    except Exception as e:
                        logger.error(f"Error stopping source {source_name}: {e}")
        
        # Add any other necessary cleanup (e.g., closing resources)
        logger.info("Audio Pipeline shutdown complete.")

if __name__ == "__main__":
    # Example usage for testing
    pass