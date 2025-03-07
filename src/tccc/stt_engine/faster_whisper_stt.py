"""
Faster Whisper STT implementation for TCCC.ai system.

This module implements Faster Whisper for speech-to-text conversion,
optimized for Jetson hardware with improved accuracy and performance.
Faster Whisper is a highly optimized implementation of OpenAI's Whisper
model that provides significantly faster inference.
"""

import os
import time
import numpy as np
import logging
import concurrent.futures
import importlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque

from tccc.utils.logging import get_logger
from tccc.stt_engine.stt_engine import TranscriptionResult, TranscriptionSegment, Word, TranscriptionConfig

# Import Jetson optimizations if available
try:
    from tccc.utils.jetson_integration import get_whisper_params, initialize_jetson_optimizations
    JETSON_INTEGRATION_AVAILABLE = True
except ImportError:
    JETSON_INTEGRATION_AVAILABLE = False

# Check if faster_whisper is available
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger(__name__)

class FasterWhisperSTT:
    """
    Implements Faster Whisper for speech recognition.
    
    This implementation provides significantly faster inference speed
    compared to standard Whisper, with optimizations for edge deployment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the faster-whisper STT engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.hardware_config = config.get('hardware', {})
        
        # Extract model settings
        self.model_size = self.model_config.get('size', 'small')
        self.model_path = self.model_config.get('path', f'models/faster-whisper-{self.model_size}')
        self.language = self.model_config.get('language', 'en')
        self.beam_size = self.model_config.get('beam_size', 5)
        self.compute_type = self.model_config.get('compute_type', 'float16')
        self.vad_filter = self.model_config.get('vad_filter', True)
        self.vad_parameters = self.model_config.get('vad_parameters', {
            'threshold': 0.5,
            'min_speech_duration_ms': 250,
            'max_speech_duration_s': 30.0,
            'min_silence_duration_ms': 500
        })
        
        # Medical domain settings
        self.use_medical_vocabulary = self.model_config.get('use_medical_vocabulary', True)
        self.vocabulary_path = self.model_config.get('vocabulary_path', 'config/vocabulary/custom_terms.txt')
        
        # Extract hardware acceleration settings
        self.enable_acceleration = self.hardware_config.get('enable_acceleration', True)
        self.cuda_device = self.hardware_config.get('cuda_device', 0)
        self.cpu_threads = self.hardware_config.get('cpu_threads', 4)
        
        # Internal state
        self.model = None
        self.initialized = False
        self.vocabulary = []
        
        # Recent transcriptions cache
        self.recent_segments = deque(maxlen=10)
        
        # Performance metrics
        self.metrics = {
            'total_audio_seconds': 0,
            'total_processing_time': 0,
            'transcript_count': 0,
            'error_count': 0,
            'avg_rtf': 0
        }
        
        # Jetson optimization variables
        self.using_chunked_processing = False
        self.audio_chunks = None
    
    def initialize(self) -> bool:
        """
        Initialize the faster-whisper model.
        
        Returns:
            Success status
        """
        if self.initialized:
            return True
            
        try:
            if not FASTER_WHISPER_AVAILABLE:
                logger.error("faster-whisper is not available. Please install it with 'pip install faster-whisper'")
                return False
                
            # Apply Jetson optimizations if available
            model_params = {}
            is_jetson = False
            
            if JETSON_INTEGRATION_AVAILABLE:
                try:
                    # Initialize Jetson optimizations
                    jetson_integration = initialize_jetson_optimizations()
                    if jetson_integration:
                        logger.info("Applying Jetson-specific optimizations")
                        # Get optimized whisper parameters
                        model_params = jetson_integration.optimize_faster_whisper(self.model_size)
                        is_jetson = jetson_integration.is_jetson
                        
                        # Override with Jetson optimized settings if available
                        if 'compute_type' in model_params:
                            self.compute_type = model_params['compute_type']
                        if 'model_size' in model_params:
                            self.model_size = model_params['model_size']
                        if 'beam_size' in model_params:
                            self.beam_size = model_params['beam_size']
                        if 'cpu_threads' in model_params:
                            self.cpu_threads = model_params['num_workers']
                            
                        # Enhanced Jetson optimizations
                        if is_jetson:
                            # Apply more aggressive memory conservation on Jetson
                            logger.info("Applying enhanced Jetson memory optimizations")
                            
                            # Detect available RAM
                            try:
                                import subprocess
                                mem_info = subprocess.check_output('free -m', shell=True).decode('utf-8')
                                mem_lines = mem_info.split('\n')
                                if len(mem_lines) > 1:
                                    mem_values = mem_lines[1].split()
                                    total_mem = int(mem_values[1])
                                    logger.info(f"Detected total memory: {total_mem}MB")
                                    
                                    # Adjust compute type based on available memory
                                    if total_mem < 4096:  # Less than 4GB RAM
                                        self.compute_type = "int8"
                                        logger.info("Limited RAM detected, forcing INT8 quantization")
                                    elif total_mem < 8192:  # Less than 8GB RAM
                                        if self.model_size in ["medium", "large", "large-v2", "large-v3"]:
                                            # Scale down model size for limited memory
                                            self.model_size = "small"
                                            logger.info("Limited RAM detected, scaling down to 'small' model")
                            except Exception as e:
                                logger.warning(f"Failed to detect memory: {str(e)}")
                except Exception as e:
                    logger.warning(f"Failed to apply Jetson optimizations: {str(e)}")
                    
            # Additional GPU detection if Jetson integration isn't available
            if not is_jetson and TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0).lower()
                    # Enhanced Jetson detection with more identifiers
                    jetson_identifiers = ["tegra", "orin", "xavier", "jetson", "nvidia"]
                    is_jetson_gpu = any(name in device_name for name in jetson_identifiers)
                    
                    if is_jetson_gpu:
                        logger.info(f"Detected Jetson GPU: {device_name}")
                        is_jetson = True
                        
                        # Apply memory-conserving settings for Jetson
                        if self.model_size in ["large", "large-v2", "large-v3"]:
                            self.model_size = "medium"
                            logger.info("Scaled down to 'medium' model for Jetson compatibility")
                        
                        # Determine Jetson hardware generation for optimal settings
                        if "orin" in device_name:
                            # Orin can handle fp16
                            self.compute_type = "float16"
                            logger.info("Detected Orin series - using FP16 precision")
                        elif "xavier" in device_name:
                            # Xavier: determine NX vs AGX
                            if "nx" in device_name:
                                # Xavier NX - limited resources
                                self.compute_type = "int8"
                                logger.info("Detected Xavier NX - using INT8 precision for better performance")
                            else:
                                # Xavier AGX - more powerful
                                self.compute_type = "float16"
                                logger.info("Detected Xavier AGX - using FP16 precision")
                        else:
                            # Other Jetson platforms work better with int8
                            self.compute_type = "int8"
                            logger.info("Detected other Jetson platform - using INT8 precision")
                            
                        logger.info(f"Selected compute type for Jetson: {self.compute_type}")
                        
                        # Try to detect RAM for better optimization
                        try:
                            import subprocess
                            mem_info = subprocess.check_output('free -m', shell=True).decode('utf-8')
                            mem_lines = mem_info.split('\n')
                            if len(mem_lines) > 1:
                                mem_values = mem_lines[1].split()
                                total_mem = int(mem_values[1])
                                logger.info(f"Detected Jetson memory: {total_mem}MB")
                                
                                # Further optimization based on available RAM
                                if total_mem < 4096:  # Less than 4GB
                                    if self.model_size != "tiny.en" and self.model_size != "tiny":
                                        self.model_size = "tiny.en"
                                        logger.info("Limited RAM detected, forcing tiny.en model")
                        except Exception as mem_e:
                            logger.debug(f"Memory detection failed: {str(mem_e)}")
                except Exception as e:
                    logger.debug(f"GPU detection failed: {str(e)}")
            
            # Determine device and compute type
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() and self.enable_acceleration else "cpu"
            compute_type = self.compute_type
            
            # Adjust settings based on device
            if device == "cpu" and compute_type == "float16":
                # Fall back to int8 for better CPU performance
                compute_type = "int8"
                logger.info("Using int8 quantization for CPU inference")
            
            # Log initialization
            logger.info(f"Initializing faster-whisper model '{self.model_size}' on {device} with compute type {compute_type}")
            
            # Create model directory if it doesn't exist
            if not os.path.exists(self.model_path) and "/" in self.model_path:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Load medical vocabulary if enabled
            if self.use_medical_vocabulary:
                self._load_vocabulary()
            
            # Set up model initialization parameters
            model_init_params = {
                "model_size_or_path": self.model_size,
                "device": device,
                "compute_type": compute_type,
                "download_root": os.path.dirname(self.model_path) if "/" in self.model_path else None,
            }
            
            # Add CPU-specific parameters
            if device == "cpu":
                # Only add one threading parameter to avoid conflicts
                # The model only accepts one of these parameters
                model_init_params["cpu_threads"] = self.cpu_threads
            
            # Log full initialization parameters
            logger.debug(f"Initializing with parameters: {model_init_params}")
            
            # Load the model
            self.model = WhisperModel(**model_init_params)
            
            self.initialized = True
            logger.info(f"Faster-whisper model '{self.model_size}' initialized successfully on {device}")
            
            # Perform a warm-up inference
            self._warm_up()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper model: {str(e)}")
            return False
    
    def _load_vocabulary(self) -> bool:
        """
        Load custom medical vocabulary for improved recognition.
        
        Returns:
            Success status
        """
        try:
            if not os.path.exists(self.vocabulary_path):
                logger.warning(f"Vocabulary file not found: {self.vocabulary_path}")
                return False
                
            # Load vocabulary file
            with open(self.vocabulary_path, 'r') as f:
                lines = f.readlines()
                
            # Process each line
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Check if line has a replacement format (original -> corrected)
                if '->' in line:
                    original, corrected = [part.strip() for part in line.split('->')]
                    self.vocabulary.append(original)
                    
                # Check if line has an abbreviation format (ABBR = Full Form)
                elif '=' in line:
                    abbr, full_form = [part.strip() for part in line.split('=')]
                    self.vocabulary.append(abbr)
                    self.vocabulary.append(full_form)
                    
                # Otherwise, treat as a simple medical term
                else:
                    term = line.strip()
                    self.vocabulary.append(term)
                    
            logger.info(f"Loaded {len(self.vocabulary)} terms into medical vocabulary")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {str(e)}")
            return False
    
    def _warm_up(self) -> bool:
        """
        Warm up the model with a short dummy audio segment.
        
        Returns:
            Success status
        """
        try:
            # Create a short dummy audio segment (0.5 seconds of silence)
            dummy_audio = np.zeros(8000, dtype=np.float32)
            
            logger.info("Warming up model with dummy inference")
            
            # Perform transcription
            self.transcribe(dummy_audio)
            
            logger.info("Model warm-up completed")
            return True
            
        except Exception as e:
            logger.warning(f"Model warm-up failed: {str(e)}")
            return False
    
    def transcribe(self, audio: np.ndarray, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio using the faster-whisper model.
        
        Args:
            audio: Audio data as numpy array
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        if not self.initialized or self.model is None:
            logger.error("Model not initialized")
            return TranscriptionResult(text="", is_partial=False, segments=[])
            
        # Handle None or empty audio input
        if audio is None or (isinstance(audio, np.ndarray) and len(audio) == 0):
            logger.warning("Empty audio segment received")
            return TranscriptionResult(text="", is_partial=False, segments=[])
        
        try:
            # Process the config
            if config is None:
                config = TranscriptionConfig()
            
            # Start timing
            start_time = time.time()
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Apply tensor optimizations for Jetson if available
            try:
                from tccc.utils.tensor_optimization import TensorOptimizer, optimize_batch_processing
                
                # Create tensor optimizer with model-specific config
                if TORCH_AVAILABLE and torch.cuda.is_available() and self.enable_acceleration:
                    # Enhanced optimizer config for Jetson hardware
                    optimizer_config = {
                        "mixed_precision": self.compute_type == "float16",
                        "memory_efficient": True,
                        "use_tensorrt": False,  # TensorRT requires specific export for whisper
                        "memory_limit_mb": 2048,  # Limit memory usage on Jetson
                        "jetson_optimized": True
                    }
                    
                    # Check if we're running on Jetson hardware
                    try:
                        import torch
                        device_name = torch.cuda.get_device_name(0).lower()
                        if any(name in device_name for name in ["tegra", "orin", "xavier"]):
                            logger.info(f"Detected Jetson hardware: {device_name}, applying specialized optimizations")
                            # Add Jetson-specific optimizations
                            optimizer_config["memory_per_sample_mb"] = 256  # Estimated memory per audio second
                            optimizer_config["stream_audio"] = True  # Process in smaller chunks
                            optimizer_config["power_efficient"] = True  # Optimize for power efficiency
                    except Exception as e:
                        logger.debug(f"Not running on Jetson or couldn't detect: {e}")
                    
                    tensor_optimizer = TensorOptimizer(optimizer_config)
                    
                    # Convert to tensor, optimize, and back to numpy
                    audio_tensor = torch.from_numpy(audio)
                    
                    # Apply memory-efficient processing for longer audio
                    if len(audio) > 80000:  # >5 seconds at 16kHz
                        logger.debug("Using chunked processing for long audio on Jetson")
                        # Calculate optimal chunk size based on available memory
                        optimal_chunk_size = optimize_batch_processing(
                            8000,  # 0.5s chunks as base size
                            optimizer_config
                        )
                        
                        # Process in chunks if very long audio
                        if len(audio) > 480000:  # >30 seconds
                            chunks = []
                            for i in range(0, len(audio_tensor), optimal_chunk_size):
                                chunk = audio_tensor[i:i+optimal_chunk_size]
                                optimized_chunk = tensor_optimizer.optimize_tensor(chunk)
                                chunks.append(optimized_chunk)
                            
                            # Use the first chunk as representative for processing approach
                            audio_tensor = chunks[0]
                            # Record that we're using chunked processing
                            self.using_chunked_processing = True
                            self.audio_chunks = chunks
                        else:
                            # Standard optimization for medium-length audio
                            audio_tensor = tensor_optimizer.optimize_tensor(audio_tensor)
                    else:
                        # Standard optimization for short audio
                        audio_tensor = tensor_optimizer.optimize_tensor(audio_tensor)
                    
                    # Check if audio tensor was moved to GPU
                    if audio_tensor.device.type == "cuda":
                        # Use GPU tensor directly when possible
                        # Some implementations support direct torch tensor input
                        try:
                            if hasattr(self.model, "supports_torch_input") and self.model.supports_torch_input:
                                # Use optimized tensor directly
                                logger.debug("Using optimized GPU tensor directly for inference")
                                # Keep as tensor for next step
                            else:
                                # Convert back to numpy for compatibility
                                audio = audio_tensor.cpu().numpy()
                        except:
                            # Fall back to numpy if direct tensor use fails
                            audio = audio_tensor.cpu().numpy()
                    else:
                        # CPU tensor, convert back to numpy
                        audio = audio_tensor.numpy()
                    
                    logger.debug("Applied enhanced tensor optimizations for Jetson")
                    
            except ImportError:
                # No tensor optimizations available, use standard processing
                pass
            except Exception as e:
                logger.warning(f"Error applying tensor optimizations: {e}, continuing with standard processing")
            
            # Set up transcription parameters
            transcription_params = {
                "language": self.language,
                "beam_size": self.beam_size,
                "word_timestamps": config.word_timestamps,
                "task": "transcribe",
                "temperature": 0.0  # Use greedy decoding for deterministic results
            }
            
            # Add VAD parameters if enabled
            # Only add vad_filter parameter - the newer versions of faster-whisper handle
            # the parameters differently than we expected
            if self.vad_filter:
                transcription_params["vad_filter"] = True
                
            # Add vocabulary if available
            if self.use_medical_vocabulary and self.vocabulary:
                transcription_params["initial_prompt"] = " ".join(self.vocabulary[:20])  # Use top terms as prompt
            
            # Check if we're using Jetson-optimized chunked processing
            if hasattr(self, 'using_chunked_processing') and self.using_chunked_processing and hasattr(self, 'audio_chunks'):
                logger.info("Using Jetson-optimized chunked processing")
                
                all_segments = []
                language_info = None
                
                # Determine optimal processing strategy based on hardware
                parallel_processing = False
                
                # Use torch.cuda properties to detect Jetson hardware capabilities
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        device_name = torch.cuda.get_device_name(0).lower()
                        # Check if this is a more powerful Jetson that can handle parallel processing
                        if "orin" in device_name and "nano" not in device_name:
                            # Orin AGX or NX can handle more parallel work
                            parallel_processing = True
                            logger.info("Using parallel chunk processing for Orin device")
                        elif "xavier" in device_name and "nx" not in device_name:
                            # Xavier AGX can handle some parallel work
                            parallel_processing = True
                            logger.info("Using parallel chunk processing for Xavier AGX")
                    except Exception as e:
                        logger.debug(f"Error detecting hardware capabilities: {e}")
                
                # Process chunks based on hardware capabilities
                if parallel_processing and len(self.audio_chunks) > 1:
                    # Process chunks in parallel for better performance on capable hardware
                    import concurrent.futures
                    
                    # Define a function to process a single chunk
                    def process_chunk(chunk_data):
                        idx, chunk = chunk_data
                        chunk_offset = idx * 8000 / 16000  # Time offset in seconds
                        
                        # Convert chunk to numpy if needed
                        if isinstance(chunk, torch.Tensor):
                            chunk_np = chunk.cpu().numpy()
                        else:
                            chunk_np = chunk
                        
                        # Transcribe the chunk
                        chunk_segments, chunk_info = self.model.transcribe(chunk_np, **transcription_params)
                        
                        # Add time offset to segments
                        for segment in chunk_segments:
                            segment.start += chunk_offset
                            segment.end += chunk_offset
                            if hasattr(segment, 'words') and segment.words:
                                for word in segment.words:
                                    word.start += chunk_offset
                                    word.end += chunk_offset
                        
                        return chunk_segments, chunk_info
                    
                    # Use ThreadPoolExecutor for parallel processing
                    # Limit max_workers based on CPU cores and chunk count
                    max_workers = min(os.cpu_count() or 2, len(self.audio_chunks), 4)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all chunks for processing
                        future_to_chunk = {
                            executor.submit(process_chunk, (i, chunk)): i 
                            for i, chunk in enumerate(self.audio_chunks)
                        }
                        
                        # Collect results as they complete
                        for future in concurrent.futures.as_completed(future_to_chunk):
                            chunk_segments, chunk_info = future.result()
                            
                            # Save the language info from the first result
                            if language_info is None:
                                language_info = chunk_info
                            
                            # Add segments to the list
                            all_segments.extend(chunk_segments)
                    
                    # Sort segments by start time to ensure correct order
                    all_segments.sort(key=lambda x: x.start)
                    
                else:
                    # Sequential processing for less capable hardware
                    # Process each chunk and accumulate results
                    for i, chunk in enumerate(self.audio_chunks):
                        chunk_offset = i * 8000 / 16000  # Time offset in seconds (8000 samples per chunk)
                        
                        # Process in numpy format for compatibility
                        if isinstance(chunk, torch.Tensor):
                            chunk_np = chunk.cpu().numpy()
                        else:
                            chunk_np = chunk
                        
                        # Transcribe the chunk
                        chunk_segments, chunk_info = self.model.transcribe(chunk_np, **transcription_params)
                        
                        # Save the language info from the first chunk
                        if language_info is None:
                            language_info = chunk_info
                        
                        # Add time offset to segments and collect
                        for segment in chunk_segments:
                            segment.start += chunk_offset
                            segment.end += chunk_offset
                            if hasattr(segment, 'words') and segment.words:
                                for word in segment.words:
                                    word.start += chunk_offset
                                    word.end += chunk_offset
                            all_segments.append(segment)
                
                # Use accumulated segments
                segments = all_segments
                info = language_info if language_info else chunk_info
                
                # Merge adjacent segments with same speaker for better readability
                if len(segments) > 1:
                    merged_segments = []
                    current = segments[0]
                    
                    for next_seg in segments[1:]:
                        # If segments are close in time (within 0.5s) and have same speaker, merge them
                        gap = next_seg.start - current.end
                        same_speaker = True  # Placeholder for speaker detection logic
                        
                        if gap < 0.5 and same_speaker:
                            # Merge segments
                            current.text += " " + next_seg.text
                            current.end = next_seg.end
                            # Combine words if available
                            if hasattr(current, 'words') and hasattr(next_seg, 'words'):
                                if current.words and next_seg.words:
                                    current.words.extend(next_seg.words)
                        else:
                            # Add current segment and move to next
                            merged_segments.append(current)
                            current = next_seg
                    
                    # Add the last segment
                    merged_segments.append(current)
                    segments = merged_segments
                
                # Reset chunked processing flag
                self.using_chunked_processing = False
                self.audio_chunks = None
                
            else:
                # Memory-efficient processing for long audio (standard approach)
                if len(audio) > 480000:  # 30 seconds at 16kHz
                    logger.debug("Using chunked processing for long audio")
                    # Set chunk parameters to split processing
                    transcription_params["chunk_size"] = 30  # 30 seconds per chunk
                    
                    # Jetson-aware batch processing
                    if self.compute_type == "float16" and TORCH_AVAILABLE and torch.cuda.is_available():
                        # Check if running on Jetson
                        try:
                            device_name = torch.cuda.get_device_name(0).lower()
                            is_jetson = any(name in device_name for name in ["tegra", "orin", "xavier"])
                            
                            # Only effective when beam_size > 1
                            if self.beam_size > 1:
                                if is_jetson:
                                    # More conservative batch size for Jetson
                                    transcription_params["batch_size"] = 8
                                else:
                                    # Standard GPU batch size
                                    transcription_params["batch_size"] = 16
                        except Exception:
                            # Default batch size if detection fails
                            if self.beam_size > 1:
                                transcription_params["batch_size"] = 12
                    else:
                        # Use more threads for CPU processing
                        transcription_params["num_workers"] = min(self.cpu_threads * 2, 8)
                
                # Transcribe with faster-whisper
                segments, info = self.model.transcribe(audio, **transcription_params)
            
            # Convert to our result format
            full_text = ""
            result_segments = []
            
            for segment in segments:
                # Get word timestamps if available
                words = []
                if hasattr(segment, "words") and segment.words:
                    for word_data in segment.words:
                        word = Word(
                            text=word_data.word,
                            start_time=word_data.start,
                            end_time=word_data.end,
                            confidence=word_data.probability,
                            speaker=None
                        )
                        words.append(word)
                
                # Create segment
                ts_segment = TranscriptionSegment(
                    text=segment.text,
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.avg_logprob,
                    words=words
                )
                
                result_segments.append(ts_segment)
                full_text += segment.text + " "
            
            # Create result
            result = TranscriptionResult(
                text=full_text.strip(),
                segments=result_segments,
                is_partial=False,
                language=info.language
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            audio_duration = len(audio) / 16000  # Assuming 16kHz
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            
            self.metrics['total_audio_seconds'] += audio_duration
            self.metrics['total_processing_time'] += processing_time
            self.metrics['transcript_count'] += 1
            
            # Update average RTF
            self.metrics['avg_rtf'] = (
                (self.metrics['avg_rtf'] * (self.metrics['transcript_count'] - 1) + real_time_factor) /
                self.metrics['transcript_count']
            )
            
            # Store segment for context
            self.recent_segments.append(result)
            
            logger.debug(f"Transcription completed in {processing_time:.2f}s, RTF: {real_time_factor:.2f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"Faster-whisper transcription error: {str(e)}")
            self.metrics['error_count'] += 1
            return TranscriptionResult(text="", is_partial=False)
    
    def transcribe_file(self, file_path: str, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio from a file.
        
        Args:
            file_path: Path to audio file
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        try:
            # Load audio file
            import soundfile as sf
            audio, sample_rate = sf.read(file_path)
            
            # Convert to mono if needed
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe
            return self.transcribe(audio, config)
            
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}")
            self.metrics['error_count'] += 1
            return TranscriptionResult(text="", is_partial=False)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the model.
        
        Returns:
            Status dictionary
        """
        # Calculate performance metrics
        avg_rtf = self.metrics['avg_rtf']
        total_seconds = self.metrics['total_audio_seconds']
        transcript_count = self.metrics['transcript_count']
        error_count = self.metrics['error_count']
        
        # Determine if we're on Jetson hardware
        is_jetson = False
        jetson_platform = "unknown"
        jetson_ram = 0
        jetson_cuda_cores = 0
        jetson_specific_model = "unknown"
        jetson_compute_cap = "unknown"
        
        # Check via multiple detection methods
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0).lower()
                # Expanded Jetson detection identifiers
                jetson_identifiers = ["tegra", "orin", "xavier", "jetson", "nvidia"]
                is_jetson = any(name in device_name for name in jetson_identifiers)
                
                if is_jetson:
                    # Identify specific platform
                    if "orin" in device_name:
                        jetson_platform = "Orin"
                        jetson_compute_cap = "8.7"  # Ampere architecture
                        
                        # Identify specific Orin model
                        if "nano" in device_name:
                            jetson_specific_model = "Orin Nano"
                            jetson_cuda_cores = 1024
                        elif "nx" in device_name:
                            jetson_specific_model = "Orin NX"
                            jetson_cuda_cores = 1536
                        else:
                            jetson_specific_model = "Orin AGX"
                            jetson_cuda_cores = 2048
                            
                    elif "xavier" in device_name:
                        jetson_platform = "Xavier"
                        jetson_compute_cap = "7.2"  # Volta architecture
                        
                        # Identify specific Xavier model
                        if "nx" in device_name:
                            jetson_specific_model = "Xavier NX"
                            jetson_cuda_cores = 384
                        else:
                            jetson_specific_model = "Xavier AGX"
                            jetson_cuda_cores = 512
                            
                    elif "tegra" in device_name:
                        jetson_platform = "Tegra"
                        # Check for TX2 or TX1
                        if "tx2" in device_name:
                            jetson_specific_model = "TX2"
                            jetson_cuda_cores = 256
                            jetson_compute_cap = "6.2"  # Pascal architecture
                        elif "tx1" in device_name:
                            jetson_specific_model = "TX1"
                            jetson_cuda_cores = 256
                            jetson_compute_cap = "5.3"  # Maxwell architecture
                        
                # Get memory info
                if is_jetson:
                    try:
                        import subprocess
                        # Get RAM information
                        mem_info = subprocess.check_output('free -m', shell=True).decode('utf-8')
                        mem_lines = mem_info.split('\n')
                        if len(mem_lines) > 1:
                            mem_values = mem_lines[1].split()
                            jetson_ram = int(mem_values[1])
                            
                        # Try to get CPU information
                        cpu_info = subprocess.check_output('cat /proc/cpuinfo | grep "model name" | head -1', shell=True).decode('utf-8')
                        if "model name" in cpu_info:
                            cpu_model = cpu_info.split(':')[1].strip()
                            
                        # Try to get CUDA core count using nvcc
                        try:
                            cuda_info = subprocess.check_output('nvcc --version', shell=True).decode('utf-8')
                            # Extract CUDA version if available
                            if "release" in cuda_info:
                                cuda_version = cuda_info.split("release")[1].split(",")[0].strip()
                        except:
                            cuda_version = "unknown"
                            
                        # Get NVIDIA driver version if available
                        try:
                            driver_info = subprocess.check_output('nvidia-smi --query-gpu=driver_version --format=csv,noheader', shell=True).decode('utf-8')
                            driver_version = driver_info.strip()
                        except:
                            driver_version = "unknown"
                            
                    except Exception as e:
                        logger.debug(f"Error getting Jetson system details: {e}")
            except Exception as e:
                logger.debug(f"GPU detection failed in status check: {e}")
        
        # Create status dictionary
        status = {
            'initialized': self.initialized,
            'model_type': 'faster-whisper',
            'model_size': self.model_size,
            'language': self.language,
            'compute_type': self.compute_type,
            'acceleration': {
                'enabled': self.enable_acceleration,
                'device': "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() and self.enable_acceleration else "cpu",
                'cpu_threads': self.cpu_threads
            },
            'vad_filter': self.vad_filter,
            'vocabulary': {
                'enabled': self.use_medical_vocabulary,
                'terms_count': len(self.vocabulary)
            },
            'performance': {
                'transcript_count': transcript_count,
                'error_count': error_count,
                'total_audio_seconds': total_seconds,
                'realtime_factor': avg_rtf,
                'accuracy_estimate': 1.0 - (error_count / (transcript_count + 1))
            },
            'optimizations': {
                'chunked_processing': hasattr(self, 'using_chunked_processing') and self.using_chunked_processing,
                'memory_efficient': True,
                'jetson_optimized': is_jetson
            }
        }
        
        # Add GPU memory usage if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                
                status['acceleration']['memory'] = {
                    'allocated_mb': round(allocated, 2),
                    'reserved_mb': round(reserved, 2)
                }
            except Exception:
                pass
        
        # Add Jetson-specific information if available
        if is_jetson or JETSON_INTEGRATION_AVAILABLE:
            # Initialize Jetson information dictionary with enhanced details
            status['jetson'] = {
                'detected': is_jetson,
                'platform': jetson_platform,
                'specific_model': jetson_specific_model,
                'ram_mb': jetson_ram,
                'cuda_cores': jetson_cuda_cores,
                'compute_capability': jetson_compute_cap,
                'optimizations_applied': is_jetson,
                'cuda_version': cuda_version if 'cuda_version' in locals() else "unknown",
                'driver_version': driver_version if 'driver_version' in locals() else "unknown",
                'cpu_model': cpu_model if 'cpu_model' in locals() else "unknown"
            }
            
            # Add detailed resource monitoring if available
            if JETSON_INTEGRATION_AVAILABLE:
                try:
                    jetson_integration = initialize_jetson_optimizations()
                    if jetson_integration:
                        # Update detection info
                        status['jetson']['detected'] = is_jetson or jetson_integration.is_jetson
                        
                        # Add resource stats if monitoring is active
                        resource_stats = jetson_integration.get_resource_stats()
                        if resource_stats:
                            status['jetson']['resources'] = resource_stats
                            
                        # Add power efficiency mode if available
                        if hasattr(jetson_integration, 'get_power_mode'):
                            power_mode = jetson_integration.get_power_mode()
                            if power_mode:
                                status['jetson']['power_mode'] = power_mode
                except Exception as e:
                    logger.warning(f"Could not get detailed Jetson status: {str(e)}")
        
        return status
        
    def transcribe_segment(self, audio: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transcribe an audio segment with optional metadata.
        
        Args:
            audio: Audio data as numpy array
            metadata: Additional metadata for transcription
            
        Returns:
            Dictionary with transcription result
        """
        if not self.initialized or self.model is None:
            logger.error("Model not initialized")
            return {'error': 'Model not initialized', 'text': ''}
            
        # Handle None or empty audio input
        if audio is None or (isinstance(audio, np.ndarray) and len(audio) == 0):
            logger.warning("Empty audio segment received")
            return {'text': '', 'segments': [], 'is_partial': False, 'language': self.language}
        
        try:
            # Start timing
            start_time = time.time()
            
            # Create TranscriptionConfig from metadata
            config = TranscriptionConfig()
            if metadata:
                if 'word_timestamps' in metadata:
                    config.word_timestamps = metadata['word_timestamps']
                if 'include_punctuation' in metadata:
                    config.include_punctuation = metadata['include_punctuation']
                if 'include_capitalization' in metadata:
                    config.include_capitalization = metadata['include_capitalization']
                if 'confidence_threshold' in metadata:
                    config.confidence_threshold = metadata['confidence_threshold']
                
                # Check if this is battlefield audio to adjust VAD parameters
                is_battlefield = metadata.get('battlefield_audio', False)
                if is_battlefield:
                    # Adjust VAD parameters for battlefield conditions
                    # More aggressive settings to better filter out battlefield noise
                    self.vad_parameters = {
                        'threshold': 0.6,  # Higher threshold to avoid false detections
                        'min_speech_duration_ms': 300,  # Longer minimum speech duration
                        'max_speech_duration_s': 30.0,
                        'min_silence_duration_ms': 700  # Longer silence before ending segment
                    }
            
            # Transcribe audio
            result = self.transcribe(audio, config)
            
            # Convert to dictionary format
            result_dict = self._result_to_dict(result)
            
            # Add performance metrics
            processing_time = time.time() - start_time
            audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            
            result_dict['metrics'] = {
                'audio_duration': audio_duration,
                'processing_time': processing_time,
                'real_time_factor': real_time_factor
            }
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Transcription segment error: {str(e)}")
            self.metrics['error_count'] += 1
            return {'error': str(e), 'text': ''}
    
    def _result_to_dict(self, result: TranscriptionResult) -> Dict[str, Any]:
        """
        Convert TranscriptionResult to dictionary.
        
        Args:
            result: TranscriptionResult object
            
        Returns:
            Dictionary representation
        """
        segments = []
        for segment in result.segments:
            seg_dict = {
                'text': segment.text,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'confidence': segment.confidence
            }
            
            if hasattr(segment, 'speaker') and segment.speaker is not None:
                seg_dict['speaker'] = segment.speaker
            
            if segment.words:
                seg_dict['words'] = [
                    {
                        'text': word.text,
                        'start_time': word.start_time,
                        'end_time': word.end_time,
                        'confidence': word.confidence,
                        'speaker': word.speaker if hasattr(word, 'speaker') else None
                    }
                    for word in segment.words
                ]
            
            segments.append(seg_dict)
        
        return {
            'text': result.text,
            'segments': segments,
            'is_partial': result.is_partial,
            'language': result.language
        }
        
    def shutdown(self) -> bool:
        """
        Shutdown the STT engine, releasing resources.
        
        Returns:
            Success status
        """
        try:
            # Release model resources
            self.model = None
            self.initialized = False
            
            # Clear caches
            self.recent_segments.clear()
            self.vocabulary.clear()
            
            logger.info("Faster-whisper STT engine shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error shutting down Faster-whisper STT engine: {str(e)}")
            return False