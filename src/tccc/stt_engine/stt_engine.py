"""
STT Engine implementation for TCCC.ai system.

This module provides real-time speech-to-text functionality with optimizations
for Jetson hardware, medical terminology, and streaming transcription.
"""

import os
import re
import time
import json
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import traceback
import asyncio
import queue
from tccc.processing_core.processing_core import ModuleState # Added import

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from tccc.utils.logging import get_logger
from tccc.utils.config import Config
from tccc.processing_core.processing_core import ModuleState

logger = get_logger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for transcription."""
    confidence_threshold: float = 0.6
    word_timestamps: bool = True
    include_punctuation: bool = True
    include_capitalization: bool = True
    format_numbers: bool = True
    segment_length: int = 30
    
    # Streaming settings
    streaming_enabled: bool = True
    partial_results_interval_ms: int = 500
    max_context_length_sec: int = 60
    stability_threshold: float = 0.8


@dataclass
class Word:
    """Word with timestamp and confidence."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker: Optional[int] = None


@dataclass
class TranscriptionSegment:
    """Segment of transcription with words and metadata."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    words: List[Word] = field(default_factory=list)
    speaker: Optional[int] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    is_partial: bool = False
    language: Optional[str] = None


class ModelManager:
    """
    Manages the Whisper model loading, optimization, and inference.
    Handles ONNX conversion and quantization for Jetson hardware.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # --- Modified to use flat config structure ---
        # Assuming the passed 'config' is the flat stt_engine section from main config
        self.model_type = config.get('type', 'whisper') # Defaulting to whisper if not specified
        self.model_size = config.get('model', 'medium') # Key 'model' holds the size (e.g., 'tiny.en')
        self.model_path = config.get('model_path', f'models/whisper-{self.model_size}-en/')
        
        # --- Workaround for tiny.en model name issue ---
        if self.model_size == 'tiny.en':
            logger.warning("Workaround: Changing requested model size from 'tiny.en' to 'tiny' due to Whisper library inconsistency.")
            self.model_size = 'tiny'
            # Adjust default path if it used the original size and wasn't explicitly set
            if config.get('model_path') is None and self.model_path == f'models/whisper-tiny.en-en/':
                self.model_path = f'models/whisper-{self.model_size}-en/'
        # --- End Workaround ---
        
        self.batch_size = config.get('batch_size', 1)
        # Map compute_type to mixed_precision if compute_type exists, else default True?
        # Let's assume compute_type dictates precision, defaulting to True if compute_type not present.
        self.mixed_precision = config.get('compute_type', 'float16') == 'float16' 
        self.language = config.get('language', 'en')
        self.beam_size = config.get('beam_size', 5)
        
        # Extract hardware acceleration settings directly from config
        self.enable_acceleration = config.get('enable_acceleration', True) # Assuming this key exists or defaults true
        self.device = config.get('device', 'cuda') # Use 'device' key from config
        self.use_tensorrt = config.get('use_tensorrt', True if self.device == 'cuda' else False) # Default based on device?
        self.cuda_streams = config.get('cuda_streams', 2)
        self.memory_limit_mb = config.get('memory_limit_mb', 4096)
        # Map compute_type to quantization? Default FP16
        self.quantization = config.get('compute_type', 'float16').upper()
        # --- End Modification ---
        
        # Model and session objects
        self.model = None
        self.processor = None
        self.onnx_session = None
        
        # Cache for initialized state
        self.initialized = False
        
        # Warmup status
        self.is_warmed_up = False
        
        # Initialize locks
        self.model_lock = threading.RLock()
    
    def initialize(self) -> bool:
        """
        Initialize the model.
        
        Returns:
            Success status
        """
        if self.initialized:
            logger.info("Model already initialized")
            return True
        
        try:
            with self.model_lock:
                # First try faster-whisper as it's optimized for CUDA and more efficient
                if self.model_type == 'faster-whisper':
                    logger.info("Initializing with faster-whisper CUDA optimized version")
                    result = self._initialize_faster_whisper()
                    
                    if result:
                        self.initialized = True
                        # Perform model warmup
                        self._warmup_model()
                        logger.info(f"Model '{self.model_type}-{self.model_size}' initialized successfully with faster-whisper")
                        return True
                    else:
                        logger.warning("faster-whisper initialization failed, falling back to PyTorch Whisper")
                
                # Fall back to PyTorch Whisper
                if self.model_type == 'whisper' and TORCH_AVAILABLE:
                    logger.info("Initializing with PyTorch Whisper (standard)")
                    result = self._initialize_whisper_torch()
                    
                    # If PyTorch initialization succeeds, skip ONNX attempts
                    if result:
                        self.initialized = True
                        # Perform model warmup
                        self._warmup_model()
                        logger.info(f"Model '{self.model_type}-{self.model_size}' initialized successfully with PyTorch")
                        return True
                
                # Only try ONNX if PyTorch failed or isn't available and ONNX is available
                if self.model_type == 'whisper' and ONNX_AVAILABLE and self.enable_acceleration:
                    logger.info("Trying ONNX Whisper as fallback")
                    result = self._initialize_whisper_onnx()
                else:
                    logger.error(f"Model type '{self.model_type}' not supported or required dependencies not available")
                    return False
                
                if result:
                    self.initialized = True
                    # Perform model warmup
                    self._warmup_model()
                    logger.info(f"Model '{self.model_type}-{self.model_size}' initialized successfully with ONNX")
                return result
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(traceback.format_exc()) # Log traceback at ERROR level
            return False
    
    def _initialize_faster_whisper(self) -> bool:
        """
        Initialize faster-whisper model with CUDA optimization.
        This implementation uses CTranslate2 for faster inference.
        
        Returns:
            Success status
        """
        try:
            if not FASTER_WHISPER_AVAILABLE:
                logger.error("faster-whisper package not available. Run 'pip install faster-whisper' to install it.")
                return False
                
            # Set up compute type based on config
            compute_type = self.config.get('compute_type', "float16" if self.mixed_precision else "float32")
            device = self.config.get('device', "cuda" if torch.cuda.is_available() and self.enable_acceleration else "cpu")
            num_workers = self.config.get('num_workers', 2)
            cpu_threads = self.config.get('cpu_threads', 4)
            
            logger.info(f"Initializing faster-whisper with model size: {self.model_size}, compute type: {compute_type}, device: {device}")
            
            # Check CUDA availability
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
            
            if device == "cuda":
                # Get CUDA device properties
                device_props = torch.cuda.get_device_properties(0)  # Assuming device 0
                logger.info(f"Using CUDA device: {device_props.name} with {device_props.total_memory/1024**3:.2f} GB memory")
            
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Create the faster-whisper model
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=self.model_path,
                cpu_threads=cpu_threads,
                num_workers=num_workers
            )
            
            logger.info(f"faster-whisper model '{self.model_size}' loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper model: {e}")
            logger.error(traceback.format_exc())  # Log the full traceback
            return False
            
    def _initialize_whisper_onnx(self) -> bool:
        """
        Initialize Whisper model with ONNX Runtime.
        
        Returns:
            Success status
        """
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Set up ONNX Runtime session options
            session_options = ort.SessionOptions()
            
            # Configure parallel execution
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.inter_op_num_threads = 4
            session_options.intra_op_num_threads = 4
            
            # Set graph optimization level
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Configure memory limits
            if self.memory_limit_mb > 0:
                session_options.enable_mem_pattern = True
                session_options.enable_mem_reuse = True
                # Handle different ONNX Runtime versions - some use set_memory_limit
                try:
                    # Convert MB to bytes
                    session_options.set_memory_limit(self.memory_limit_mb * 1024 * 1024)
                except AttributeError:
                    # Newer versions use set_memory_limit_by_device
                    try:
                        # Default to CPU device (0)
                        session_options.add_session_config_entry("session.max_mem_target", str(self.memory_limit_mb * 1024 * 1024))
                    except AttributeError:
                        logger.warning("Unable to set memory limit for ONNX Runtime - continuing without limit")
            
            # Get available providers from system
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")
            
            # Set up providers based on what's available
            providers = []
            provider_options = []
            
            # Check if acceleration is enabled and appropriate providers are available
            if self.enable_acceleration and self.device == 'cuda':
                # Try to use TensorRT if available and requested
                if self.use_tensorrt and 'TensorrtExecutionProvider' in available_providers:
                    # TensorRT execution provider options
                    trt_options = {
                        'device_id': 0, # Default to device 0
                        'trt_max_workspace_size': self.memory_limit_mb * 1024 * 1024,
                        'trt_fp16_enable': '1' if self.mixed_precision else '0',
                    }
                    providers.append('TensorrtExecutionProvider')
                    provider_options.append(trt_options)
                    logger.info("Using TensorRT provider for acceleration")
                
                # Try to use CUDA if available
                if 'CUDAExecutionProvider' in available_providers:
                    # CUDA execution provider options
                    cuda_options = {
                        'device_id': 0, # Default to device 0
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': self.memory_limit_mb * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': '1',
                    }
                    providers.append('CUDAExecutionProvider')
                    provider_options.append(cuda_options)
                    logger.info("Using CUDA provider for acceleration")
            
            # Always add CPU provider as fallback
            providers.append('CPUExecutionProvider')
            provider_options.append({})
            
            # Add other available providers if they exist
            for provider in available_providers:
                if provider not in providers and provider != 'CPUExecutionProvider':
                    providers.append(provider)
                    provider_options.append({})
                    logger.info(f"Added available provider: {provider}")
            
            logger.info(f"Using ONNX providers: {providers}")
            
            # Get model paths for encoder and decoder
            encoder_path = os.path.join(self.model_path, 'encoder.onnx')
            decoder_path = os.path.join(self.model_path, 'decoder.onnx')
            
            logger.debug(f"Checking for ONNX models: encoder='{encoder_path}', decoder='{decoder_path}'")
            # Check if models exist, if not we need to convert them
            if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                logger.info(f"ONNX models not found at {self.model_path}, attempting conversion")
                logger.debug("Calling _convert_whisper_to_onnx")
                conversion_success = self._convert_whisper_to_onnx()
                
                # If conversion failed, fallback to PyTorch
                if not conversion_success:
                    logger.warning("ONNX conversion failed, falling back to PyTorch mode")
                    return False
            
            # Check again after potential conversion
            if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                logger.error(f"ONNX models still not found at {self.model_path} after conversion attempt")
                return False
            
            # Create ONNX Runtime sessions
            try:
                logger.debug(f"Creating ONNX encoder session from: {encoder_path}")
                self.encoder_session = ort.InferenceSession(
                    encoder_path, 
                    sess_options=session_options, 
                    providers=providers, 
                    provider_options=provider_options
                )
                
                logger.debug(f"Creating ONNX decoder session from: {decoder_path}")
                self.decoder_session = ort.InferenceSession(
                    decoder_path, 
                    sess_options=session_options, 
                    providers=providers, 
                    provider_options=provider_options
                )
            except Exception as e:
                logger.error(f"Failed to create ONNX sessions: {e}")
                # If session creation failed, fallback to PyTorch
                return False
            
            # Load tokenizer and other necessary components
            logger.debug("Loading Whisper processor for ONNX")
            self._load_whisper_processor()
            
            logger.info(f"Whisper ONNX model loaded with providers: {providers}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper ONNX model: {e}")
            logger.error(traceback.format_exc()) # Log traceback at ERROR level
            return False
    
    def _initialize_whisper_torch(self) -> bool:
        """
        Initialize Whisper model with PyTorch.
        
        Returns:
            Success status
        """
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available, cannot initialize Whisper model")
                return False
            
            import whisper
            from transformers import WhisperProcessor
            
            # Load model
            logger.debug(f"Loading Whisper PyTorch model: {self.model_size}")
            device = "cuda" if torch.cuda.is_available() and self.enable_acceleration else "cpu"
            self.model = whisper.load_model(self.model_size, device=device)
            logger.debug("Whisper PyTorch model loaded successfully")
            
            # Load processor
            logger.debug(f"Loading Whisper processor from Hugging Face: openai/whisper-{self.model_size}")
            self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{self.model_size}")
            logger.debug("Whisper processor loaded successfully")
            
            logger.info(f"Whisper PyTorch model loaded on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper PyTorch model: {e}")
            logger.error("Whisper PyTorch initialization failed", exc_info=True)
            return False
    
    def _convert_whisper_to_onnx(self) -> bool:
        """
        Convert Whisper PyTorch model to ONNX format.
        
        Returns:
            Success status
        """
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available, cannot convert Whisper model to ONNX")
                return False
            
            import whisper
            
            # Create model directory
            os.makedirs(self.model_path, exist_ok=True)
            
            # Load model
            logger.info(f"Loading Whisper model {self.model_size} for ONNX conversion")
            model = whisper.load_model(self.model_size)
            
            # Export encoder and decoder to ONNX
            encoder_path = os.path.join(self.model_path, 'encoder.onnx')
            decoder_path = os.path.join(self.model_path, 'decoder.onnx')
            
            logger.info("Converting Whisper encoder to ONNX")
            # Define dummy input for encoder
            dummy_input = torch.zeros((1, 80, 3000), dtype=torch.float32)
            
            # Temporarily disable torch.compiler and other features that may cause ONNX issues
            if hasattr(torch, 'compiler'):
                old_compiler_state = torch._dynamo.config.dynamic_shapes
                torch._dynamo.config.dynamic_shapes = False
            
            # Export encoder with lower opset version to avoid attention issues
            try:
                torch.onnx.export(
                    model.encoder,
                    dummy_input,
                    encoder_path,
                    export_params=True,
                    opset_version=12,  # Lower opset version to avoid attention issues
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size', 2: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}}
                )
            except Exception as e:
                logger.warning(f"Failed to export encoder with opset 12, trying fallback: {e}")
                # Fallback to PyTorch mode if ONNX conversion fails
                self.enable_acceleration = False
                return False
            
            logger.info("Converting Whisper decoder to ONNX")
            # Define dummy inputs for decoder
            dummy_audio_features = torch.zeros((1, 1500, 512), dtype=torch.float32)
            dummy_tokens = torch.zeros((1, 1), dtype=torch.int64)
            
            # Export decoder with lower opset version
            try:
                torch.onnx.export(
                    model.decoder,
                    (dummy_tokens, dummy_audio_features),
                    decoder_path,
                    export_params=True,
                    opset_version=12,  # Lower opset version to avoid attention issues
                    input_names=['tokens', 'audio_features'],
                    output_names=['output'],
                    dynamic_axes={'tokens': {0: 'batch_size', 1: 'sequence_length'},
                                'audio_features': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}}
                )
            except Exception as e:
                logger.warning(f"Failed to export decoder with opset 12, fallback to PyTorch mode: {e}")
                # If decoder export fails, remove encoder file to ensure we don't have partial conversion
                if os.path.exists(encoder_path):
                    os.remove(encoder_path)
                # Fallback to PyTorch mode if ONNX conversion fails
                self.enable_acceleration = False
                return False
                
            # Restore torch compiler state if we changed it
            if hasattr(torch, 'compiler') and 'old_compiler_state' in locals():
                torch._dynamo.config.dynamic_shapes = old_compiler_state
            
            # Export tokenizer and configuration
            model_config = {
                'language': self.language,
                'task': 'transcribe',
                'beam_size': self.beam_size,
                'temperature': 0.0,
                'model_size': self.model_size
            }
            
            with open(os.path.join(self.model_path, 'config.json'), 'w') as f:
                json.dump(model_config, f)
            
            logger.info(f"Whisper model converted to ONNX and saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert Whisper model to ONNX: {e}")
            return False
    
    def _load_whisper_processor(self) -> bool:
        """
        Load Whisper tokenizer and processor.
        
        Returns:
            Success status
        """
        try:
            from transformers import WhisperProcessor
            
            self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{self.model_size}")
            
            # Load model configuration
            config_path = os.path.join(self.model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.model_config.update(json.load(f))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Whisper processor: {e}")
            return False
    
    def _warmup_model(self) -> bool:
        """
        Warm up the model with a dummy input.
        
        Returns:
            Success status
        """
        if self.is_warmed_up:
            return True
        
        try:
            logger.info("Warming up model with dummy input")
            
            # Create dummy audio input (0.5 seconds of silence)
            sample_rate = 16000
            dummy_audio = np.zeros(sample_rate // 2, dtype=np.float32)
            
            # Run inference with dummy input
            if self.onnx_session:
                # ONNX Runtime inference
                self.transcribe(dummy_audio)
            elif self.model:
                # PyTorch inference
                self.transcribe(dummy_audio)
            
            self.is_warmed_up = True
            logger.info("Model warmup completed")
            return True
            
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            return False
    
    def transcribe(self, audio: np.ndarray, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio using the model.
        
        Args:
            audio: Audio data as numpy array
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        if not self.initialized:
            logger.error("Model not initialized")
            return TranscriptionResult(text="", is_partial=False)
        
        try:
            # Use faster-whisper model if available (preferred for CUDA optimization)
            if self.model_type == 'faster-whisper' and hasattr(self, 'model') and self.model is not None:
                return self._transcribe_with_faster_whisper(audio, config)
            
            # Use the encoder-decoder model for ONNX inference
            elif hasattr(self, 'encoder_session') and hasattr(self, 'decoder_session'):
                return self._transcribe_with_onnx(audio, config)
            
            # Use the PyTorch model
            elif hasattr(self, 'model') and self.model is not None:
                return self._transcribe_with_torch(audio, config)
            
            else:
                logger.error("No model available for transcription")
                return TranscriptionResult(text="", is_partial=False)
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return TranscriptionResult(text="", is_partial=False)
    
    def _transcribe_with_onnx(self, audio: np.ndarray, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio using ONNX Runtime.
        
        Args:
            audio: Audio data as numpy array
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        # Process the config
        if config is None:
            # Create default config
            config = TranscriptionConfig()
        
        try:
            # Preprocess audio (ensure correct format and sample rate)
            audio_features = self._preprocess_audio(audio)
            
            # Run encoder
            encoder_output = self.encoder_session.run(
                ['output'], 
                {'input': audio_features.astype(np.float32)}
            )[0]
            
            # Initialize token sequence with initial token
            tokens = np.array([[1]], dtype=np.int64)  # 1 is the BOS token
            
            # Maximum generation length
            max_length = 448  # A reasonable maximum for most utterances
            
            # Store generated tokens
            generated_tokens = [1]  # Start with BOS token
            
            # Perform decoding
            for i in range(max_length - 1):
                # Run decoder
                decoder_output = self.decoder_session.run(
                    ['output'], 
                    {'tokens': tokens, 'audio_features': encoder_output}
                )[0]
                
                # Get last token probabilities
                next_token_logits = decoder_output[0, -1, :]
                
                # Get next token (greedy decoding for simplicity)
                next_token = np.argmax(next_token_logits)
                
                # Add token to list
                generated_tokens.append(int(next_token))
                
                # Check for end-of-sequence token
                if next_token == 50257:  # EOS token
                    break
                
                # Update tokens for next iteration
                tokens = np.array([generated_tokens], dtype=np.int64)
            
            # Decode tokens to text
            text = self.processor.decode(generated_tokens)
            
            # Create result
            result = TranscriptionResult(
                text=text, 
                is_partial=False, 
                language=self.language
            )
            
            # Create a single segment (simplified for now)
            segment = TranscriptionSegment(
                text=text,
                start_time=0.0,
                end_time=len(audio) / 16000.0,  # Assuming 16kHz sample rate
                confidence=0.8  # Simplified confidence
            )
            result.segments.append(segment)
            
            return result
            
        except Exception as e:
            logger.error(f"ONNX transcription error: {e}")
            return TranscriptionResult(text="", is_partial=False)
    
    def _transcribe_with_faster_whisper(self, audio: np.ndarray, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio using faster-whisper with CUDA optimization.
        
        Args:
            audio: Audio data as numpy array
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        # Process the config
        if config is None:
            # Create default config
            config = TranscriptionConfig()
            
        try:
            # Ensure audio is the right format (float32)
            if audio.dtype != np.float32:
                # If int16, explicitly convert to float32 and normalize
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                else:
                    # For other types, convert to float32
                    audio = audio.astype(np.float32)
                
                # Ensure values are within [-1.0, 1.0]
                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()
            
            # Log audio stats for debugging
            logger.debug(f"Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, mean={np.abs(audio).mean():.6f}, " 
                         f"shape={audio.shape}, dtype={audio.dtype}")
            
            # Extract settings from the config
            beam_size = config.beam_size if hasattr(config, 'beam_size') else 5
            language = self.language  # Default to model's language
            
            # Run transcription with faster-whisper
            logger.debug(f"Running faster-whisper transcription with beam_size={beam_size}, language={language}")
            segments, info = self.model.transcribe(
                audio=audio,
                language=language,
                task="transcribe",
                beam_size=beam_size,
                word_timestamps=config.word_timestamps,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Convert segments to list for easier processing
            segment_list = list(segments)
            
            # Create result object
            result = TranscriptionResult(
                text=" ".join(seg.text for seg in segment_list),
                is_partial=False,
                language=info.language
            )
            
            # Create segments
            for i, seg in enumerate(segment_list):
                # Process words if available
                words = []
                if hasattr(seg, 'words') and seg.words:
                    for word_data in seg.words:
                        word = Word(
                            text=word_data.word,
                            start_time=word_data.start,
                            end_time=word_data.end,
                            confidence=word_data.probability
                        )
                        words.append(word)
                
                # Create segment
                segment = TranscriptionSegment(
                    text=seg.text,
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence=seg.avg_logprob,  # This is the log probability, not exactly confidence
                    words=words
                )
                result.segments.append(segment)
            
            logger.debug(f"Transcription complete: {result.text[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"faster-whisper transcription error: {e}")
            logger.error(traceback.format_exc())
            return TranscriptionResult(text="", is_partial=False)
    
    def _transcribe_with_torch(self, audio: np.ndarray, config: Optional[TranscriptionConfig] = None) -> TranscriptionResult:
        """
        Transcribe audio using PyTorch.
        
        Args:
            audio: Audio data as numpy array
            config: Transcription configuration
            
        Returns:
            TranscriptionResult object
        """
        # Process the config
        if config is None:
            # Create default config
            config = TranscriptionConfig()
        
        try:
            import whisper
        
            # Ensure audio is the right format (float32)
            if audio.dtype != np.float32:
                # If int16, explicitly convert to float32 and normalize
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                else:
                    # For other types, convert to float32
                    audio = audio.astype(np.float32)
                
                # Ensure values are within [-1.0, 1.0]
                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()
            
            # Ensure the sample rate is 16kHz (Whisper requirement)
            # If we had resampling code here we would use it, but for now we assume 16kHz
        
            # --- MODIFICATION: Always use simpler transcription settings to reduce load ---
            logger.debug("_transcribe_with_torch: Forcing simple transcription settings (beam_size=1, no timestamps)")
            try:
                # Use a simpler approach without beam search or timestamps
                simple_text = self.model.transcribe(
                    audio,
                    language=self.language,
                    temperature=0.0,
                    fp16=False,
                    beam_size=1,  # Simple beam size
                    word_timestamps=False  # No timestamps
                )["text"]
            
                whisper_result = {
                    "text": simple_text,
                    "segments": [
                        {
                            "text": simple_text, # Reuse transcribed text
                            "start": 0.0,
                            "end": len(audio) / 16000.0,
                            "confidence": 0.8, # Assign default confidence
                            "words": [] # Empty words list as timestamps are off
                        }
                    ],
                    "language": self.language
                }
                logger.debug(f"_transcribe_with_torch: Simple transcription successful. Text: '{simple_text[:50]}...' ")
            except Exception as e:
                 logger.exception(f"_transcribe_with_torch: Simple transcription attempt FAILED: {e}")
                 # Return empty result on failure
                 return TranscriptionResult(text="", is_partial=False)
            # --- END MODIFICATION ---

            # Extract text and segments
            text = whisper_result["text"]
        
            # Create TranscriptionResult
            result = TranscriptionResult(
                text=text,
                is_partial=False,
                language=whisper_result.get('language', 'en')
            )
        
            # Process segments
            for i, segment in enumerate(whisper_result["segments"]):
                # Create a TranscriptionSegment
                ts_segment = TranscriptionSegment(
                    text=segment["text"],
                    start_time=segment["start"],
                    end_time=segment["end"],
                    confidence=segment.get("confidence", 0.8)
                )
            
                result.segments.append(ts_segment)
        
            return result
        
        except Exception as e:
            logger.error(f"PyTorch transcription error: {e}")
            return TranscriptionResult(text="", is_partial=False)
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for model input.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Preprocessed audio features
        """
        try:
            import whisper
            
            # Convert to float32 if not already
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize if needed (whisper expects values in [-1, 1])
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Apply Whisper's preprocessing (log-mel spectrogram)
            if hasattr(whisper, "log_mel_spectrogram"):
                mel = whisper.log_mel_spectrogram(audio)
                return mel.unsqueeze(0).cpu().numpy()
            else:
                # Fallback to manual computation if not available
                return self._compute_mel_spectrogram(audio)
                
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            # Return empty spectrogram
            return np.zeros((1, 80, 3000), dtype=np.float32)
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute log-mel spectrogram manually.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Log-mel spectrogram
        """
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available for spectrogram computation")
                return np.zeros((1, 80, 3000), dtype=np.float32)
            
            # Whisper parameters
            sample_rate = 16000
            n_fft = 400
            n_mels = 80
            hop_length = 160
            
            # Convert to tensor
            waveform = torch.from_numpy(audio).unsqueeze(0)
            
            # Compute mel spectrogram
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=hop_length,
            )(waveform)
            
            # Convert to log scale
            log_mel_spec = torch.log(mel_spec + 1e-9)
            
            # Normalize
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
            
            return log_mel_spec.unsqueeze(0).cpu().numpy()
            
        except Exception as e:
            logger.error(f"Mel spectrogram computation error: {e}")
            return np.zeros((1, 80, 3000), dtype=np.float32)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the model.
        
        Returns:
            Status dictionary
        """
        # Determine the status based on initialization state
        current_status = ModuleState.READY if self.initialized else ModuleState.UNINITIALIZED
        
        try:
            status_details = {
                'status': current_status,
                'initialized': self.initialized,
                'model_type': self.model_type,
                'model_size': self.model_size,
                'language': self.language,
                'acceleration': {
                    'enabled': self.enable_acceleration,
                    'cuda_device': 0, # Default to device 0
                    'tensorrt': self.use_tensorrt
                }
            }
            
            # Add provider info if using ONNX
            if hasattr(self, 'encoder_session') and self.encoder_session is not None:
                status_details['providers'] = self.encoder_session.get_providers()
            
            # Add device info if using PyTorch
            if hasattr(self, 'model') and hasattr(self.model, 'device'):
                status_details['device'] = str(self.model.device)
            
            return status_details
        
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            # Return an error status if an exception occurs during status retrieval
            return {
                'status': ModuleState.ERROR,
                'initialized': self.initialized,
                'error': str(e)
            }


class SpeakerDiarizer:
    """
    Handles speaker diarization (identifying different speakers in audio).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the diarizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.diarization_config = config.get('diarization', {})
        
        # Extract diarization settings
        self.enabled = self.diarization_config.get('enabled', True)
        self.num_speakers = self.diarization_config.get('num_speakers', 0)
        self.min_speakers = self.diarization_config.get('min_speakers', 1)
        self.max_speakers = self.diarization_config.get('max_speakers', 10)
        self.clustering_method = self.diarization_config.get('clustering_method', 'spectral')
        self.embeddings_model = self.diarization_config.get('embeddings_model', 'speechbrain/spkrec-ecapa-voxceleb')
        
        # Internal state
        self.initialized = False
        self.pipeline = None
    
    def initialize(self) -> bool:
        """
        Initialize the diarizer.
        
        Returns:
            Success status
        """
        if not self.enabled:
            logger.info("Speaker diarization is disabled")
            return True
            
        if self.initialized:
            return True
            
        try:
            # Check if torch.compiler exists - pyannote requires it
            if not hasattr(torch, 'compiler'):
                logger.warning("torch.compiler not available - pyannote may not work correctly")
                # Add a placeholder attribute to prevent attribute errors
                # This is a temporary fix to allow the code to run
                import types
                setattr(torch, 'compiler', types.ModuleType('compiler'))
            
            # Patch torch.compiler.cudagraph module
            if not hasattr(torch.compiler, 'cudagraph'):
                # Create empty module to prevent attribute errors
                import types
                torch.compiler.cudagraph = types.ModuleType('cudagraph')
                
            # Patch cudagraph_impl if it doesn't exist
            if not hasattr(torch.compiler.cudagraph, 'cudagraph_impl'):
                torch.compiler.cudagraph.cudagraph_impl = types.ModuleType('cudagraph_impl') 
                
            # Add the cudagraphify function if missing
            if not hasattr(torch.compiler.cudagraph.cudagraph_impl, 'cudagraphify'):
                torch.compiler.cudagraph.cudagraph_impl.cudagraphify = lambda func, **kwargs: func
            
            # Continue with normal initialization
            try:
                # Lazy import so we don't require these dependencies if not using diarization
                from pyannote.audio import Pipeline
                
                # Initialize pipeline
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=os.environ.get("HF_TOKEN")
                )
                
                # Set the device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipeline = self.pipeline.to(device)
                
                self.initialized = True
                logger.info(f"Speaker diarization initialized on {device}")
                return True
            except ImportError:
                logger.warning("pyannote.audio not available, disabling speaker diarization")
                self.enabled = False
                return True
            except Exception as e:
                logger.error(f"Failed to initialize speaker diarization pipeline: {e}")
                self.enabled = False
                return True
            
        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization: {e}")
            self.enabled = False
            return True  # Return true but disable functionality
    
    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Perform speaker diarization.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with speaker segments
        """
        if not self.enabled or not self.initialized:
            # Return dummy diarization with single speaker
            return {'speakers': [0], 'segments': [{'speaker': 0, 'start': 0, 'end': len(audio)/sample_rate}]}
            
        try:
            # Convert audio to waveform
            waveform = {'waveform': torch.tensor(audio).unsqueeze(0), 'sample_rate': sample_rate}
            
            # Set parameters
            params = {}
            if self.num_speakers > 0:
                params['num_speakers'] = self.num_speakers
            else:
                params['min_speakers'] = self.min_speakers
                params['max_speakers'] = self.max_speakers
            
            # Run diarization
            diarization = self.pipeline(waveform, **params)
            
            # Process results
            speakers = []
            segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = int(speaker.split('_')[1])
                if speaker_id not in speakers:
                    speakers.append(speaker_id)
                
                segments.append({
                    'speaker': speaker_id,
                    'start': turn.start,
                    'end': turn.end
                })
            
            return {
                'speakers': speakers,
                'segments': segments
            }
            
        except Exception as e:
            logger.error(f"Diarization error: {e}")
            # Return dummy diarization with single speaker
            return {'speakers': [0], 'segments': [{'speaker': 0, 'start': 0, 'end': len(audio)/sample_rate}]}


class MedicalTermProcessor:
    """
    Processes medical terminology to improve transcription accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the medical term processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vocabulary_config = config.get('vocabulary', {})
        
        # Extract vocabulary settings
        self.enabled = self.vocabulary_config.get('enabled', True)
        self.vocabulary_path = self.vocabulary_config.get('path', 'config/vocabulary/custom_terms.txt')
        self.boost = self.vocabulary_config.get('boost', 10.0)
        
        # Load medical terms
        self.medical_terms = {}
        self.abbreviations = {}
        self.term_regexes = []
        
        if self.enabled:
            self._load_vocabulary()
    
    def _load_vocabulary(self) -> bool:
        """
        Load medical vocabulary from file.
        
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
                    self.medical_terms[original.lower()] = corrected
                    
                    # Create regex pattern for word boundary-aware replacement
                    pattern = r'\b' + re.escape(original.lower()) + r'\b'
                    self.term_regexes.append((re.compile(pattern, re.IGNORECASE), corrected))
                
                # Check if line has an abbreviation format (ABBR = Full Form)
                elif '=' in line:
                    abbr, full_form = [part.strip() for part in line.split('=')]
                    self.abbreviations[abbr.lower()] = full_form
                    
                    # Create regex pattern for abbreviation replacement
                    pattern = r'\b' + re.escape(abbr) + r'\b'
                    self.term_regexes.append((re.compile(pattern, re.IGNORECASE), full_form))
                
                # Otherwise, treat as a simple medical term to keep as-is
                else:
                    term = line.strip()
                    self.medical_terms[term.lower()] = term
            
            logger.info(f"Loaded {len(self.medical_terms)} medical terms and {len(self.abbreviations)} abbreviations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            return False
    
    def correct_text(self, text: str) -> str:
        """
        Correct medical terminology in text.
        
        Args:
            text: Input text
            
        Returns:
            Corrected text
        """
        if not self.enabled or not self.term_regexes:
            return text
            
        try:
            # Apply all regex replacements
            corrected = text
            for pattern, replacement in self.term_regexes:
                corrected = pattern.sub(replacement, corrected)
            
            return corrected
            
        except Exception as e:
            logger.error(f"Text correction error: {e}")
            return text
    
    def correct_segment(self, segment: TranscriptionSegment) -> TranscriptionSegment:
        """
        Correct medical terminology in a segment.
        
        Args:
            segment: TranscriptionSegment to correct
            
        Returns:
            Corrected segment
        """
        if not self.enabled:
            return segment
            
        try:
            # Correct the segment text
            segment.text = self.correct_text(segment.text)
            
            # Correct individual words
            for word in segment.words:
                word.text = self.correct_text(word.text)
            
            return segment
            
        except Exception as e:
            logger.error(f"Segment correction error: {e}")
            return segment
    
    def correct_result(self, result: TranscriptionResult) -> TranscriptionResult:
        """
        Correct medical terminology in transcription result.
        
        Args:
            result: TranscriptionResult to correct
            
        Returns:
            Corrected result
        """
        if not self.enabled:
            return result
            
        try:
            # Correct the full text
            result.text = self.correct_text(result.text)
            
            # Correct each segment
            for i, segment in enumerate(result.segments):
                result.segments[i] = self.correct_segment(segment)
            
            return result
            
        except Exception as e:
            logger.error(f"Result correction error: {e}")
            return result


class STTEngine:
    """
    Speech-to-Text Engine implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the STT engine."""
        logger.debug("STTEngine.__init__: Entering constructor")
        try:
            logger.debug("STTEngine.__init__: Initializing base attributes...")
            self.config = config or {} # Store the passed config or use an empty dict
            logger.debug(f"STTEngine.__init__: Config assigned: {self.config}") # Log assigned config
            self.initialized = False
            self.model_manager = None
            self.diarizer = None
            self.term_processor = None
            self.state = ModuleState.INITIALIZING # Set initial status
            self._tccc_system_ref = None # Reference to TCCCSystem for callbacks
            logger.debug("STTEngine.__init__: Base attributes initialized.")
            
            # --- ADDED: Queue and Worker Thread setup ---
            logger.debug("STTEngine.__init__: Initializing audio queue and worker thread components...")
            self.audio_queue = queue.Queue()
            self._worker_thread = None
            self._stop_event = threading.Event()
            logger.debug("STTEngine.__init__: Queue and worker components initialized.")
            # --- END ADDED ---

            # Streaming context
            logger.debug("STTEngine.__init__: Initializing streaming context...")
            self.context = ""
            self.context_max_length = 1000
            logger.debug("STTEngine.__init__: Streaming context initialized.")
            
            # Recent audio buffer for streaming
            logger.debug("STTEngine.__init__: Creating audio buffer deque...")
            self.audio_buffer = deque(maxlen=100)
            logger.debug("STTEngine.__init__: Audio buffer deque created.")
            
            # Recent transcriptions
            logger.debug("STTEngine.__init__: Creating recent segments deque...")
            self.recent_segments = deque(maxlen=10)
            logger.debug("STTEngine.__init__: Recent segments deque created.")
            
            # Performance metrics
            logger.debug("STTEngine.__init__: Initializing metrics dictionary...")
            self.metrics = {
                'total_audio_seconds': 0,
                'total_processing_time': 0,
                'transcript_count': 0,
                'error_count': 0,
                'avg_confidence': 0
            }
            logger.debug("STTEngine.__init__: Metrics dictionary initialized.")

            logger.debug("STTEngine.__init__: Constructor finished successfully.")
            # Note: Actual initialization (model loading etc.) happens in the async initialize method
            # We don't set status to IDLE here, as async init needs to run first.

        except Exception as e:
            logger.error(f"STTEngine.__init__: CRITICAL FAILURE DURING __init__: {e}\n{traceback.format_exc()}")
            self.state = ModuleState.ERROR # Ensure status reflects error
            # Optional: Re-raise if needed, but logging might be sufficient for __init__ failure
            # raise
        finally:
            logger.debug("STTEngine.__init__: Exiting constructor (finally block)")
            
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the STT engine with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Success status
        """
        try:
            # Use a copy of config to avoid modifying the original
            if config is None:
                logger.warning("No configuration provided, using default configuration")
                config = {
                    "model": {
                        "type": "whisper",
                        "size": "tiny.en",
                        "path": "models/stt"
                    },
                    "streaming": {
                        "enabled": True,
                        "max_context_length_sec": 60
                    },
                    "hardware": {
                        "enable_acceleration": False,
                        "cuda_device": -1
                    }
                }
                
            self.config = config
            
            # Create model directory if it doesn't exist
            model_path = config.get('model_path', 'models/stt') 
            os.makedirs(model_path, exist_ok=True)
            
            # Track component initialization status
            components_initialized = True
            
            # Initialize model manager with error handling
            try:
                logger.info("Initializing STT model manager")
                self.model_manager = ModelManager(config)
                model_init = self.model_manager.initialize()
                
                if not model_init:
                    logger.warning("Model manager initialization returned False")
                    components_initialized = False
                else:
                    logger.info("Model manager initialized successfully")
            except Exception as mm_error:
                logger.error(f"Failed to initialize model manager: {mm_error}")
                # Create a minimal mock model manager
                self.model_manager = self._create_minimal_model_manager(config or {})
                components_initialized = False
            
            # Initialize speaker diarizer with error handling
            try:
                logger.info("Initializing speaker diarizer")
                self.diarizer = SpeakerDiarizer(config)
                diarizer_init = self.diarizer.initialize()
                
                if not diarizer_init and config.get('diarization', {}).get('enabled', True):
                    logger.warning("Speaker diarization initialization failed")
                    components_initialized = False
                else:
                    logger.info("Speaker diarizer initialized successfully")
            except Exception as sd_error:
                logger.error(f"Failed to initialize speaker diarizer: {sd_error}")
                # Create minimal speaker diarizer
                self.diarizer = self._create_minimal_diarizer(config or {})
                components_initialized = False
            
            # Initialize medical term processor with error handling
            try:
                logger.info("Initializing medical term processor")
                self.term_processor = MedicalTermProcessor(config)
                logger.info("Medical term processor initialized successfully")
            except Exception as mtp_error:
                logger.error(f"Failed to initialize medical term processor: {mtp_error}")
                # Create minimal term processor
                self.term_processor = self._create_minimal_term_processor(config or {})
                components_initialized = False
            
            # Set streaming context parameters
            try:
                streaming_config = config.get('streaming', {})
                self.context_max_length = streaming_config.get('max_context_length_sec', 60) * 16000
                logger.info(f"Streaming context max length: {self.context_max_length / 16000:.1f} seconds")
            except Exception as sc_error:
                logger.warning(f"Error setting streaming parameters: {sc_error}")
                self.context_max_length = 60 * 16000  # Default to 60 seconds
            
            # Mark as initialized, even with limited functionality
            self.initialized = True
            
            if components_initialized:
                logger.info(f"STT Engine initialized with model: {self.model_manager.model_type}-{self.model_manager.model_size}")
            else:
                logger.warning("STT Engine initialized with limited functionality - some components may not work properly")
            
            # Setup cache directory for models if needed
            try:
                from tccc.stt_engine.model_cache_manager import get_model_cache_manager
                cache_manager = get_model_cache_manager()
                if cache_manager:
                    logger.info("Model cache manager is available")
            except ImportError:
                logger.warning("Model cache manager not available")
            
            # Subscribe to audio events from the Audio Pipeline
            try:
                self.subscribe_to_audio_events()
            except Exception as e:
                logger.warning(f"Failed to subscribe to audio events: {e}")
                
            # Start the background processing worker thread
            if self._worker_thread is None or not self._worker_thread.is_alive():
                logger.info("STTEngine.initialize: Starting STT processing worker thread...")
                self._stop_event.clear()
                self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True, name="STTWorkerThread")
                self._worker_thread.start()
                logger.info("STTEngine.initialize: STT processing worker thread started.")
            else:
                logger.warning("STTEngine.initialize: STT processing worker thread already running.")

            # Final status update
            self.state = ModuleState.IDLE
            return True

        except Exception as e:
            logger.error(f"Failed to initialize STT Engine: {e}")
            self.state = ModuleState.ERROR
            return False
    
    def set_system_reference(self, system_ref):
        """Store a reference to the TCCCSystem instance for scheduling callbacks."""
        logger.debug("STTEngine.set_system_reference: Setting TCCCSystem reference.")
        self._tccc_system_ref = system_ref

    def enqueue_audio(self, audio_data: np.ndarray, metadata: Dict[str, Any], event_context: Dict[str, Any]):
        """Add audio data and context to the processing queue."""
        if not self.initialized:
            logger.warning("STTEngine not initialized, cannot enqueue audio.")
            return
            
        try:
            # logger.debug(f"STTEngine.enqueue_audio: Queuing audio chunk (seq {event_context.get('sequence', 'N/A')}) for processing.")
            self.audio_queue.put((audio_data, metadata, event_context))
        except Exception as e:
            logger.exception(f"STTEngine.enqueue_audio: Failed to queue audio chunk: {e}")

    def _processing_loop(self):
        """Worker thread method to process audio chunks from the queue sequentially."""
        logger.info("STT Worker Thread: Starting processing loop.")
        
        while not self._stop_event.is_set():
            try:
                # Wait for an item from the queue
                audio_data, metadata, event_context = self.audio_queue.get(timeout=1)
                sequence = event_context.get('sequence', 'N/A')
                # logger.debug(f"STT Worker Thread: Dequeued audio chunk (seq {sequence}). Starting transcription...")
                
                # Perform synchronous transcription
                transcription_result = self.transcribe_segment(audio_data, metadata)
                # logger.debug(f"STT Worker Thread: Transcription finished for seq {sequence}.")
                
                # Check if transcription is valid and has text
                if transcription_result and not transcription_result.get('error') and transcription_result.get('text', '').strip():
                    logger.info(f"STT Worker Thread: Transcription successful (seq {sequence}). Text: '{transcription_result.get('text', '')[:50]}...' Scheduling event processing.")
                    
                    # Convert result to standard event format using the adapter
                    try:
                        transcription_event = STTEngineAdapter.convert_transcription_to_event(
                            transcription_result, event_context
                        )
                    except Exception as adapter_error:
                        logger.exception(f"STT Worker Thread: Error converting transcription using STTEngineAdapter (seq {sequence}): {adapter_error}")
                        continue # Skip processing this event further
                        
                    # Schedule the process_event coroutine on the main event loop
                    if self._tccc_system_ref and hasattr(self._tccc_system_ref, '_main_event_loop') and self._tccc_system_ref._main_event_loop:
                        if self._tccc_system_ref._main_event_loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                self._tccc_system_ref.process_event(transcription_event),
                                self._tccc_system_ref._main_event_loop
                            )
                            
                            # Optional: Add callback to future to log result/errors after execution
                            def log_event_result(f):
                                try:
                                    result = f.result()
                                    # logger.debug(f"STT Worker Thread: Main loop processed transcription event (seq {sequence}). Result: {result}")
                                except Exception as e_future:
                                    logger.error(f"STT Worker Thread: Error processing transcription event in main loop (seq {sequence}): {e_future}")
                            
                            future.add_done_callback(log_event_result)
                        else:
                            logger.error(f"STT Worker Thread: Main event loop is not running. Cannot schedule event processing for seq {sequence}.")
                    else:
                         logger.error(f"STT Worker Thread: TCCCSystem reference or main event loop not available. Cannot schedule event processing for seq {sequence}.")
                         
                elif transcription_result and transcription_result.get('error'):
                     logger.error(f"STT Worker Thread: Transcription failed with error (seq {sequence}): {transcription_result['error']}")
                     # Optionally emit an error event here?
                     
                else:
                    # No transcription text (silence or VAD filter?)
                    # logger.debug(f"STT Worker Thread: No transcription text returned for seq {sequence}.")
                    pass # Just continue to the next chunk
                    
            except queue.Empty:
                # Timeout waiting for queue item, just loop again
                continue
            except Exception as e:
                logger.exception(f"STT Worker Thread: Unexpected error in processing loop: {e}")
                # Optional: short sleep to prevent rapid looping on persistent errors
                time.sleep(0.1)
                
        logger.info("STT Worker Thread: Stop event received. Exiting processing loop.")

    def update_context(self, context: str) -> bool:
        """
        Update context for improved transcription accuracy.
        
        Args:
            context: Context string
            
        Returns:
            Success status
        """
        try:
            # Limit context size
            if len(context) > self.context_max_length:
                context = context[-self.context_max_length:]
            
            self.context = context
            return True
            
        except Exception as e:
            logger.error(f"Failed to update context: {e}")
            return False

    def transcribe_segment(self, audio: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transcribe audio segment.
        
        Args:
            audio: Audio data as numpy array
            metadata: Additional metadata for transcription
            
        Returns:
            Dictionary with transcription result
        """
        if not self.initialized:
            logger.error("STT Engine not initialized")
            return {'error': 'STT Engine not initialized', 'text': ''}
            
        try:
            # Track performance
            start_time = time.time()
            
            # --- VAD or other pre-processing can happen here ---
            # Example: Check if audio energy is above a threshold
            # if np.sqrt(np.mean(audio**2)) < self.config.get('vad_threshold', 0.01):
            #     logger.debug("Audio below VAD threshold, skipping transcription.")
            #     return {'text': '', 'is_silent': True} # Indicate silence
            
            # Update audio buffer for context (if using streaming logic)
            # if len(audio) > 0:
            #     self.audio_buffer.append(audio)
            
            # Parse metadata
            if metadata is None:
                metadata = {}
            
            # TODO: Adapt metadata usage if needed for new sequential flow
            # is_partial = metadata.get('is_partial', False) 
            # include_diarization = metadata.get('diarization', self.diarizer.enabled if self.diarizer else False)
            include_diarization = False # Diarization disabled for now

            # Process audio
            audio_duration = len(audio) / 16000 # Assuming 16kHz sample rate
            
            # Create transcription configuration (simplified for now)
            # transcription_config = self._create_transcription_config(metadata)
            transcription_config = None # Use ModelManager defaults or simplified config
            
            # Transcribe audio using the model manager
            # logger.debug(f"Transcribing segment of duration {audio_duration:.2f}s...")
            result = self.model_manager.transcribe(audio, config=transcription_config)
            # logger.debug(f"Transcription result received: Text='{result.text[:30]}...'")

            # Perform diarization if enabled (currently disabled)
            # if include_diarization and not is_partial:
            #     diarization = self.diarizer.diarize(audio)
            #     self._apply_diarization(result, diarization)
            
            # Apply medical terminology correction (if enabled)
            # result = self.term_processor.correct_result(result)
            
            # Update context (if using context logic)
            # if result.text and not is_partial:
            #     self._update_internal_context(result.text)
            
            # Store segment for context (if needed)
            # if not is_partial:
            #     self.recent_segments.append(result)
            
            # Track performance metrics
            processing_time = time.time() - start_time
            
            self.metrics['total_audio_seconds'] += audio_duration
            self.metrics['total_processing_time'] += processing_time
            self.metrics['transcript_count'] += 1
            
            # Calculate average confidence (handle potential lack of segments)
            if result.segments:
                avg_confidence = sum(getattr(segment, 'confidence', 0.0) for segment in result.segments) / len(result.segments)
                self.metrics['avg_confidence'] = (self.metrics['avg_confidence'] * (self.metrics['transcript_count'] - 1) + avg_confidence) / self.metrics['transcript_count']
            else:
                avg_confidence = 0.0 # Default if no segments

            # Convert result to dictionary (use adapter for consistency)
            # We need the original event context here if using the adapter fully.
            # For now, create a basic dict. Context will be added later in worker loop.
            result_dict = STTEngineAdapter.convert_raw_result_to_dict(result)

            # Add performance metrics to the dictionary
            result_dict['metrics'] = {
                'audio_duration': audio_duration,
                'processing_time': processing_time,
                'real_time_factor': processing_time / audio_duration if audio_duration > 0 else 0
            }
            
            # logger.debug(f"Transcription segment processed in {processing_time:.2f}s.")
            return result_dict
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True) # Log traceback
            self.metrics['error_count'] += 1
            return {'error': str(e), 'text': ''}

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the STT engine.
        
        Returns:
            Status dictionary (using ModuleState enum for 'status').
        """
        overall_status = self.state # Use the instance state attribute
            
        try:
            status_details = {
                'initialized': self.initialized,
                'status': overall_status.name, # Reflect the internal state
                'worker_thread_alive': self._worker_thread.is_alive() if self._worker_thread else False,
                'queue_size': self.audio_queue.qsize(),
                'metrics': {
                    'total_audio_seconds': self.metrics['total_audio_seconds'],
                    'total_processing_time': self.metrics['total_processing_time'],
                    'transcript_count': self.metrics['transcript_count'],
                    'error_count': self.metrics['error_count'],
                    'avg_confidence': self.metrics['avg_confidence'],
                    'real_time_factor': self.metrics['total_processing_time'] / self.metrics['total_audio_seconds'] if self.metrics['total_audio_seconds'] > 0 else 0
                }
            }
            
            # Add model status if available
            if self.model_manager:
                model_status = self.model_manager.get_status()
                status_details['model'] = model_status
                # Update overall status if model manager is in error
                if model_status.get('status') == ModuleState.ERROR and overall_status not in [ModuleState.STANDBY, ModuleState.SHUTDOWN]:
                    overall_status = ModuleState.ERROR
            else:
                status_details['model'] = {'status': ModuleState.UNINITIALIZED.name}
                if overall_status not in [ModuleState.STOPPING, ModuleState.STOPPED]:
                    overall_status = ModuleState.ERROR # Critical component missing
            
            # Add diarization status (simplified, as it's less critical/integrated now)
            diarizer_status = ModuleState.UNINITIALIZED
            diarizer_enabled = False
            if self.diarizer:
                diarizer_enabled = self.diarizer.enabled
                diarizer_status = self.diarizer.status
                if diarizer_status == ModuleState.ERROR and overall_status not in [ModuleState.STANDBY, ModuleState.SHUTDOWN]:
                    overall_status = ModuleState.WARNING # Diarizer error might not be fatal

            status_details['diarization'] = {
                'status': diarizer_status.name,
                'enabled': diarizer_enabled,
            }
            
            # Add vocabulary status (simplified)
            term_proc_enabled = False
            if self.term_processor:
                term_proc_enabled = self.term_processor.enabled
            status_details['vocabulary'] = {
                'enabled': term_proc_enabled
            }

            # Final overall status check
            final_status = {"status": overall_status.name}
            final_status.update(status_details)
            return final_status

        except Exception as e:
            logger.error(f"Error getting STTEngine status: {e}", exc_info=True)
            # Return an error status if an exception occurs during status retrieval
            return {
                "status": ModuleState.ERROR.name, # Report error status
                "initialized": self.initialized,
                "error": str(e)
            }

    def shutdown(self) -> bool:
        """
        Properly shut down the STT engine, releasing resources.
        
        Returns:
            Success status
        """
        logger.info("STTEngine.shutdown: Beginning shutdown...")
        logger.debug(f"STTEngine.shutdown: Checking state before shutdown logic. Current state: {self.state}, Type: {type(self.state)}")
        if self.state in [ModuleState.STANDBY, ModuleState.SHUTDOWN]: # Using correct ModuleState values
            logger.warning(f"STTEngine.shutdown called when already in state: {self.state}")
            return True
 
        self.state = ModuleState.STANDBY
        success = True

        try:
            # --- Stop the worker thread FIRST ---
            logger.info("STTEngine.shutdown: Signaling worker thread to stop...")
            self._stop_event.set()
            # Ensure queue is cleared or worker handles empty queue gracefully during shutdown
            # Optional: Put a sentinel value in the queue if worker blocks indefinitely on get()
            # try:
            #     self.audio_queue.put(None, block=False) # Signal worker loop to exit
            # except queue.Full:
            #     logger.warning("STTEngine.shutdown: Audio queue full while trying to add sentinel.")
                
            if self._worker_thread is not None and self._worker_thread.is_alive():
                logger.info("STTEngine.shutdown: Waiting for worker thread to join...")
                self._worker_thread.join(timeout=5.0) # Wait up to 5 seconds
                if self._worker_thread.is_alive():
                    logger.warning("STTEngine.shutdown: Worker thread did not exit cleanly after 5 seconds.")
                    success = False # Indicate unclean shutdown
                else:
                    logger.info("STTEngine.shutdown: Worker thread joined successfully.")
            else:
                logger.info("STTEngine.shutdown: Worker thread was not running or already stopped.")
            # --- Worker thread stopped --- #
            
            # Unsubscribe from events (if using event bus)
            # Example: self.event_bus.unsubscribe(self._handle_audio_event)
            # logger.info("STTEngine.shutdown: Unsubscribed from events (if applicable).")
            
            # Release model resources (ensure model_manager exists)
            if hasattr(self, 'model_manager') and self.model_manager:
                # Add specific shutdown/release logic for ModelManager if it exists
                if hasattr(self.model_manager, 'shutdown'):
                     self.model_manager.shutdown()
                else:
                     # Basic cleanup if no dedicated shutdown
                     if hasattr(self.model_manager, 'model'): self.model_manager.model = None
                     if hasattr(self.model_manager, 'encoder_session'): self.model_manager.encoder_session = None
                     if hasattr(self.model_manager, 'decoder_session'): self.model_manager.decoder_session = None
                logger.info("STTEngine.shutdown: Released model resources.")
            
            # Release diarizer resources (ensure diarizer exists)
            if hasattr(self, 'diarizer') and self.diarizer:
                if hasattr(self.diarizer, 'shutdown'):
                    self.diarizer.shutdown()
                elif hasattr(self.diarizer, 'pipeline'):
                     self.diarizer.pipeline = None
                logger.info("STTEngine.shutdown: Released diarizer resources.")
            
            # Clear buffers (if used)
            # self.audio_buffer.clear()
            # self.recent_segments.clear()
            # logger.info("STTEngine.shutdown: Cleared internal buffers.")
            
            # Reset initialization flag and status
            self.initialized = False
            self.state = ModuleState.SHUTDOWN
            logger.info("STTEngine shutdown complete.")
            
        except Exception as e:
            logger.exception(f"STTEngine.shutdown: Error during shutdown: {e}")
            self.state = ModuleState.ERROR # Mark as error state after failed shutdown
            success = False
            
        return success

class STTEngineAdapter:
    """Adapts data between TCCCSystem/Events and STTEngine."""

    @staticmethod
    def convert_audio_event_to_input(event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts an AUDIO_SEGMENT event payload into input for STTEngine.transcribe_segment."""
        # logger.debug(f"Adapter: Converting audio event keys: {event_data.keys()}")
        try:
            # Extract the core data dictionary
            core_data = event_data.get('data', {})
            # Extract audio numpy array
            audio_data = core_data.get('audio_data') 
            if audio_data is None:
                 raise ValueError("Audio event data missing 'audio_data'")
                 
            # Extract metadata dictionary
            metadata = core_data.get('metadata', {})
            
            # Add context from event if needed (session_id, sequence)
            metadata['session_id'] = event_data.get('session_id')
            metadata['sequence'] = event_data.get('sequence')

            # logger.debug(f"Adapter: Extracted audio shape: {audio_data.shape}, metadata keys: {metadata.keys()}")
            return {
                "audio": audio_data,
                "metadata": metadata
            }
        except Exception as e:
            logger.exception(f"STTEngineAdapter: Error converting audio event: {e}")
            raise # Re-raise to be caught by the caller

    @staticmethod
    def convert_transcription_to_event(transcription_result: Dict[str, Any], original_event_context: Dict[str, Any]) -> Dict[str, Any]:
        """Converts the result from STTEngine.transcribe_segment into a TRANSCRIPTION event payload."""
        # logger.debug(f"Adapter: Converting transcription result keys: {transcription_result.keys()}")
        try:
            # Extract essential info
            text = transcription_result.get('text', '')
            segments = transcription_result.get('segments', [])
            language = transcription_result.get('language', 'en')
            is_partial = transcription_result.get('is_partial', False) # Whisper doesn't usually do partials this way
            metrics = transcription_result.get('metrics', {})
            
            # Calculate overall confidence (example: average of segments)
            confidence = 0.0
            if segments:
                confidences = [seg.get('confidence', 0.0) for seg in segments]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
                    
            # Prepare metadata for the event
            event_metadata = {
                'processing_time': metrics.get('processing_time'),
                'real_time_factor': metrics.get('real_time_factor'),
                'audio_duration': metrics.get('audio_duration')
                # Add model info if available and needed
                # 'model': self.model_manager.model_size if self.model_manager else "unknown"
            }

            # Create the event payload dictionary
            event_payload = {
                "source": "stt_engine", # Indicate the source is the STT engine itself
                "type": EventType.TRANSCRIPTION.value,
                "timestamp": time.time(), # Timestamp of event creation
                "session_id": original_event_context.get('session_id'),
                "sequence": original_event_context.get('sequence'), # Carry over sequence number
                "data": {
                    "text": text,
                    "segments": segments, # Include detailed segments
                    "language": language,
                    "confidence": confidence,
                    "is_partial": is_partial,
                    "metadata": event_metadata # Attach processing metrics
                }
            }
            # logger.debug(f"Adapter: Created TRANSCRIPTION event payload keys: {event_payload.keys()}")
            return event_payload
            
        except Exception as e:
            logger.exception(f"STTEngineAdapter: Error converting transcription result: {e}")
            # Return a minimal error structure or raise?
            # For now, raise to indicate failure in conversion
            raise

    @staticmethod
    def convert_raw_result_to_dict(result: TranscriptionResult) -> Dict[str, Any]:
        """Converts the raw TranscriptionResult object to a dictionary format.
           Needed because the object itself might not be directly serializable or suitable.
        """
        segments_list = []
        if result.segments:
            for segment in result.segments:
                seg_dict = {
                    'text': getattr(segment, 'text', ''),
                    'start_time': getattr(segment, 'start_time', 0.0),
                    'end_time': getattr(segment, 'end_time', 0.0),
                    'confidence': getattr(segment, 'confidence', 0.0),
                    # Add speaker info if available
                    'speaker': getattr(segment, 'speaker', None)
                }
                # Add word timings if available
                if hasattr(segment, 'words') and segment.words:
                    seg_dict['words'] = [
                        {
                            'text': getattr(word, 'text', ''),
                            'start_time': getattr(word, 'start_time', 0.0),
                            'end_time': getattr(word, 'end_time', 0.0),
                            'confidence': getattr(word, 'confidence', 0.0),
                            'speaker': getattr(word, 'speaker', None)
                        }
                        for word in segment.words
                    ]
                segments_list.append(seg_dict)

        return {
            'text': getattr(result, 'text', ''),
            'segments': segments_list,
            'is_partial': getattr(result, 'is_partial', False),
            'language': getattr(result, 'language', 'en')
        }