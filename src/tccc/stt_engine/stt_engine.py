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

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
        self.model_config = config.get('model', {})
        self.hardware_config = config.get('hardware', {})
        
        # Extract model settings
        self.model_type = self.model_config.get('type', 'whisper')
        self.model_size = self.model_config.get('size', 'medium')
        self.model_path = self.model_config.get('path', f'models/whisper-{self.model_size}-en/')
        
        # --- Workaround for tiny.en model name issue ---
        if self.model_size == 'tiny.en':
            logger.warning("Workaround: Changing requested model size from 'tiny.en' to 'tiny' due to Whisper library inconsistency.")
            self.model_size = 'tiny'
            # Adjust default path if it used the original size
            if self.model_path == f'models/whisper-tiny.en-en/':
                 self.model_path = f'models/whisper-{self.model_size}-en/'
        # --- End Workaround ---
        
        self.batch_size = self.model_config.get('batch_size', 1)
        self.mixed_precision = self.model_config.get('mixed_precision', True)
        self.language = self.model_config.get('language', 'en')
        self.beam_size = self.model_config.get('beam_size', 5)
        
        # Extract hardware acceleration settings
        self.enable_acceleration = self.hardware_config.get('enable_acceleration', True)
        self.cuda_device = self.hardware_config.get('cuda_device', 0)
        self.use_tensorrt = self.hardware_config.get('use_tensorrt', True)
        self.cuda_streams = self.hardware_config.get('cuda_streams', 2)
        self.memory_limit_mb = self.hardware_config.get('memory_limit_mb', 4096)
        self.quantization = self.hardware_config.get('quantization', 'FP16')
        
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
                # First try PyTorch directly as it's more reliable
                if self.model_type == 'whisper' and TORCH_AVAILABLE:
                    logger.info("Initializing with PyTorch Whisper (more reliable)")
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
            logger.debug(traceback.format_exc()) # Add traceback for model init failure
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
            if self.enable_acceleration and self.cuda_device >= 0:
                # Try to use TensorRT if available and requested
                if self.use_tensorrt and 'TensorrtExecutionProvider' in available_providers:
                    # TensorRT execution provider options
                    trt_options = {
                        'device_id': self.cuda_device,
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
                        'device_id': self.cuda_device,
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
            logger.debug(traceback.format_exc()) # Add traceback for ONNX init failure
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
            logger.exception("Traceback for Whisper PyTorch initialization failure:") # Log exception with traceback
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
            # Use the encoder-decoder model for ONNX inference
            if hasattr(self, 'encoder_session') and hasattr(self, 'decoder_session'):
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
            
            # Use Whisper's built-in transcription with explicit timeout
            try:
                whisper_result = self.model.transcribe(
                    audio,
                    language=self.language,
                    task="transcribe",
                    beam_size=self.beam_size,
                    temperature=0.0,
                    word_timestamps=config.word_timestamps,
                    fp16=False  # Force FP32 to avoid GPU-specific issues
                )
            except RuntimeError as e:
                # If there's a specific torch error, try again with a simpler approach
                logger.warning(f"Initial PyTorch transcription failed: {e}, trying simpler approach")
                # Use a simpler approach without beam search or timestamps
                whisper_result = {
                    "text": self.model.transcribe(
                        audio,
                        language=self.language,
                        temperature=0.0,
                        fp16=False,
                        beam_size=1,  # Simple beam size
                        word_timestamps=False  # No timestamps
                    )["text"],
                    "segments": [
                        {
                            "text": self.model.transcribe(audio, language=self.language, fp16=False)["text"],
                            "start": 0.0,
                            "end": len(audio) / 16000.0,
                            "confidence": 0.8
                        }
                    ],
                    "language": self.language
                }
            
            # Extract text and segments
            text = whisper_result["text"]
            
            # Create TranscriptionResult
            result = TranscriptionResult(
                text=text,
                is_partial=False,
                language=whisper_result.get("language", self.language)
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
                
                # Add words if available
                if "words" in segment and config.word_timestamps:
                    for word_info in segment["words"]:
                        word = Word(
                            text=word_info["word"],
                            start_time=word_info["start"],
                            end_time=word_info["end"],
                            confidence=word_info.get("confidence", 0.8)
                        )
                        ts_segment.words.append(word)
                
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
        overall_status = ModuleState.UNINITIALIZED
        if self.initialized:
             # Assume READY, check components below
            overall_status = ModuleState.READY
            
        try:
            status_details = {
                'initialized': self.initialized,
                'model_type': self.model_type,
                'model_size': self.model_size,
                'language': self.language,
                'acceleration': {
                    'enabled': self.enable_acceleration,
                    'cuda_device': self.cuda_device,
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
            return {
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
    
    def __init__(self):
        """Initialize the STT engine."""
        self.initialized = False
        self.config = None
        self.model_manager = None
        self.diarizer = None
        self.term_processor = None
        
        # Streaming context
        self.context = ""
        self.context_max_length = 1000
        
        # Recent audio buffer for streaming
        self.audio_buffer = deque(maxlen=100)
        
        # Recent transcriptions
        self.recent_segments = deque(maxlen=10)
        
        # Performance metrics
        self.metrics = {
            'total_audio_seconds': 0,
            'total_processing_time': 0,
            'transcript_count': 0,
            'error_count': 0,
            'avg_confidence': 0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
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
            model_path = config.get('model', {}).get('path', 'models/stt')
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
                
            # Return the actual initialization status
            return components_initialized
        
        except Exception as e:
            logger.error(f"Failed to initialize STT Engine: {e}")
            
            # Set up minimal functioning state
            try:
                logger.warning("Setting up minimal STT Engine after initialization error")
                # Mock components
                self.model_manager = self._create_minimal_model_manager(config or {})
                self.diarizer = self._create_minimal_diarizer(config or {})
                self.term_processor = self._create_minimal_term_processor(config or {})
                
                # Default context length
                self.context_max_length = 60 * 16000
                
                # Mark as initialized with limited functionality
                self.initialized = True
                logger.warning("STT Engine initialized with minimal functionality after error")
                return True
            except:
                # Complete failure
                self.initialized = False
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
            
            # Update audio buffer for context
            if len(audio) > 0:
                self.audio_buffer.append(audio)
            
            # Parse metadata
            if metadata is None:
                metadata = {}
            
            is_partial = metadata.get('is_partial', False)
            include_diarization = metadata.get('diarization', self.diarizer.enabled)
            
            # Process audio
            audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
            
            # Create transcription configuration
            transcription_config = self._create_transcription_config(metadata)
            
            # Transcribe audio
            result = self.model_manager.transcribe(audio, transcription_config)
            
            # Perform diarization if enabled
            if include_diarization and not is_partial:
                diarization = self.diarizer.diarize(audio)
                self._apply_diarization(result, diarization)
            
            # Apply medical terminology correction
            result = self.term_processor.correct_result(result)
            
            # Update context
            if result.text and not is_partial:
                self._update_internal_context(result.text)
            
            # Store segment for context
            if not is_partial:
                self.recent_segments.append(result)
            
            # Track performance metrics
            processing_time = time.time() - start_time
            
            self.metrics['total_audio_seconds'] += audio_duration
            self.metrics['total_processing_time'] += processing_time
            self.metrics['transcript_count'] += 1
            
            # Calculate average confidence
            avg_confidence = sum(segment.confidence for segment in result.segments) / len(result.segments) if result.segments else 0
            self.metrics['avg_confidence'] = (self.metrics['avg_confidence'] * (self.metrics['transcript_count'] - 1) + avg_confidence) / self.metrics['transcript_count']
            
            # Convert result to dictionary
            result_dict = self._result_to_dict(result)
            
            # Add performance metrics
            result_dict['metrics'] = {
                'audio_duration': audio_duration,
                'processing_time': processing_time,
                'real_time_factor': processing_time / audio_duration if audio_duration > 0 else 0
            }
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.metrics['error_count'] += 1
            return {'error': str(e), 'text': ''}
    
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
    
    def _update_internal_context(self, text: str) -> None:
        """
        Update internal context with new text.
        
        Args:
            text: New text to add to context
        """
        current_context = self.context + " " + text
        if len(current_context) > self.context_max_length:
            current_context = current_context[-self.context_max_length:]
        
        self.context = current_context.strip()
    
    def _create_transcription_config(self, metadata: Dict[str, Any]) -> TranscriptionConfig:
        """
        Create transcription configuration from metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            TranscriptionConfig object
        """
        # Get transcription settings from config
        transcription_config = self.config.get('transcription', {})
        
        # Create config object
        config = TranscriptionConfig(
            confidence_threshold=transcription_config.get('confidence_threshold', 0.6),
            word_timestamps=transcription_config.get('word_timestamps', True),
            include_punctuation=transcription_config.get('include_punctuation', True),
            include_capitalization=transcription_config.get('include_capitalization', True),
            format_numbers=transcription_config.get('format_numbers', True),
            segment_length=transcription_config.get('segment_length', 30)
        )
        
        # Override with metadata if provided
        if 'confidence_threshold' in metadata:
            config.confidence_threshold = metadata['confidence_threshold']
        if 'word_timestamps' in metadata:
            config.word_timestamps = metadata['word_timestamps']
        if 'include_punctuation' in metadata:
            config.include_punctuation = metadata['include_punctuation']
        if 'include_capitalization' in metadata:
            config.include_capitalization = metadata['include_capitalization']
        if 'format_numbers' in metadata:
            config.format_numbers = metadata['format_numbers']
        
        return config
    
    def _apply_diarization(self, result: TranscriptionResult, diarization: Dict[str, Any]) -> None:
        """
        Apply diarization results to transcription.
        
        Args:
            result: TranscriptionResult to update
            diarization: Diarization results
        """
        # Process each segment
        for segment in result.segments:
            segment_center = (segment.start_time + segment.end_time) / 2
            
            # Find matching diarization segment
            for dia_segment in diarization['segments']:
                if dia_segment['start'] <= segment_center <= dia_segment['end']:
                    segment.speaker = dia_segment['speaker']
                    break
            
            # Apply speaker to words
            if segment.speaker is not None:
                for word in segment.words:
                    word.speaker = segment.speaker
    
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
            
            if segment.speaker is not None:
                seg_dict['speaker'] = segment.speaker
            
            if segment.words:
                seg_dict['words'] = [
                    {
                        'text': word.text,
                        'start_time': word.start_time,
                        'end_time': word.end_time,
                        'confidence': word.confidence,
                        'speaker': word.speaker
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
    
    def subscribe_to_audio_events(self) -> bool:
        """
        Subscribe to audio events from the Audio Pipeline.
        
        Returns:
            Success status
        """
        try:
            # Import required modules
            from tccc.utils.event_bus import get_event_bus
            from tccc.utils.event_schema import EventType
            
            # Get event bus
            event_bus = get_event_bus()
            
            # Subscribe to audio segment events
            success = event_bus.subscribe(
                subscriber="stt_engine",
                event_types=[EventType.AUDIO_SEGMENT],
                callback=self._handle_audio_event
            )
            
            if success:
                logger.info("STT Engine subscribed to audio events")
            else:
                logger.warning("Failed to subscribe to audio events")
                
            return success
            
        except ImportError:
            logger.warning("Event bus or schema not available, cannot subscribe to audio events")
            return False
        except Exception as e:
            logger.error(f"Error subscribing to audio events: {e}")
            return False
            
    def _handle_audio_event(self, event):
        """
        Handle incoming audio segment events.
        
        Args:
            event: AudioSegmentEvent to process
        """
        try:
            # Check if event has audio data and is speech
            if not hasattr(event, 'data') or not event.data.get('is_speech', False):
                return
                
            # Extract audio data
            audio_data = event.audio_data if hasattr(event, 'audio_data') else None
            if audio_data is None:
                logger.warning("Received audio event without audio data")
                return
                
            # Process the audio
            result = self.transcribe_segment(audio_data)
            
            # If we got a valid result, emit a transcription event
            if result and 'text' in result and result['text']:
                self._emit_transcription_event(result, event.session_id, event.sequence)
                
        except Exception as e:
            logger.error(f"Error handling audio event: {e}")
            
            # Emit error event
            try:
                self._emit_error_event(
                    "audio_processing_error",
                    f"Error processing audio event: {e}",
                    "stt_engine",
                    True  # Recoverable
                )
            except Exception:
                # Just log if event emission fails
                pass
    
    def _emit_transcription_event(self, transcription: Dict[str, Any], session_id: str = None, sequence: int = None):
        """
        Emit a TranscriptionEvent.
        
        Args:
            transcription: Transcription result dictionary
            session_id: Session identifier from audio event
            sequence: Sequence number from audio event
        """
        try:
            # Import event schema items only when needed
            from tccc.utils.event_schema import TranscriptionEvent
            
            # Get event bus
            event_bus = self._get_event_bus()
            if not event_bus:
                return
                
            # Create event
            event = TranscriptionEvent(
                source="stt_engine",
                text=transcription['text'],
                segments=transcription['segments'],
                language=transcription.get('language', 'en'),
                confidence=transcription['segments'][0]['confidence'] if transcription['segments'] else 0.0,
                is_partial=transcription.get('is_partial', False),
                metadata={
                    'processing_time': transcription.get('metrics', {}).get('processing_time', 0),
                    'model': self.model_manager.model_size if self.model_manager else "unknown"
                },
                session_id=session_id,
                sequence=sequence
            )
            
            # Publish event
            event_bus.publish(event)
            logger.debug(f"Emitted transcription event: {transcription['text']}")
            
        except ImportError:
            logger.warning("Event schema not available, cannot emit transcription event")
        except Exception as e:
            logger.error(f"Error emitting transcription event: {e}")
    
    def _emit_error_event(
        self, 
        error_code: str, 
        message: str, 
        component: str,
        recoverable: bool = False
    ):
        """
        Emit an ErrorEvent.
        
        Args:
            error_code: Error code identifier
            message: Error message
            component: Component that experienced the error
            recoverable: Whether the error is recoverable
        """
        try:
            # Import event schema items only when needed
            from tccc.utils.event_schema import ErrorEvent, ErrorSeverity
            
            # Get event bus
            event_bus = self._get_event_bus()
            if not event_bus:
                return
            
            # Create event
            event = ErrorEvent(
                source="stt_engine",
                error_code=error_code,
                message=message,
                severity=ErrorSeverity.ERROR,
                component=component,
                recoverable=recoverable
            )
            
            # Publish event
            event_bus.publish(event)
            
        except ImportError:
            logger.warning("Event schema not available, cannot emit error event")
        except Exception as e:
            logger.error(f"Error emitting error event: {e}")
    
    def _get_event_bus(self):
        """Get the event bus instance, if available."""
        try:
            from tccc.utils.event_bus import get_event_bus
            return get_event_bus()
        except ImportError:
            logger.warning("Event bus not available")
            return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the STT engine.
        
        Returns:
            Status dictionary (using ModuleState enum for 'status').
        """
        overall_status = ModuleState.UNINITIALIZED
        if self.initialized:
             # Assume READY, check components below
            overall_status = ModuleState.READY
            
        try:
            status_details = {
                'initialized': self.initialized,
                'metrics': {
                    'total_audio_seconds': self.metrics['total_audio_seconds'],
                    'total_processing_time': self.metrics['total_processing_time'],
                    'transcript_count': self.metrics['transcript_count'],
                    'error_count': self.metrics['error_count'],
                    'avg_confidence': self.metrics['avg_confidence'],
                    'real_time_factor': self.metrics['total_processing_time'] / self.metrics['total_audio_seconds'] if self.metrics['total_audio_seconds'] > 0 else 0
                }
            }
            
            # Add model status
            model_manager_state = ModuleState.ERROR # Default if not available
            if self.model_manager:
                model_status = self.model_manager.get_status()
                status_details['model'] = model_status
                model_manager_state = model_status.get('status', ModuleState.ERROR)
            
            # Add diarization status
            diarizer_state = ModuleState.UNINITIALIZED # Default if not configured
            diarizer_enabled = False
            diarizer_initialized = False
            if self.diarizer:
                diarizer_enabled = self.diarizer.enabled
                diarizer_initialized = self.diarizer.initialized
                if not diarizer_enabled:
                    diarizer_state = ModuleState.READY # Disabled is considered ready/ok for the engine
                elif diarizer_initialized:
                    diarizer_state = ModuleState.READY
                else:
                    # Enabled but not initialized. Check if init failed.
                    # The initialize method sets enabled=False on failure, but let's be safe.
                    if hasattr(self.diarizer, 'pipeline') and self.diarizer.pipeline is None and self.diarizer.enabled:
                         diarizer_state = ModuleState.ERROR # Initialization likely failed
                    else:
                         diarizer_state = ModuleState.UNINITIALIZED # Or still initializing

                status_details['diarization'] = {
                    'status': diarizer_state.name, # Use the derived state
                    'enabled': diarizer_enabled,
                    'initialized': diarizer_initialized
                }
            else:
                 # No diarizer configured, treat as READY/Not Applicable
                 diarizer_state = ModuleState.READY
                 status_details['diarization'] = {
                     'status': diarizer_state.name,
                     'enabled': False,
                     'initialized': False
                 }
            
            # Add vocabulary status
            # Term processor might not have a complex state, check initialization
            term_processor_ready = False
            if self.term_processor:
                 term_processor_ready = self.term_processor.enabled # Simple check for now
                 status_details['vocabulary'] = {
                     'enabled': self.term_processor.enabled,
                     'medical_terms': len(self.term_processor.medical_terms),
                     'abbreviations': len(self.term_processor.abbreviations)
                 }

            # Determine overall status based on components
            if model_manager_state == ModuleState.ERROR:
                overall_status = ModuleState.ERROR
            elif diarizer_state == ModuleState.ERROR: # Check our derived state
                 overall_status = ModuleState.ERROR
            # Add other critical component checks here
            
            # If initialized but critical components aren't READY, reflect that
            elif self.initialized and model_manager_state != ModuleState.READY:
                overall_status = model_manager_state # Reflect model manager state (e.g., INITIALIZING)
            elif self.initialized and diarizer_state not in [ModuleState.READY, ModuleState.UNINITIALIZED]: # If enabled, needs to be READY
                 # If diarizer is enabled and not READY (e.g., ERROR, INITIALIZING), reflect its state
                 if diarizer_enabled:
                    overall_status = diarizer_state

            status = {"status": overall_status}
            status.update(status_details)
            return status

        except Exception as e:
            logger.error(f"Error getting STTEngine status: {e}", exc_info=True)
            return {
                "status": ModuleState.ERROR,
                "initialized": self.initialized,
                "error": str(e)
            }
    
    def shutdown(self) -> bool:
        """
        Properly shut down the STT engine, releasing resources.
        
        Returns:
            Success status
        """
        try:
            logger.info("Shutting down STT Engine")
            
            # Unsubscribe from event bus
            try:
                from tccc.utils.event_bus import get_event_bus
                event_bus = get_event_bus()
                if event_bus:
                    event_bus.unsubscribe("stt_engine")
                    logger.info("Unsubscribed from event bus")
            except Exception as e:
                logger.warning(f"Error unsubscribing from event bus: {e}")
            
            # Release model resources
            if hasattr(self, 'model_manager') and self.model_manager:
                if hasattr(self.model_manager, 'model') and self.model_manager.model:
                    self.model_manager.model = None
                    
                # Clear ONNX sessions
                if hasattr(self.model_manager, 'encoder_session'):
                    self.model_manager.encoder_session = None
                if hasattr(self.model_manager, 'decoder_session'):
                    self.model_manager.decoder_session = None
                    
                logger.info("Released model resources")
            
            # Release diarizer resources
            if hasattr(self, 'diarizer') and self.diarizer and hasattr(self.diarizer, 'pipeline'):
                self.diarizer.pipeline = None
                logger.info("Released diarizer resources")
            
            # Clear buffers
            self.audio_buffer.clear()
            self.recent_segments.clear()
            
            # Reset initialization flag for clean restart
            self.initialized = False
            
            logger.info("STT Engine shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during STT Engine shutdown: {e}")
            return False

    def _create_minimal_model_manager(self, config: Dict[str, Any]) -> Any:
        """Create a minimal model manager with basic functionality.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Minimal model manager object
        """
        class MinimalModelManager:
            def __init__(self, config):
                self.config = config
                self.model_type = "mock-whisper"
                self.model_size = "tiny"
                self.initialized = True
                self.is_warmed_up = True
            
            def initialize(self):
                return True
                
            def transcribe(self, audio, config=None):
                # Return a mock transcription result
                return TranscriptionResult(
                    text="[This is placeholder text from minimal STT Engine]",
                    segments=[
                        TranscriptionSegment(
                            text="[This is placeholder text from minimal STT Engine]",
                            start_time=0.0,
                            end_time=len(audio) / 16000.0 if isinstance(audio, (list, np.ndarray)) else 1.0,
                            confidence=0.8
                        )
                    ]
                )
                
            def get_status(self):
                return {
                    'initialized': True,
                    'warmed_up': True,
                    'model_type': 'mock-whisper',
                    'model_size': 'tiny',
                    'language': 'en',
                    'acceleration': {
                        'enabled': False,
                        'cuda_device': -1,
                        'tensorrt': False
                    }
                }
        
        return MinimalModelManager(config)
    
    def _create_minimal_diarizer(self, config: Dict[str, Any]) -> Any:
        """Create a minimal speaker diarizer with basic functionality.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Minimal speaker diarizer object
        """
        class MinimalSpeakerDiarizer:
            def __init__(self, config):
                self.config = config
                self.enabled = False
                self.initialized = True
            
            def initialize(self):
                return True
                
            def diarize(self, audio, sample_rate=16000):
                # Return a simple diarization result with single speaker
                return {
                    'speakers': [0],
                    'segments': [
                        {
                            'speaker': 0,
                            'start': 0,
                            'end': len(audio) / sample_rate if isinstance(audio, (list, np.ndarray)) else 1.0
                        }
                    ]
                }
        
        return MinimalSpeakerDiarizer(config)
    
    def _create_minimal_term_processor(self, config: Dict[str, Any]) -> Any:
        """Create a minimal medical term processor with basic functionality.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Minimal medical term processor object
        """
        class MinimalTermProcessor:
            def __init__(self, config):
                self.config = config
                self.enabled = False
                self.medical_terms = {}
                self.abbreviations = {}
                self.term_regexes = []
            
            def correct_text(self, text):
                return text  # No-op, just pass through
                
            def correct_segment(self, segment):
                return segment  # No-op, just pass through
                
            def correct_result(self, result):
                return result  # No-op, just pass through
        
        return MinimalTermProcessor(config)