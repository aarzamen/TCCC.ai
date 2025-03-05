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
                # Check if we're using ONNX Runtime
                if self.model_type == 'whisper' and ONNX_AVAILABLE and self.enable_acceleration:
                    result = self._initialize_whisper_onnx()
                # Fallback to PyTorch if ONNX is not available or not enabled
                elif self.model_type == 'whisper' and TORCH_AVAILABLE:
                    result = self._initialize_whisper_torch()
                else:
                    logger.error(f"Model type '{self.model_type}' not supported or required dependencies not available")
                    return False
                
                if result:
                    self.initialized = True
                    # Perform model warmup
                    self._warmup_model()
                    logger.info(f"Model '{self.model_type}-{self.model_size}' initialized successfully")
                return result
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
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
                # Convert MB to bytes
                session_options.set_memory_limit(self.memory_limit_mb * 1024 * 1024)
            
            # Set up providers
            providers = []
            provider_options = []
            
            if self.enable_acceleration and self.cuda_device >= 0:
                # CUDA execution provider options
                cuda_options = {
                    'device_id': self.cuda_device,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': self.memory_limit_mb * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': '1',
                }
                
                if self.use_tensorrt:
                    # TensorRT execution provider options
                    trt_options = {
                        'device_id': self.cuda_device,
                        'trt_max_workspace_size': self.memory_limit_mb * 1024 * 1024,
                        'trt_fp16_enable': '1' if self.mixed_precision else '0',
                    }
                    providers.append('TensorrtExecutionProvider')
                    provider_options.append(trt_options)
                
                providers.append('CUDAExecutionProvider')
                provider_options.append(cuda_options)
            
            # Always add CPU provider as fallback
            providers.append('CPUExecutionProvider')
            provider_options.append({})
            
            # Get model paths for encoder and decoder
            encoder_path = os.path.join(self.model_path, 'encoder.onnx')
            decoder_path = os.path.join(self.model_path, 'decoder.onnx')
            
            # Check if models exist, if not we need to convert them
            if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                logger.info(f"ONNX models not found at {self.model_path}, attempting conversion")
                self._convert_whisper_to_onnx()
            
            # Create ONNX Runtime sessions
            self.encoder_session = ort.InferenceSession(
                encoder_path, 
                sess_options=session_options, 
                providers=providers, 
                provider_options=provider_options
            )
            
            self.decoder_session = ort.InferenceSession(
                decoder_path, 
                sess_options=session_options, 
                providers=providers, 
                provider_options=provider_options
            )
            
            # Load tokenizer and other necessary components
            self._load_whisper_processor()
            
            logger.info(f"Whisper ONNX model loaded with providers: {providers}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper ONNX model: {e}")
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
            device = "cuda" if torch.cuda.is_available() and self.enable_acceleration else "cpu"
            self.model = whisper.load_model(self.model_size, device=device)
            
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{self.model_size}")
            
            logger.info(f"Whisper PyTorch model loaded on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper PyTorch model: {e}")
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
            
            # Export encoder
            torch.onnx.export(
                model.encoder,
                dummy_input,
                encoder_path,
                export_params=True,
                opset_version=14,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size', 2: 'sequence_length'},
                              'output': {0: 'batch_size', 1: 'sequence_length'}}
            )
            
            logger.info("Converting Whisper decoder to ONNX")
            # Define dummy inputs for decoder
            dummy_audio_features = torch.zeros((1, 1500, 512), dtype=torch.float32)
            dummy_tokens = torch.zeros((1, 1), dtype=torch.int64)
            
            # Export decoder
            torch.onnx.export(
                model.decoder,
                (dummy_tokens, dummy_audio_features),
                decoder_path,
                export_params=True,
                opset_version=14,
                input_names=['tokens', 'audio_features'],
                output_names=['output'],
                dynamic_axes={'tokens': {0: 'batch_size', 1: 'sequence_length'},
                             'audio_features': {0: 'batch_size', 1: 'sequence_length'},
                             'output': {0: 'batch_size', 1: 'sequence_length'}}
            )
            
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
            
            # Use Whisper's built-in transcription
            whisper_result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                beam_size=self.beam_size,
                temperature=0.0,
                word_timestamps=config.word_timestamps
            )
            
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
        status = {
            'initialized': self.initialized,
            'warmed_up': self.is_warmed_up,
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
            status['providers'] = self.encoder_session.get_providers()
        
        # Add device info if using PyTorch
        if hasattr(self, 'model') and hasattr(self.model, 'device'):
            status['device'] = str(self.model.device)
        
        return status


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
            
        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization: {e}")
            return False
    
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
            self.config = config
            
            # Initialize model manager
            self.model_manager = ModelManager(config)
            model_init = self.model_manager.initialize()
            
            # Initialize speaker diarizer
            self.diarizer = SpeakerDiarizer(config)
            diarizer_init = self.diarizer.initialize()
            
            # Initialize medical term processor
            self.term_processor = MedicalTermProcessor(config)
            
            # Set streaming context parameters
            streaming_config = config.get('streaming', {})
            self.context_max_length = streaming_config.get('max_context_length_sec', 60) * 16000
            
            # Set initialized flag
            self.initialized = model_init
            
            logger.info(f"STT Engine initialized with model: {self.model_manager.model_type}-{self.model_manager.model_size}")
            return self.initialized
            
        except Exception as e:
            logger.error(f"Failed to initialize STT Engine: {e}")
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
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the STT engine.
        
        Returns:
            Status dictionary
        """
        status = {
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
        if self.model_manager:
            status['model'] = self.model_manager.get_status()
        
        # Add diarization status
        if self.diarizer:
            status['diarization'] = {
                'enabled': self.diarizer.enabled,
                'initialized': self.diarizer.initialized
            }
        
        # Add vocabulary status
        if self.term_processor:
            status['vocabulary'] = {
                'enabled': self.term_processor.enabled,
                'medical_terms': len(self.term_processor.medical_terms),
                'abbreviations': len(self.term_processor.abbreviations)
            }
        
        return status