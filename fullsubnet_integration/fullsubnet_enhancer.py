#!/usr/bin/env python3
"""
FullSubNet Speech Enhancer for TCCC Project

This module integrates the FullSubNet speech enhancement model into the TCCC
audio processing pipeline, providing enhanced speech quality in noisy battlefield 
environments, leveraging Nvidia GPU acceleration.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Tuple, Optional, Union, Any
import soundfile as sf
from scipy import signal
import threading
import queue
import collections

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fullsubnet'))

# Import TCCC components
from tccc.utils.logging import get_logger
from tccc.utils.config_manager import ConfigManager

# Import FullSubNet components when available
try:
    from fullsubnet.audio_zen.model.full_subnet.model import FullSubNet
    from fullsubnet.audio_zen.utils import prepare_empty_dir, load_checkpoint
    from fullsubnet.audio_zen.acoustics.feature import stft, istft
    HAS_FULLSUBNET = False  # Temporarily disabled for offline testing
except ImportError:
    HAS_FULLSUBNET = False
    print("Warning: FullSubNet import failed. Please run fullsubnet_setup.sh first.")

# Configure logging
logger = get_logger("fullsubnet_enhancer")

class FullSubNetEnhancer:
    """
    Speech enhancement using the FullSubNet deep learning model.
    
    This class provides a wrapper around the FullSubNet model for real-time
    speech enhancement optimized for Jetson hardware with CUDA acceleration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the FullSubNet enhancer.
        
        Args:
            config: Configuration dictionary
        """
        # Check if FullSubNet is available
        if not HAS_FULLSUBNET:
            raise ImportError("FullSubNet not installed. Run fullsubnet_setup.sh first.")
        
        # Load default config if none provided
        if config is None:
            try:
                config_manager = ConfigManager()
                # Try to load from audio_pipeline first
                config = config_manager.load_config("audio_pipeline") 
                
                # Check for fullsubnet section, if not available load specific config
                if 'fullsubnet' not in config:
                    fullsubnet_config = {}
                    try:
                        # Try to load specific fullsubnet config
                        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                              'fullsubnet_config.yaml'), 'r') as f:
                            import yaml
                            fullsubnet_config = yaml.safe_load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load FullSubNet configuration: {e}")
                    
                    # Add fullsubnet config to main config
                    config['fullsubnet'] = fullsubnet_config.get('fullsubnet', {})
            except Exception as e:
                logger.warning(f"Failed to load configuration, using defaults: {e}")
                config = {'fullsubnet': self._get_default_config()}
        
        self.config = config
        self.fullsubnet_config = config.get('fullsubnet', self._get_default_config())
        
        # CUDA availability check with detailed diagnostics
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"CUDA is available with {gpu_count} devices. Using: {gpu_name}")
            
            # Additional Jetson-specific CUDA diagnostics
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                logger.info(f"GPU memory: {gpu_mem:.1f} MB")
            except Exception:
                pass
        else:
            logger.warning("CUDA is not available. Using CPU for inference (slower).")
        
        # Use GPU only if available and enabled in config
        self.use_gpu = self.cuda_available and self.fullsubnet_config.get('use_gpu', True)
        
        # Set device
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Initialize model
        self._initialize_model()
        
        # Processing history for quality metrics
        self.processing_metrics = {
            'processing_times': collections.deque(maxlen=100),
            'input_levels': collections.deque(maxlen=20),
            'output_levels': collections.deque(maxlen=20)
        }
        
        # Runtime stats
        self.stats = {
            'total_processed': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0,
            'start_time': time.time()
        }
        
        logger.info("FullSubNet speech enhancer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for FullSubNet."""
        return {
            'model_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      'models/fullsubnet_best_model_58epochs.pth'),
            'use_gpu': True,
            'sample_rate': 16000,
            'batch_size': 1,
            'chunk_size': 16000,  # 1 second of audio at 16kHz
            'n_fft': 512,
            'hop_length': 256,
            'win_length': 512,
            'normalized_input': True,
            'normalized_output': True,
            'gpu_acceleration': True,
            'fallback_to_cpu': True
        }
    
    def _initialize_model(self):
        """Initialize the FullSubNet model."""
        logger.info("Initializing FullSubNet model...")
        
        try:
            # Create model instance
            self.model = FullSubNet(
                num_freqs=self.fullsubnet_config.get('n_fft', 512) // 2 + 1,
                look_ahead=2,
                sequence_model="LSTM",
                fb_num_neighbors=0,
                sb_num_neighbors=15,
                fb_output_activate_function="ReLU",
                sb_output_activate_function=None,
                fb_model_hidden_size=512,
                sb_model_hidden_size=384,
                weight_init=True,
                norm_type="offline_laplace_norm",
                num_groups_in_drop_band=2
            )
            
            # Move to appropriate device
            self.model = self.model.to(self.device)
            
            # Load pre-trained weights
            model_path = self.fullsubnet_config.get('model_path')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            checkpoint = load_checkpoint(model_path, self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set to evaluation mode
            self.model.eval()
            
            # Extra optimization for CUDA if available
            if self.use_gpu:
                try:
                    # For Jetson optimal performance
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                    
                    # Enable TensorRT if available for Jetson optimization
                    if hasattr(torch, 'jit') and hasattr(torch.jit, 'trace'):
                        logger.info("TorchScript JIT available for optimization")
                    
                    if self.fullsubnet_config.get('gpu_acceleration', True):
                        # Use mixed precision for faster inference
                        if hasattr(torch.cuda, 'amp'):
                            logger.info("Using mixed precision for faster inference")
                            self.use_amp = True
                        else:
                            self.use_amp = False
                    else:
                        self.use_amp = False
                except Exception as e:
                    logger.warning(f"Failed to optimize for CUDA: {e}")
                    self.use_amp = False
            else:
                self.use_amp = False
            
            logger.info("FullSubNet model initialized successfully")
            
            # Set up STFT parameters
            self.n_fft = self.fullsubnet_config.get('n_fft', 512)
            self.hop_length = self.fullsubnet_config.get('hop_length', 256)
            self.win_length = self.fullsubnet_config.get('win_length', 512)
            
            # For overlap-add processing
            self.overlap_chunk_size = self.fullsubnet_config.get('chunk_size', 16000)
            self.overlap = 2048  # Overlap between chunks to avoid boundary artifacts
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FullSubNet model: {e}")
            if self.fullsubnet_config.get('fallback_to_cpu', True) and self.use_gpu:
                logger.info("Attempting to fall back to CPU")
                self.use_gpu = False
                self.device = torch.device("cpu")
                try:
                    return self._initialize_model()
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {e2}")
            
            raise RuntimeError("Failed to initialize FullSubNet model")
    
    @torch.no_grad()
    def _enhance_spectrogram(self, noisy_complex_spec):
        """
        Enhance a complex spectrogram using the FullSubNet model.
        
        Args:
            noisy_complex_spec: Complex noisy spectrogram
            
        Returns:
            Enhanced complex spectrogram
        """
        # Get magnitude and phase
        noisy_mag = torch.abs(noisy_complex_spec)
        noisy_phase = torch.angle(noisy_complex_spec)
        
        # Normalize magnitude
        if self.fullsubnet_config.get('normalized_input', True):
            mean = torch.mean(noisy_mag, dim=[1, 2], keepdim=True)
            std = torch.std(noisy_mag, dim=[1, 2], keepdim=True)
            noisy_mag = (noisy_mag - mean) / (std + 1e-8)
        
        # Prepare batch for model
        batch = {
            "noisy_mag": noisy_mag,
            "noisy_phase": noisy_phase,
        }
        
        # Use mixed precision if available
        if self.use_amp and hasattr(torch.cuda, 'amp'):
            with torch.cuda.amp.autocast():
                enhanced_mag = self.model(batch)
        else:
            enhanced_mag = self.model(batch)
        
        # Denormalize magnitude if needed
        if self.fullsubnet_config.get('normalized_output', True) and self.fullsubnet_config.get('normalized_input', True):
            enhanced_mag = enhanced_mag * std + mean
        
        # Combine enhanced magnitude with original phase
        enhanced_complex_spec = enhanced_mag * torch.exp(1j * noisy_phase)
        
        return enhanced_complex_spec
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, bool]:
        """
        Process audio with FullSubNet enhancement.
        
        Args:
            audio_data: Input audio data (numpy array)
            sample_rate: Sample rate of audio in Hz
            
        Returns:
            Tuple of (processed audio data, is_speech flag)
        """
        start_time = time.time()
        
        # Handle empty input gracefully
        if len(audio_data) == 0:
            logger.warning("Empty audio data provided to FullSubNet enhancer")
            return audio_data, False
        
        # Ensure proper format (float32 in range [-1, 1])
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32767.0
        
        # Resample if needed
        target_sample_rate = self.fullsubnet_config.get('sample_rate', 16000)
        if sample_rate != target_sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        
        # Store input level for metrics
        input_level = np.sqrt(np.mean(audio_data ** 2))
        self.processing_metrics['input_levels'].append(input_level)
        
        # Process audio
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_data).to(self.device)
            
            # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)  # [1, audio_len]
            
            # Compute STFT
            noisy_complex_spec = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(self.device),
                return_complex=True
            )
            
            # Run enhancement
            enhanced_complex_spec = self._enhance_spectrogram(noisy_complex_spec)
            
            # Convert back to time domain
            enhanced_audio = torch.istft(
                enhanced_complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(self.device),
                length=audio_tensor.shape[1]
            )
            
            # Remove batch dimension
            enhanced_audio = enhanced_audio.squeeze(0)
            
            # Convert to numpy array
            enhanced_audio_np = enhanced_audio.cpu().numpy()
            
            # Ensure output length matches input length
            if len(enhanced_audio_np) != len(audio_data):
                enhanced_audio_np = enhanced_audio_np[:len(audio_data)]
                if len(enhanced_audio_np) < len(audio_data):
                    enhanced_audio_np = np.pad(enhanced_audio_np, (0, len(audio_data) - len(enhanced_audio_np)))
            
            # For speech detection, we can use a simple energy-based method
            # A more sophisticated VAD could be implemented if needed
            is_speech = self._detect_speech(enhanced_audio_np)
            
            # Store output level for metrics
            output_level = np.sqrt(np.mean(enhanced_audio_np ** 2))
            self.processing_metrics['output_levels'].append(output_level)
            
            # Update stats
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_metrics['processing_times'].append(processing_time)
            self.stats['total_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = np.mean(self.processing_metrics['processing_times'])
            
            return enhanced_audio_np, is_speech
            
        except Exception as e:
            logger.error(f"Error in FullSubNet processing: {e}", exc_info=True)
            # Return original audio in case of error
            return audio_data, self._detect_speech(audio_data)
    
    def _resample_audio(self, audio_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """
        Resample audio to a different sample rate.
        
        Args:
            audio_data: Input audio data (numpy array)
            src_rate: Source sample rate
            dst_rate: Destination sample rate
            
        Returns:
            Resampled audio data
        """
        if src_rate == dst_rate:
            return audio_data
        
        try:
            # Use torchaudio for GPU-accelerated resampling if available
            if self.use_gpu and hasattr(torchaudio, 'transforms'):
                resampler = torchaudio.transforms.Resample(
                    orig_freq=src_rate,
                    new_freq=dst_rate
                ).to(self.device)
                
                # Convert to tensor
                audio_tensor = torch.FloatTensor(audio_data).to(self.device)
                
                # Resample
                resampled_tensor = resampler(audio_tensor)
                
                # Convert back to numpy
                return resampled_tensor.cpu().numpy()
            else:
                # Use scipy for CPU resampling
                return signal.resample_poly(
                    audio_data,
                    dst_rate,
                    src_rate
                )
        except Exception as e:
            logger.warning(f"Error in resampling, using scipy fallback: {e}")
            # Fallback to scipy
            return signal.resample_poly(
                audio_data,
                dst_rate,
                src_rate
            )
    
    def _detect_speech(self, audio_data: np.ndarray) -> bool:
        """
        Detect speech in audio data.
        
        Args:
            audio_data: Audio data (numpy array)
            
        Returns:
            True if speech is detected, False otherwise
        """
        # Simple energy-based VAD
        energy = np.mean(audio_data ** 2)
        
        # Estimate noise floor from processing history
        if len(self.processing_metrics['input_levels']) > 0:
            noise_floor = np.percentile(list(self.processing_metrics['input_levels']), 10)
            noise_floor = max(1e-6, noise_floor)  # Avoid zero
        else:
            noise_floor = 1e-6
        
        # Calculate adaptive threshold
        threshold = noise_floor * 3
        
        # Detect speech based on energy
        return energy > threshold
    
    def process_file(self, input_file: str, output_file: str) -> bool:
        """
        Process an audio file with FullSubNet enhancement.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            
        Returns:
            Success status
        """
        try:
            # Load audio file
            logger.info(f"Loading audio file: {input_file}")
            audio_data, sample_rate = sf.read(input_file)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Process audio in chunks for memory efficiency
            logger.info(f"Processing audio file with sample rate {sample_rate}Hz")
            
            # Process in overlapping chunks for better results
            chunk_size = self.overlap_chunk_size
            overlap = self.overlap
            pos = 0
            processed_chunks = []
            
            # Show progress
            total_chunks = (len(audio_data) - overlap) // (chunk_size - overlap) + 1
            progress_interval = max(1, total_chunks // 10)  # Show progress ~10 times
            
            chunk_counter = 0
            speech_chunks = 0
            
            while pos < len(audio_data):
                # Get chunk (with overlap for smooth processing)
                end_pos = min(pos + chunk_size, len(audio_data))
                chunk = audio_data[pos:end_pos]
                
                # Process chunk
                processed_chunk, is_speech = self.process_audio(chunk, sample_rate)
                
                # Apply fade in/out at chunk boundaries to avoid clicking
                if pos > 0 and len(processed_chunks) > 0:
                    fade_len = min(overlap, len(processed_chunk))
                    fade_in = np.linspace(0, 1, fade_len)
                    processed_chunk[:fade_len] = processed_chunk[:fade_len] * fade_in
                    
                    # Crossfade with previous chunk
                    prev_chunk = processed_chunks[-1]
                    fade_out = np.linspace(1, 0, fade_len)
                    overlap_region = prev_chunk[-fade_len:]
                    overlap_region = overlap_region * fade_out + processed_chunk[:fade_len] * fade_in
                    
                    # Update previous chunk
                    prev_chunk[-fade_len:] = overlap_region
                    
                    # Remove overlapped region from current chunk
                    processed_chunk = processed_chunk[fade_len:]
                
                # Add to processed data
                processed_chunks.append(processed_chunk)
                
                # Update counters
                chunk_counter += 1
                if is_speech:
                    speech_chunks += 1
                
                # Show progress
                if chunk_counter % progress_interval == 0:
                    logger.info(f"Progress: {pos/len(audio_data)*100:.1f}% - " +
                               f"Speech detected in {speech_chunks}/{chunk_counter} chunks")
                
                # Move position with overlap
                pos = end_pos - overlap
                if pos >= len(audio_data) - overlap:
                    break
            
            # Combine processed chunks
            processed_audio = np.concatenate(processed_chunks)
            
            # Ensure output length matches input
            if len(processed_audio) != len(audio_data):
                if len(processed_audio) > len(audio_data):
                    processed_audio = processed_audio[:len(audio_data)]
                else:
                    processed_audio = np.pad(processed_audio, (0, len(audio_data) - len(processed_audio)))
            
            # Convert to int16 for output
            processed_audio_int16 = np.clip(processed_audio * 32767, -32768, 32767).astype(np.int16)
            
            # Save to output file
            sf.write(output_file, processed_audio_int16, sample_rate)
            
            # Get processing stats
            avg_processing_time = np.mean(self.processing_metrics['processing_times'])
            
            logger.info(f"Audio processing complete: {output_file}")
            logger.info(f"Processed {chunk_counter} chunks, " +
                      f"detected speech in {speech_chunks} chunks " +
                      f"({speech_chunks/chunk_counter*100:.1f}%)")
            logger.info(f"Average processing time: {avg_processing_time:.2f}ms per chunk")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the FullSubNet enhancer.
        
        Returns:
            Dictionary of performance statistics
        """
        # Calculate estimated SNR improvement
        if (len(self.processing_metrics['input_levels']) > 0 and 
            len(self.processing_metrics['output_levels']) > 0):
            
            avg_input = np.mean(list(self.processing_metrics['input_levels']))
            avg_output = np.mean(list(self.processing_metrics['output_levels']))
            
            # Simple SNR improvement estimate based on level changes
            if avg_input > 0:
                snr_improvement = 10 * np.log10(avg_output / avg_input)
            else:
                snr_improvement = 0
        else:
            snr_improvement = 0
        
        # Gather stats
        stats = {
            'average_processing_time_ms': self.stats['avg_processing_time'],
            'total_chunks_processed': self.stats['total_processed'],
            'estimated_snr_improvement_db': snr_improvement,
            'cuda_available': self.cuda_available,
            'using_gpu': self.use_gpu,
            'using_mixed_precision': getattr(self, 'use_amp', False),
            'total_runtime_seconds': time.time() - self.stats['start_time'],
            'processing_rate': self.stats['total_processed'] / max(1, time.time() - self.stats['start_time'])
        }
        
        # Add GPU stats if available
        if self.use_gpu:
            try:
                stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
                stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            except Exception:
                pass
        
        return stats

def compare_enhancers(input_file, output_dir=None, visualize=False):
    """
    Compare FullSubNet with Battlefield enhancer on the same audio file.
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output files (default: same as input)
        visualize: Whether to create visualization (requires matplotlib)
    
    Returns:
        Dictionary with comparison results
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Output filenames
    fullsubnet_output = os.path.join(output_dir, f"{base_name}_fullsubnet.wav")
    battlefield_output = os.path.join(output_dir, f"{base_name}_battlefield.wav")
    
    # Load input audio
    audio_data, sample_rate = sf.read(input_file)
    
    # Initialize enhancers
    try:
        # FullSubNet enhancer
        fullsubnet_enhancer = FullSubNetEnhancer()
        
        # Battlefield enhancer
        from battlefield_audio_enhancer import BattlefieldAudioEnhancer
        battlefield_enhancer = BattlefieldAudioEnhancer()
        
        # Process with both enhancers
        logger.info("Processing with FullSubNet enhancer...")
        fullsubnet_success = fullsubnet_enhancer.process_file(input_file, fullsubnet_output)
        
        logger.info("Processing with Battlefield enhancer...")
        battlefield_success = battlefield_enhancer.process_file(input_file, battlefield_output)
        
        # Calculate metrics
        results = {
            "fullsubnet_success": fullsubnet_success,
            "battlefield_success": battlefield_success,
            "fullsubnet_stats": fullsubnet_enhancer.get_performance_stats(),
            "battlefield_stats": battlefield_enhancer.get_performance_stats() if battlefield_success else {}
        }
        
        # Create visualization if requested
        if visualize:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                
                # Load enhanced audio files
                fullsubnet_audio, _ = sf.read(fullsubnet_output)
                battlefield_audio, _ = sf.read(battlefield_output)
                
                # Create spectrograms
                fig, axs = plt.subplots(3, 1, figsize=(10, 12))
                
                # Original audio spectrogram
                axs[0].specgram(audio_data, Fs=sample_rate, cmap='viridis')
                axs[0].set_title("Original Audio")
                axs[0].set_xlabel("Time (s)")
                axs[0].set_ylabel("Frequency (Hz)")
                
                # FullSubNet enhanced spectrogram
                axs[1].specgram(fullsubnet_audio, Fs=sample_rate, cmap='viridis')
                axs[1].set_title("FullSubNet Enhanced")
                axs[1].set_xlabel("Time (s)")
                axs[1].set_ylabel("Frequency (Hz)")
                
                # Battlefield enhanced spectrogram
                axs[2].specgram(battlefield_audio, Fs=sample_rate, cmap='viridis')
                axs[2].set_title("Battlefield Enhanced")
                axs[2].set_xlabel("Time (s)")
                axs[2].set_ylabel("Frequency (Hz)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{base_name}_comparison.png"))
                plt.close()
                
                logger.info(f"Visualization saved to {os.path.join(output_dir, f'{base_name}_comparison.png')}")
            except Exception as e:
                logger.error(f"Error creating visualization: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        return {"error": str(e)}


def main():
    """Command-line interface for FullSubNet enhancer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FullSubNet Speech Enhancer")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--file", "-f", action="store_true",
                         help="Process an audio file")
    mode_group.add_argument("--compare", "-c", action="store_true",
                         help="Compare FullSubNet with Battlefield enhancer")
    
    # File processing options
    parser.add_argument("--input", "-i", type=str,
                      help="Input audio file")
    parser.add_argument("--output", "-o", type=str,
                      help="Output audio file")
    
    # Comparison options
    parser.add_argument("--visualize", "-v", action="store_true",
                      help="Create visualization of comparison")
    
    args = parser.parse_args()
    
    # Process file
    if args.file:
        if not args.input:
            print("Error: Input file required for file mode")
            return 1
        
        output_file = args.output or f"{os.path.splitext(args.input)[0]}_enhanced.wav"
        
        try:
            enhancer = FullSubNetEnhancer()
            print(f"Processing file: {args.input}")
            success = enhancer.process_file(args.input, output_file)
            
            if success:
                print(f"Processing complete: {output_file}")
                # Print performance stats
                stats = enhancer.get_performance_stats()
                print(f"Average processing time: {stats['average_processing_time_ms']:.2f}ms per chunk")
                print(f"Estimated SNR improvement: {stats['estimated_snr_improvement_db']:.1f}dB")
                return 0
            else:
                print("Processing failed")
                return 1
                
        except Exception as e:
            print(f"Error processing file: {e}")
            return 1
    
    # Compare enhancers
    if args.compare:
        if not args.input:
            print("Error: Input file required for comparison mode")
            return 1
        
        try:
            results = compare_enhancers(args.input, os.path.dirname(args.output) if args.output else None, args.visualize)
            
            print("\nComparison Results:")
            print("-" * 60)
            
            if "error" in results:
                print(f"Error: {results['error']}")
                return 1
            
            print("FullSubNet Enhancement:")
            print(f"  Success: {results['fullsubnet_success']}")
            print(f"  Processing time: {results['fullsubnet_stats'].get('average_processing_time_ms', 0):.2f}ms per chunk")
            print(f"  SNR improvement: {results['fullsubnet_stats'].get('estimated_snr_improvement_db', 0):.1f}dB")
            
            print("\nBattlefield Enhancement:")
            print(f"  Success: {results['battlefield_success']}")
            if results['battlefield_success']:
                print(f"  Processing time: {results['battlefield_stats'].get('average_processing_time_ms', 0):.2f}ms per chunk")
                print(f"  SNR improvement: {results['battlefield_stats'].get('estimated_snr_db', 0):.1f}dB")
            
            return 0
            
        except Exception as e:
            print(f"Error in comparison: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
