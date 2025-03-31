#!/usr/bin/env python3
"""
Test script for the FullSubNet speech enhancement model in the TCCC project.
This script tests the FullSubNet enhancer with various audio inputs and
compares performance with the existing Battlefield Audio Enhancer.
"""

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import torch
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from scipy import signal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import TCCC components
from tccc.utils.logging import get_logger
logger = get_logger("fullsubnet_test")

# Import enhancers
try:
    from fullsubnet_enhancer import FullSubNetEnhancer
    HAS_FULLSUBNET = True
except ImportError:
    HAS_FULLSUBNET = False
    logger.warning("FullSubNet enhancer not available. Run fullsubnet_setup.sh first.")

try:
    from battlefield_audio_enhancer import BattlefieldAudioEnhancer
    HAS_BATTLEFIELD = True
except ImportError:
    HAS_BATTLEFIELD = False
    logger.warning("Battlefield Audio Enhancer not available.")


def generate_test_audio(output_file: str, duration: float = 5.0, 
                       sample_rate: int = 16000, speech_file: str = None):
    """
    Generate test audio by mixing clean speech with noise.
    
    Args:
        output_file: Path to save the test audio
        duration: Duration of the test audio in seconds
        sample_rate: Sample rate of the test audio
        speech_file: Path to clean speech file (if None, uses test_data/test_speech.wav)
    """
    # Default speech file
    if speech_file is None:
        speech_file = os.path.join(project_root, "test_data/test_speech.wav")
    
    # Check if speech file exists
    if not os.path.exists(speech_file):
        logger.error(f"Speech file not found: {speech_file}")
        return False
    
    try:
        # Load speech file
        speech, speech_rate = sf.read(speech_file)
        
        # Convert to mono if stereo
        if len(speech.shape) > 1 and speech.shape[1] > 1:
            speech = speech.mean(axis=1)
        
        # Resample if needed
        if speech_rate != sample_rate:
            speech = signal.resample_poly(speech, sample_rate, speech_rate)
        
        # Truncate or pad to match duration
        target_samples = int(duration * sample_rate)
        if len(speech) > target_samples:
            speech = speech[:target_samples]
        elif len(speech) < target_samples:
            # Pad with zeros
            speech = np.pad(speech, (0, target_samples - len(speech)))
        
        # Generate noise
        noise_types = ["white", "pink", "brown", "vehicle", "wind"]
        noise_functions = {
            "white": lambda n: np.random.normal(0, 0.1, n),
            "pink": lambda n: np.array(
                signal.lfilter([1], [1, -0.9], np.random.normal(0, 0.1, n+100))
            )[100:],
            "brown": lambda n: np.array(
                signal.lfilter([1], [1, -0.98], np.random.normal(0, 0.05, n+100))
            )[100:],
            "vehicle": lambda n: _generate_vehicle_noise(n, sample_rate),
            "wind": lambda n: _generate_wind_noise(n, sample_rate)
        }
        
        # Choose random noise type
        noise_type = np.random.choice(noise_types)
        logger.info(f"Generating test audio with {noise_type} noise")
        
        # Generate noise
        noise = noise_functions[noise_type](target_samples)
        
        # Normalize noise level
        noise_level = np.random.uniform(0.05, 0.3)  # Random SNR
        speech_power = np.mean(speech ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Scale noise to desired level
        if noise_power > 0:
            scale = np.sqrt(speech_power / noise_power) * noise_level
            noise = noise * scale
        
        # Mix speech and noise
        mixed = speech + noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.99:
            mixed = mixed / max_val * 0.9
        
        # Save to file
        sf.write(output_file, mixed, sample_rate)
        
        logger.info(f"Generated test audio with {noise_type} noise: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating test audio: {e}")
        return False

def _generate_vehicle_noise(n_samples: int, sample_rate: int) -> np.ndarray:
    """Generate simulated vehicle noise."""
    # Create engine rumble (harmonic content plus noise)
    engine_freq = np.random.uniform(50, 120)  # Engine fundamental frequency
    t = np.arange(n_samples) / sample_rate
    
    # Create harmonics
    engine = np.zeros(n_samples)
    for harmonic in range(1, 6):
        amplitude = 0.2 / harmonic
        engine += amplitude * np.sin(2 * np.pi * engine_freq * harmonic * t)
    
    # Add some fluctuation
    fluctuation = 2 + np.sin(2 * np.pi * 0.5 * t)  # Slow fluctuation
    engine = engine * fluctuation
    
    # Add brown noise for rumble
    brown = signal.lfilter([1], [1, -0.98], np.random.normal(0, 0.1, n_samples))
    
    # Combine
    noise = 0.7 * engine + 0.3 * brown
    return noise * 0.15  # Scale to reasonable level

def _generate_wind_noise(n_samples: int, sample_rate: int) -> np.ndarray:
    """Generate simulated wind noise."""
    # Wind noise is mostly low-frequency with some gusts
    base_noise = np.random.normal(0, 0.1, n_samples)
    
    # Apply strong low-pass filter
    b, a = signal.butter(3, 400 / (sample_rate/2), 'low')
    wind = signal.lfilter(b, a, base_noise)
    
    # Add some gusts
    t = np.arange(n_samples) / sample_rate
    num_gusts = int(np.random.uniform(2, 5))
    
    for _ in range(num_gusts):
        # Random gust time and duration
        gust_time = np.random.uniform(0, len(t)/sample_rate - 1)
        gust_duration = np.random.uniform(0.2, 1.0)
        
        # Create gust envelope
        gust_center = gust_time * sample_rate
        gust_width = gust_duration * sample_rate
        gust_env = np.exp(-0.5 * ((np.arange(n_samples) - gust_center) / gust_width) ** 2)
        
        # Add gust
        wind += gust_env * np.random.uniform(0.1, 0.3) * wind
    
    return wind * 0.3  # Scale to reasonable level

def benchmark_enhancers(audio_file: str, output_dir: str = None):
    """
    Benchmark both enhancers on the same audio file and compare results.
    
    Args:
        audio_file: Path to input audio file
        output_dir: Directory to save output files (default: same as input)
    
    Returns:
        Dictionary with benchmark results
    """
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    # Output filenames
    fullsubnet_output = os.path.join(output_dir, f"{base_name}_fullsubnet.wav")
    battlefield_output = os.path.join(output_dir, f"{base_name}_battlefield.wav")
    
    results = {
        "fullsubnet_available": HAS_FULLSUBNET,
        "battlefield_available": HAS_BATTLEFIELD,
        "fullsubnet": {},
        "battlefield": {}
    }
    
    # Load input audio for reference
    try:
        audio_data, sample_rate = sf.read(audio_file)
        results["input_duration"] = len(audio_data) / sample_rate
        results["sample_rate"] = sample_rate
        
        # Calculate input audio stats
        input_rms = np.sqrt(np.mean(audio_data ** 2))
        input_peak = np.max(np.abs(audio_data))
        
        results["input_stats"] = {
            "rms": float(input_rms),
            "peak": float(input_peak),
            "dynamic_range_db": float(20 * np.log10(input_peak / (input_rms + 1e-8)))
        }
    except Exception as e:
        logger.error(f"Error loading input audio: {e}")
        results["error"] = f"Failed to load input audio: {str(e)}"
        return results
    
    # Benchmark FullSubNet
    if HAS_FULLSUBNET:
        try:
            logger.info("Testing FullSubNet enhancer...")
            
            # Check CUDA availability for FullSubNet
            cuda_available = torch.cuda.is_available()
            results["fullsubnet"]["cuda_available"] = cuda_available
            
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                results["fullsubnet"]["gpu_info"] = {
                    "name": gpu_name,
                    "memory_gb": float(gpu_memory)
                }
            
            # Create enhancer
            start_time = time.time()
            enhancer = FullSubNetEnhancer()
            init_time = time.time() - start_time
            results["fullsubnet"]["initialization_time_seconds"] = init_time
            
            # Process audio
            logger.info(f"Processing with FullSubNet: {audio_file}")
            start_time = time.time()
            success = enhancer.process_file(audio_file, fullsubnet_output)
            processing_time = time.time() - start_time
            
            results["fullsubnet"]["success"] = success
            results["fullsubnet"]["processing_time_seconds"] = processing_time
            results["fullsubnet"]["processing_ratio"] = processing_time / results["input_duration"]
            
            # Get enhancer stats
            results["fullsubnet"]["stats"] = enhancer.get_performance_stats()
            
            # Analyze enhanced audio
            if success and os.path.exists(fullsubnet_output):
                enhanced_audio, _ = sf.read(fullsubnet_output)
                
                # Calculate enhanced audio stats
                enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
                enhanced_peak = np.max(np.abs(enhanced_audio))
                
                results["fullsubnet"]["output_stats"] = {
                    "rms": float(enhanced_rms),
                    "peak": float(enhanced_peak),
                    "dynamic_range_db": float(20 * np.log10(enhanced_peak / (enhanced_rms + 1e-8))),
                    "level_change_db": float(20 * np.log10(enhanced_rms / (input_rms + 1e-8)))
                }
        except Exception as e:
            logger.error(f"Error benchmarking FullSubNet: {e}")
            results["fullsubnet"]["error"] = str(e)
    
    # Benchmark Battlefield Enhancer
    if HAS_BATTLEFIELD:
        try:
            logger.info("Testing Battlefield Audio Enhancer...")
            
            # Create enhancer
            start_time = time.time()
            enhancer = BattlefieldAudioEnhancer()
            init_time = time.time() - start_time
            results["battlefield"]["initialization_time_seconds"] = init_time
            
            # Process audio
            logger.info(f"Processing with Battlefield enhancer: {audio_file}")
            start_time = time.time()
            success = enhancer.process_file(audio_file, battlefield_output)
            processing_time = time.time() - start_time
            
            results["battlefield"]["success"] = success
            results["battlefield"]["processing_time_seconds"] = processing_time
            results["battlefield"]["processing_ratio"] = processing_time / results["input_duration"]
            
            # Get enhancer stats
            results["battlefield"]["stats"] = enhancer.get_performance_stats()
            
            # Analyze enhanced audio
            if success and os.path.exists(battlefield_output):
                enhanced_audio, _ = sf.read(battlefield_output)
                
                # Calculate enhanced audio stats
                enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
                enhanced_peak = np.max(np.abs(enhanced_audio))
                
                results["battlefield"]["output_stats"] = {
                    "rms": float(enhanced_rms),
                    "peak": float(enhanced_peak),
                    "dynamic_range_db": float(20 * np.log10(enhanced_peak / (enhanced_rms + 1e-8))),
                    "level_change_db": float(20 * np.log10(enhanced_rms / (input_rms + 1e-8)))
                }
        except Exception as e:
            logger.error(f"Error benchmarking Battlefield enhancer: {e}")
            results["battlefield"]["error"] = str(e)
    
    # Create comparison visualizations
    try:
        if (results["fullsubnet"].get("success", False) and 
            results["battlefield"].get("success", False)):
            
            logger.info("Creating comparison visualizations...")
            
            # Create output directory for visualizations
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Create comparison visualization
            _create_comparison_visualization(
                audio_file,
                fullsubnet_output,
                battlefield_output,
                os.path.join(viz_dir, f"{base_name}_comparison.png")
            )
            
            # Create spectrogram comparison
            _create_spectrogram_comparison(
                audio_file,
                fullsubnet_output,
                battlefield_output,
                os.path.join(viz_dir, f"{base_name}_spectrograms.png")
            )
            
            # Add visualization paths to results
            results["visualizations"] = {
                "comparison": os.path.join(viz_dir, f"{base_name}_comparison.png"),
                "spectrograms": os.path.join(viz_dir, f"{base_name}_spectrograms.png")
            }
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        results["visualization_error"] = str(e)
    
    return results

def _create_comparison_visualization(input_file: str, 
                                    fullsubnet_file: str, 
                                    battlefield_file: str,
                                    output_file: str):
    """Create waveform comparison visualization."""
    try:
        # Load audio files
        input_audio, sr_input = sf.read(input_file)
        fullsubnet_audio, sr_fs = sf.read(fullsubnet_file)
        battlefield_audio, sr_bf = sf.read(battlefield_file)
        
        # Ensure same length
        min_length = min(len(input_audio), len(fullsubnet_audio), len(battlefield_audio))
        input_audio = input_audio[:min_length]
        fullsubnet_audio = fullsubnet_audio[:min_length]
        battlefield_audio = battlefield_audio[:min_length]
        
        # Create time axis
        time_axis = np.arange(min_length) / sr_input
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot input audio
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, input_audio)
        plt.title("Original Noisy Audio")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # Plot FullSubNet audio
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, fullsubnet_audio)
        plt.title("FullSubNet Enhanced Audio")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # Plot Battlefield audio
        plt.subplot(3, 1, 3)
        plt.plot(time_axis, battlefield_audio)
        plt.title("Battlefield Enhanced Audio")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Saved waveform comparison to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating comparison visualization: {e}")
        return False

def _create_spectrogram_comparison(input_file: str, 
                                  fullsubnet_file: str, 
                                  battlefield_file: str,
                                  output_file: str):
    """Create spectrogram comparison visualization."""
    try:
        # Load audio files
        input_audio, sr_input = sf.read(input_file)
        fullsubnet_audio, sr_fs = sf.read(fullsubnet_file)
        battlefield_audio, sr_bf = sf.read(battlefield_file)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot input spectrogram
        plt.subplot(3, 1, 1)
        plt.specgram(input_audio, NFFT=512, Fs=sr_input, noverlap=256, cmap='viridis')
        plt.title("Original Noisy Audio Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(format='%+2.0f dB')
        
        # Plot FullSubNet spectrogram
        plt.subplot(3, 1, 2)
        plt.specgram(fullsubnet_audio, NFFT=512, Fs=sr_fs, noverlap=256, cmap='viridis')
        plt.title("FullSubNet Enhanced Audio Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(format='%+2.0f dB')
        
        # Plot Battlefield spectrogram
        plt.subplot(3, 1, 3)
        plt.specgram(battlefield_audio, NFFT=512, Fs=sr_bf, noverlap=256, cmap='viridis')
        plt.title("Battlefield Enhanced Audio Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Saved spectrogram comparison to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating spectrogram comparison: {e}")
        return False

def test_with_whisper(audio_file: str, enhanced_file: str):
    """
    Test transcription accuracy with Whisper STT.
    
    Args:
        audio_file: Path to original audio file
        enhanced_file: Path to enhanced audio file
    
    Returns:
        Dictionary with transcription results
    """
    try:
        # Ensure path to TCCC STT engine
        sys.path.insert(0, os.path.join(project_root, 'src'))
        
        # Import STT engine
        from tccc.stt_engine import create_stt_engine
        
        logger.info("Initializing Whisper STT engine...")
        
        # Create STT engine
        stt_engine = create_stt_engine("faster-whisper")
        
        # Initialize engine
        stt_config = {
            "model": {
                "size": "tiny",
                "compute_type": "int8",
                "vad_filter": False
            }
        }
        stt_engine.initialize(stt_config)
        
        # Transcribe original audio
        logger.info(f"Transcribing original audio: {audio_file}")
        orig_result = stt_engine.transcribe(audio_file)
        
        # Transcribe enhanced audio
        logger.info(f"Transcribing enhanced audio: {enhanced_file}")
        enhanced_result = stt_engine.transcribe(enhanced_file)
        
        # Get results
        orig_text = orig_result.get('text', '')
        enhanced_text = enhanced_result.get('text', '')
        
        results = {
            "original": {
                "text": orig_text,
                "word_count": len(orig_text.split()) if orig_text else 0
            },
            "enhanced": {
                "text": enhanced_text,
                "word_count": len(enhanced_text.split()) if enhanced_text else 0
            }
        }
        
        # Calculate simple similarity score (could be improved)
        if orig_text and enhanced_text:
            orig_words = set(orig_text.lower().split())
            enhanced_words = set(enhanced_text.lower().split())
            
            common_words = orig_words.intersection(enhanced_words)
            union_words = orig_words.union(enhanced_words)
            
            if union_words:
                similarity = len(common_words) / len(union_words)
                results["similarity"] = similarity
        
        # Shutdown engine
        stt_engine.shutdown()
        
        return results
    except Exception as e:
        logger.error(f"Error testing with Whisper: {e}")
        return {"error": str(e)}

def main():
    """Main function for testing FullSubNet enhancer."""
    parser = argparse.ArgumentParser(description="Test FullSubNet Speech Enhancer")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--benchmark", "-b", action="store_true",
                         help="Benchmark enhancers on audio file")
    mode_group.add_argument("--generate", "-g", action="store_true",
                         help="Generate test audio with noise")
    mode_group.add_argument("--transcribe", "-t", action="store_true",
                         help="Test transcription accuracy")
    
    # File options
    parser.add_argument("--input", "-i", type=str,
                      help="Input audio file")
    parser.add_argument("--output", "-o", type=str,
                      help="Output directory")
    parser.add_argument("--speech", "-s", type=str,
                      help="Clean speech file (for generate mode)")
    
    # Generation options
    parser.add_argument("--duration", "-d", type=float, default=5.0,
                      help="Duration of generated test audio in seconds")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output is None:
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Generate test audio
    if args.generate:
        if args.input is None:
            args.input = os.path.join(args.output, f"test_audio_{int(time.time())}.wav")
        
        print(f"Generating test audio: {args.input}")
        result = generate_test_audio(args.input, args.duration, speech_file=args.speech)
        
        if result:
            print(f"Test audio generated: {args.input}")
            print(f"Duration: {args.duration} seconds")
            return 0
        else:
            print("Failed to generate test audio")
            return 1
    
    # Benchmark enhancers
    if args.benchmark:
        if not args.input:
            print("Error: Input file required for benchmark mode")
            return 1
        
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1
        
        print(f"Benchmarking enhancers on: {args.input}")
        results = benchmark_enhancers(args.input, args.output)
        
        print("\nBenchmark Results:")
        print("-" * 60)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return 1
        
        print(f"Input audio duration: {results['input_duration']:.2f} seconds")
        print(f"Input audio RMS level: {results['input_stats']['rms']:.4f}")
        print(f"Input audio dynamic range: {results['input_stats']['dynamic_range_db']:.1f} dB")
        
        if results["fullsubnet_available"]:
            print("\nFullSubNet Enhancer:")
            fs_results = results["fullsubnet"]
            if "error" in fs_results:
                print(f"  Error: {fs_results['error']}")
            else:
                print(f"  CUDA available: {fs_results.get('cuda_available', False)}")
                if fs_results.get("gpu_info"):
                    print(f"  GPU: {fs_results['gpu_info'].get('name')}")
                    print(f"  GPU memory: {fs_results['gpu_info'].get('memory_gb', 0):.1f} GB")
                print(f"  Initialization time: {fs_results.get('initialization_time_seconds', 0):.2f} seconds")
                print(f"  Processing time: {fs_results.get('processing_time_seconds', 0):.2f} seconds")
                print(f"  Real-time factor: {fs_results.get('processing_ratio', 0):.2f}x")
                
                if fs_results.get("output_stats"):
                    print(f"  Output RMS level: {fs_results['output_stats'].get('rms', 0):.4f}")
                    print(f"  Level change: {fs_results['output_stats'].get('level_change_db', 0):.1f} dB")
                
                if fs_results.get("stats"):
                    print(f"  Avg. processing time: {fs_results['stats'].get('average_processing_time_ms', 0):.2f} ms/chunk")
        
        if results["battlefield_available"]:
            print("\nBattlefield Enhancer:")
            bf_results = results["battlefield"]
            if "error" in bf_results:
                print(f"  Error: {bf_results['error']}")
            else:
                print(f"  Initialization time: {bf_results.get('initialization_time_seconds', 0):.2f} seconds")
                print(f"  Processing time: {bf_results.get('processing_time_seconds', 0):.2f} seconds")
                print(f"  Real-time factor: {bf_results.get('processing_ratio', 0):.2f}x")
                
                if bf_results.get("output_stats"):
                    print(f"  Output RMS level: {bf_results['output_stats'].get('rms', 0):.4f}")
                    print(f"  Level change: {bf_results['output_stats'].get('level_change_db', 0):.1f} dB")
                
                if bf_results.get("stats"):
                    print(f"  Avg. processing time: {bf_results['stats'].get('average_processing_time_ms', 0):.2f} ms/chunk")
        
        if results.get("visualizations"):
            print("\nVisualizations:")
            print(f"  Waveform comparison: {results['visualizations'].get('comparison')}")
            print(f"  Spectrogram comparison: {results['visualizations'].get('spectrograms')}")
        
        return 0
    
    # Test transcription
    if args.transcribe:
        if not args.input:
            print("Error: Input file required for transcription test")
            return 1
        
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1
        
        # Process with both enhancers
        fs_output = os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.input))[0]}_fullsubnet.wav")
        bf_output = os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.input))[0]}_battlefield.wav")
        
        # Process with FullSubNet if available
        if HAS_FULLSUBNET:
            try:
                print(f"Processing with FullSubNet: {args.input}")
                enhancer = FullSubNetEnhancer()
                enhancer.process_file(args.input, fs_output)
                
                # Test transcription
                print("Testing transcription with FullSubNet enhancement...")
                fs_results = test_with_whisper(args.input, fs_output)
                
                print("\nTranscription Results (FullSubNet):")
                print("-" * 60)
                if "error" in fs_results:
                    print(f"Error: {fs_results['error']}")
                else:
                    print("Original transcription:")
                    print(f"  {fs_results['original']['text']}")
                    print(f"  Word count: {fs_results['original']['word_count']}")
                    
                    print("\nEnhanced transcription:")
                    print(f"  {fs_results['enhanced']['text']}")
                    print(f"  Word count: {fs_results['enhanced']['word_count']}")
                    
                    if "similarity" in fs_results:
                        print(f"\nSimilarity score: {fs_results['similarity']:.2f}")
            except Exception as e:
                print(f"Error testing FullSubNet transcription: {e}")
        
        # Process with Battlefield enhancer if available
        if HAS_BATTLEFIELD:
            try:
                print(f"Processing with Battlefield enhancer: {args.input}")
                enhancer = BattlefieldAudioEnhancer()
                enhancer.process_file(args.input, bf_output)
                
                # Test transcription
                print("Testing transcription with Battlefield enhancement...")
                bf_results = test_with_whisper(args.input, bf_output)
                
                print("\nTranscription Results (Battlefield):")
                print("-" * 60)
                if "error" in bf_results:
                    print(f"Error: {bf_results['error']}")
                else:
                    print("Original transcription:")
                    print(f"  {bf_results['original']['text']}")
                    print(f"  Word count: {bf_results['original']['word_count']}")
                    
                    print("\nEnhanced transcription:")
                    print(f"  {bf_results['enhanced']['text']}")
                    print(f"  Word count: {bf_results['enhanced']['word_count']}")
                    
                    if "similarity" in bf_results:
                        print(f"\nSimilarity score: {bf_results['similarity']:.2f}")
            except Exception as e:
                print(f"Error testing Battlefield transcription: {e}")
        
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
