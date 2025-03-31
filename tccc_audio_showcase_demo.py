#!/usr/bin/env python3
"""
TCCC Audio Showcase Demo

A comprehensive demonstration of the TCCC audio pipeline capabilities:
- High-quality audio capture with configurable settings
- Real-time audio processing with battlefield enhancement
- Voice activity detection with visual feedback
- Real-time transcription with performance metrics
- Multiple audio enhancement options
- Visual audio level monitoring and spectral display
- Performance benchmarking and comparison

This app represents a production-quality implementation of the audio pipeline.
"""

import os
import sys
import time
import wave
import queue
import threading
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from enum import Enum
from collections import deque
import logging
import platform
import subprocess
from scipy import signal

# Set up proper path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Force use of real implementations rather than mock versions
os.environ["USE_MOCK_STT"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_showcase_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TCCC.AudioShowcase")

# Import TCCC modules
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine
from tccc.utils.config import Config
from tccc.utils.vad_manager import VADManager

# Try to import optional battlefield enhancer
try:
    from battlefield_audio_enhancer import BattlefieldAudioEnhancer
    HAS_BATTLEFIELD_ENHANCER = True
except ImportError:
    HAS_BATTLEFIELD_ENHANCER = False
    logger.warning("Battlefield audio enhancer not available. Some features will be disabled.")

# Try to import optional FullSubNet enhancer
try:
    from fullsubnet_integration.fullsubnet_enhancer import FullSubNetEnhancer
    HAS_FULLSUBNET_ENHANCER = True
except ImportError:
    HAS_FULLSUBNET_ENHANCER = False
    logger.warning("FullSubNet enhancer not available. Some features will be disabled.")

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = 'int16'
BUFFER_SIZE = 10  # Number of chunks to buffer
DEFAULT_DEVICE = 0  # Default microphone device ID

# Output files
OUTPUT_DIR = "audio_showcase_output"
RAW_AUDIO_FILE = os.path.join(OUTPUT_DIR, "raw_audio.wav")
ENHANCED_AUDIO_FILE = os.path.join(OUTPUT_DIR, "enhanced_audio.wav")
TRANSCRIPTION_FILE = os.path.join(OUTPUT_DIR, "transcription.txt")
BENCHMARK_FILE = os.path.join(OUTPUT_DIR, "benchmark_results.txt")

# Enhancement modes
class EnhancementMode(Enum):
    NONE = "none"
    BASIC = "basic"
    BATTLEFIELD = "battlefield"
    FULLSUBNET = "fullsubnet"
    COMBINED = "combined"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ASCII Art for terminal UI
ASCII_TITLE = r"""
  _____  _____ _____ _____      _           _ _       
 |_   _|/ ____|_   _/ ____|    | |         | (_)      
   | | | |      | || |    _____| |__   __ _| |_ _ __  
   | | | |      | || |   |_____| '_ \ / _` | | | '_ \ 
  _| |_| |____ _| || |____     | | | | (_| | | | | | |
 |_____\_____|_____\_____|    |_| |_|\__,_|_|_|_| |_|
                                                      
     _             _ _           _____ _                                    
    / \  _   _  __| (_) ___     / ___|| |__   _____      _____ __ _ ___  ___ 
   / _ \| | | |/ _` | |/ _ \   | |  _ | '_ \ / _ \ \ /\ / / __/ _` / __|/ _ \
  / ___ \ |_| | (_| | | (_) |  | |_| || | | | (_) \ V  V / (_| (_| \__ \  __/
 /_/   \_\__,_|\__,_|_|\___/    \____|_| |_|\___/ \_/\_/ \___\__,_|___/\___|
"""

# Status colors for terminal
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Shared buffers and state
audio_buffer = deque(maxlen=100)  # Store recent audio for visualization
spectrum_buffer = deque(maxlen=100)  # Store recent spectra for visualization
level_buffer = deque(maxlen=50)  # Store recent audio levels
is_speaking = False  # Voice activity detection state
transcription_buffer = []  # Store recent transcriptions
current_mode = EnhancementMode.BASIC  # Default enhancement mode
benchmark_results = {mode: {"rtf": [], "latency": []} for mode in EnhancementMode}
recording_active = False
paused = False
exit_event = threading.Event()

# Visualization figure and axes
fig = None
ax_waveform = None
ax_spectrum = None
ax_levels = None
animation = None

# Performance tracking
perf_stats = {
    "processing_times": [],
    "cpu_usage": [],
    "latency": [],
    "memory_usage": [],
}

def create_terminal_ui():
    """Create an ASCII terminal UI for the demo."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(TermColors.BLUE + ASCII_TITLE + TermColors.END)
    print(TermColors.BOLD + "=" * 80 + TermColors.END)
    print(TermColors.BOLD + "TCCC Audio Pipeline Showcase Demo" + TermColors.END)
    print(TermColors.BOLD + "=" * 80 + TermColors.END)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Show audio hardware info
    try:
        if platform.system() == "Linux":
            audio_info = subprocess.check_output("arecord -l", shell=True).decode()
            print("\nAudio Hardware:")
            print(audio_info.split("\n")[0])
        else:
            print("\nAudio Hardware: [Import PyAudio to display audio device info]")
    except:
        print("\nAudio Hardware: [Unable to detect audio hardware]")
    
    # Show enhancement capabilities
    print("\nAvailable Enhancement Modes:")
    print(f" • Basic: ✓ (Spectral subtraction, normalization)")
    print(f" • Battlefield: {'✓' if HAS_BATTLEFIELD_ENHANCER else '✗'} (Specialized for tactical environments)")
    print(f" • FullSubNet: {'✓' if HAS_FULLSUBNET_ENHANCER else '✗'} (Deep learning-based speech enhancement)")
    
    # STT engine info
    print("\nSpeech Recognition:")
    print(" • Engine: Faster-Whisper (tiny model)")
    print(" • Features: Medical term detection, noise robustness, multilingual support")
    
    print("\nControls:")
    print(" • [Space] - Start/pause recording")
    print(" • [1-5]   - Switch enhancement modes")
    print(" • [V]     - Toggle visualization")
    print(" • [B]     - Run benchmark comparison")
    print(" • [Q]     - Quit")
    
    print(TermColors.BOLD + "\n" + "=" * 80 + TermColors.END)
    print(TermColors.GREEN + "Ready to start! Press SPACE to begin recording." + TermColors.END)

def display_audio_level(level, width=40, use_color=True):
    """
    Create a visual audio level meter with improved rendering and color.
    
    Args:
        level: Audio level (0.0-1.0)
        width: Width of the meter bar
        use_color: Whether to use ANSI color codes
        
    Returns:
        String containing the visual meter
    """
    filled = int(level * width)
    symbol = '▓'
    empty = '░'
    meter = symbol * filled + empty * (width - filled)
    db_level = 20 * np.log10(level + 1e-10)  # Convert to dB for better display
    
    # Color coding based on levels (green-yellow-red)
    if use_color:
        if level < 0.3:  # Low level - green
            color_code = TermColors.GREEN
        elif level < 0.7:  # Medium level - yellow
            color_code = TermColors.YELLOW
        else:  # High level - red
            color_code = TermColors.RED
            
        reset_code = TermColors.END
        meter = f"{color_code}|{meter}|{reset_code}"
    else:
        meter = f"|{meter}|"
    
    # Add a visual indicator for voice detection
    voice_indicator = ""
    if is_speaking:
        if use_color:
            voice_indicator = TermColors.BOLD + TermColors.GREEN + " SPEAKING " + TermColors.END
        else:
            voice_indicator = "SPEAKING"
    
    return f"{meter} {db_level:.1f} dB ({level*100:.1f}%) {voice_indicator}"

def update_status_display(message, mode=None, metrics=None):
    """Update the status display with the current state."""
    # Move cursor to line 15 (after the header)
    print("\033[15;1H")
    # Clear to the end of screen
    print("\033[J")
    
    # Print status message
    print(TermColors.BOLD + f"Status: {message}" + TermColors.END)
    
    # Show current mode
    mode_str = mode.value if mode else current_mode.value
    mode_colors = {
        "none": TermColors.RED,
        "basic": TermColors.YELLOW,
        "battlefield": TermColors.GREEN,
        "fullsubnet": TermColors.BLUE,
        "combined": TermColors.HEADER
    }
    mode_color = mode_colors.get(mode_str, TermColors.END)
    print(f"Enhancement Mode: {mode_color}{mode_str.upper()}{TermColors.END}")
    
    # Show metrics if available
    if metrics:
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f" • {key}: {value}")
    
    # Show recent transcriptions (last 5)
    if transcription_buffer:
        print("\nRecent Transcriptions:")
        for timestamp, text in transcription_buffer[-5:]:
            print(f"[{timestamp}] {text}")
    
    # Add recording status indicator
    status = "⏺ RECORDING" if recording_active and not paused else "⏸ PAUSED" if paused else "⏹ STOPPED"
    status_color = TermColors.RED if recording_active and not paused else TermColors.YELLOW if paused else TermColors.BLUE
    print(f"\n{status_color}{status}{TermColors.END}")

def init_audio_pipeline():
    """Initialize the audio pipeline with the appropriate configuration."""
    # Create audio pipeline configuration
    audio_config = {
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "format": AUDIO_FORMAT,
            "chunk_size": CHUNK_SIZE,
            "buffer_size": BUFFER_SIZE
        },
        "io": {
            "input_sources": [
                {
                    "name": "microphone",
                    "type": "microphone",
                    "device_id": DEFAULT_DEVICE
                }
            ],
            "default_input": "microphone"
        },
        "noise_reduction": {
            "enabled": True,
            "strength": 0.7
        },
        "vad": {
            "enabled": True,
            "sensitivity": 2,
            "frame_duration_ms": 30
        }
    }
    
    # Initialize the audio pipeline
    pipeline = AudioPipeline()
    success = pipeline.initialize(audio_config)
    if not success:
        logger.error("Failed to initialize audio pipeline")
        return None
    
    return pipeline

def init_stt_engine():
    """Initialize the STT engine with appropriate configuration."""
    # Create STT configuration
    stt_config = {
        "model": {
            "size": "tiny",
            "compute_type": "int8",
            "vad_filter": False  # We'll use our enhanced VAD instead
        }
    }
    
    # Create and initialize the STT engine
    stt_engine = create_stt_engine("faster-whisper")
    success = stt_engine.initialize(stt_config)
    if not success:
        logger.error("Failed to initialize STT engine")
        return None
    
    return stt_engine

def init_enhancers():
    """Initialize available audio enhancers."""
    enhancers = {}
    
    # Basic enhancer is always available (spectral subtraction + normalization)
    enhancers[EnhancementMode.BASIC] = {
        "name": "Basic Enhancement",
        "instance": None  # Uses internal functions
    }
    
    # Initialize battlefield enhancer if available
    if HAS_BATTLEFIELD_ENHANCER:
        try:
            battlefield_config = {
                "audio": {
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                    "format": AUDIO_FORMAT,
                    "chunk_size": CHUNK_SIZE
                },
                "battlefield_filtering": {
                    "enabled": True,
                    "outdoor_mode": True,
                    "transient_protection": True,
                    "distance_compensation": True
                },
                "voice_isolation": {
                    "enabled": True,
                    "strength": 0.8,
                    "focus_width": 200,
                    "voice_boost_db": 6
                }
            }
            
            battlefield_enhancer = BattlefieldAudioEnhancer(battlefield_config)
            enhancers[EnhancementMode.BATTLEFIELD] = {
                "name": "Battlefield Enhancer",
                "instance": battlefield_enhancer
            }
            logger.info("Battlefield enhancer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize battlefield enhancer: {e}")
    
    # Initialize FullSubNet enhancer if available
    if HAS_FULLSUBNET_ENHANCER:
        try:
            fullsubnet_config = {
                "fullsubnet": {
                    "enabled": True,
                    "use_gpu": True,
                    "sample_rate": SAMPLE_RATE,
                    "chunk_size": CHUNK_SIZE
                }
            }
            
            fullsubnet_enhancer = FullSubNetEnhancer(fullsubnet_config)
            enhancers[EnhancementMode.FULLSUBNET] = {
                "name": "FullSubNet Enhancer",
                "instance": fullsubnet_enhancer
            }
            logger.info("FullSubNet enhancer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FullSubNet enhancer: {e}")
    
    # Add combined mode if both enhancers are available
    if EnhancementMode.BATTLEFIELD in enhancers and EnhancementMode.FULLSUBNET in enhancers:
        enhancers[EnhancementMode.COMBINED] = {
            "name": "Combined Enhancement",
            "instance": None  # Will use both enhancers in sequence
        }
    
    return enhancers

def apply_basic_enhancement(audio_data, noise_profile=None):
    """Apply basic audio enhancement (spectral subtraction and normalization)."""
    # Ensure input is in float32 format for processing
    if audio_data.dtype != np.float32:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.copy()
    
    # Apply spectral subtraction if noise profile is available
    if noise_profile is not None:
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_float)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Apply spectral subtraction
        # Subtract noise profile from magnitude, ensuring it doesn't go below a minimum threshold
        threshold = 0.01  # Minimum spectral floor
        reduction_strength = 0.7  # Adjustable strength
        magnitude = np.maximum(magnitude - reduction_strength * noise_profile, threshold * np.max(magnitude))
        
        # Reconstruct signal with reduced noise magnitude but original phase
        fft_reduced = magnitude * np.exp(1j * phase)
        
        # Convert back to time domain
        audio_float = np.fft.irfft(fft_reduced)
        
        # Ensure same length as input
        audio_float = audio_float[:len(audio_data)]
    
    # Apply normalization
    # Calculate current RMS level
    rms = np.sqrt(np.mean(audio_float**2))
    if rms > 1e-8:  # Avoid division by zero
        # Convert to dB
        current_db = 20 * np.log10(rms)
        
        # Calculate gain needed (target -18dB)
        target_level = -18
        gain_db = target_level - current_db
        gain_linear = 10 ** (gain_db/20)
        
        # Apply gain
        audio_float = audio_float * gain_linear
        
        # Prevent clipping
        max_val = np.max(np.abs(audio_float))
        if max_val > 0.95:
            audio_float = audio_float * 0.95 / max_val
    
    # Convert back to original format if needed
    if audio_data.dtype != np.float32:
        return (audio_float * 32767.0).astype(audio_data.dtype)
    return audio_float

def enhance_audio(audio_data, mode, enhancers, noise_profile=None):
    """
    Enhance audio using the selected enhancement mode.
    
    Args:
        audio_data: Audio data as numpy array
        mode: Enhancement mode to use
        enhancers: Dictionary of available enhancers
        noise_profile: Optional noise profile for basic enhancement
        
    Returns:
        Enhanced audio data, speaking flag
    """
    # Convert to float32 for processing if needed
    if audio_data.dtype != np.float32:
        float_audio = audio_data.astype(np.float32) / 32767.0
    else:
        float_audio = audio_data.copy()
    
    speaking = False
    
    # No enhancement
    if mode == EnhancementMode.NONE:
        processed_audio = float_audio
        # Simple energy-based VAD
        energy = np.sqrt(np.mean(float_audio ** 2))
        speaking = energy > 0.01  # Simple threshold
        
    # Basic enhancement
    elif mode == EnhancementMode.BASIC:
        processed_audio = apply_basic_enhancement(float_audio, noise_profile)
        # Simple energy-based VAD
        energy = np.sqrt(np.mean(processed_audio ** 2))
        speaking = energy > 0.01  # Simple threshold
        
    # Battlefield enhancement
    elif mode == EnhancementMode.BATTLEFIELD and EnhancementMode.BATTLEFIELD in enhancers:
        battlefield_enhancer = enhancers[EnhancementMode.BATTLEFIELD]["instance"]
        processed_audio, speaking = battlefield_enhancer.process_audio(float_audio, SAMPLE_RATE)
        
    # FullSubNet enhancement
    elif mode == EnhancementMode.FULLSUBNET and EnhancementMode.FULLSUBNET in enhancers:
        fullsubnet_enhancer = enhancers[EnhancementMode.FULLSUBNET]["instance"]
        processed_audio, speaking = fullsubnet_enhancer.process_audio(float_audio, SAMPLE_RATE)
        
    # Combined enhancement (Battlefield + FullSubNet)
    elif mode == EnhancementMode.COMBINED and EnhancementMode.COMBINED in enhancers:
        # Process with battlefield enhancer first
        battlefield_enhancer = enhancers[EnhancementMode.BATTLEFIELD]["instance"]
        processed_audio, speaking_bf = battlefield_enhancer.process_audio(float_audio, SAMPLE_RATE)
        
        # Then process with FullSubNet
        fullsubnet_enhancer = enhancers[EnhancementMode.FULLSUBNET]["instance"]
        processed_audio, speaking_fs = fullsubnet_enhancer.process_audio(processed_audio, SAMPLE_RATE)
        
        # Combine speaking flags (either detected speech)
        speaking = speaking_bf or speaking_fs
        
    # Fallback to basic enhancement
    else:
        processed_audio = apply_basic_enhancement(float_audio, noise_profile)
        # Simple energy-based VAD
        energy = np.sqrt(np.mean(processed_audio ** 2))
        speaking = energy > 0.01  # Simple threshold
    
    # Convert back to original format if needed
    if audio_data.dtype != np.float32:
        return (processed_audio * 32767.0).astype(audio_data.dtype), speaking
    return processed_audio, speaking

def calculate_spectrum(audio_data, sample_rate=SAMPLE_RATE):
    """Calculate the spectrum of the audio data using STFT."""
    # Use short-time Fourier transform (STFT)
    f, t, Zxx = signal.stft(
        audio_data, 
        fs=sample_rate, 
        window='hann', 
        nperseg=min(512, len(audio_data)), 
        noverlap=128
    )
    # Convert to dB scale
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    return f, t, Zxx_db

def init_visualization():
    """Initialize the visualization plots."""
    global fig, ax_waveform, ax_spectrum, ax_levels
    
    # Create figure with 3 subplots
    fig, (ax_waveform, ax_spectrum, ax_levels) = plt.subplots(3, 1, figsize=(10, 8))
    
    # Set up waveform plot
    ax_waveform.set_title('Audio Waveform')
    ax_waveform.set_ylabel('Amplitude')
    ax_waveform.set_ylim(-32768, 32768)
    ax_waveform.grid(True)
    waveform_line, = ax_waveform.plot(np.zeros(1024), 'b-')
    
    # Set up spectrum plot
    ax_spectrum.set_title('Audio Spectrum')
    ax_spectrum.set_ylabel('Frequency (Hz)')
    ax_spectrum.set_xlabel('Time')
    
    # Create an empty spectrogram
    spectrogram = ax_spectrum.imshow(
        np.zeros((100, 100)), 
        aspect='auto', 
        origin='lower',
        cmap='viridis'
    )
    
    # Set up audio level plot
    ax_levels.set_title('Audio Level')
    ax_levels.set_ylabel('Level (dB)')
    ax_levels.set_xlabel('Time')
    ax_levels.set_ylim(-60, 0)
    ax_levels.grid(True)
    levels_line, = ax_levels.plot(np.zeros(50), 'g-')
    
    # Adjust layout
    fig.tight_layout()
    
    return waveform_line, spectrogram, levels_line

def update_visualization(frame):
    """Update the visualization with new audio data."""
    global audio_buffer, spectrum_buffer, level_buffer
    global waveform_line, spectrogram, levels_line
    
    # Update waveform plot
    if audio_buffer:
        latest_audio = audio_buffer[-1]
        waveform_line.set_ydata(latest_audio)
        waveform_line.set_xdata(np.arange(len(latest_audio)))
        ax_waveform.set_xlim(0, len(latest_audio))
    
    # Update spectrum plot
    if spectrum_buffer:
        # Get the spectrum data
        all_spectra = np.array(list(spectrum_buffer))
        spectrogram.set_data(all_spectra)
        spectrogram.set_extent([0, all_spectra.shape[0], 0, all_spectra.shape[1]])
        
        # Update colorbar limits
        vmin = np.min(all_spectra)
        vmax = np.max(all_spectra)
        spectrogram.set_clim(vmin, vmax)
    
    # Update level plot
    if level_buffer:
        levels = np.array(level_buffer)
        levels_line.set_ydata(levels)
        levels_line.set_xdata(np.arange(len(levels)))
        ax_levels.set_xlim(0, len(levels))
        
        # Color the line based on is_speaking
        if is_speaking:
            levels_line.set_color('g')
            ax_levels.set_title('Audio Level (SPEECH DETECTED)')
        else:
            levels_line.set_color('b')
            ax_levels.set_title('Audio Level')
    
    return waveform_line, spectrogram, levels_line

def start_visualization():
    """Start the visualization animation."""
    global fig, animation, waveform_line, spectrogram, levels_line
    
    if fig is None:
        waveform_line, spectrogram, levels_line = init_visualization()
    
    # Create animation
    animation = FuncAnimation(
        fig, 
        update_visualization, 
        interval=100,
        blit=True
    )
    
    # Show plot
    plt.show(block=False)

def stop_visualization():
    """Stop the visualization animation."""
    global animation, fig
    
    if animation:
        animation.event_source.stop()
        animation = None
    
    if fig:
        plt.close(fig)
        fig = None

def run_benchmark(audio_pipeline, stt_engine, enhancers, num_samples=10):
    """
    Run a benchmark comparing all available enhancement modes.
    
    Args:
        audio_pipeline: Initialized audio pipeline
        stt_engine: Initialized STT engine
        enhancers: Dictionary of available enhancers
        num_samples: Number of audio samples to process for each mode
        
    Returns:
        Dictionary with benchmark results
    """
    # Store benchmark results
    results = {}
    
    # Create a consistent audio source for benchmarking
    audio_source = audio_pipeline.get_available_sources()[0]["name"]
    audio_pipeline.start_capture(audio_source)
    
    # Update status display
    update_status_display("Running benchmark...", None, {"Samples": num_samples})
    
    # Collect some audio samples for processing
    audio_samples = []
    for _ in range(num_samples):
        audio_stream = audio_pipeline.get_audio_stream()
        if audio_stream:
            audio_data = audio_stream.read()
            if audio_data is not None and len(audio_data) > 0:
                audio_samples.append(audio_data)
        time.sleep(0.1)
    
    # If we couldn't collect enough samples, use the ones we have multiple times
    while len(audio_samples) < num_samples:
        audio_samples.append(audio_samples[0] if audio_samples else np.zeros(CHUNK_SIZE, dtype=np.int16))
    
    # Benchmark each enhancement mode
    for mode in EnhancementMode:
        # Skip modes that aren't available
        if mode in [EnhancementMode.BATTLEFIELD, EnhancementMode.FULLSUBNET, EnhancementMode.COMBINED]:
            mode_name = mode.value
            if mode not in enhancers:
                continue
        
        results[mode.value] = {
            "processing_time_ms": [],
            "rtf": [],
            "latency_ms": []
        }
        
        update_status_display(f"Benchmarking {mode.value} mode...", mode)
        
        # Process each sample with the current enhancement mode
        for audio_data in audio_samples:
            # Record start time
            start_time = time.time()
            
            # Process audio with the enhancement mode
            enhanced_audio, _ = enhance_audio(audio_data, mode, enhancers)
            
            # Calculate processing time
            enhancement_time = time.time() - start_time
            
            # Send to STT engine
            stt_start_time = time.time()
            stt_result = stt_engine.transcribe_segment(enhanced_audio)
            stt_time = time.time() - stt_start_time
            
            # Calculate metrics
            audio_duration = len(audio_data) / SAMPLE_RATE
            processing_time_ms = enhancement_time * 1000
            total_time = enhancement_time + stt_time
            rtf = total_time / audio_duration
            latency_ms = total_time * 1000
            
            # Store results
            results[mode.value]["processing_time_ms"].append(processing_time_ms)
            results[mode.value]["rtf"].append(rtf)
            results[mode.value]["latency_ms"].append(latency_ms)
    
    # Calculate average metrics
    for mode in results:
        for metric in results[mode]:
            results[mode][metric] = sum(results[mode][metric]) / len(results[mode][metric])
    
    # Stop audio capture
    audio_pipeline.stop_capture()
    
    # Write results to file
    with open(BENCHMARK_FILE, 'w') as f:
        f.write("TCCC Audio Pipeline Benchmark Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples per mode: {num_samples}\n\n")
        
        f.write("Summary:\n")
        for mode in results:
            f.write(f"{mode.upper()} Mode:\n")
            for metric, value in results[mode].items():
                f.write(f"  {metric}: {value:.2f}\n")
            f.write("\n")
    
    # Update status display with results
    metrics = {
        f"{mode.upper()} RTF": f"{results[mode]['rtf']:.2f}x" 
        for mode in results
    }
    update_status_display("Benchmark complete!", None, metrics)
    
    # Return results
    return results

def process_audio_callback(audio_data, stt_engine, enhancers, mode, noise_profile=None):
    """
    Process audio data and display transcription results.
    
    Args:
        audio_data: Audio data as numpy array
        stt_engine: STT engine instance
        enhancers: Dictionary of available enhancers
        mode: Enhancement mode to use
        noise_profile: Optional noise profile for basic enhancement
    """
    global audio_buffer, spectrum_buffer, level_buffer, is_speaking, transcription_buffer
    global perf_stats
    
    # Skip if paused
    if paused:
        return
    
    # Record start time for performance tracking
    start_time = time.time()
    
    # Store original audio for raw output
    if len(audio_buffer) >= 100:
        audio_buffer.popleft()
    audio_buffer.append(audio_data)
    
    # Process audio with selected enhancement
    enhanced_audio, speaking = enhance_audio(audio_data, mode, enhancers, noise_profile)
    is_speaking = speaking
    
    # Calculate and store spectrum for visualization
    _, _, spectrum = calculate_spectrum(enhanced_audio)
    if spectrum.shape[0] > 0 and spectrum.shape[1] > 0:
        # Take the first time slice of the spectrum
        spectrum_slice = spectrum[:, 0]
        spectrum_buffer.append(spectrum_slice)
    
    # Calculate and store level for visualization
    level = np.max(np.abs(enhanced_audio)) / 32767.0
    level_db = 20 * np.log10(level + 1e-10)
    level_buffer.append(level_db)
    
    # Process with STT if speaking is detected
    if speaking:
        # Process audio with STT engine
        stt_result = stt_engine.transcribe_segment(enhanced_audio)
        
        # Get transcription result
        if stt_result and 'text' in stt_result and stt_result['text'].strip():
            text = stt_result['text'].strip()
            
            # Get metrics if available
            rtf = stt_result.get('metrics', {}).get('real_time_factor', 0)
            
            # Get timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Add to transcription buffer
            transcription_buffer.append((timestamp, text))
            
            # Write to transcription file
            with open(TRANSCRIPTION_FILE, 'a') as f:
                f.write(f"[{timestamp}] {text}\n")
            
            # Update perf stats
            processing_time = time.time() - start_time
            perf_stats["processing_times"].append(processing_time)
            perf_stats["latency"].append(processing_time * 1000)  # Convert to ms
            
            # Update status display
            metrics = {
                "RTF": f"{rtf:.2f}x",
                "Latency": f"{processing_time*1000:.1f} ms",
                "Level": f"{level_db:.1f} dB"
            }
            update_status_display("Processing audio...", mode, metrics)
    else:
        # Update status display with just the level
        metrics = {
            "Level": f"{level_db:.1f} dB"
        }
        update_status_display("Listening...", mode, metrics)
    
    # Display audio level
    level_str = display_audio_level(level)
    print(f"\r{level_str}", end='', flush=True)

def calibrate_noise(audio_pipeline, calibration_time=2.0):
    """
    Calibrate noise profile for basic enhancement.
    
    Args:
        audio_pipeline: Initialized audio pipeline
        calibration_time: Calibration duration in seconds
        
    Returns:
        Noise profile for spectral subtraction
    """
    # Update status
    update_status_display("Calibrating noise profile... Please remain silent.", None)
    
    # Start audio capture
    audio_source = audio_pipeline.get_available_sources()[0]["name"]
    audio_pipeline.start_capture(audio_source)
    
    # Collect noise samples
    noise_samples = []
    start_time = time.time()
    
    while time.time() - start_time < calibration_time:
        audio_stream = audio_pipeline.get_audio_stream()
        if audio_stream:
            audio_data = audio_stream.read()
            if audio_data is not None and len(audio_data) > 0:
                noise_samples.append(audio_data)
        time.sleep(0.1)
    
    # Stop audio capture
    audio_pipeline.stop_capture()
    
    # Calculate noise profile
    if noise_samples:
        noise_concat = np.concatenate(noise_samples)
        noise_fft = np.abs(np.fft.rfft(noise_concat))
        # Average the noise spectrum
        noise_profile = noise_fft / len(noise_fft)
        
        # Scale noise profile to match future FFT sizes
        target_size = CHUNK_SIZE // 2 + 1
        if len(noise_profile) > target_size:
            noise_profile = noise_profile[:target_size]
        elif len(noise_profile) < target_size:
            noise_profile = np.pad(noise_profile, (0, target_size - len(noise_profile)))
        
        # Calculate average noise level
        avg_noise_level = np.sqrt(np.mean(noise_concat**2)) / 32767.0
        db_level = 20 * np.log10(avg_noise_level + 1e-10)
        
        # Update status
        update_status_display(f"Calibration complete. Noise level: {db_level:.1f} dB", None)
        
        return noise_profile
    
    # Return None if no noise samples were collected
    update_status_display("Calibration failed. Using default noise profile.", None)
    return None

def keyboard_listener():
    """Listen for keyboard commands."""
    global recording_active, paused, current_mode, exit_event
    
    while not exit_event.is_set():
        try:
            # Get a single character
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
            # Process commands
            if ch.lower() == 'q':
                exit_event.set()
            elif ch == ' ':  # space
                if not recording_active:
                    recording_active = True
                    paused = False
                    update_status_display("Starting recording...", current_mode)
                else:
                    paused = not paused
                    update_status_display("Paused" if paused else "Resumed", current_mode)
            elif ch == 'v':
                # Toggle visualization
                if fig is None:
                    start_visualization()
                    update_status_display("Visualization started", current_mode)
                else:
                    stop_visualization()
                    update_status_display("Visualization stopped", current_mode)
            elif ch == 'b':
                # Run benchmark
                run_benchmark(audio_pipeline, stt_engine, enhancers)
            elif ch in '12345':
                # Switch enhancement mode
                mode_map = {
                    '1': EnhancementMode.NONE,
                    '2': EnhancementMode.BASIC,
                    '3': EnhancementMode.BATTLEFIELD,
                    '4': EnhancementMode.FULLSUBNET,
                    '5': EnhancementMode.COMBINED
                }
                new_mode = mode_map[ch]
                
                # Check if mode is available
                if new_mode in [EnhancementMode.BATTLEFIELD, EnhancementMode.FULLSUBNET, EnhancementMode.COMBINED]:
                    if new_mode not in enhancers:
                        update_status_display(f"{new_mode.value} mode not available", current_mode)
                        continue
                
                current_mode = new_mode
                update_status_display(f"Switched to {current_mode.value} mode", current_mode)
        except Exception as e:
            logger.error(f"Error in keyboard listener: {e}")
            time.sleep(0.1)

def main():
    """Main entry point for the audio showcase demo."""
    global audio_pipeline, stt_engine, enhancers, current_mode, recording_active, exit_event
    
    # Create terminal UI
    create_terminal_ui()
    
    # Create output files
    with open(TRANSCRIPTION_FILE, 'w') as f:
        f.write("=== TCCC Audio Showcase Demo Transcription ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Initialize components
    audio_pipeline = init_audio_pipeline()
    if not audio_pipeline:
        print("Error initializing audio pipeline. Exiting.")
        return 1
    
    stt_engine = init_stt_engine()
    if not stt_engine:
        print("Error initializing STT engine. Exiting.")
        return 1
    
    enhancers = init_enhancers()
    
    # Initialize VAD
    vad_manager = VADManager()
    vad_manager.initialize({
        "sample_rate": SAMPLE_RATE,
        "frame_duration_ms": 30,
        "sensitivity": 2
    })
    
    # Calibrate noise profile
    noise_profile = calibrate_noise(audio_pipeline)
    
    # Create keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # Main processing loop
    try:
        while not exit_event.is_set():
            # Check if recording is active
            if recording_active and not paused:
                # Start audio capture if not already running
                if not audio_pipeline.is_capturing():
                    audio_source = audio_pipeline.get_available_sources()[0]["name"]
                    audio_pipeline.start_capture(audio_source)
                    
                    # Initialize WAV files for recording
                    # Raw audio
                    raw_wav = wave.open(RAW_AUDIO_FILE, 'wb')
                    raw_wav.setnchannels(CHANNELS)
                    raw_wav.setsampwidth(2)  # 16-bit
                    raw_wav.setframerate(SAMPLE_RATE)
                    
                    # Enhanced audio
                    enhanced_wav = wave.open(ENHANCED_AUDIO_FILE, 'wb')
                    enhanced_wav.setnchannels(CHANNELS)
                    enhanced_wav.setsampwidth(2)  # 16-bit
                    enhanced_wav.setframerate(SAMPLE_RATE)
                
                # Get audio from the pipeline
                audio_stream = audio_pipeline.get_audio_stream()
                if audio_stream:
                    audio_data = audio_stream.read()
                    if audio_data is not None and len(audio_data) > 0:
                        # Process the audio
                        process_audio_callback(audio_data, stt_engine, enhancers, current_mode, noise_profile)
                        
                        # Save raw audio
                        raw_wav.writeframes(audio_data.tobytes())
                        
                        # Process and save enhanced audio
                        enhanced_audio, _ = enhance_audio(audio_data, current_mode, enhancers, noise_profile)
                        enhanced_wav.writeframes(enhanced_audio.tobytes())
            else:
                # Stop audio capture if running
                if audio_pipeline.is_capturing():
                    audio_pipeline.stop_capture()
                    
                    # Close WAV files
                    if 'raw_wav' in locals():
                        raw_wav.close()
                    if 'enhanced_wav' in locals():
                        enhanced_wav.close()
                
                # Just update status and sleep
                update_status_display("Ready to record. Press SPACE to start.", current_mode)
                time.sleep(0.1)
            
            # Small sleep to prevent tight loop
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Stop visualization
        stop_visualization()
        
        # Stop audio capture
        if audio_pipeline.is_capturing():
            audio_pipeline.stop_capture()
            
            # Close WAV files
            if 'raw_wav' in locals():
                raw_wav.close()
            if 'enhanced_wav' in locals():
                enhanced_wav.close()
        
        # Shut down components
        stt_engine.shutdown()
        
        # Print final status
        print("\n" + "=" * 80)
        print("TCCC Audio Showcase Demo Complete")
        print("=" * 80)
        print(f"Raw audio saved to: {RAW_AUDIO_FILE}")
        print(f"Enhanced audio saved to: {ENHANCED_AUDIO_FILE}")
        print(f"Transcription saved to: {TRANSCRIPTION_FILE}")
        
        # Show performance stats
        if perf_stats["processing_times"]:
            avg_processing_time = sum(perf_stats["processing_times"]) / len(perf_stats["processing_times"])
            avg_latency = sum(perf_stats["latency"]) / len(perf_stats["latency"])
            print("\nPerformance Statistics:")
            print(f"Average processing time: {avg_processing_time*1000:.2f} ms")
            print(f"Average latency: {avg_latency:.2f} ms")
        
        # If benchmark was run, show results
        if os.path.exists(BENCHMARK_FILE):
            print(f"\nBenchmark results saved to: {BENCHMARK_FILE}")
    
    return 0

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TCCC Audio Showcase Demo")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="Microphone device ID")
    parser.add_argument("--no-visualization", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    # Override defaults with command-line arguments
    DEFAULT_DEVICE = args.device
    
    # Run the demo
    sys.exit(main())