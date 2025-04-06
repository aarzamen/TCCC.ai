#!/usr/bin/env python3
"""
Simple STT demo that shows real-time transcription from microphone input.
This script focuses on showing clear audio processing and transcription results.
"""

import os
import sys
import time
import json
import traceback
import numpy as np
import queue
import threading
import argparse
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description="Simple STT demo")
parser.add_argument("--mock", action="store_true", help="Use mock STT engine instead of real engine")
parser.add_argument("--device", type=int, help="Audio device index (default: auto-detect)")
parser.add_argument("--device-name", type=str, help="Audio device name (partial match, e.g., 'webcam')")
parser.add_argument("--list-devices", action="store_true", help="List all available audio devices and exit")
parser.add_argument("--duration", type=int, default=30, help="Demo duration in seconds (default: 30)")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

# Configure debug logging if requested
if args.debug:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set environment variables based on args
if args.mock:
    os.environ["USE_MOCK_STT"] = "1"
    print("Using mock STT engine (simulated responses)")
else:
    os.environ["USE_MOCK_STT"] = "0"
    print("Using real STT engine (Faster Whisper)")

# TCCC imports
from tccc.audio_pipeline.audio_pipeline import AudioPipeline
from tccc.stt_engine.stt_engine import STTEngine

# Function to list all available audio devices
def list_audio_devices():
    """
    List all available audio input devices and their properties.
    """
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print("\nAvailable audio input devices:")
        print("-" * 60)
        print(f"{'Index':<6} {'Channels':<10} {'Sample Rate':<15} {'Name':<30}")
        print("-" * 60)
        
        # Find all input devices
        input_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            # Only list devices with input channels
            if device_info.get('maxInputChannels', 0) > 0:
                input_devices.append({
                    'index': i,
                    'name': device_info.get('name', 'Unknown'),
                    'channels': device_info.get('maxInputChannels', 0),
                    'sample_rate': int(device_info.get('defaultSampleRate', 0))
                })
        
        # Print device information
        for device in input_devices:
            print(f"{device['index']:<6} {device['channels']:<10} {device['sample_rate']:<15} {device['name']:<30}")
        
        # Note about device selection
        print("\nTo use a specific device, run with:")
        print("  --device INDEX         # Use device by index number")
        print("  --device-name NAME    # Use device by name (partial match)")
        
        p.terminate()
        return True
    except Exception as e:
        print(f"\nError listing audio devices: {e}")
        return False

# Check if we should just list devices and exit
if args.list_devices:
    if list_audio_devices():
        sys.exit(0)
    else:
        sys.exit(1)

# Simple system class to handle connection between audio pipeline and STT
class SimpleSystem:
    def __init__(self, stt_engine, use_mock=False, debug=False):
        self.stt_engine = stt_engine
        self.audio_queue = queue.Queue(maxsize=100)
        self.transcription_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        self.use_mock = use_mock
        self.audio_count = 0
        self.debug = debug
    
    def enqueue_audio(self, audio_data):
        """Method called by AudioPipeline to forward audio data"""
        try:
            # Skip None or empty data
            if audio_data is None or len(audio_data) == 0:
                return
                
            # Calculate audio level
            if isinstance(audio_data, np.ndarray):
                level = np.abs(audio_data).mean() / 32768.0 * 100
            else:
                try:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    level = np.abs(audio_array).mean() / 32768.0 * 100
                except Exception as e:
                    logger.error(f"Error processing audio data: {e}, type={type(audio_data)}")
                    return
            
            # Always process the audio, regardless of level (don't apply a threshold)
            # This ensures we capture even very quiet audio signals
            try:
                self.audio_queue.put(audio_data, block=False)
                self.audio_count += 1
                
                # Show every 5th audio frame for diagnostics
                if self.audio_count % 5 == 0:
                    # Use different emoji indicators for audio levels
                    level_indicator = "🔴" if level < 0.1 else "🟡" if level < 0.5 else "🟢"
                    logger.info(f"Audio capture {level_indicator} #{self.audio_count}: level={level:.2f}%, queue={self.audio_queue.qsize()}")
                    
                    # Save raw audio samples periodically for debugging
                    if self.audio_count % 30 == 0 and self.debug:
                        os.makedirs("audio_samples", exist_ok=True)
                        if isinstance(audio_data, np.ndarray):
                            np.save(f"audio_samples/raw_audio_{self.audio_count}.npy", audio_data)
                        else:
                            # Convert to numpy array first if needed
                            try:
                                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                                np.save(f"audio_samples/raw_audio_{self.audio_count}.npy", audio_array)
                            except Exception as e:
                                logger.error(f"Could not save audio sample: {e}")
                                
            except queue.Full:
                logger.warning("Audio queue full, dropping frame")
                
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
        except Exception as e:
            logger.error(f"Error enqueueing audio: {e}")
    
    def start_processing(self):
        """Start processing audio data in a background thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
    
    def _processing_loop(self):
        """Background thread that processes audio data and gets transcriptions"""
        segment_count = 0
        while self.is_running:
            try:
                # Get audio data from the queue with a timeout
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Skip processing if no audio data
                if audio_data is None or len(audio_data) == 0:
                    logger.debug("Skipping empty audio data")
                    continue
                
                # Generate fake transcription if using mock mode
                if self.use_mock:
                    segment_count += 1
                    # Generate a mock transcription every ~10 audio chunks
                    if segment_count % 10 == 0:
                        mock_text = f"This is test audio segment {segment_count}"
                        self.transcription_queue.put({
                            'text': mock_text,
                            'timestamp': time.time()
                        })
                        logger.info(f"Generated mock transcription: {mock_text}")
                    continue
                    
                # Process through STT engine
                try:
                    # Log audio processing attempt
                    segment_count += 1
                    if segment_count % 5 == 0:
                        logger.info(f"Processing audio segment {segment_count}")
                    
                    # Ensure audio_data is correctly formatted for the STT engine
                    if not isinstance(audio_data, np.ndarray):
                        try:
                            audio_data = np.frombuffer(audio_data, dtype=np.int16)
                            logger.debug(f"Converted audio data to numpy array, shape={audio_data.shape}")
                        except Exception as e:
                            logger.error(f"Failed to convert audio data: {e}, type={type(audio_data)}")
                            continue
                    
                    # Check audio data statistics
                    mean_level = np.abs(audio_data).mean()
                    max_level = np.abs(audio_data).max()
                    
                    # Enhanced audio level reporting - make this visible for diagnostics
                    has_signal = max_level > 500  # Even quiet audio should exceed this
                    is_very_quiet = max_level < 1000
                    
                    # Report audio levels more prominently for diagnostics
                    if segment_count % 5 == 0:
                        status = "🔴 NO SIGNAL" if not has_signal else "🟡 QUIET" if is_very_quiet else "🟢 GOOD"
                        logger.info(f"AUDIO LEVELS: {status} - mean: {mean_level:.1f}, max: {max_level:.1f}")
                        
                        # Save a sample of the raw audio data periodically for debugging
                        if segment_count % 20 == 0:
                            try:
                                # Create diagnostic directory if needed
                                os.makedirs("audio_diagnostics", exist_ok=True)
                                # Save raw audio data for inspection
                                np.save(f"audio_diagnostics/audio_segment_{segment_count}.npy", audio_data)
                                logger.info(f"Saved diagnostic audio sample to audio_diagnostics/audio_segment_{segment_count}.npy")
                            except Exception as e:
                                logger.error(f"Error saving diagnostic audio: {e}")
                                
                    logger.debug(f"Audio stats - mean: {mean_level:.2f}, max: {max_level:.2f}, shape: {audio_data.shape}")
                    
                    # Normalize if needed (STT engine may expect float32 data in [-1.0, 1.0] range)
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Get transcription
                    logger.debug(f"Transcribing segment {segment_count}, shape={audio_data.shape}, dtype={audio_data.dtype}")
                    result = self.stt_engine.transcribe_segment(audio_data)
                    
                    # Debug result
                    if result:
                        logger.debug(f"Got transcription result: {result}")
                    else:
                        logger.debug("No transcription result returned")
                    
                    # If we got a result, put it in the transcription queue
                    if result and isinstance(result, dict) and 'text' in result and result['text'].strip():
                        self.transcription_queue.put({
                            'text': result['text'].strip(),
                            'timestamp': time.time()
                        })
                        logger.info(f"Got transcription: {result['text'].strip()}")
                except Exception as e:
                    logger.error(f"Error transcribing segment: {e}")
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")

async def run_stt_demo():
    """Run a simple STT demo showing real-time transcription from the microphone."""
    print("\n===== TCCC Real-Time Transcription Demo =====")
    
    # Initialize STT engine
    print("\nInitializing STT engine...")
    engine_type = "mock" if args.mock else "faster-whisper"
    stt_engine = STTEngine()
    
    # Check for CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if args.debug:
            print(f"CUDA Available: {cuda_available}")
            if cuda_available:
                device = torch.cuda.get_device_properties(0)
                print(f"CUDA Device: {device.name}, Compute Capability: {device.major}.{device.minor}")
    except ImportError:
        cuda_available = False
        if args.debug:
            print("PyTorch not installed or CUDA not available")
    
    # Flattened configuration structure based on ModelManager expectations
    stt_config = {
        # Model parameters directly at the root level
        "type": "faster-whisper",     # Explicitly use faster-whisper for potential CUDA optimization
        "model": "tiny.en",          # Model size as a direct string value
        "use_model_cache": True,     # Enable model caching
        
        # Hardware settings - auto-detect best options
        "enable_acceleration": True,  # Enable acceleration (CPU or GPU)
        "device": "cuda" if cuda_available else "cpu", # Use CUDA if available, otherwise CPU
        "compute_type": "float16" if cuda_available else "int8", # Use mixed precision on GPU, int8 on CPU
        "cpu_threads": 4,           # CPU threads for processing
        "cuda_device": 0,           # Use first CUDA device
        
        # Additional parameters
        "streaming": {
            "enabled": True,
            "max_context_length_sec": 30
        }
    }
    
    if args.debug:
        print(f"\nSTT configuration: {json.dumps(stt_config, indent=2)}")
    
    # Initialize STT engine
    print("Initializing STT engine...")
    await stt_engine.initialize(stt_config)
    print("STT engine initialized!")
    
    # Create our simple system to connect components
    system = SimpleSystem(stt_engine, use_mock=args.mock, debug=args.debug)
    
    # Initialize audio pipeline
    print("\nInitializing Audio Pipeline...")
    audio_pipeline = AudioPipeline()
    
    # Important: Set up system reference properly for the audio pipeline
    # This is what allows audio data to flow to our SimpleSystem.enqueue_audio method
    audio_pipeline.system = system
    
    # Print audio device info
    if not args.mock:
        import pyaudio
        p = pyaudio.PyAudio()
        print("\nAudio device information:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                print(f"Device {i}: {dev_info['name']}")
                print(f"  Input channels: {dev_info['maxInputChannels']}")
                print(f"  Sample rate: {int(dev_info['defaultSampleRate'])}")
        print(f"\nUsing device index: {args.device}")
        p.terminate()
    
    # Configure audio pipeline with enhanced device selection
    audio_input_config = {
        "name": "microphone",
        "type": "microphone"
    }
    
    # Apply device selection based on command line arguments
    if args.device is not None:
        # Use explicit device index if provided
        audio_input_config["device_index"] = args.device
        logger.info(f"Using explicit microphone device_index: {args.device}")
    elif args.device_name:
        # Use device name for more user-friendly selection
        audio_input_config["device_name"] = args.device_name
        logger.info(f"Looking for microphone with name: {args.device_name}")
    else:
        # Let the enhanced auto-detection in MicrophoneSource handle it
        logger.info("No specific device selected, will auto-detect best available microphone")
    
    audio_config = {
        "io": {
            "input_sources": [audio_input_config],
            "default_input": "microphone"
        },
        "processing": {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "enable_vad": False,
            "noise_reduction": True
        }
    }
    
    if args.debug:
        print(f"\nAudio configuration: {json.dumps(audio_config, indent=2)}")
    
    # Initialize audio pipeline
    print("Initializing audio pipeline...")
    await audio_pipeline.initialize(audio_config)
    print("Audio pipeline initialized!")
    
    # Set up timestamp for transcript output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_file = f"transcript_{timestamp}.txt"
    
    # Create output file
    with open(transcript_file, "w") as f:
        f.write("=== Transcription Results ===\n\n")
    
    # Start the audio processing
    system.start_processing()
    
    # Start the audio capture
    print("\nStarting microphone capture...")
    if not audio_pipeline.start_capture():
        print("Failed to start audio capture!")
        return
    
    print("\n\033[1m===== START SPEAKING NOW =====\033[0m")
    print("Say simple phrases clearly into the microphone")
    print("Press Ctrl+C to exit\n")
    
    last_text = ""
    end_time = time.time() + args.duration
    
    try:
        # Main loop to process transcriptions
        while time.time() < end_time:
            try:
                # Print audio processing status every few seconds
                if time.time() % 3 < 0.1:
                    audio_count = getattr(system, 'audio_count', 0)
                    queue_size = system.audio_queue.qsize()
                    print(f"Audio stats: {audio_count} chunks processed, queue size: {queue_size}")
                
                # Check for new transcriptions
                try:
                    trans = system.transcription_queue.get(block=False)
                    text = trans['text']
                    
                    # Only show if different from last text
                    if text != last_text:
                        last_text = text
                        
                        # Display with timestamp
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"\033[32m[{timestamp}]\033[0m \033[1m{text}\033[0m")
                        
                        # Save to transcript file
                        with open(transcript_file, "a") as f:
                            f.write(f"[{timestamp}] {text}\n")
                except queue.Empty:
                    # No new transcriptions, just wait
                    pass
                
                # Show time remaining
                remaining = int(end_time - time.time())
                if remaining % 5 == 0:
                    sys.stdout.write(f"\r\033[90mTime remaining: {remaining}s\033[0m")
                    sys.stdout.flush()
                
                # Short sleep to prevent CPU overuse
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    
    finally:
        # Clean up
        print("\nShutting down...")
        
        # Stop processing and audio capture
        system.stop_processing()
        audio_pipeline.stop_capture()
        
        # Shutdown components - note: shutdown() is not async
        stt_engine.shutdown()
        
        print(f"\nTranscript saved to: {transcript_file}")
        print("\nDemo completed.")

if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(run_stt_demo())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")