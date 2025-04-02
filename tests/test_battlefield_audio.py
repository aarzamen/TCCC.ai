#!/usr/bin/env python3
"""
Test script for the enhanced battlefield audio pipeline.

This script tests the audio pipeline's ability to handle
battlefield noise conditions and maintain high speech intelligibility.
"""

import os
import sys
import time
import wave
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import audio pipeline
from tccc.utils.config_manager import ConfigManager
from tccc.audio_pipeline import AudioPipeline
from tccc.utils.logging import get_logger

# Configure logging
logger = get_logger("battlefield_audio_test")

def save_audio_to_wav(audio_data, sample_rate, filename):
    """Save audio data to WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    logger.info(f"Saved audio to {filename}")

def process_test_file(test_file, output_folder, config_path=None):
    """Process a test audio file with the enhanced audio pipeline."""
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load config
    config_manager = ConfigManager()
    if config_path:
        config = config_manager.load_config_from_file(config_path)
    else:
        config = config_manager.load_config("audio_pipeline")
    
    # Create output filenames
    base_name = Path(test_file).stem
    processed_file = os.path.join(output_folder, f"{base_name}_processed.wav")
    
    # Initialize audio pipeline
    audio_pipeline = AudioPipeline()
    success = audio_pipeline.initialize(config)
    
    if not success:
        logger.error("Failed to initialize audio pipeline")
        return False
    
    # Update config to use the test file as source
    config["io"]["input_sources"] = [
        {
            "name": "test_file",
            "type": "file",
            "path": test_file,
            "loop": False
        }
    ]
    config["io"]["default_input"] = "test_file"
    
    # Collected processed audio chunks
    processed_chunks = []
    
    # Callback to capture processed audio
    def audio_callback(chunk):
        processed_chunks.append(chunk)
    
    # Configure audio pipeline to process test file
    source_name = "test_file"
    audio_pipeline.sources[source_name].file_path = test_file
    
    # Start capturing and processing
    logger.info(f"Processing audio file: {test_file}")
    
    try:
        # Get stream buffer to read processed audio
        stream_buffer = audio_pipeline.get_audio_stream()
        
        # Start processing
        audio_pipeline.start_capture(source_name)
        
        # Wait for processing to complete (with timeout)
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        # Read from stream buffer until we get no more data (timeout or EOF)
        while time.time() - start_time < max_wait_time:
            # Read a chunk from the buffer
            chunk = stream_buffer.read()
            
            # If we got data, add it to our collection
            if len(chunk) > 0:
                processed_chunks.append(chunk)
                # Reset timeout when we get data
                start_time = time.time()
            else:
                # Short sleep to avoid busy waiting
                time.sleep(0.01)
                
                # Check if processing is still running
                if not audio_pipeline.is_running:
                    logger.info("Processing completed")
                    break
        
        # Stop capturing and processing
        audio_pipeline.stop_capture()
        
        # Check if we collected any processed audio
        if not processed_chunks:
            logger.error("No processed audio collected")
            return False
        
        # Combine chunks and save to file
        combined_audio = np.concatenate(processed_chunks)
        save_audio_to_wav(combined_audio, config["audio"]["sample_rate"], processed_file)
        
        # Log processing stats
        stats = audio_pipeline.stats
        logger.info(f"Processed {stats['chunks_processed']} chunks")
        logger.info(f"Detected {stats['speech_chunks']} speech chunks")
        logger.info(f"Average processing time: {stats['average_processing_ms']:.2f}ms per chunk")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure we stop the pipeline
        if audio_pipeline.is_running:
            audio_pipeline.stop_capture()

def add_battlefield_noise(test_file, noise_type, output_folder):
    """Add simulated battlefield noise to a test file for more realistic testing."""
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the test file
    with wave.open(test_file, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
    
    # Normalize audio to float [-1, 1]
    audio_float = audio_data.astype(np.float32) / 32767.0
    
    # Generate noise based on type
    duration = len(audio_float) / sample_rate
    noise = np.zeros_like(audio_float)
    
    if noise_type == 'gunshot':
        # Simulate gunshots (sharp peaks)
        num_shots = int(duration / 3) + 1  # One shot every 3 seconds on average
        for _ in range(num_shots):
            # Random position
            pos = np.random.randint(0, len(noise) - sample_rate // 10)
            # Brief, high-intensity pulse
            shot_len = np.random.randint(sample_rate // 100, sample_rate // 40) 
            # Create shape with fast attack, slower decay
            shot = np.random.randn(shot_len) * 0.8  # Random noise
            # Apply envelope
            env = np.exp(-np.linspace(0, 10, len(shot)))
            shot = shot * env
            # Add to noise
            noise[pos:pos+len(shot)] += shot
    
    elif noise_type == 'explosion':
        # Simulate explosion (low rumble + broadband noise)
        num_explosions = max(1, int(duration / 10))  # One explosion every 10 seconds
        for _ in range(num_explosions):
            # Random position
            pos = np.random.randint(0, len(noise) - sample_rate)
            # Low frequency rumble
            explosion_len = np.random.randint(sample_rate, sample_rate * 3)
            if pos + explosion_len > len(noise):
                explosion_len = len(noise) - pos
            
            # Create explosion sound (filtered noise)
            from scipy import signal
            # Generate white noise
            explosion = np.random.randn(explosion_len)
            # Apply lowpass filter for rumble
            b, a = signal.butter(4, 150 / (sample_rate/2), 'lowpass')
            rumble = signal.lfilter(b, a, explosion)
            # Apply envelope
            env = np.exp(-np.linspace(0, 5, len(rumble)))
            rumble = rumble * env * 0.7
            # Add to noise
            noise[pos:pos+len(rumble)] += rumble
    
    elif noise_type == 'vehicle':
        # Simulate vehicle noise (continuous low-frequency)
        # Create engine rumble
        from scipy import signal
        # Base frequencies for engine
        engine_freq = 80  # Hz
        # Generate sine wave
        t = np.linspace(0, duration, len(audio_float))
        engine = 0.3 * np.sin(2 * np.pi * engine_freq * t)
        # Add harmonics
        engine += 0.15 * np.sin(2 * np.pi * engine_freq * 2 * t)
        engine += 0.1 * np.sin(2 * np.pi * engine_freq * 3 * t)
        # Add some noise
        engine += 0.1 * np.random.randn(len(engine))
        # Apply lowpass filter
        b, a = signal.butter(4, 300 / (sample_rate/2), 'lowpass')
        engine = signal.lfilter(b, a, engine)
        # Add to noise
        noise += engine
    
    elif noise_type == 'wind':
        # Simulate wind noise (filtered noise with time-varying intensity)
        from scipy import signal
        # Generate base noise
        wind = np.random.randn(len(audio_float))
        # Apply bandpass filter
        b, a = signal.butter(4, [20 / (sample_rate/2), 200 / (sample_rate/2)], 'bandpass')
        wind = signal.lfilter(b, a, wind)
        # Apply time-varying intensity (gusts)
        t = np.linspace(0, duration, len(audio_float))
        gust_envelope = 0.3 + 0.2 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.sin(2 * np.pi * 0.05 * t)
        wind = wind * gust_envelope
        # Add to noise
        noise += wind * 0.4
    
    elif noise_type == 'mixed':
        # Create a mix of battlefield noises
        # Add some vehicle noise as background
        t = np.linspace(0, duration, len(audio_float))
        engine = 0.15 * np.sin(2 * np.pi * 80 * t) + 0.05 * np.sin(2 * np.pi * 160 * t)
        
        # Add some filtered wind noise
        from scipy import signal
        wind = np.random.randn(len(audio_float)) * 0.2
        b, a = signal.butter(4, [20 / (sample_rate/2), 200 / (sample_rate/2)], 'bandpass')
        wind = signal.lfilter(b, a, wind)
        
        # Add occasional gunshots
        num_shots = int(duration / 5) + 1
        for _ in range(num_shots):
            pos = np.random.randint(0, len(noise) - sample_rate // 10)
            shot_len = np.random.randint(sample_rate // 100, sample_rate // 40)
            shot = np.random.randn(shot_len)
            env = np.exp(-np.linspace(0, 10, len(shot)))
            shot = shot * env * 0.7
            if pos + len(shot) < len(noise):
                noise[pos:pos+len(shot)] += shot
        
        # Add one explosion
        if len(noise) > sample_rate * 2:
            pos = np.random.randint(0, len(noise) - sample_rate * 2)
            explosion_len = int(sample_rate * 1.5)
            explosion = np.random.randn(explosion_len)
            b, a = signal.butter(4, 150 / (sample_rate/2), 'lowpass')
            rumble = signal.lfilter(b, a, explosion)
            env = np.exp(-np.linspace(0, 5, len(rumble)))
            rumble = rumble * env * 0.5
            noise[pos:pos+len(rumble)] += rumble
        
        # Combine all noise types
        noise += engine + wind
    
    # Mix noise with original audio (adjustable level)
    noise_level = 0.7
    noisy_audio = audio_float + noise * noise_level
    
    # Clipping prevention
    noisy_audio = np.clip(noisy_audio, -0.99, 0.99)
    
    # Convert back to int16
    noisy_audio_int16 = (noisy_audio * 32767).astype(np.int16)
    
    # Save noisy audio
    output_file = os.path.join(output_folder, f"{Path(test_file).stem}_with_{noise_type}_noise.wav")
    save_audio_to_wav(noisy_audio_int16, sample_rate, output_file)
    
    return output_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test enhanced battlefield audio pipeline")
    parser.add_argument("--input", "-i", type=str, default="test_data/test_speech.wav",
                      help="Input audio file")
    parser.add_argument("--output", "-o", type=str, default="test_data/output",
                      help="Output folder for processed files")
    parser.add_argument("--noise", "-n", type=str, default="none",
                      choices=["none", "gunshot", "explosion", "vehicle", "wind", "mixed"],
                      help="Type of battlefield noise to add")
    parser.add_argument("--config", "-c", type=str, default=None,
                      help="Configuration file path (optional)")
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Add noise if requested
    input_file = args.input
    if args.noise != "none":
        logger.info(f"Adding {args.noise} noise to the input file")
        input_file = add_battlefield_noise(args.input, args.noise, args.output)
    
    # Process the file
    logger.info(f"Processing file: {input_file}")
    success = process_test_file(input_file, args.output, args.config)
    
    if success:
        logger.info("Test completed successfully")
        return 0
    else:
        logger.error("Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())