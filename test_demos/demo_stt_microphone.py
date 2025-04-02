#!/usr/bin/env python3
"""
Demo script for STT Engine using microphone input.

This script demonstrates the TCCC.ai STT engine with real-time microphone input.
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path
import logging
import numpy as np

# Set up proper path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TCCC.Demo")

# Import TCCC modules
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine
from tccc.utils.config import Config

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")

def process_audio_callback(audio_data, stt_engine):
    """
    Process audio data and display transcription results.
    
    Args:
        audio_data: Audio data as numpy array
        stt_engine: STT engine instance
    """
    # Process audio with STT engine
    result = stt_engine.transcribe_segment(audio_data)
    
    # Print results if there's text
    if result and 'text' in result and result['text'].strip():
        # Clear line and print the text
        print("\r" + " " * 80, end="\r")  # Clear line
        print(f"\rTranscription: {result['text']}")
        
        # Print metrics if available
        if 'metrics' in result:
            metrics = result['metrics']
            rtf = metrics.get('real_time_factor', 0)
            print(f"  [RTF: {rtf:.2f}x]", end="")
    
    # Always return True to continue processing
    return True

def main():
    """Main demo function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Demo STT Engine with microphone input")
    parser.add_argument("--engine", choices=["mock", "faster-whisper", "whisper"], default="mock",
                        help="STT engine type to use for demo")
    parser.add_argument("--list-microphones", action="store_true", help="List available microphones and exit")
    parser.add_argument("--device", type=int, default=0, help="Microphone device ID to use")
    args = parser.parse_args()
    
    # Handle microphone listing request
    if args.list_microphones:
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            print_separator("Available Audio Devices")
            info = p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            for i in range(numdevices):
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    print(f"Device ID {i}: {device_info.get('name')}")
            
            p.terminate()
            return 0
        except ImportError:
            print("PyAudio not installed. Cannot list microphones.")
            return 1
        except Exception as e:
            print(f"Error listing microphones: {e}")
            return 1
    
    print_separator("STT Microphone Demo")
    print(f"Selected STT engine: {args.engine}")
    print(f"Using microphone device ID: {args.device}")
    
    # Configure mock functionality if needed
    if args.engine == "mock":
        os.environ["USE_MOCK_STT"] = "1"
    else:
        os.environ["USE_MOCK_STT"] = "0"
    
    # Load configurations
    try:
        # Load YAML files manually since the load_yaml method is not a class method
        with open(os.path.join(os.path.dirname(__file__), 'config', 'stt_engine.yaml'), 'r') as f:
            stt_config = yaml.safe_load(f)
        
        with open(os.path.join(os.path.dirname(__file__), 'config', 'audio_pipeline.yaml'), 'r') as f:
            audio_config = yaml.safe_load(f)
        
        # Modify config for demo (use smaller model for faster results)
        if args.engine != "mock":
            stt_config['model']['size'] = 'tiny'  # Use smallest model for demo
        
        # Set microphone device ID
        if 'io' not in audio_config:
            audio_config['io'] = {}
        if 'input_sources' not in audio_config['io']:
            audio_config['io']['input_sources'] = []
        
        # Add or update microphone source with specified device ID
        mic_source_found = False
        for source in audio_config['io']['input_sources']:
            if source.get('type') == 'microphone':
                source['device_id'] = args.device
                mic_source_found = True
                break
        
        if not mic_source_found:
            audio_config['io']['input_sources'].append({
                'name': 'microphone',
                'type': 'microphone',
                'device_id': args.device
            })
        
        # Set default input to microphone
        audio_config['io']['default_input'] = 'microphone'
        
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Initialize components
    try:
        # Create STT Engine
        print("Initializing STT Engine...")
        stt_engine = create_stt_engine(args.engine, stt_config)
        if not stt_engine.initialize(stt_config):
            print("Failed to initialize STT Engine")
            return 1
        print("STT Engine initialized")
        
        # Create Audio Pipeline
        print("Initializing Audio Pipeline...")
        audio_pipeline = AudioPipeline()
        if not audio_pipeline.initialize(audio_config):
            print("Failed to initialize Audio Pipeline")
            return 1
        print("Audio Pipeline initialized")
        
        # Display audio source info
        sources = audio_pipeline.get_available_sources()
        print("\nAvailable audio sources:")
        for source in sources:
            print(f"  - {source['name']} ({source['type']})")
        
        # Start demo
        print_separator("Starting Demo")
        print("Speak into your microphone...\n")
        print("Press Ctrl+C to stop the demo")
        
        # Buffer for audio data
        audio_buffer = []
        
        # Define wrapper function for the callback
        def audio_callback(audio_data):
            # Start processing after receiving some data to avoid false starts
            if len(audio_buffer) < 3:
                audio_buffer.append(audio_data)
                return
            
            # Process the audio with STT
            process_audio_callback(audio_data, stt_engine)
        
        # Start audio capture with the callback
        # Get the first microphone source from available sources
        mic_source = None
        for source in sources:
            if source['type'] == 'microphone':
                mic_source = source['name']
                break
        
        if not mic_source:
            print("No microphone source found")
            return 1
            
        print(f"Using microphone source: {mic_source}")
        if not audio_pipeline.start_capture(mic_source):
            print("Failed to start audio capture")
            return 1
        
        # Main loop
        try:
            while True:
                # Get audio from pipeline
                try:
                    # Get the audio stream directly
                    audio_stream = audio_pipeline.get_audio_stream()
                    if audio_stream:
                        audio_data = audio_stream.read()
                        if audio_data is not None and len(audio_data) > 0:
                            audio_callback(audio_data)
                except Exception as e:
                    print(f"\rError getting audio: {e}", end="")
                
                # Sleep to avoid tight loop
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping demo...")
        finally:
            # Stop audio capture
            audio_pipeline.stop_capture()
        
        # Print final status
        print_separator("Demo Complete")
        status = stt_engine.get_status()
        
        if 'metrics' in status:
            metrics = status['metrics']
            print(f"Metrics:")
            print(f"  Total audio processed: {metrics.get('total_audio_seconds', 0):.2f}s")
            print(f"  Total processing time: {metrics.get('total_processing_time', 0):.2f}s")
            print(f"  Transcripts generated: {metrics.get('transcript_count', 0)}")
            print(f"  Errors: {metrics.get('error_count', 0)}")
            print(f"  Average confidence: {metrics.get('avg_confidence', 0):.2f}")
            print(f"  Average real-time factor: {metrics.get('real_time_factor', 0):.2f}x")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())