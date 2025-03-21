#!/usr/bin/env python3
"""
End-to-end verification script for Audio Pipeline to STT integration.

This script provides a comprehensive test of the audio-to-transcription workflow,
testing both the Mock STT engine and the real Faster Whisper-based engine.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("E2EVerification")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

def main():
    """Main verification function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Verify audio to STT integration")
    parser.add_argument("--mock", action="store_true", help="Use mock STT engine")
    parser.add_argument("--file", action="store_true", help="Use file input instead of microphone")
    parser.add_argument("--model", choices=["tiny", "tiny.en", "base", "small", "medium", "large"], 
                        default="tiny.en", help="Model size for faster-whisper")
    parser.add_argument("--cache", action="store_true", help="Test model caching")
    args = parser.parse_args()
    
    print_separator("Audio STT End-to-End Verification")
    
    # Import components after setting environment variables as needed
    if args.mock:
        os.environ["USE_MOCK_STT"] = "1"
        print("Using mock STT engine")
    else:
        os.environ["USE_MOCK_STT"] = "0"
        print(f"Using real STT engine with model size: {args.model}")
    
    try:
        from tccc.audio_pipeline.audio_pipeline import AudioPipeline
        from tccc.stt_engine.stt_engine import STTEngine
        if not args.mock:
            from tccc.stt_engine.faster_whisper_stt import FasterWhisperSTT
            if args.cache:
                try:
                    from tccc.stt_engine.model_cache_manager import get_model_cache_manager, ModelCacheManager
                    MODEL_CACHE_AVAILABLE = True
                except ImportError:
                    MODEL_CACHE_AVAILABLE = False
                    print("WARNING: Model cache manager not available")
    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("Make sure virtual environment is activated and dependencies are installed.")
        return 1
    
    print_separator("Environment Setup")
    # Check virtual environment
    venv_active = True if os.environ.get('VIRTUAL_ENV') else False
    print(f"Virtual environment active: {venv_active}")
    
    # Ensure test directory exists
    test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_dir, exist_ok=True)
    
    # Test file path
    test_file_path = os.path.join(test_dir, 'test_speech.wav')
    
    # Create test file if it doesn't exist and file input is requested
    if args.file and not os.path.exists(test_file_path):
        create_test_wav_file(test_file_path)
    
    print_separator("Initializing Components")
    
    # Initialize Audio Pipeline
    audio_pipeline = AudioPipeline()
    audio_config = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "int16",
            "chunk_size": 1024
        },
        "io": {
            "input_sources": [
                {
                    "name": "test_file",
                    "type": "file",
                    "path": test_file_path
                }
            ],
            "default_input": "test_file" if args.file else "system"
        }
    }
    audio_init = audio_pipeline.initialize(audio_config)
    print(f"Audio Pipeline initialized: {audio_init}")
    
    # Initialize STT Engine
    stt_engine = STTEngine()
    stt_config = {
        "model": {
            "type": "whisper",
            "size": args.model,
            "use_model_cache": args.cache  # Enable caching if requested
        },
        "hardware": {
            "enable_acceleration": True  # Enable GPU if available
        }
    }
    stt_init = stt_engine.initialize(stt_config)
    print(f"STT Engine initialized: {stt_init}")
    
    if not audio_init or not stt_init:
        print("ERROR: Failed to initialize components")
        return 1
    
    # Handle model caching test if requested
    if args.cache and not args.mock and MODEL_CACHE_AVAILABLE:
        print_separator("Model Cache Testing")
        # Check cache manager
        cache_available = False
        try:
            cache_manager = get_model_cache_manager()
            cache_available = True
            print(f"Model cache manager available: {cache_available}")
            
            # Get initial cache status
            cache_status = cache_manager.get_status()
            print(f"Initial cache status: {cache_status['cache_size']} models cached")
            
            # Create a second STT engine with same config to test caching
            print("\nCreating second STT engine instance (should use cached model)...")
            stt_engine2 = STTEngine()
            
            start_time = time.time()
            stt_engine2.initialize(stt_config)
            init_time = time.time() - start_time
            
            print(f"Second engine initialized in {init_time:.2f} seconds")
            
            # Get updated cache status
            cache_status = cache_manager.get_status()
            print(f"Updated cache status: {cache_status['cache_size']} models cached")
            
            # Check if cache is working
            if cache_status['cache_size'] > 0:
                print("✓ Model caching is working correctly")
            else:
                print("✗ Model cache appears to be empty")
                
            # Clean up second engine
            stt_engine2.shutdown()
            print("Second engine shutdown complete")
            
        except Exception as e:
            print(f"Error testing model cache: {e}")
    
    print_separator("Audio Capture Test")
    
    # Start audio capture
    audio_pipeline.start_capture()
    print("Started audio capture")
    
    # Wait for audio processing to start
    time.sleep(2)
    
    # Get audio segment
    try:
        audio_segment = None
        if hasattr(audio_pipeline, 'get_audio'):
            audio_segment = audio_pipeline.get_audio()
        elif hasattr(audio_pipeline, 'get_audio_segment'):
            audio_segment = audio_pipeline.get_audio_segment()
        
        # Ensure audio_segment is not None and has content
        if audio_segment is not None and len(audio_segment) > 0:
            print(f"✓ Successfully captured audio: {len(audio_segment)} samples")
            print(f"  Audio type: {type(audio_segment)}, dtype: {audio_segment.dtype}")
            
            # Verify that audio contains non-zero data
            if isinstance(audio_segment, np.ndarray) and np.any(audio_segment != 0):
                print("  Audio contains valid signal (non-zero values detected)")
            else:
                print("  Warning: Audio may contain only zeros or silence")
        else:
            print("✗ Failed to capture audio")
            # Create dummy audio segment as fallback
            audio_segment = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            print("  Created fallback audio segment for testing")
    except Exception as e:
        print(f"Error capturing audio: {e}")
        # Create dummy audio segment as fallback
        audio_segment = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        print("  Created fallback audio segment after error")
        
    # Stop audio capture
    audio_pipeline.stop_capture()
    print("Stopped audio capture")
    
    print_separator("Transcription Test")
    
    # Create a new audio segment for testing
    if args.file:
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(test_file_path)
            
            # Convert to mono if needed
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)
                
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                try:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
                except ImportError:
                    # Simple resampling fallback
                    audio = np.interp(
                        np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)), 
                        np.arange(len(audio)), 
                        audio
                    )
            
            print(f"Loaded test audio file: {len(audio)} samples at 16kHz")
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            # Create synthetic audio as fallback
            audio = create_sine_wave(3, 16000)
            print("Created synthetic audio instead")
    else:
        # Use the captured audio if available, otherwise create synthetic audio
        if audio_segment is not None and len(audio_segment) > 0:
            audio = audio_segment
            print("Using captured audio for transcription")
        else:
            # Create synthetic audio
            audio = create_sine_wave(3, 16000)
            print("Created synthetic audio for testing")
    
    # Ensure audio is in float32 format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize audio if needed
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    
    # Measure transcription time
    start_time = time.time()
    result = stt_engine.transcribe_segment(audio)
    transcription_time = time.time() - start_time
    
    # Check result
    if result and "text" in result:
        text = result["text"]
        print(f"✓ Successfully transcribed audio in {transcription_time:.2f} seconds")
        print(f"  Transcription: \"{text}\"")
        
        # Print segment information if available
        if "segments" in result and result["segments"]:
            print(f"  Segments: {len(result['segments'])}")
            print(f"  First segment: {result['segments'][0]['text']}")
    else:
        print("✗ Failed to transcribe audio")
        if "error" in result:
            print(f"  Error: {result['error']}")
        
    print_separator("STT Engine Status")
    
    # Get STT engine status
    status = stt_engine.get_status()
    
    # Print relevant status information
    if 'model' in status:
        model_status = status['model']
        print(f"Model type: {model_status.get('model_type', 'unknown')}")
        print(f"Model size: {model_status.get('model_size', 'unknown')}")
    else:
        print("Model status not available")
    
    if 'metrics' in status:
        perf = status['metrics']
        print(f"Performance metrics:")
        print(f"  Transcripts: {perf.get('transcript_count', 0)}")
        print(f"  Real-time factor: {perf.get('real_time_factor', 0):.2f}x")
        
    # Print cache information if available
    if args.cache and 'cache_manager' in status:
        cache = status['cache_manager']
        print(f"Cache status:")
        print(f"  Cached models: {cache.get('cached_models_count', 0)}")
        print(f"  Max cache size: {cache.get('max_cache_size', 0)}")
        print(f"  Memory usage: {cache.get('memory_usage', 0):.2f}%")
        
    print_separator("Event Integration Test")
    
    # Test event system integration
    try:
        from tccc.utils.event_bus import get_event_bus
        from tccc.utils.event_schema import EventType
        
        event_bus = get_event_bus()
        if event_bus:
            print("Testing STT-Event system integration...")
            
            # Check for event bus methods and use appropriate ones
            if hasattr(event_bus, 'get_subscribers_for_event_type'):
                print("Using direct subscriber lookup")
                # Check if STT engine is subscribed to audio events
                subscriptions = event_bus.get_subscribers_for_event_type(EventType.AUDIO_SEGMENT)
                if "stt_engine" in subscriptions:
                    print("✓ STT Engine is subscribed to audio events")
                else:
                    print("✗ STT Engine is not subscribed to audio events")
                    
                # Check for transcription event handlers
                transcription_subscribers = event_bus.get_subscribers_for_event_type(EventType.TRANSCRIPTION)
                if transcription_subscribers:
                    print(f"✓ Transcription events have {len(transcription_subscribers)} subscribers")
                    print(f"  Subscribers: {', '.join(transcription_subscribers)}")
                else:
                    print("✗ No subscribers for transcription events")
            elif hasattr(event_bus, '_subscribers'):
                print("Using internal subscribers dictionary")
                # Direct access to internal state (less ideal but works for testing)
                subscribers = event_bus._subscribers
                
                # Check if STT engine is subscribed to audio events
                audio_subscribed = False
                for event_type, subs in subscribers.items():
                    if event_type == EventType.AUDIO_SEGMENT or (isinstance(event_type, str) and event_type == 'audio_segment'):
                        if 'stt_engine' in subs:
                            audio_subscribed = True
                            break
                
                if audio_subscribed:
                    print("✓ STT Engine is subscribed to audio events")
                else:
                    print("✗ STT Engine is not subscribed to audio events")
                    
                # Check for transcription event handlers
                transcription_subs = []
                for event_type, subs in subscribers.items():
                    if event_type == EventType.TRANSCRIPTION or (isinstance(event_type, str) and event_type == 'transcription'):
                        transcription_subs = list(subs.keys())
                        break
                
                if transcription_subs:
                    print(f"✓ Transcription events have {len(transcription_subs)} subscribers")
                    print(f"  Subscribers: {', '.join(transcription_subs)}")
                else:
                    print("✗ No subscribers for transcription events")
            else:
                # Fallback test method using available information
                print("Using basic event bus verification")
                # Check that event bus exists and is initialized
                if hasattr(event_bus, 'publish') and callable(event_bus.publish):
                    print("✓ Event bus is initialized with publish method")
                    
                    if hasattr(event_bus, 'subscribe') and callable(event_bus.subscribe):
                        print("✓ Event bus has subscribe method")
                    else:
                        print("✗ Event bus missing subscribe method")
                else:
                    print("✗ Event bus does not have basic required methods")
            
            print("Event integration test complete")
        else:
            print("Event bus not available for testing")
            
    except ImportError:
        print("Event bus or schema not available, skipping event integration test")
    except Exception as e:
        print(f"Error testing event integration: {e}")
    
    print_separator("Shutdown")
    
    # Shutdown components
    stt_shutdown = stt_engine.shutdown()
    print(f"STT Engine shutdown: {stt_shutdown}")
    
    # Write verification result to filesystem
    try:
        success = result and "text" in result and result["text"]
        with open("audio_stt_integration_verified.txt", "w") as f:
            if success:
                f.write(f"VERIFICATION PASSED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Transcription: {result['text']}\n")
            else:
                f.write(f"VERIFICATION FAILED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Could not successfully transcribe audio\n")
                if "error" in result:
                    f.write(f"Error: {result['error']}\n")
        
        print(f"Verification {'succeeded' if success else 'failed'}")
        print(f"Result saved to audio_stt_integration_verified.txt")
    except Exception as e:
        print(f"Failed to write verification result: {e}")
    
    print("\nVerification complete!")
    return 0 if result and "text" in result and result["text"] else 1
    
def create_test_wav_file(file_path, duration=3.0, sample_rate=16000):
    """Create a test WAV file with a speech-like sine wave."""
    try:
        import wave
        import struct
        
        # Create audio data
        audio = create_sine_wave(duration, sample_rate)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(file_path, 'w') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            
            # Convert samples to bytes
            sample_bytes = audio_int16.tobytes()
            wf.writeframes(sample_bytes)
            
        print(f"Created test WAV file: {file_path}")
        print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")
        return True
        
    except Exception as e:
        print(f"Error creating test WAV file: {e}")
        return False

def create_sine_wave(duration, sample_rate):
    """Create a sine wave audio signal."""
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create sine wave with frequency sweep
    frequencies = np.linspace(200, 1000, len(t))
    audio = 0.5 * np.sin(2 * np.pi * frequencies * t / sample_rate)
    
    # Add some noise
    audio += 0.01 * np.random.normal(size=len(audio))
    
    return audio.astype(np.float32)

if __name__ == "__main__":
    sys.exit(main())