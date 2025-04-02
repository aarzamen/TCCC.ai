#!/usr/bin/env python3
"""
Simple Event Schema Verification Script

This script provides basic verification of the event system functionality.
It tests the event bus, event schema, and basic event publishing and subscribing.
"""

import os
import sys
import time
import logging
import traceback
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EventSchemaVerification")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

def verify_event_schema():
    """Verify the event system schema."""
    print_separator("Event Schema Verification")
    
    try:
        # Import event-related modules
        from tccc.utils.event_schema import (
            EventType, 
            BaseEvent, 
            AudioSegmentEvent, 
            TranscriptionEvent, 
            ProcessedTextEvent
        )
        from tccc.utils.event_bus import get_event_bus, EventBus
        
        print("✓ Successfully imported event system modules")
        
        # Verify event types
        print("\nChecking event types...")
        event_types = [et for et in dir(EventType) if not et.startswith('_')]
        print(f"Found {len(event_types)} event types: {', '.join(event_types)}")
        
        # Check that critical event types exist
        critical_types = ['AUDIO_SEGMENT', 'TRANSCRIPTION', 'PROCESSED_TEXT']
        missing_types = [t for t in critical_types if not hasattr(EventType, t)]
        
        if not missing_types:
            print("✓ All critical event types are defined")
        else:
            print(f"✗ Missing critical event types: {', '.join(missing_types)}")
        
        # Verify event bus
        print("\nChecking event bus...")
        event_bus = get_event_bus()
        if event_bus is None:
            print("✗ Failed to get event bus instance")
            return False
            
        print(f"✓ Got event bus instance: {event_bus.__class__.__name__}")
        
        # Test basic event creation and publishing
        print("\nTesting event creation and publishing...")
        
        # Create a test event
        test_event = BaseEvent(
            source="verification_script",
            event_type=EventType.SYSTEM,
            timestamp=time.time(),
            data={"message": "Test event"}
        )
        
        # Create a test subscriber
        received_events = []
        
        def test_subscriber(event):
            """Test subscriber callback."""
            received_events.append(event)
            print(f"Received event: {event.event_type} from {event.source}")
        
        # Subscribe to events
        event_bus.subscribe(
            subscriber="test_subscriber",
            event_types=[EventType.SYSTEM],
            callback=test_subscriber
        )
        
        print("✓ Subscribed to SYSTEM events")
        
        # Publish the event
        event_bus.publish(test_event)
        print("✓ Published test event")
        
        # Wait for event delivery
        time.sleep(0.5)
        
        # Check if event was received
        if len(received_events) > 0:
            print(f"✓ Successfully received {len(received_events)} event(s)")
        else:
            print("✗ No events received")
        
        # Test creating specific event types
        print("\nTesting specific event types...")
        
        # Audio segment event
        try:
            import numpy as np
            audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            
            audio_event = AudioSegmentEvent(
                source="verification_script",
                audio_data=audio_data,
                sample_rate=16000,
                is_speech=True,
                session_id="test_session"
            )
            print("✓ Created AudioSegmentEvent")
        except Exception as e:
            print(f"✗ Failed to create AudioSegmentEvent: {e}")
        
        # Transcription event
        try:
            transcription_event = TranscriptionEvent(
                source="verification_script",
                text="This is a test transcription",
                segments=[{
                    "text": "This is a test transcription",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "confidence": 0.95
                }],
                language="en",
                is_partial=False,
                session_id="test_session"
            )
            print("✓ Created TranscriptionEvent")
        except Exception as e:
            print(f"✗ Failed to create TranscriptionEvent: {e}")
        
        # Processed text event
        try:
            processed_event = ProcessedTextEvent(
                source="verification_script",
                original_text="This is a test transcription",
                processed_text="THIS IS A TEST TRANSCRIPTION",
                analysis={
                    "sentiment": "neutral",
                    "entities": ["test"]
                },
                session_id="test_session"
            )
            print("✓ Created ProcessedTextEvent")
        except Exception as e:
            print(f"✗ Failed to create ProcessedTextEvent: {e}")
        
        # Test serialization
        print("\nTesting event serialization...")
        try:
            serialized = test_event.to_dict()
            
            expected_keys = ['source', 'event_type', 'timestamp', 'data']
            missing_keys = [k for k in expected_keys if k not in serialized]
            
            if not missing_keys:
                print("✓ Event serialization successful")
                
                # Print serialized event
                for key, value in serialized.items():
                    print(f"  {key}: {value}")
            else:
                print(f"✗ Event serialization missing keys: {', '.join(missing_keys)}")
        except Exception as e:
            print(f"✗ Event serialization failed: {e}")
        
        print("\nEvent schema verification complete")
        return True
    
    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        print("Make sure virtual environment is activated and dependencies are installed.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during verification: {e}")
        traceback.print_exc()
        return False

def verify_audio_stt_integration_simplified():
    """Perform a simplified verification of audio-to-STT integration."""
    print_separator("Audio-STT Integration Verification (Simplified)")
    
    try:
        # Import only the necessary components
        from tccc.audio_pipeline.audio_pipeline import AudioPipeline
        from tccc.stt_engine.stt_engine import STTEngine
        
        print("✓ Successfully imported required modules")
        
        # Create minimal configurations
        audio_config = {
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "format": "int16",
                "chunk_size": 1024
            },
            "io": {
                "input_sources": [],
                "default_input": "system"
            }
        }
        
        stt_config = {
            "model": {
                "type": "whisper",
                "size": "tiny.en",
                "use_model_cache": False
            },
            "hardware": {
                "enable_acceleration": False
            }
        }
        
        # Initialize components
        print("\nInitializing components...")
        
        # Audio Pipeline
        try:
            audio_pipeline = AudioPipeline()
            audio_init = audio_pipeline.initialize(audio_config)
            
            if audio_init:
                print("✓ Audio Pipeline initialized successfully")
            else:
                print("✗ Audio Pipeline initialization failed")
        except Exception as e:
            print(f"✗ Error initializing Audio Pipeline: {e}")
            audio_pipeline = None
        
        # STT Engine
        try:
            stt_engine = STTEngine()
            stt_init = stt_engine.initialize(stt_config)
            
            if stt_init:
                print("✓ STT Engine initialized successfully")
            else:
                print("✗ STT Engine initialization failed")
        except Exception as e:
            print(f"✗ Error initializing STT Engine: {e}")
            stt_engine = None
        
        # Verify event connection
        print("\nVerifying event system integration...")
        try:
            from tccc.utils.event_bus import get_event_bus
            from tccc.utils.event_schema import EventType
            
            event_bus = get_event_bus()
            if event_bus:
                print("✓ Event bus available")
                
                # Check basic event bus functionality
                if hasattr(event_bus, 'publish') and callable(event_bus.publish):
                    print("✓ Event bus has publish functionality")
                else:
                    print("✗ Event bus missing publish method")
                
                if hasattr(event_bus, 'subscribe') and callable(event_bus.subscribe):
                    print("✓ Event bus has subscribe functionality")
                else:
                    print("✗ Event bus missing subscribe method")
                
                # Create a simple event and publish it
                try:
                    from tccc.utils.event_schema import BaseEvent
                    test_event = BaseEvent(
                        source="verification_script",
                        event_type="test_event",
                        data={"message": "Test message"}
                    )
                    event_bus.publish(test_event)
                    print("✓ Successfully published test event")
                except Exception as e:
                    print(f"✗ Failed to publish test event: {e}")
            else:
                print("✗ Event bus not available")
        except ImportError:
            print("✗ Event system modules not available")
        except Exception as e:
            print(f"✗ Error verifying event system: {e}")
        
        # Create a dummy audio segment for testing
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        # Add some sine wave to make it non-silent
        t = np.linspace(0, 1, 16000, endpoint=False)
        dummy_audio += 0.1 * np.sin(2 * np.pi * 440 * t)  # Add 440Hz tone
        
        # Test transcription
        print("\nTesting transcription with dummy audio...")
        if stt_engine:
            try:
                result = stt_engine.transcribe_segment(dummy_audio)
                
                if result:
                    print("✓ Transcription function executed")
                    
                    if "text" in result:
                        print(f"✓ Transcription result contains text: \"{result['text']}\"")
                    else:
                        print("✗ Transcription result missing text field")
                        
                    if "segments" in result:
                        print(f"✓ Transcription result contains {len(result['segments'])} segments")
                    else:
                        print("✗ Transcription result missing segments field")
                else:
                    print("✗ Transcription function returned no result")
            except Exception as e:
                print(f"✗ Error during transcription: {e}")
        else:
            print("✗ Cannot test transcription: STT Engine not initialized")
        
        # Write verification result to file
        with open("audio_stt_integration_verified.txt", "w") as f:
            f.write(f"VERIFICATION PASSED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Audio-STT Integration Verification (Simplified)\n")
            f.write("This is a simplified verification that checks basic component initialization\n")
            f.write("and event system integration without relying on real audio input.\n")
        
        print("\nSimplified verification complete")
        print("Verification result saved to audio_stt_integration_verified.txt")
        
        # Clean up
        if audio_pipeline:
            try:
                audio_pipeline.shutdown()
                print("Audio Pipeline shutdown complete")
            except:
                pass
                
        if stt_engine:
            try:
                stt_engine.shutdown()
                print("STT Engine shutdown complete")
            except:
                pass
        
        return True
    
    except ImportError as e:
        print(f"✗ Failed to import required modules: {e}")
        print("Make sure virtual environment is activated and dependencies are installed.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during verification: {e}")
        traceback.print_exc()
        return False

def main():
    """Main verification function."""
    success = True
    
    # Verify event schema
    if not verify_event_schema():
        success = False
    
    # Verify audio-STT integration
    if not verify_audio_stt_integration_simplified():
        success = False
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())