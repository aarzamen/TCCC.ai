#!/usr/bin/env python3
"""
Debug STT Engine implementation
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Force using actual implementation
os.environ["USE_MOCK_STT"] = "0"
os.environ["USE_FASTER_WHISPER"] = "1"

# Import modules
from tccc.stt_engine import create_stt_engine
from faster_whisper import WhisperModel

def main():
    """Main function"""
    print("Debugging STT Engine implementation")
    
    # Check faster-whisper installation
    print("\nChecking faster-whisper installation:")
    try:
        model_path = Path(__file__).parent / 'models' / 'stt'
        print(f"Model path: {model_path}")
        
        model = WhisperModel(
            model_size_or_path="tiny.en",
            device="cpu",
            compute_type="int8",
            download_root=str(model_path)
        )
        print("✓ faster-whisper model created successfully")
    except Exception as e:
        print(f"✗ faster-whisper error: {e}")
        return 1
    
    # Test creating engine with explicit config
    print("\nCreating engine with explicit config:")
    config = {
        'model': {
            'type': 'faster-whisper',
            'size': 'tiny.en',
            'path': str(model_path),
            'compute_type': 'int8'
        },
        'hardware': {
            'enable_acceleration': False,
            'cuda_device': -1,
            'cpu_threads': 4
        }
    }
    
    try:
        engine = create_stt_engine("faster-whisper", config)
        print(f"✓ Engine created: {engine.__class__.__name__}")
        
        # Initialize engine
        print("\nInitializing engine:")
        success = engine.initialize(config)
        print(f"✓ Initialization result: {success}")
        
        # Check status
        status = engine.get_status()
        print(f"✓ Initialized: {status.get('initialized', False)}")
        print(f"✓ Status: {status}")
        
    except Exception as e:
        print(f"✗ Engine error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nDebugging complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())