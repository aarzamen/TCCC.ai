#!/usr/bin/env python3
"""
Test the STT engine with a recorded audio file.
"""
import os
import sys

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Import required components
from tccc.stt_engine import create_stt_engine
from tccc.utils.config_manager import ConfigManager

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STT.Test")

def main():
    """Test the STT engine with a recorded audio file."""
    # Load configuration
    config_manager = ConfigManager()
    stt_config = config_manager.load_config("stt_engine")
    
    # Initialize STT Engine (faster-whisper)
    logger.info("Initializing STT Engine (faster-whisper)...")
    stt_engine = create_stt_engine("faster-whisper", stt_config)
    if not stt_engine.initialize(stt_config):
        logger.error("Failed to initialize STT Engine")
        return False
    
    # Transcribe the audio file
    audio_file = os.path.join(project_dir, "test_mic.wav")
    logger.info(f"Transcribing audio file: {audio_file}")
    
    # Load the audio file
    import soundfile as sf
    audio_data, _ = sf.read(audio_file)
    
    # Transcribe the audio data
    result = stt_engine.transcribe_segment(audio_data)
    
    if result and 'text' in result and result['text'].strip():
        text = result['text']
        print(f"\nTranscribed text: \"{text}\"")
        return True
    else:
        print("\nNo transcription result or empty text")
        return False

if __name__ == "__main__":
    main()