#!/usr/bin/env python3
"""
Test script for VAD Manager integration between Audio Pipeline and STT Engine.

This script verifies that both the Audio Pipeline and the STT Engine
are correctly using the centralized VAD Manager for speech detection.
"""

import os
import sys
import time
import numpy as np
import json
import argparse
from pathlib import Path

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required components
from src.tccc.audio_pipeline import AudioPipeline
from src.tccc.stt_engine.faster_whisper_stt import FasterWhisperSTT
from src.tccc.utils.vad_manager import get_vad_manager, VADMode
from src.tccc.utils.logging import get_logger, configure_logging

# Configure logging
configure_logging(log_level="INFO")
logger = get_logger("vad_integration_test")


def create_test_audio(duration_seconds=2.0, sample_rate=16000, with_speech=True):
    """Create a synthetic test audio with or without speech-like content."""
    samples = int(duration_seconds * sample_rate)
    
    if with_speech:
        # Generate speech-like signal (simple sine waves at speech frequencies)
        t = np.arange(samples) / sample_rate
        signal = 0.5 * np.sin(2 * np.pi * 150 * t)  # Fundamental frequency
        signal += 0.25 * np.sin(2 * np.pi * 300 * t)  # First harmonic
        signal += 0.125 * np.sin(2 * np.pi * 450 * t)  # Second harmonic
    else:
        # Generate noise
        signal = np.random.normal(0, 0.05, samples)
    
    return signal


def test_vad_manager_shared_instance():
    """Test that both components get the same VAD Manager instance."""
    logger.info("Testing VAD Manager shared instance...")
    
    # Create Audio Pipeline
    audio_pipeline = AudioPipeline()
    audio_config = {
        'audio': {
            'sample_rate': 16000,
            'channels': 1,
            'format': 'float32',
            'chunk_size': 1024
        },
        'vad': {
            'enabled': True,
            'sensitivity': 2,
            'energy_threshold': 0.01
        }
    }
    audio_pipeline.initialize(audio_config)
    
    # Create STT Engine
    stt_engine = FasterWhisperSTT({
        'model': {
            'size': 'tiny',
            'vad_filter': True
        }
    })
    
    # Initialize the STT engine
    if not stt_engine.initialize():
        logger.error("Failed to initialize STT engine")
        return False
    
    # Get VAD Manager instances
    # Extract audio processor's VAD Manager
    audio_processor = audio_pipeline.audio_processor
    audio_vad_manager = getattr(audio_processor, 'vad_manager', None)
    
    # Extract STT engine's VAD Manager
    stt_vad_manager = getattr(stt_engine, 'vad_manager', None)
    
    # Check if both have VAD Managers
    if audio_vad_manager is None:
        logger.error("Audio Processor doesn't have a VAD Manager")
        return False
    
    if stt_vad_manager is None:
        logger.error("STT Engine doesn't have a VAD Manager")
        return False
    
    # Verify they are the same instance
    if audio_vad_manager is stt_vad_manager:
        logger.info("✅ Both components share the same VAD Manager instance")
        return True
    else:
        logger.error("❌ Components have different VAD Manager instances")
        return False


def test_vad_detection_consistency():
    """Test that both components detect speech consistently."""
    logger.info("Testing VAD detection consistency...")
    
    # Create Audio Pipeline
    audio_pipeline = AudioPipeline()
    audio_config = {
        'audio': {
            'sample_rate': 16000,
            'channels': 1,
            'format': 'float32',
            'chunk_size': 1024
        },
        'vad': {
            'enabled': True,
            'sensitivity': 2,
            'energy_threshold': 0.01
        }
    }
    audio_pipeline.initialize(audio_config)
    
    # Get audio processor
    audio_processor = audio_pipeline.audio_processor
    
    # Create STT Engine
    stt_engine = FasterWhisperSTT({
        'model': {
            'size': 'tiny',
            'vad_filter': True
        }
    })
    
    # Initialize the STT engine
    if not stt_engine.initialize():
        logger.error("Failed to initialize STT engine")
        return False
    
    # Create test audio with speech
    speech_audio = create_test_audio(with_speech=True)
    
    # Create test audio without speech (just noise)
    noise_audio = create_test_audio(with_speech=False)
    
    # Process speech audio with both components
    # 1. Audio Pipeline
    audio_pipeline_speech_result = audio_processor.enhanced_speech_detection(speech_audio)
    
    # 2. STT Engine (use VAD Manager directly)
    if hasattr(stt_engine, 'vad_manager'):
        stt_vad_result = stt_engine.vad_manager.detect_speech(speech_audio, "whisper_stt")
        stt_speech_result = stt_vad_result.is_speech
    else:
        # Fallback - just use direct transcription
        trans_result = stt_engine.transcribe_segment(speech_audio)
        stt_speech_result = len(trans_result.get('text', '')) > 0
    
    # Compare speech detection results
    logger.info(f"Speech audio - Audio Pipeline detection: {audio_pipeline_speech_result}")
    logger.info(f"Speech audio - STT Engine detection: {stt_speech_result}")
    
    # Process noise audio with both components
    # 1. Audio Pipeline
    audio_pipeline_noise_result = audio_processor.enhanced_speech_detection(noise_audio)
    
    # 2. STT Engine (use VAD Manager directly)
    if hasattr(stt_engine, 'vad_manager'):
        stt_vad_result = stt_engine.vad_manager.detect_speech(noise_audio, "whisper_stt")
        stt_noise_result = stt_vad_result.is_speech
    else:
        # Fallback - just use direct transcription
        trans_result = stt_engine.transcribe_segment(noise_audio)
        stt_noise_result = len(trans_result.get('text', '')) > 0
    
    # Compare noise detection results
    logger.info(f"Noise audio - Audio Pipeline detection: {audio_pipeline_noise_result}")
    logger.info(f"Noise audio - STT Engine detection: {stt_noise_result}")
    
    # The detection should be consistent between components
    consistent_speech = audio_pipeline_speech_result == stt_speech_result
    consistent_noise = audio_pipeline_noise_result == stt_noise_result
    
    # Check for proper speech detection (should detect speech in speech audio)
    correct_speech = audio_pipeline_speech_result and stt_speech_result
    
    # Check for proper noise rejection (should not detect speech in noise audio)
    correct_noise = not audio_pipeline_noise_result and not stt_noise_result
    
    # Log results
    if consistent_speech:
        logger.info("✅ Both components have consistent speech detection for speech audio")
    else:
        logger.error("❌ Components have inconsistent speech detection for speech audio")
    
    if consistent_noise:
        logger.info("✅ Both components have consistent speech detection for noise audio")
    else:
        logger.error("❌ Components have inconsistent speech detection for noise audio")
    
    if correct_speech:
        logger.info("✅ Both components correctly detect speech in speech audio")
    else:
        logger.error("❌ One or both components failed to detect speech in speech audio")
    
    if correct_noise:
        logger.info("✅ Both components correctly reject noise audio")
    else:
        logger.error("❌ One or both components incorrectly detected speech in noise audio")
    
    return consistent_speech and consistent_noise and correct_speech and correct_noise


def test_vad_battlefield_mode():
    """Test VAD Manager battlefield mode propagation."""
    logger.info("Testing VAD battlefield mode propagation...")
    
    # Create Audio Pipeline
    audio_pipeline = AudioPipeline()
    audio_config = {
        'audio': {
            'sample_rate': 16000,
            'channels': 1,
            'format': 'float32',
            'chunk_size': 1024
        },
        'vad': {
            'enabled': True,
            'sensitivity': 2,
            'energy_threshold': 0.01
        }
    }
    audio_pipeline.initialize(audio_config)
    
    # Get audio processor
    audio_processor = audio_pipeline.audio_processor
    
    # Create STT Engine
    stt_engine = FasterWhisperSTT({
        'model': {
            'size': 'tiny',
            'vad_filter': True
        }
    })
    
    # Initialize the STT engine
    if not stt_engine.initialize():
        logger.error("Failed to initialize STT engine")
        return False
    
    # Set battlefield mode in Audio Pipeline
    if hasattr(audio_processor, 'vad_manager'):
        # Import VADMode
        from src.tccc.utils.vad_manager import VADMode
        
        # Set battlefield mode using integer value (3)
        audio_processor.vad_manager.set_mode(3)  # BATTLEFIELD mode value
        logger.info("Set battlefield mode on Audio Pipeline's VAD Manager")
        
        # Check if STT Engine's VAD Manager is also in battlefield mode
        if hasattr(stt_engine, 'vad_manager'):
            stt_vad_status = stt_engine.vad_manager.get_status()
            audio_vad_status = audio_processor.vad_manager.get_status()
            
            # Compare battlefield mode
            logger.info(f"Audio Pipeline battlefield mode: {audio_vad_status.get('battlefield_mode', False)}")
            logger.info(f"STT Engine battlefield mode: {stt_vad_status.get('battlefield_mode', False)}")
            
            # They should be the same if sharing the VAD Manager
            if audio_vad_status.get('battlefield_mode', False) == stt_vad_status.get('battlefield_mode', False):
                logger.info("✅ Battlefield mode successfully propagated to both components")
                return True
            else:
                logger.error("❌ Battlefield mode did not propagate correctly")
                return False
        else:
            logger.error("STT Engine doesn't have a VAD Manager")
            return False
    else:
        logger.error("Audio Processor doesn't have a VAD Manager")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test VAD Manager integration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        configure_logging(log_level="DEBUG")
    
    logger.info("Starting VAD Manager integration test")
    
    # Run tests
    all_passed = True
    
    # Test 1: Check shared VAD Manager instance
    logger.info("\n=== Test 1: Shared VAD Manager Instance ===")
    try:
        if not test_vad_manager_shared_instance():
            all_passed = False
    except Exception as e:
        logger.error(f"Error in shared instance test: {e}")
        all_passed = False
    
    # Test 2: Check VAD detection consistency
    logger.info("\n=== Test 2: VAD Detection Consistency ===")
    try:
        if not test_vad_detection_consistency():
            all_passed = False
    except Exception as e:
        logger.error(f"Error in detection consistency test: {e}")
        all_passed = False
    
    # Test 3: Check battlefield mode propagation
    logger.info("\n=== Test 3: Battlefield Mode Propagation ===")
    try:
        if not test_vad_battlefield_mode():
            all_passed = False
    except Exception as e:
        logger.error(f"Error in battlefield mode test: {e}")
        all_passed = False
    
    # Final result
    if all_passed:
        logger.info("\n✅ All VAD Manager integration tests passed!")
        return 0
    else:
        logger.error("\n❌ Some VAD Manager integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())