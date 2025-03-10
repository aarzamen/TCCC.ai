#!/usr/bin/env python3
"""
Verification script for the VAD Manager module.
Tests the thread-safe integration between components that need voice activity detection.
"""

import os
import sys
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tccc.utils.vad_manager import VADManager, get_vad_manager, VADMode, VADResult
from src.tccc.utils.logging import get_logger, configure_logging

# Configure logging
configure_logging()
logger = get_logger("vad_verification")

def create_synthetic_audio(duration_seconds=1.0, sample_rate=16000, speech=True, noise_level=0.01):
    """Create synthetic audio sample with or without speech-like content."""
    num_samples = int(duration_seconds * sample_rate)
    
    if speech:
        # Generate speech-like audio (multiple sine waves at voice frequencies)
        t = np.arange(num_samples) / sample_rate
        signal = 0.3 * np.sin(2 * np.pi * 150 * t)  # Fundamental frequency
        signal += 0.2 * np.sin(2 * np.pi * 300 * t)  # 1st harmonic
        signal += 0.1 * np.sin(2 * np.pi * 450 * t)  # 2nd harmonic
        signal += 0.05 * np.sin(2 * np.pi * 600 * t)  # 3rd harmonic
        
        # Add some noise
        noise = np.random.normal(0, noise_level, num_samples)
        return signal + noise
    else:
        # Generate just noise
        return np.random.normal(0, noise_level, num_samples)

def test_basic_operation():
    """Test basic operations of the VAD Manager."""
    logger.info("Testing basic VAD Manager operations")
    
    # Create a VAD Manager
    config = {
        'vad': {
            'enabled': True,
            'sensitivity': 2,
            'energy_threshold': 0.01
        },
        'audio': {
            'sample_rate': 16000
        }
    }
    
    vad_manager = VADManager(config)
    
    # Test speech detection with synthetic audio
    speech_audio = create_synthetic_audio(speech=True)
    noise_audio = create_synthetic_audio(speech=False)
    
    # Check speech detection on speech audio
    result_speech = vad_manager.detect_speech(speech_audio, "test_component")
    logger.info(f"Speech detection on speech audio: {result_speech.is_speech} (confidence: {result_speech.confidence:.2f})")
    
    # Check speech detection on noise audio
    result_noise = vad_manager.detect_speech(noise_audio, "test_component")
    logger.info(f"Speech detection on noise audio: {result_noise.is_speech} (confidence: {result_noise.confidence:.2f})")
    
    # Verify correct detection
    assert result_speech.is_speech, "Failed to detect speech in synthetic audio"
    assert not result_noise.is_speech, "Incorrectly detected speech in noise audio"
    
    # Test segment detection
    mixed_audio = np.concatenate([
        create_synthetic_audio(0.5, speech=False),  # 0.5s silence
        create_synthetic_audio(1.0, speech=True),   # 1.0s speech
        create_synthetic_audio(0.3, speech=False),  # 0.3s silence
        create_synthetic_audio(0.7, speech=True)    # 0.7s speech
    ])
    
    segments = vad_manager.get_speech_segments(mixed_audio, "test_component")
    logger.info(f"Detected {len(segments)} speech segments in mixed audio")
    
    for i, (start, end) in enumerate(segments):
        logger.info(f"  Segment {i+1}: {start/16000:.2f}s - {end/16000:.2f}s")
    
    # Verify we detected at least one segment
    assert len(segments) >= 1, "Failed to detect speech segments in mixed audio"
    
    # Test mode setting
    vad_manager.set_mode(VADMode.BATTLEFIELD)
    status = vad_manager.get_status()
    logger.info(f"VAD status after mode change: {status['mode_name']}")
    
    assert status['battlefield_mode'], "Failed to enable battlefield mode"
    
    # Test config update
    new_config = {
        'vad': {
            'sensitivity': 3,
            'energy_threshold': 0.02
        }
    }
    vad_manager.update_config(new_config)
    status = vad_manager.get_status()
    
    assert status['sensitivity'] == 3, "Failed to update sensitivity"
    assert status['energy_threshold'] == 0.02, "Failed to update energy threshold"
    
    logger.info("Basic VAD Manager tests passed")
    return True

def test_component_isolation():
    """Test that each component gets its own isolated VAD instance."""
    logger.info("Testing component isolation in VAD Manager")
    
    # Create a VAD Manager
    config = {
        'vad': {
            'enabled': True,
            'sensitivity': 2
        },
        'audio': {
            'sample_rate': 16000
        }
    }
    
    vad_manager = VADManager(config)
    
    # Get VAD instances for different components
    vad1 = vad_manager.get_vad_instance("component1")
    vad2 = vad_manager.get_vad_instance("component2")
    
    # Verify they are separate instances
    assert vad1 is not vad2, "VAD instances are not properly isolated"
    
    # Verify each component has its own state tracking
    speech_audio = create_synthetic_audio(speech=True)
    
    # Process with first component multiple times
    for _ in range(5):
        vad_manager.detect_speech(speech_audio, "component1")
    
    # Process with second component once
    vad_manager.detect_speech(speech_audio, "component2")
    
    # Get status and check component counters
    status = vad_manager.get_status()
    logger.info(f"Component status: {status['components']}")
    
    comp1_speech = status['components']['component1']['speech_counter']
    comp2_speech = status['components']['component2']['speech_counter']
    
    assert comp1_speech != comp2_speech, "Component state tracking is not isolated"
    
    logger.info("Component isolation tests passed")
    return True

def test_thread_safety():
    """Test thread safety of VAD Manager with concurrent access."""
    logger.info("Testing thread safety of VAD Manager")
    
    # Create a shared VAD Manager via the singleton
    config = {
        'vad': {'enabled': True},
        'audio': {'sample_rate': 16000}
    }
    vad_manager = get_vad_manager(config)
    
    # Create speech and noise samples
    speech_audio = create_synthetic_audio(speech=True)
    noise_audio = create_synthetic_audio(speech=False)
    
    # Function for thread testing
    def process_audio(component_name, iterations=20):
        results = []
        for i in range(iterations):
            # Alternate between speech and noise
            audio = speech_audio if i % 2 == 0 else noise_audio
            result = vad_manager.detect_speech(audio, component_name)
            results.append(result.is_speech)
            time.sleep(0.01)  # Small delay to increase thread interleaving
        return results
    
    # Test with multiple threads representing different components
    components = ["component1", "component2", "component3", "component4"]
    
    with ThreadPoolExecutor(max_workers=len(components)) as executor:
        futures = [executor.submit(process_audio, comp) for comp in components]
        results = [future.result() for future in futures]
    
    # Verify each component processed the expected pattern (alternating speech/noise)
    for i, (comp, comp_results) in enumerate(zip(components, results)):
        expected_pattern = [j % 2 == 0 for j in range(len(comp_results))]
        matches = sum(1 for a, b in zip(comp_results, expected_pattern) if a == b)
        match_ratio = matches / len(comp_results)
        
        logger.info(f"{comp} detection accuracy: {match_ratio:.2f}")
        
        # We allow some detection errors due to the simplistic synthetic audio
        assert match_ratio > 0.7, f"Thread safety issue detected in {comp}"
    
    # Check status to verify all components are tracked
    status = vad_manager.get_status()
    assert len(status['components']) == len(components), "Not all components were tracked"
    
    logger.info("Thread safety tests passed")
    return True

def test_singleton_pattern():
    """Test the singleton pattern of the VAD Manager."""
    logger.info("Testing singleton pattern of VAD Manager")
    
    # Create first instance with specific config
    config1 = {
        'vad': {
            'enabled': True,
            'sensitivity': 1,
            'energy_threshold': 0.01
        }
    }
    vad1 = get_vad_manager(config1)
    
    # Try to create second instance with different config
    config2 = {
        'vad': {
            'enabled': True,
            'sensitivity': 3,
            'energy_threshold': 0.05
        }
    }
    vad2 = get_vad_manager(config2)
    
    # Verify they are the same instance
    assert vad1 is vad2, "get_vad_manager did not return the same instance"
    
    # Verify the config from the first initialization is used
    status = vad1.get_status()
    logger.info(f"VAD Manager status: {status}")
    
    assert status['sensitivity'] == 1, "Singleton lost original configuration"
    
    # Update config explicitly
    vad1.update_config(config2)
    
    # Verify config was updated
    status = vad1.get_status()
    assert status['sensitivity'] == 3, "Failed to update singleton configuration"
    
    logger.info("Singleton pattern tests passed")
    return True

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Verify VAD Manager functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbosity
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level)
    
    logger.info("Starting VAD Manager verification...")
    
    try:
        # Run tests
        tests = [
            test_basic_operation,
            test_component_isolation,
            test_thread_safety,
            test_singleton_pattern
        ]
        
        all_passed = True
        for test in tests:
            try:
                passed = test()
                if not passed:
                    all_passed = False
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with error: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("✅ All VAD Manager verification tests passed!")
            return 0
        else:
            logger.error("❌ Some VAD Manager verification tests failed!")
            return 1
        
    except Exception as e:
        logger.error(f"Verification script failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())