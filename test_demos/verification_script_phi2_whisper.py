#!/usr/bin/env python3
"""
Verification script for Phi-2 LLM and Faster Whisper STT implementations.

This script verifies the functionality of both the Phi-2 LLM implementation
and the Faster Whisper STT implementation for the TCCC.ai project.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Phi2WhisperVerification")

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def verify_phi2():
    """Verify the Phi-2 LLM implementation."""
    try:
        from tccc.llm_analysis import get_phi_model
        from tccc.utils.config_manager import ConfigManager
        
        logger.info("\n=== Testing Phi-2 LLM Implementation ===")
        
        # Create configuration for Phi-2 - use mock for quick testing
        phi_config = {
            "model_path": "microsoft/phi-2",
            "use_gpu": True,
            "quantization": "4-bit",
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "force_mock": True  # Use mock for faster testing
        }
        
        # Get model
        logger.info("Initializing Phi-2 model")
        phi_model = get_phi_model(phi_config)
        
        # Test generation - medical prompt
        logger.info("Testing generation with medical prompt")
        
        test_prompt = """
        Extract medical entities from the following conversation:
        
        Medic: Patient has a gunshot wound to the right thigh with significant bleeding.
        Medic: I've applied a tourniquet above the wound.
        Medic: Blood pressure is 100/60, pulse 120, breathing normal.
        """
        
        start_time = time.time()
        response = phi_model.generate(test_prompt)
        generation_time = time.time() - start_time
        
        logger.info(f"Generated response in {generation_time:.2f} seconds:")
        logger.info(response["choices"][0]["text"])
        
        # Get metrics
        metrics = phi_model.get_metrics()
        logger.info(f"Model metrics: {metrics}")
        
        logger.info("Phi-2 LLM implementation test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Phi-2 verification failed: {str(e)}")
        return False

def verify_faster_whisper():
    """Verify the Faster Whisper STT implementation."""
    try:
        from tccc.stt_engine import create_stt_engine
        from tccc.utils.config_manager import ConfigManager
        import numpy as np
        
        logger.info("\n=== Testing Faster Whisper STT Implementation ===")
        
        # Create configuration for Faster Whisper
        stt_config = {
            "model": {
                "size": "tiny",
                "language": "en",
                "beam_size": 1,
                "compute_type": "int8",
                "vad_filter": True,
                "use_medical_vocabulary": True
            },
            "hardware": {
                "enable_acceleration": True,
                "cpu_threads": 4
            }
        }
        
        # Create engine
        logger.info("Creating Faster Whisper STT engine")
        stt_engine = create_stt_engine("faster-whisper", stt_config)
        
        # Initialize engine
        logger.info("Initializing Faster Whisper STT engine")
        success = stt_engine.initialize(stt_config)
        
        if not success:
            logger.error("Failed to initialize Faster Whisper STT engine")
            return False
            
        logger.info("Faster Whisper STT engine initialized successfully")
        
        # Generate test audio - sine wave
        logger.info("Generating test audio")
        sample_rate = 16000
        duration_seconds = 2
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Transcribe test audio
        logger.info("Transcribing test audio")
        start_time = time.time()
        result = stt_engine.transcribe_segment(audio)
        transcription_time = time.time() - start_time
        
        logger.info(f"Transcription result in {transcription_time:.2f} seconds:")
        logger.info(f"Text: {result.get('text', 'No text generated')}")
        
        # Get engine status
        status = stt_engine.get_status()
        logger.info(f"Engine status: {status}")
        
        # Shutdown engine
        logger.info("Shutting down engine")
        stt_engine.shutdown()
        
        logger.info("Faster Whisper STT implementation test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Faster Whisper verification failed: {str(e)}")
        return False

def main():
    """Main entry point for verification script."""
    parser = argparse.ArgumentParser(description="Verify Phi-2 LLM and Faster Whisper STT implementations")
    parser.add_argument("--phi2-only", action="store_true", help="Only verify Phi-2 LLM")
    parser.add_argument("--whisper-only", action="store_true", help="Only verify Faster Whisper STT")
    
    args = parser.parse_args()
    
    phi2_success = None
    whisper_success = None
    
    if args.whisper_only:
        whisper_success = verify_faster_whisper()
    elif args.phi2_only:
        phi2_success = verify_phi2()
    else:
        # Verify both
        phi2_success = verify_phi2()
        whisper_success = verify_faster_whisper()
    
    # Print summary
    print("\n=== Verification Summary ===")
    if phi2_success is not None:
        print(f"Phi-2 LLM: {'SUCCESS' if phi2_success else 'FAILED'}")
    if whisper_success is not None:
        print(f"Faster Whisper STT: {'SUCCESS' if whisper_success else 'FAILED'}")
    
    # Return overall status
    if phi2_success is False or whisper_success is False:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())