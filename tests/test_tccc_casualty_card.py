#!/usr/bin/env python3
"""
Test script for TCCC Casualty Card workflow with real model implementations.

This script demonstrates the complete TCCC Casualty Card workflow using the
Phi-2 LLM for analysis and Faster Whisper for speech recognition.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TCCCCasualtyCard")

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TCCCCasualtyCardDemo:
    """Demo for the TCCC Casualty Card workflow."""
    
    def __init__(self, use_mock=False):
        """Initialize the demo."""
        self.use_mock = use_mock
        self.stt_engine = None
        self.llm_model = None
        
    def setup(self):
        """Set up the necessary components."""
        try:
            # Import necessary components
            from tccc.stt_engine import create_stt_engine
            from tccc.llm_analysis import get_phi_model
            
            # Configure STT engine
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
            
            # Create STT engine
            logger.info("Creating Faster Whisper STT engine")
            engine_type = "standard" if self.use_mock else "faster-whisper"
            self.stt_engine = create_stt_engine(engine_type, stt_config)
            
            # Initialize STT engine
            logger.info("Initializing STT engine")
            try:
                success = self.stt_engine.initialize(stt_config)
                if hasattr(self.stt_engine, "is_initialized") and not self.stt_engine.is_initialized:
                    logger.error("STT engine not fully initialized")
                    return False
                logger.info("STT engine initialization successful")
            except Exception as e:
                logger.error(f"STT engine initialization failed: {str(e)}")
                return False
            
            # Configure LLM
            llm_config = {
                "model_path": "microsoft/phi-2",
                "use_gpu": True,
                "quantization": "4-bit",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "force_mock": self.use_mock
            }
            
            # Create LLM model
            logger.info("Creating Phi-2 LLM model")
            self.llm_model = get_phi_model(llm_config)
            
            logger.info("Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False
    
    def process_audio(self, audio_file):
        """Process an audio file with STT."""
        try:
            import numpy as np
            import soundfile as sf
            
            logger.info(f"Processing audio file: {audio_file}")
            
            # Load audio file
            audio, sample_rate = sf.read(audio_file)
            
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Transcribe audio
            logger.info("Transcribing audio")
            start_time = time.time()
            
            # First try transcribe method, fall back to transcribe_segment if needed
            if hasattr(self.stt_engine, "transcribe"):
                result = self.stt_engine.transcribe(audio)
            elif hasattr(self.stt_engine, "transcribe_segment"):
                result = self.stt_engine.transcribe_segment(audio)
            else:
                raise ValueError("STT engine has no transcribe or transcribe_segment method")
                
            transcription_time = time.time() - start_time
            
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return None
    
    def analyze_transcript(self, transcript):
        """Analyze transcript with LLM."""
        try:
            logger.info("Analyzing transcript with LLM")
            
            prompt = f"""
            Extract medical entities from the following conversation and format them as a TCCC Casualty Card:
            
            {transcript}
            
            Format the output as structured JSON with the following sections:
            1. Patient Information
            2. Mechanism of Injury
            3. Injuries
            4. Vital Signs
            5. Treatments
            6. Medications
            7. Notes
            """
            
            start_time = time.time()
            response = self.llm_model.generate(prompt)
            analysis_time = time.time() - start_time
            
            logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
            
            return response["choices"][0]["text"]
            
        except Exception as e:
            logger.error(f"Transcript analysis failed: {str(e)}")
            return None
    
    def run_demo(self, audio_file):
        """Run the complete demo workflow."""
        try:
            # Process audio file
            transcription_result = self.process_audio(audio_file)
            if not transcription_result:
                logger.error("Transcription failed")
                return False
            
            # Handle different response formats from different engines
            if isinstance(transcription_result, dict) and "text" in transcription_result:
                transcript = transcription_result["text"]
            elif hasattr(transcription_result, "text"):
                transcript = transcription_result.text
            elif isinstance(transcription_result, str):
                transcript = transcription_result
            else:
                logger.warning(f"Unknown transcription result format: {type(transcription_result)}")
                transcript = str(transcription_result)
            logger.info(f"Transcription: {transcript}")
            
            # Analyze transcript
            analysis_result = self.analyze_transcript(transcript)
            if not analysis_result:
                logger.error("Analysis failed")
                return False
            
            logger.info("Analysis result:")
            logger.info(analysis_result)
            
            # Generate casualty card
            casualty_card = {
                "transcript": transcript,
                "analysis": analysis_result,
                "timestamp": time.time()
            }
            
            # Save casualty card
            output_file = f"casualty_card_{int(time.time())}.json"
            with open(output_file, "w") as f:
                json.dump(casualty_card, f, indent=2)
            
            logger.info(f"Casualty card saved to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Demo workflow failed: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.stt_engine:
                logger.info("Shutting down STT engine")
                self.stt_engine.shutdown()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

def main():
    """Main entry point for test script."""
    parser = argparse.ArgumentParser(description="Test TCCC Casualty Card workflow")
    parser.add_argument("--audio-file", type=str, default="test_data/test_speech.wav",
                      help="Path to the audio file to process")
    parser.add_argument("--use-mock", action="store_true",
                      help="Use mock implementations instead of real models")
    
    args = parser.parse_args()
    
    # Create demo
    demo = TCCCCasualtyCardDemo(use_mock=args.use_mock)
    
    # Set up components
    if not demo.setup():
        logger.error("Setup failed")
        return 1
    
    # Run demo
    success = demo.run_demo(args.audio_file)
    
    # Clean up
    demo.cleanup()
    
    if success:
        logger.info("TCCC Casualty Card workflow completed successfully")
        return 0
    else:
        logger.error("TCCC Casualty Card workflow failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())