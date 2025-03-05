#!/usr/bin/env python3
"""
Simplified test script for TCCC workflow using mock implementations only.
"""

import os
import sys
import time
import logging
import json
import uuid
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MockTCCCWorkflow")

class MockSTTEngine:
    """Simple mock STT engine for testing."""
    
    def __init__(self):
        self.is_initialized = False
    
    def initialize(self, config):
        """Initialize the mock STT engine."""
        logger.info("Initializing mock STT engine")
        self.is_initialized = True
        return True
    
    def transcribe(self, audio):
        """Mock transcription."""
        logger.info("Performing mock transcription")
        time.sleep(0.5)  # Simulate processing
        
        # Always return a realistic medical transcript
        return {
            "text": "Patient has a gunshot wound to the right thigh with significant bleeding. "
                    "I've applied a tourniquet above the wound. "
                    "Blood pressure is 100/60, pulse 120, breathing normal. "
                    "Patient is conscious, GCS 14. "
                    "I've administered 10mg morphine IV and started a fluid line."
        }
    
    def shutdown(self):
        """Shutdown the mock STT engine."""
        logger.info("Shutting down mock STT engine")
        self.is_initialized = False
        return True

class MockLLMModel:
    """Simple mock LLM model for testing."""
    
    def __init__(self, config):
        """Initialize the mock LLM model."""
        self.config = config
        logger.info("Initializing mock LLM model")
    
    def generate(self, prompt):
        """Generate text based on the prompt."""
        logger.info("Generating mock LLM response")
        time.sleep(1.0)  # Simulate processing
        
        # Return a realistic TCCC card
        return {
            "id": str(uuid.uuid4()),
            "choices": [{
                "text": json.dumps({
                    "Patient_Information": {
                        "id": "unknown",
                        "age": "adult",
                        "gender": "unknown",
                        "status": "conscious"
                    },
                    "Mechanism_of_Injury": {
                        "type": "gunshot wound",
                        "location": "right thigh",
                        "time": "unknown"
                    },
                    "Injuries": [
                        {
                            "type": "gunshot wound",
                            "location": "right thigh",
                            "severity": "significant",
                            "bleeding": "controlled with tourniquet"
                        }
                    ],
                    "Vital_Signs": [
                        {
                            "type": "blood pressure",
                            "value": "100/60",
                            "assessment": "low"
                        },
                        {
                            "type": "pulse",
                            "value": "120",
                            "assessment": "elevated"
                        },
                        {
                            "type": "breathing",
                            "value": "normal",
                            "assessment": "stable"
                        },
                        {
                            "type": "GCS",
                            "value": "14",
                            "assessment": "mild impairment"
                        }
                    ],
                    "Treatments": [
                        {
                            "procedure": "tourniquet application",
                            "location": "right thigh",
                            "time": "unknown",
                            "outcome": "bleeding controlled"
                        },
                        {
                            "procedure": "IV access",
                            "location": "unknown",
                            "time": "unknown",
                            "outcome": "successful"
                        }
                    ],
                    "Medications": [
                        {
                            "name": "morphine",
                            "dosage": "10mg",
                            "route": "IV",
                            "time": "unknown",
                            "purpose": "pain management"
                        }
                    ],
                    "Notes": [
                        "Patient requires immediate evacuation for surgical intervention",
                        "Monitor vital signs closely",
                        "Reassess tourniquet every 30 minutes"
                    ]
                }, indent=2)
            }]
        }

def run_mock_tccc_workflow():
    """Run the complete mock TCCC workflow."""
    try:
        # Create mock components
        logger.info("Creating mock components")
        stt_engine = MockSTTEngine()
        llm_model = MockLLMModel({"temperature": 0.7})
        
        # Initialize STT engine
        stt_engine.initialize({})
        
        # Simulate audio input
        logger.info("Simulating audio input")
        mock_audio = [0.0] * 16000  # 1 second of silence at 16kHz
        
        # Perform transcription
        logger.info("Performing transcription")
        transcription_result = stt_engine.transcribe(mock_audio)
        transcript = transcription_result["text"]
        logger.info(f"Transcription: {transcript}")
        
        # Analyze transcript
        logger.info("Analyzing transcript")
        prompt = f"""
        Extract medical entities from the following conversation and format them as a TCCC Casualty Card:
        
        {transcript}
        
        Format the output as structured JSON.
        """
        
        analysis_result = llm_model.generate(prompt)
        analysis_text = analysis_result["choices"][0]["text"]
        logger.info("Analysis result:")
        logger.info(analysis_text)
        
        # Generate casualty card
        casualty_card = {
            "transcript": transcript,
            "analysis": analysis_text,
            "timestamp": time.time()
        }
        
        # Save casualty card
        output_file = f"mock_casualty_card_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(casualty_card, f, indent=2)
        
        logger.info(f"Casualty card saved to {output_file}")
        
        # Cleanup
        logger.info("Shutting down mock STT engine")
        stt_engine.shutdown()
        
        logger.info("Mock TCCC workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Mock TCCC workflow failed: {str(e)}")
        return False

def main():
    """Main entry point."""
    success = run_mock_tccc_workflow()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())