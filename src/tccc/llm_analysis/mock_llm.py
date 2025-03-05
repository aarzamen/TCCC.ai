"""
Mock LLM module for TCCC.ai system.

This module provides a mock implementation of the Phi-2 Instruct model
for development and testing without requiring the actual model.
"""

import os
import json
import time
import logging
import threading
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re
import hashlib
import uuid

# Utilities
from tccc.utils.config import Config
from tccc.utils.logging import get_logger

logger = get_logger(__name__)

class MockPhiModel:
    """
    Mock implementation of the Phi-2 Instruct model using predefined responses
    for medical conversation analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock Phi model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Initializing Mock Phi-2 Instruct model")
        
        # Register pre-defined responses
        self._init_responses()
        
        # Track metrics
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency": 0.0
        }
    
    def _init_responses(self):
        """Initialize predefined responses for different prompt types."""
        # Load responses from file if available
        responses_path = os.path.join(os.path.dirname(__file__), "mock_responses.json")
        
        if os.path.exists(responses_path):
            with open(responses_path, "r") as f:
                self.responses = json.load(f)
            logger.info(f"Loaded {len(self.responses)} mock responses from file")
        else:
            # Default responses for different medical analysis tasks
            self.responses = {
                "entity_extraction": [
                    {
                        "type": "procedure",
                        "value": "tourniquet application",
                        "time": "0930 hours",
                        "context": "significant bleeding from right thigh"
                    },
                    {
                        "type": "procedure",
                        "value": "needle decompression",
                        "time": "0935 hours",
                        "context": "suspected tension pneumothorax, right chest"
                    },
                    {
                        "type": "medication",
                        "value": "morphine",
                        "time": "0940 hours",
                        "context": "10mg IV"
                    },
                    {
                        "type": "medication",
                        "value": "ceftriaxone",
                        "time": "after morphine",
                        "context": "1g IV antibiotics"
                    },
                    {
                        "type": "procedure",
                        "value": "IV access",
                        "time": "before fluid administration",
                        "context": "two large-bore IVs"
                    },
                    {
                        "type": "medication",
                        "value": "Hextend",
                        "time": "after IV access",
                        "context": "100ml/hour fluid resuscitation"
                    }
                ],
                
                "temporal_extraction": [
                    {
                        "event_id": "evt001",
                        "event": "scene arrival",
                        "timestamp": "2023-07-15T09:45:00",
                        "relative_time": "0945 hours",
                        "sequence": "first event",
                        "confidence": "high"
                    },
                    {
                        "event_id": "evt002",
                        "event": "tourniquet application",
                        "timestamp": "2023-07-15T09:30:00",
                        "relative_time": "0930 hours",
                        "sequence": "before needle decompression",
                        "confidence": "high"
                    },
                    {
                        "event_id": "evt003",
                        "event": "needle decompression",
                        "timestamp": "2023-07-15T09:35:00",
                        "relative_time": "0935 hours",
                        "sequence": "after tourniquet, before morphine",
                        "confidence": "high"
                    },
                    {
                        "event_id": "evt004",
                        "event": "morphine administration",
                        "timestamp": "2023-07-15T09:40:00",
                        "relative_time": "0940 hours",
                        "sequence": "after needle decompression",
                        "confidence": "high"
                    },
                    {
                        "event_id": "evt005",
                        "event": "antibiotics administration",
                        "timestamp": null,
                        "relative_time": "after morphine",
                        "sequence": "after morphine",
                        "confidence": "medium"
                    },
                    {
                        "event_id": "evt006",
                        "event": "IV access",
                        "timestamp": null,
                        "relative_time": "before fluid administration",
                        "sequence": "before Hextend",
                        "confidence": "medium"
                    },
                    {
                        "event_id": "evt007",
                        "event": "MEDEVAC arrangement",
                        "timestamp": null,
                        "relative_time": "after status report",
                        "sequence": "after vitals stabilizing",
                        "confidence": "medium"
                    }
                ],
                
                "vital_signs": [
                    {
                        "type": "blood_pressure",
                        "value": "100/60",
                        "unit": "mmHg",
                        "time": "first assessment",
                        "trend": "low"
                    },
                    {
                        "type": "heart_rate",
                        "value": "120",
                        "unit": "bpm",
                        "time": "first assessment",
                        "trend": "elevated"
                    },
                    {
                        "type": "respiratory_rate",
                        "value": "24",
                        "unit": "breaths/min",
                        "time": "first assessment",
                        "trend": "elevated"
                    },
                    {
                        "type": "oxygen_saturation",
                        "value": "92",
                        "unit": "%",
                        "time": "first assessment",
                        "trend": "slightly decreased"
                    },
                    {
                        "type": "glasgow_coma_scale",
                        "value": "14",
                        "unit": "points",
                        "time": "current assessment",
                        "trend": "improving"
                    },
                    {
                        "type": "glasgow_coma_scale",
                        "value": "12",
                        "unit": "points",
                        "time": "initial assessment",
                        "trend": "decreased"
                    },
                    {
                        "type": "blood_pressure",
                        "value": "110/70",
                        "unit": "mmHg",
                        "time": "after fluid resuscitation",
                        "trend": "improving"
                    },
                    {
                        "type": "heart_rate",
                        "value": "115",
                        "unit": "bpm",
                        "time": "after fluid resuscitation",
                        "trend": "still elevated"
                    }
                ],
                
                "medication": [
                    {
                        "name": "morphine",
                        "dosage": "10mg",
                        "route": "IV",
                        "time": "0940 hours",
                        "frequency": "once",
                        "purpose": "pain management"
                    },
                    {
                        "name": "ceftriaxone",
                        "dosage": "1g",
                        "route": "IV",
                        "time": "after morphine",
                        "frequency": "once",
                        "purpose": "antibiotic prophylaxis"
                    },
                    {
                        "name": "Hextend",
                        "dosage": "100ml/hour",
                        "route": "IV",
                        "time": "after IV access",
                        "frequency": "continuous",
                        "purpose": "fluid resuscitation"
                    }
                ],
                
                "procedures": [
                    {
                        "name": "tourniquet application",
                        "status": "completed",
                        "time": "0930 hours",
                        "performer": "Medic 1-2",
                        "outcome": "bleeding controlled",
                        "details": "applied to right thigh for significant hemorrhage"
                    },
                    {
                        "name": "needle decompression",
                        "status": "completed",
                        "time": "0935 hours",
                        "performer": "Medic 1-2",
                        "outcome": "tension pneumothorax resolved",
                        "details": "performed on right chest"
                    },
                    {
                        "name": "IV access",
                        "status": "completed",
                        "time": "before fluid administration",
                        "performer": "Medic 1-2",
                        "outcome": "successful",
                        "details": "established two large-bore IVs"
                    },
                    {
                        "name": "fluid resuscitation",
                        "status": "in_progress",
                        "time": "after IV access",
                        "performer": "Medic 1-2",
                        "outcome": "BP improving to 110/70",
                        "details": "Hextend at 100ml/hour"
                    },
                    {
                        "name": "hypothermia prevention",
                        "status": "in_progress",
                        "time": "latest intervention",
                        "performer": "Medic 1-2",
                        "outcome": "ongoing",
                        "details": "applying measures to prevent hypothermia"
                    },
                    {
                        "name": "MEDEVAC",
                        "status": "planned",
                        "time": "ETA 15 minutes",
                        "performer": "not specified",
                        "outcome": "pending",
                        "details": "urgent surgical case, pickup at LZ Bravo"
                    }
                ],
                
                "medevac": """MEDEVAC REQUEST

Line 1: LZ Bravo, grid coordinates to be transmitted on secure channel
Line 2: Freq: MEDEVAC Net, Call Sign: DUSTOFF 6
Line 3: 1 patient, Urgent Surgical (bleeding controlled, requires surgery)
Line 4: Special equipment required: None
Line 5: 1 litter patient
Line 6: Security at pickup site: Secure
Line 7: Site marked with smoke signal
Line 8: Patient is US military personnel
Line 9: No NBC contamination

PATIENT DETAILS:
28-year-old male with blast injuries to right leg
Tourniquet applied, needle decompression performed
BP 110/70, HR 115, RR 24, SpO2 92%, GCS 14
MEDS: Morphine 10mg IV, Ceftriaxone 1g IV, Hextend infusion
STATUS: Currently stable but requires urgent surgical intervention""",
                
                "zmist": """ZMIST REPORT

Z - MECHANISM OF INJURY:
IED blast with primary and secondary blast injuries to right leg

M - INJURIES SUSTAINED:
- Severe right leg injury with significant hemorrhage (controlled with tourniquet)
- Suspected tension pneumothorax, right chest (resolved with needle decompression)
- Possible traumatic brain injury (GCS improved from 12 to 14)

I - SIGNS:
- Initial: BP 100/60, HR 120, RR 24, SpO2 92%, GCS 12
- Current: BP 110/70, HR 115, RR 24, SpO2 92%, GCS 14

S - TREATMENT:
- Tourniquet applied to right thigh at 0930hrs
- Needle decompression right chest at 0935hrs
- 10mg morphine IV at 0940hrs
- 1g ceftriaxone IV
- Two large-bore IVs established
- Hextend infusion at 100ml/hr
- Hypothermia prevention measures

T - TRENDS:
- Blood pressure improving from 100/60 to 110/70
- GCS improving from 12 to 14
- Bleeding controlled
- Tension pneumothorax resolved
- Requires urgent surgical intervention""",
                
                "soap": """SOAP NOTE

S - SUBJECTIVE:
Patient is a 28-year-old male who sustained injuries from an IED blast. Patient was initially unresponsive at scene but is now responding. No subjective complaints noted as patient was initially unconscious.

O - OBJECTIVE:
- Vital Signs:
  * BP: Initially 100/60, now 110/70
  * HR: Initially 120, now 115
  * RR: 24
  * SpO2: 92%
  * GCS: Initially 12, now 14

- Physical Exam:
  * Right leg with blast injury and significant bleeding (controlled with tourniquet)
  * Right chest with suspected tension pneumothorax (treated with needle decompression)
  * No other visible external injuries noted in report

- Interventions:
  * Tourniquet applied to right thigh at 0930 hours
  * Needle decompression on right chest at 0935 hours
  * 10mg morphine IV at 0940 hours
  * 1g ceftriaxone IV
  * Two large-bore IVs established
  * Hextend infusion at 100ml/hour
  * Hypothermia prevention measures

A - ASSESSMENT:
1. Blast injury to right leg with hemorrhage, currently controlled
2. Resolved tension pneumothorax, right chest
3. Possible traumatic brain injury with improving mental status
4. Hypovolemic shock, responding to fluid resuscitation

P - PLAN:
1. Continue fluid resuscitation with Hextend
2. Monitor vital signs every 5 minutes
3. Urgent evacuation to Role 2 facility for surgical intervention
4. Continue pain management as needed
5. Reassess tourniquet and chest status frequently
6. Prepare MEDEVAC report and establish LZ Bravo for evacuation""",
                
                "tccc": """TACTICAL COMBAT CASUALTY CARE (TCCC) CARD

1. CASUALTY INFORMATION:
   Name: Not provided
   Rank/ID: Not provided
   Unit: Not provided
   Age: 28
   Sex: Male
   Time of Injury: Prior to 0930 hours

2. MECHANISM OF INJURY:
   IED blast
   Significant blast force affecting right leg

3. INJURIES:
   - Severe right leg injury with hemorrhage
   - Tension pneumothorax, right chest (resolved)
   - Possible TBI (GCS improved from 12 to 14)

4. SIGNS AND SYMPTOMS:
   Initial VS:
   - BP: 100/60
   - HR: 120
   - RR: 24
   - SpO2: 92%
   - GCS: 12 (initial), 14 (current)
   
   Current VS:
   - BP: 110/70
   - HR: 115
   - RR: 24
   - SpO2: 92%
   - GCS: 14

5. TREATMENTS:
   - T: Tourniquet applied to right thigh @ 0930 hours
   - A: Needle decompression right chest @ 0935 hours
   - C: Bleeding controlled with tourniquet
   - C: Hypothermia prevention measures applied
   - E: No documented eye injuries or treatments

6. MEDICATIONS:
   - Morphine 10mg IV @ 0940 hours
   - Ceftriaxone 1g IV (time not specified)

7. FLUID THERAPY:
   - Two large-bore IVs established
   - Hextend 100ml/hour
   - Volume infused: Not specified

8. NOTES:
   - Initially unresponsive at scene
   - Mental status improving
   - Classified as Urgent surgical case
   - MEDEVAC requested to LZ Bravo
   - ETA for MEDEVAC: 15 minutes from last report
   - Plan to continue monitoring q5min until evacuation"""
            }
            
            logger.info("Initialized default mock responses")
    
    def generate(self, prompt: str, max_tokens: int = None, 
                temperature: float = None, top_p: float = None) -> Dict[str, Any]:
        """Generate text based on the prompt.
        
        This mock implementation identifies the prompt type and returns
        a predefined response.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling value
            
        Returns:
            Dictionary with generated text
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        # Add some latency to simulate real model inference
        latency = 0.5 + (0.5 * temperature if temperature else 0.5)
        time.sleep(latency)
        
        # Determine prompt type
        prompt_type = self._identify_prompt_type(prompt)
        response_text = self._get_response_for_type(prompt_type)
        
        # Add some tokens for metrics
        estimated_tokens = len(response_text.split()) * 1.3
        self.metrics["total_tokens"] += estimated_tokens
        
        # Update average latency
        elapsed = time.time() - start_time
        self.metrics["avg_latency"] = (
            (self.metrics["avg_latency"] * (self.metrics["total_requests"] - 1) + elapsed) / 
            self.metrics["total_requests"]
        )
        
        return {
            "id": str(uuid.uuid4()),
            "choices": [{"text": response_text}],
            "usage": {
                "prompt_tokens": len(prompt.split()) * 1.3,
                "completion_tokens": estimated_tokens,
                "total_tokens": (len(prompt.split()) + len(response_text.split())) * 1.3
            },
            "model": "phi-2-instruct-mock",
            "latency": elapsed
        }
    
    def _identify_prompt_type(self, prompt: str) -> str:
        """Identify the type of prompt to determine response.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Prompt type string
        """
        prompt_lower = prompt.lower()
        
        # Check for specific indicators in the prompt
        if "medical entities" in prompt_lower and "extract" in prompt_lower:
            return "entity_extraction"
        elif "temporal references" in prompt_lower and "identify" in prompt_lower:
            return "temporal_extraction"
        elif "vital sign" in prompt_lower and "extract" in prompt_lower:
            return "vital_signs"
        elif "medication" in prompt_lower and "extract" in prompt_lower:
            return "medication"
        elif "procedure" in prompt_lower and "extract" in prompt_lower:
            return "procedures"
        elif "medevac" in prompt_lower and "generate" in prompt_lower:
            return "medevac"
        elif "zmist" in prompt_lower and "generate" in prompt_lower:
            return "zmist"
        elif "soap" in prompt_lower and "generate" in prompt_lower:
            return "soap"
        elif "tccc" in prompt_lower and "generate" in prompt_lower:
            return "tccc"
        else:
            # Default to entity extraction if can't determine
            return "entity_extraction"
    
    def _get_response_for_type(self, prompt_type: str) -> str:
        """Get predefined response for prompt type.
        
        Args:
            prompt_type: Type of prompt
            
        Returns:
            Response text
        """
        if prompt_type in self.responses:
            response_data = self.responses[prompt_type]
            
            # Handle string responses (like reports)
            if isinstance(response_data, str):
                return response_data
            
            # Handle structured data responses (like entity extraction)
            elif isinstance(response_data, list):
                return json.dumps(response_data, indent=2)
            
        # Fallback response
        return json.dumps([{"error": "No matching response template found"}], indent=2)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model usage metrics.
        
        Returns:
            Dictionary with usage metrics
        """
        return {
            "total_requests": self.metrics["total_requests"],
            "total_tokens": int(self.metrics["total_tokens"]),
            "avg_latency": round(self.metrics["avg_latency"], 3),
            "model": "phi-2-instruct-mock"
        }