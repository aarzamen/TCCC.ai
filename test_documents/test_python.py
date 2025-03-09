#!/usr/bin/env python3
"""
Test Python file for RAG document processing.
This file contains medical terminology and typical code patterns.
"""

import os
import sys
from typing import Dict, List, Optional

# Medical terminology definitions
MEDICAL_TERMS = {
    "hemorrhage": "Severe bleeding that is difficult to control",
    "tourniquet": "Device used to apply pressure to a limb to control bleeding",
    "tension_pneumothorax": "Life-threatening collapse of the lung",
    "triage": "Process of determining priority of treatment based on severity",
    "hypoxia": "Deficiency in oxygen reaching tissues"
}

class Patient:
    """Representation of a patient with medical attributes."""
    
    def __init__(self, name: str, age: int):
        """Initialize a new patient record."""
        self.name = name
        self.age = age
        self.vital_signs = {
            "pulse": 0,
            "bp_systolic": 0,
            "bp_diastolic": 0,
            "respiration": 0,
            "temperature": 0,
            "o2_sat": 0
        }
        self.injuries = []
        self.treatments = []
    
    def add_injury(self, injury_type: str, location: str, severity: int) -> None:
        """
        Add an injury to the patient record.
        
        Parameters:
        - injury_type: Type of injury (e.g., "laceration", "fracture")
        - location: Body location (e.g., "left arm", "chest")
        - severity: Injury severity on scale of 1-5
        """
        self.injuries.append({
            "type": injury_type,
            "location": location,
            "severity": severity,
            "timestamp": "current_time_placeholder"
        })
    
    def apply_treatment(self, treatment: str, notes: Optional[str] = None) -> None:
        """
        Record a treatment applied to the patient.
        
        Parameters:
        - treatment: Treatment applied (e.g., "tourniquet", "chest seal")
        - notes: Additional treatment notes
        """
        self.treatments.append({
            "treatment": treatment,
            "notes": notes,
            "timestamp": "current_time_placeholder"
        })
    
    def triage_category(self) -> str:
        """
        Determine triage category based on injuries and vital signs.
        
        Returns:
        - Triage category ("immediate", "delayed", "minimal", "expectant")
        """
        # Check for life-threatening conditions
        for injury in self.injuries:
            if injury["severity"] >= 4:
                return "immediate"
            
        # Check vital signs
        if (self.vital_signs["pulse"] < 50 or 
            self.vital_signs["pulse"] > 120 or
            self.vital_signs["bp_systolic"] < 90 or
            self.vital_signs["respiration"] < 8 or
            self.vital_signs["respiration"] > 30 or
            self.vital_signs["o2_sat"] < 90):
            return "immediate"
            
        # Check for moderate severity
        if any(injury["severity"] >= 3 for injury in self.injuries):
            return "delayed"
            
        # Check for minimal severity
        if self.injuries:
            return "minimal"
            
        return "minimal"

def hemorrhage_control(patient: Patient, location: str) -> bool:
    """
    Apply hemorrhage control measures.
    
    Parameters:
    - patient: Patient object
    - location: Body location requiring hemorrhage control
    
    Returns:
    - Success status of the procedure
    """
    # Check if tourniquet is needed (arm or leg)
    if location.lower() in ["left arm", "right arm", "left leg", "right leg"]:
        patient.apply_treatment(
            f"tourniquet applied to {location}",
            "Applied 2 inches above wound, time marked"
        )
    else:
        # Apply wound packing and pressure dressing
        patient.apply_treatment(
            f"hemostatic gauze and pressure dressing to {location}",
            "Direct pressure applied for 3 minutes"
        )
    
    return True

# Example usage
if __name__ == "__main__":
    # Create a test patient
    test_patient = Patient("Test Patient", 30)
    
    # Set vital signs
    test_patient.vital_signs = {
        "pulse": 110,
        "bp_systolic": 100,
        "bp_diastolic": 70,
        "respiration": 18,
        "temperature": 98.6,
        "o2_sat": 95
    }
    
    # Add injuries
    test_patient.add_injury("laceration", "right arm", 3)
    test_patient.add_injury("contusion", "chest", 2)
    
    # Apply treatments
    hemorrhage_control(test_patient, "right arm")
    
    # Print patient status
    print(f"Patient triage category: {test_patient.triage_category()}")
    print(f"Treatments applied: {len(test_patient.treatments)}")