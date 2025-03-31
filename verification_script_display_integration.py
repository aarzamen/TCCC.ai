#!/usr/bin/env python3
"""
TCCC Display Integration Verification Script
-------------------------------------------
Verifies the integration between display components and the event system.
"""

import os
import sys
import time
import pygame
import logging
from datetime import datetime
import argparse
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import TCCC components
from tccc.utils.event_bus import get_event_bus
from tccc.utils.event_schema import (
    BaseEvent, EventType, TranscriptionEvent, 
    LLMAnalysisEvent, ProcessedTextEvent
)
from tccc.display.visualization.timeline import TimelineVisualization, TimelineEvent
from tccc.display.visualization.vital_signs import VitalSignsVisualization
from tccc.display.visualization.event_adapter import DisplayEventAdapter
from tccc.display.display_config import get_display_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayIntegrationVerification")


class DisplayIntegrationVerification:
    """Verifies the integration between display components and the event system."""
    
    def __init__(self, fullscreen=False, headless=False):
        """
        Initialize the verification script.
        
        Args:
            fullscreen: Whether to run in fullscreen mode
            headless: Whether to run in headless mode (no display)
        """
        self.fullscreen = fullscreen
        self.headless = headless
        self.running = False
        self.screen = None
        self.clock = None
        self.timeline = None
        self.vitals = None
        self.adapter = None
        self.event_bus = get_event_bus()
        self.display_config = get_display_config()
        self.test_results = {}
        
    def setup(self):
        """Set up the verification environment."""
        logger.info("Setting up display integration verification...")
        
        if not self.headless:
            # Initialize pygame
            pygame.init()
            
            # Create window
            if self.fullscreen:
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                width, height = self.screen.get_size()
            else:
                width, height = 1280, 720
                self.screen = pygame.display.set_mode((width, height))
                
            pygame.display.set_caption("TCCC Display Integration Verification")
            
            # Create clock
            self.clock = pygame.time.Clock()
            
            # Create visualization components
            timeline_rect = pygame.Rect(50, 50, width - 100, height // 2 - 75)
            self.timeline = TimelineVisualization(self.screen, timeline_rect)
            
            vitals_rect = pygame.Rect(50, height // 2 + 25, width - 100, height // 2 - 75)
            self.vitals = VitalSignsVisualization(self.screen, vitals_rect)
        else:
            # Create components without display
            self.timeline = TimelineVisualization()
            self.vitals = VitalSignsVisualization()
            
        # Create adapter
        self.adapter = DisplayEventAdapter(self.timeline, self.vitals, "verification_display")
        
        logger.info("Setup complete")
        
    def test_transcription_events(self):
        """Test handling of transcription events."""
        logger.info("Testing transcription event handling...")
        
        # Create test transcription event
        event = TranscriptionEvent(
            source="verification_script",
            text="Patient has severe respiratory distress with heart rate 130, blood pressure 90/60, and oxygen saturation 85%",
            segments=[],
            language="en",
            confidence=0.92,
            is_partial=False,
            session_id="verification_session"
        )
        
        # Publish event
        self.event_bus.publish(event)
        
        # Give time for event processing
        time.sleep(0.5)
        
        # Verify results
        if not self.headless:
            self.update_display()
        
        # Check if events were added to timeline
        timeline_updated = len(self.timeline.events) > 0
        
        # Check if vitals were updated
        vitals_updated = (
            self.vitals.vital_signs["heart_rate"]["current"] == 130 and
            self.vitals.vital_signs["blood_pressure"]["current"] == (90, 60) and
            self.vitals.vital_signs["spo2"]["current"] == 85
        )
        
        # Record results
        self.test_results["transcription_events"] = {
            "timeline_updated": timeline_updated,
            "vitals_updated": vitals_updated,
            "passed": timeline_updated and vitals_updated
        }
        
        logger.info(f"Transcription event test: {'PASSED' if timeline_updated and vitals_updated else 'FAILED'}")
        
    def test_llm_analysis_events(self):
        """Test handling of LLM analysis events."""
        logger.info("Testing LLM analysis event handling...")
        
        # Create test LLM analysis event
        event = LLMAnalysisEvent(
            source="verification_script",
            summary="Patient presents with signs of tension pneumothorax requiring immediate intervention.",
            topics=["tension pneumothorax", "chest trauma", "respiratory distress"],
            medical_terms=[
                {
                    "term": "tension pneumothorax",
                    "category": "medical_condition",
                    "severity": "critical",
                    "confidence": 0.95
                }
            ],
            actions=[
                {
                    "type": "treatment",
                    "priority": "immediate",
                    "description": "Perform needle decompression followed by chest tube placement"
                }
            ],
            session_id="verification_session"
        )
        
        # Get initial event count
        initial_count = len(self.timeline.events)
        
        # Publish event
        self.event_bus.publish(event)
        
        # Give time for event processing
        time.sleep(0.5)
        
        # Verify results
        if not self.headless:
            self.update_display()
        
        # Count new events
        new_events = len(self.timeline.events) - initial_count
        
        # Verify at least 2 new events (summary and critical condition)
        llm_events_added = new_events >= 2
        
        # Check event types (should include critical and treatment)
        event_types = [e.event_type for e in list(self.timeline.events)[:new_events]]
        critical_found = "critical" in event_types
        treatment_found = "treatment" in event_types
        
        # Record results
        self.test_results["llm_analysis_events"] = {
            "new_events": new_events,
            "critical_found": critical_found,
            "treatment_found": treatment_found,
            "passed": llm_events_added and critical_found and treatment_found
        }
        
        test_passed = llm_events_added and critical_found and treatment_found
        logger.info(f"LLM analysis event test: {'PASSED' if test_passed else 'FAILED'}")
        
    def test_processed_text_events(self):
        """Test handling of processed text events."""
        logger.info("Testing processed text event handling...")
        
        # Create test processed text event
        event = ProcessedTextEvent(
            source="verification_script",
            text="Patient requires morphine for pain management",
            entities=[
                {
                    "text": "morphine",
                    "type": "MEDICATION",
                    "start": 16,
                    "end": 24,
                    "confidence": 0.98
                }
            ],
            intent={
                "name": "request_treatment",
                "confidence": 0.85,
                "slots": {
                    "medication": "morphine",
                    "purpose": "pain management"
                }
            },
            session_id="verification_session"
        )
        
        # Get initial event count
        initial_count = len(self.timeline.events)
        
        # Publish event
        self.event_bus.publish(event)
        
        # Give time for event processing
        time.sleep(0.5)
        
        # Verify results
        if not self.headless:
            self.update_display()
        
        # Count new events
        new_events = len(self.timeline.events) - initial_count
        
        # Verify at least 1 new event for the medication
        processed_events_added = new_events >= 1
        
        # Check for medication event
        event_types = [e.event_type for e in list(self.timeline.events)[:new_events]]
        medication_found = "medication" in event_types or "treatment" in event_types
        
        # Record results
        self.test_results["processed_text_events"] = {
            "new_events": new_events,
            "medication_found": medication_found,
            "passed": processed_events_added and medication_found
        }
        
        test_passed = processed_events_added and medication_found
        logger.info(f"Processed text event test: {'PASSED' if test_passed else 'FAILED'}")
        
    def simulate_event_flow(self):
        """Simulate a realistic flow of events from multiple components."""
        logger.info("Simulating full event flow...")
        
        # Create a thread for publishing events
        def event_publisher():
            # Simulate initializing the system
            time.sleep(1)
            
            # Initial audio capture
            logger.info("Publishing initial transcription...")
            self.event_bus.publish(TranscriptionEvent(
                source="stt_engine",
                text="I need to perform a patient assessment. The casualty has multiple injuries from an IED blast.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="flow_session"
            ))
            
            time.sleep(2)
            
            # Initial assessment
            logger.info("Publishing processed text for assessment...")
            self.event_bus.publish(ProcessedTextEvent(
                source="processing_core",
                text="I need to perform a patient assessment. The casualty has multiple injuries from an IED blast.",
                entities=[
                    {
                        "text": "patient assessment",
                        "type": "PROCEDURE",
                        "start": 16,
                        "end": 34,
                        "confidence": 0.95
                    },
                    {
                        "text": "IED blast",
                        "type": "INJURY_MECHANISM",
                        "start": 65,
                        "end": 74,
                        "confidence": 0.92
                    }
                ],
                intent={
                    "name": "initiate_assessment",
                    "confidence": 0.93,
                    "slots": {
                        "assessment_type": "initial",
                        "mechanism": "IED blast"
                    }
                },
                session_id="flow_session"
            ))
            
            time.sleep(2)
            
            # LLM analysis of assessment
            logger.info("Publishing LLM analysis for initial assessment...")
            self.event_bus.publish(LLMAnalysisEvent(
                source="llm_analysis",
                summary="Initial assessment indicates potential blast injuries requiring systematic MARCH evaluation.",
                topics=["blast injury", "IED", "MARCH protocol"],
                medical_terms=[
                    {
                        "term": "blast injury",
                        "category": "mechanism_of_injury",
                        "severity": "critical",
                        "confidence": 0.95
                    }
                ],
                actions=[
                    {
                        "type": "assessment",
                        "priority": "immediate",
                        "description": "Conduct MARCH assessment: Massive hemorrhage, Airway, Respiration, Circulation, Head/Hypothermia"
                    }
                ],
                session_id="flow_session"
            ))
            
            time.sleep(3)
            
            # Vital signs reporting
            logger.info("Publishing transcription for vital signs...")
            self.event_bus.publish(TranscriptionEvent(
                source="stt_engine",
                text="Vital signs are: heart rate 125, blood pressure 95/65, respiratory rate 22, oxygen saturation 92%, temperature 98.6 F",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="flow_session"
            ))
            
            time.sleep(2)
            
            # Critical finding
            logger.info("Publishing transcription for critical finding...")
            self.event_bus.publish(TranscriptionEvent(
                source="stt_engine",
                text="Patient has significant bleeding from left thigh with arterial spurting. Applying tourniquet now.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="flow_session"
            ))
            
            time.sleep(2)
            
            # LLM analysis of bleeding
            logger.info("Publishing LLM analysis for bleeding...")
            self.event_bus.publish(LLMAnalysisEvent(
                source="llm_analysis",
                summary="Patient has arterial hemorrhage from left thigh requiring immediate tourniquet application.",
                topics=["arterial bleeding", "hemorrhage control", "tourniquet"],
                medical_terms=[
                    {
                        "term": "arterial hemorrhage",
                        "category": "medical_condition",
                        "severity": "critical",
                        "confidence": 0.98
                    }
                ],
                actions=[
                    {
                        "type": "treatment",
                        "priority": "immediate",
                        "description": "Apply tourniquet 2-3 inches above wound, tighten until bleeding stops, note time of application"
                    }
                ],
                document_results=[
                    {
                        "title": "Tourniquet Application Protocol",
                        "relevance": 0.95,
                        "snippet": "For arterial bleeding, apply tourniquet 2-3 inches proximal to wound..."
                    }
                ],
                session_id="flow_session"
            ))
            
            time.sleep(3)
            
            # Treatment confirmation
            logger.info("Publishing transcription for treatment confirmation...")
            self.event_bus.publish(TranscriptionEvent(
                source="stt_engine",
                text="Tourniquet applied successfully at 14:32. Bleeding has stopped. Moving on to assess airway.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="flow_session"
            ))
            
            time.sleep(2)
            
            # Processed text for treatment
            logger.info("Publishing processed text for treatment...")
            self.event_bus.publish(ProcessedTextEvent(
                source="processing_core",
                text="Tourniquet applied successfully at 14:32. Bleeding has stopped. Moving on to assess airway.",
                entities=[
                    {
                        "text": "Tourniquet",
                        "type": "MEDICAL_DEVICE",
                        "start": 0,
                        "end": 10,
                        "confidence": 0.97
                    },
                    {
                        "text": "Bleeding",
                        "type": "MEDICAL_CONDITION",
                        "start": 40,
                        "end": 48,
                        "confidence": 0.95
                    },
                    {
                        "text": "airway",
                        "type": "BODY_PART",
                        "start": 76,
                        "end": 82,
                        "confidence": 0.96
                    }
                ],
                intent={
                    "name": "report_treatment_status",
                    "confidence": 0.92,
                    "slots": {
                        "treatment": "tourniquet",
                        "status": "successful",
                        "time": "14:32"
                    }
                },
                session_id="flow_session"
            ))
            
        # Start the publisher thread
        event_thread = threading.Thread(target=event_publisher)
        event_thread.daemon = True
        event_thread.start()
        
        # Run the display loop for 20 seconds
        start_time = time.time()
        while time.time() - start_time < 20 and not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
            
            self.update_display()
            
        # Wait for the event thread to finish
        event_thread.join(timeout=2.0)
        
        # Verify that enough events were generated
        event_count = len(self.timeline.events)
        vitals_updated = (
            self.vitals.vital_signs["heart_rate"]["current"] == 125 and
            self.vitals.vital_signs["blood_pressure"]["current"] == (95, 65) and
            self.vitals.vital_signs["respiratory_rate"]["current"] == 22 and
            self.vitals.vital_signs["spo2"]["current"] == 92
        )
        
        # Record results
        self.test_results["event_flow"] = {
            "event_count": event_count,
            "vitals_updated": vitals_updated,
            "passed": event_count >= 5 and vitals_updated
        }
        
        test_passed = event_count >= 5 and vitals_updated
        logger.info(f"Event flow simulation: {'PASSED' if test_passed else 'FAILED'} - {event_count} events generated")
        
    def update_display(self):
        """Update the display."""
        self.screen.fill((0, 0, 0))
        self.timeline.draw()
        self.vitals.draw()
        pygame.display.flip()
        self.clock.tick(30)
        
    def run_verification(self):
        """Run all verification tests."""
        logger.info("Starting display integration verification...")
        self.setup()
        
        # Run individual tests
        self.test_transcription_events()
        self.test_llm_analysis_events()
        self.test_processed_text_events()
        
        # Run full event flow simulation
        self.simulate_event_flow()
        
        # Clean up
        self.cleanup()
        
        # Report results
        self.report_results()
        
    def cleanup(self):
        """Clean up resources."""
        if not self.headless:
            pygame.quit()
            
        # Unregister from event bus
        if self.adapter:
            self.adapter.unregister()
            
        logger.info("Cleanup complete")
        
    def report_results(self):
        """Report verification results."""
        logger.info("=== Display Integration Verification Results ===")
        
        all_passed = True
        for test_name, results in self.test_results.items():
            passed = results.get("passed", False)
            all_passed = all_passed and passed
            status = "PASSED" if passed else "FAILED"
            logger.info(f"{test_name}: {status}")
            
        logger.info(f"Overall verification: {'PASSED' if all_passed else 'FAILED'}")
        logger.info("=== End of Verification Results ===")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TCCC Display Integration Verification")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display)")
    args = parser.parse_args()
    
    verification = DisplayIntegrationVerification(
        fullscreen=args.fullscreen,
        headless=args.headless
    )
    
    success = verification.run_verification()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())