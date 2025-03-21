#!/bin/bash
#
# TCCC Display Integration Launcher
# Runs the TCCC display components with event system integration
#

# Source virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create a new Python script for the integrated display
cat > tccc_integrated_display.py << 'EOF'
#!/usr/bin/env python3
"""
TCCC Integrated Display
----------------------
Integrated display components with event system connectivity.
"""

import os
import sys
import time
import pygame
import logging
import argparse
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import TCCC components
from tccc.utils.event_bus import get_event_bus
from tccc.utils.event_schema import BaseEvent, EventType
from tccc.display.visualization.timeline import TimelineVisualization
from tccc.display.visualization.vital_signs import VitalSignsVisualization
from tccc.display.visualization.event_adapter import DisplayEventAdapter
from tccc.display.display_config import get_display_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegratedDisplay")


class TCCCIntegratedDisplay:
    """Integrated display application with event bus integration."""
    
    def __init__(self, fullscreen=False, demo_mode=False):
        """
        Initialize the integrated display.
        
        Args:
            fullscreen: Whether to run in fullscreen mode
            demo_mode: Whether to generate demo events
        """
        self.fullscreen = fullscreen
        self.demo_mode = demo_mode
        self.running = True
        self.screen = None
        self.clock = None
        self.timeline = None
        self.vitals = None
        self.adapter = None
        self.event_bus = get_event_bus()
        self.display_config = get_display_config()
        self.last_demo_event = 0
        self.demo_events = []
        self.demo_thread = None
        
    def setup(self):
        """Set up the display components and event integration."""
        logger.info("Setting up integrated display...")
        
        # Initialize pygame
        pygame.init()
        
        # Create window
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            width, height = self.screen.get_size()
        else:
            width, height = 1280, 720
            self.screen = pygame.display.set_mode((width, height))
            
        pygame.display.set_caption("TCCC Integrated Display")
        
        # Create clock
        self.clock = pygame.time.Clock()
        
        # Create visualization components
        timeline_rect = pygame.Rect(50, 50, width - 100, height // 2 - 75)
        self.timeline = TimelineVisualization(self.screen, timeline_rect)
        
        vitals_rect = pygame.Rect(50, height // 2 + 25, width - 100, height // 2 - 75)
        self.vitals = VitalSignsVisualization(self.screen, vitals_rect)
        
        # Create adapter
        self.adapter = DisplayEventAdapter(self.timeline, self.vitals, "integrated_display")
        
        # Set up demo events if in demo mode
        if self.demo_mode:
            self.setup_demo_events()
            self.start_demo_thread()
            
        logger.info("Setup complete")
        
    def setup_demo_events(self):
        """Set up demo events for simulation."""
        from tccc.utils.event_schema import (
            TranscriptionEvent, LLMAnalysisEvent, ProcessedTextEvent
        )
        
        # Create a series of events that simulates a realistic field medical scenario
        self.demo_events = [
            # Initial assessment
            (0, TranscriptionEvent(
                source="stt_engine",
                text="Beginning patient assessment. Casualty is a 25-year-old male with injuries from an IED blast.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # Initial vitals
            (3, TranscriptionEvent(
                source="stt_engine",
                text="Initial vital signs: heart rate 135, blood pressure 85/60, respiratory rate 28, oxygen saturation 90%.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # Primary injury
            (6, TranscriptionEvent(
                source="stt_engine",
                text="Patient has significant bleeding from left thigh with arterial spurting. Applying tourniquet now.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # LLM analysis of bleeding
            (7, LLMAnalysisEvent(
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
                session_id="demo_session"
            )),
            
            # Treatment confirmation
            (10, TranscriptionEvent(
                source="stt_engine",
                text="Tourniquet applied successfully at 14:32. Bleeding has stopped. Moving on to assess airway.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # Secondary assessment
            (13, TranscriptionEvent(
                source="stt_engine",
                text="Checking airway. Patient is responsive but has difficulty breathing. Possible tension pneumothorax on right side.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # LLM analysis of pneumothorax
            (14, LLMAnalysisEvent(
                source="llm_analysis",
                summary="Patient exhibits signs consistent with tension pneumothorax requiring immediate decompression.",
                topics=["tension pneumothorax", "chest injury", "needle decompression"],
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
                        "description": "Perform needle decompression at the second intercostal space, midclavicular line on the affected side"
                    }
                ],
                session_id="demo_session"
            )),
            
            # Treatment for pneumothorax
            (17, TranscriptionEvent(
                source="stt_engine",
                text="Performing needle decompression on right side. Inserted 14-gauge needle at second intercostal space, midclavicular line.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # Improved vitals
            (20, TranscriptionEvent(
                source="stt_engine",
                text="Vital signs improving: heart rate 115, blood pressure 95/65, respiratory rate 22, oxygen saturation 94%.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # Pain management
            (23, TranscriptionEvent(
                source="stt_engine",
                text="Patient reporting pain level of 8 out of 10. Administering 10mg morphine IV for pain management.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # Processed text for pain management
            (24, ProcessedTextEvent(
                source="processing_core",
                text="Patient reporting pain level of 8 out of 10. Administering 10mg morphine IV for pain management.",
                entities=[
                    {
                        "text": "pain level of 8",
                        "type": "MEDICAL_FINDING",
                        "start": 17,
                        "end": 32,
                        "confidence": 0.93
                    },
                    {
                        "text": "10mg morphine",
                        "type": "MEDICATION",
                        "start": 46,
                        "end": 59,
                        "confidence": 0.97
                    }
                ],
                intent={
                    "name": "administer_medication",
                    "confidence": 0.91,
                    "slots": {
                        "medication": "morphine",
                        "dosage": "10mg",
                        "route": "IV",
                        "reason": "pain management"
                    }
                },
                session_id="demo_session"
            )),
            
            # Evacuation planning
            (27, TranscriptionEvent(
                source="stt_engine",
                text="Requesting MEDEVAC. Patient is stable enough for transport. ETA for helicopter is 10 minutes.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            )),
            
            # Final vitals
            (30, TranscriptionEvent(
                source="stt_engine",
                text="Final vital signs before transport: heart rate 105, blood pressure 100/70, respiratory rate 18, oxygen saturation 96%.",
                segments=[],
                language="en",
                confidence=0.95,
                is_partial=False,
                session_id="demo_session"
            ))
        ]
    
    def start_demo_thread(self):
        """Start a thread to publish demo events."""
        if self.demo_thread is not None:
            return
            
        self.demo_thread = threading.Thread(target=self.publish_demo_events)
        self.demo_thread.daemon = True
        self.demo_thread.start()
        
    def publish_demo_events(self):
        """Publish demo events with timing."""
        logger.info("Starting demo event publication...")
        
        start_time = time.time()
        event_index = 0
        
        while event_index < len(self.demo_events) and self.running:
            current_time = time.time() - start_time
            event_time, event = self.demo_events[event_index]
            
            if current_time >= event_time:
                # Time to publish this event
                logger.info(f"Publishing demo event {event_index+1}/{len(self.demo_events)}")
                self.event_bus.publish(event)
                event_index += 1
                
            time.sleep(0.1)
            
        logger.info("Demo event publication complete")
        
    def run(self):
        """Run the integrated display application."""
        self.setup()
        
        # Main loop
        while self.running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keypress(event.key)
                    
            # Update display
            self.screen.fill((0, 0, 0))
            if self.timeline:
                self.timeline.draw()
            if self.vitals:
                self.vitals.draw()
                
            # Draw status info
            self.draw_status_info()
                
            # Update screen
            pygame.display.flip()
            
            # Cap frame rate
            self.clock.tick(30)
            
        # Clean up
        self.cleanup()
        
    def handle_keypress(self, key):
        """Handle keyboard input."""
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_F11:
            # Toggle fullscreen
            pygame.display.toggle_fullscreen()
        elif key == pygame.K_t:
            # Toggle timeline compact mode
            if self.timeline:
                self.timeline.toggle_compact_mode()
        elif key == pygame.K_v:
            # Toggle vital signs compact mode
            if self.vitals:
                self.vitals.toggle_compact_mode()
        elif key == pygame.K_c:
            # Toggle critical event filter
            if self.timeline:
                if self.timeline.filter_type == "critical":
                    self.timeline.set_filter(None)
                else:
                    self.timeline.set_filter("critical")
        elif key == pygame.K_d:
            # Generate a demo event (if not in demo mode)
            if not self.demo_mode and time.time() - self.last_demo_event > 1.0:
                self.generate_random_event()
                self.last_demo_event = time.time()
                
    def draw_status_info(self):
        """Draw status information overlay."""
        if not self.screen:
            return
            
        # Create a small font for status info
        try:
            font = pygame.font.SysFont('Arial', 16)
        except:
            font = pygame.font.Font(None, 16)
            
        # Draw event bus status
        stats = self.event_bus.get_stats()
        status_text = f"Events: {stats['events_published']} | Subscribers: {stats['subscriber_count']}"
        
        if self.demo_mode:
            status_text += " | DEMO MODE ACTIVE"
            
        status_surface = font.render(status_text, True, (200, 200, 200))
        self.screen.blit(status_surface, (10, self.screen.get_height() - 30))
        
        # Draw controls help
        controls_text = "ESC: Exit | T: Toggle Timeline | V: Toggle Vitals | C: Filter Critical | F11: Fullscreen"
        controls_surface = font.render(controls_text, True, (200, 200, 200))
        self.screen.blit(controls_surface, (10, self.screen.get_height() - 50))
        
    def generate_random_event(self):
        """Generate a random event for testing."""
        import random
        from tccc.utils.event_schema import TranscriptionEvent
        
        # Create a random transcription event with medical information
        vital_texts = [
            "Heart rate is now 120 bpm",
            "Blood pressure reading 110/70",
            "Oxygen saturation at 95%",
            "Respiratory rate 18 breaths per minute",
            "Temperature 37.2 degrees Celsius"
        ]
        
        event_text = random.choice(vital_texts)
        
        # Create and publish event
        event = TranscriptionEvent(
            source="demo_generator",
            text=event_text,
            segments=[],
            language="en",
            confidence=0.95,
            is_partial=False,
            session_id="manual_demo"
        )
        
        self.event_bus.publish(event)
        
    def cleanup(self):
        """Clean up resources."""
        # Stop demo thread
        self.running = False
        if self.demo_thread:
            self.demo_thread.join(timeout=1.0)
            
        # Unregister from event bus
        if self.adapter:
            self.adapter.unregister()
            
        # Quit pygame
        pygame.quit()
        
        logger.info("Display cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TCCC Integrated Display")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with simulated events")
    args = parser.parse_args()
    
    display = TCCCIntegratedDisplay(
        fullscreen=args.fullscreen,
        demo_mode=args.demo
    )
    
    display.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF

# Make the script executable
chmod +x tccc_integrated_display.py

# Make the verification script executable too
chmod +x verification_script_display_integration.py

# Launch the integrated display
if [ "$1" == "--test" ]; then
    echo "Running display integration verification..."
    python verification_script_display_integration.py
    exit $?
elif [ "$1" == "--fullscreen" ]; then
    echo "Launching TCCC integrated display in fullscreen mode..."
    python tccc_integrated_display.py --fullscreen --demo
else
    echo "Launching TCCC integrated display..."
    python tccc_integrated_display.py --demo
fi