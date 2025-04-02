#!/usr/bin/env python3
"""
TCCC.ai Enhanced Display Demo
----------------------------
Demonstrates the enhanced display components with timeline and vital signs visualization.
For use on Jetson Nano with HDMI display or Waveshare touchscreen.
"""

import os
import sys
import time
import random
import argparse
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Configure path to find TCCC modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, "src"))

try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("pygame not installed. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        import pygame
        from pygame.locals import *
        print("pygame installed successfully")
    except Exception as e:
        print(f"Failed to install pygame: {e}")
        raise

# Import TCCC display modules
from tccc.display.display_config import get_display_config
from tccc.display.display_interface import DisplayInterface
from tccc.display.visualization.vital_signs import VitalSignsMonitor
from tccc.display.visualization.timeline import TimelineVisualization, TimelineEvent


class EnhancedDisplayDemo:
    """
    TCCC.ai Enhanced Display Demo with Timeline and Vital Signs
    
    Shows the enhanced display components with real-time updates and simulated data
    for demonstration purposes.
    """
    
    def __init__(self, fullscreen=False, resolution=None, test_data=True):
        """
        Initialize the enhanced display demo.
        
        Args:
            fullscreen: Whether to run in fullscreen mode
            resolution: Optional resolution override (width, height)
            test_data: Whether to generate random test data
        """
        self.fullscreen = fullscreen
        self.custom_resolution = resolution
        self.generate_test_data = test_data
        
        # Get display configuration
        self.config = get_display_config().get_config()
        
        # Set up pygame
        pygame.init()
        
        # Initialize display
        self.setup_display()
        
        # Initialize components
        self.display_interface = None
        self.vital_signs = None
        self.timeline = None
        
        # Initialize state
        self.running = False
        self.clock = pygame.time.Clock()
        self.fps = self.config["performance"]["fps_limit"]
        self.last_test_data_time = time.time()
        self.test_data_interval = 5.0  # seconds between test data updates
        
        # Create threading event for clean shutdown
        self.stop_event = threading.Event()
        
    def setup_display(self):
        """Set up the display with the configured resolution."""
        # Get resolution from config or use custom resolution
        if self.custom_resolution:
            width, height = self.custom_resolution
        else:
            width, height = get_display_config().get_display_dimensions()
            
        # Set up display flags
        display_flags = 0
        if self.fullscreen:
            display_flags |= pygame.FULLSCREEN
            
        # Additional flags based on platform
        if os.environ.get('SDL_RENDERER_DRIVER') != 'software':
            display_driver = pygame.display.get_driver() if pygame.display.get_init() else "unknown"
            if display_driver in ("kmsdrm", "wayland", "x11"):
                display_flags |= pygame.HWSURFACE | pygame.DOUBLEBUF
                
        # Create display
        self.screen = pygame.display.set_mode((width, height), display_flags)
        pygame.display.set_caption("TCCC.ai Enhanced Display Demo")
        
        # Set dimensions
        self.width, self.height = width, height
        
        print(f"Display initialized at {width}x{height}")
        
    def initialize_components(self):
        """Initialize all display components."""
        # Create main display interface
        self.display_interface = DisplayInterface(
            width=self.width,
            height=self.height, 
            fullscreen=self.fullscreen,
            config=self.config
        )
        
        # Initialize display interface
        if not self.display_interface.initialize():
            print("Failed to initialize display interface")
            return False
            
        # Create vital signs component
        vitals_rect = pygame.Rect(
            20, 
            80, 
            self.width // 3 - 30, 
            200
        )
        self.vital_signs = VitalSignsMonitor(self.screen, vitals_rect, self.config)
        
        # Create timeline component
        timeline_rect = pygame.Rect(
            20,
            300,
            self.width - 40,
            self.height - 350
        )
        self.timeline = TimelineVisualization(self.screen, timeline_rect, self.config)
        
        # Add some initial sample data
        self._initialize_sample_data()
        
        return True
        
    def _initialize_sample_data(self):
        """Initialize with sample medical data for demonstration."""
        # Add sample vital signs
        self.vital_signs.update_vital("hr", 88)
        self.vital_signs.update_vital("sbp", 125)
        self.vital_signs.update_vital("dbp", 78)
        self.vital_signs.update_vital("rr", 16)
        self.vital_signs.update_vital("spo2", 96)
        self.vital_signs.update_vital("temp", 37.2)
        
        # Add sample timeline events
        now = datetime.now()
        
        # Initial assessment
        self.timeline.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=2),
            event_type="assessment",
            title="Initial Assessment",
            description="Patient responsive but disoriented. Multiple injuries identified."
        ))
        
        # Injuries
        self.timeline.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=1, seconds=30),
            event_type="critical",
            title="Primary Injury",
            description="GSW to left thigh with arterial bleeding."
        ))
        
        # Treatment
        self.timeline.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=1),
            event_type="treatment",
            title="Hemorrhage Control",
            description="CAT tourniquet applied to left thigh, 5cm above wound."
        ))
        
        # Vitals
        self.timeline.add_event(TimelineEvent(
            timestamp=now - timedelta(seconds=30),
            event_type="vital",
            title="Initial Vital Signs",
            description="HR 94, BP 126/84, RR 18, SpO2 96%, GCS 14"
        ))
        
        # Add sample card data to display interface
        self.display_interface.update_card_data({
            "name": "SMITH, JOHN",
            "rank": "SGT",
            "unit": "1st Medical Battalion",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "mechanism_of_injury": "GSW, left thigh",
            "injuries": "Gunshot wound left thigh, arterial bleeding, minor facial lacerations",
            "vital_signs": "HR 94, BP 126/84, RR 18, SpO2 96%",
            "treatment_given": "Tourniquet to left thigh, hemostatic gauze applied, IV 18G right AC",
            "medications": "Morphine 10mg IV at 1432",
            "evacuation_priority": "Priority"
        })
        
        # Add transcription samples
        self.display_interface.update_transcription("Starting patient assessment now.")
        self.display_interface.update_transcription("Patient has a gunshot wound to the left thigh with arterial bleeding.")
        self.display_interface.update_transcription("Applying tourniquet now.")
        self.display_interface.update_transcription("Tourniquet applied. Bleeding is controlled.")
        self.display_interface.update_transcription("Let's check vital signs. Heart rate is 94, blood pressure is 126/84.")
        
        # Add significant events
        self.display_interface.add_significant_event({
            "time": datetime.now().strftime("%H:%M"),
            "description": "GSW left thigh identified"
        })
        
        self.display_interface.add_significant_event({
            "time": datetime.now().strftime("%H:%M"),
            "description": "Arterial bleeding - significant blood loss"
        })
        
        self.display_interface.add_significant_event({
            "time": datetime.now().strftime("%H:%M"),
            "description": "Tourniquet applied to left thigh"
        })
        
        self.display_interface.add_significant_event({
            "time": datetime.now().strftime("%H:%M"),
            "description": "Bleeding controlled successfully"
        })
        
    def generate_random_data(self):
        """Generate random medical data updates for demonstration."""
        # Only generate data at defined intervals
        current_time = time.time()
        if current_time - self.last_test_data_time < self.test_data_interval:
            return
            
        self.last_test_data_time = current_time
        
        # Random vital sign updates with trends
        hr_base = self.vital_signs.vital_data["hr"]["current"]
        hr_new = max(40, min(160, hr_base + random.randint(-5, 5)))
        self.vital_signs.update_vital("hr", hr_new)
        
        sbp_base = self.vital_signs.vital_data["sbp"]["current"]
        sbp_new = max(70, min(200, sbp_base + random.randint(-8, 8)))
        self.vital_signs.update_vital("sbp", sbp_new)
        
        dbp_base = self.vital_signs.vital_data["dbp"]["current"]
        dbp_new = max(40, min(110, dbp_base + random.randint(-5, 5)))
        self.vital_signs.update_vital("dbp", dbp_new)
        
        rr_base = self.vital_signs.vital_data["rr"]["current"]
        rr_new = max(8, min(30, rr_base + random.randint(-2, 2)))
        self.vital_signs.update_vital("rr", rr_new)
        
        spo2_base = self.vital_signs.vital_data["spo2"]["current"]
        spo2_new = max(85, min(100, spo2_base + random.randint(-2, 2)))
        self.vital_signs.update_vital("spo2", spo2_new)
        
        # Update transcription with new entries occasionally
        if random.random() < 0.3:
            transcription_options = [
                "Checking capillary refill...",
                "Patient GCS is now 15, fully alert.",
                "IV fluids running at 100ml/hr.",
                f"Updated vitals: HR {hr_new}, BP {sbp_new}/{dbp_new}.",
                "Patient reporting pain level 6/10.",
                "Preparing for transport to medical facility.",
                "Securing patient for movement.",
                "Administering additional analgesia.",
                "Reassessing airway status.",
                "Documenting all interventions."
            ]
            new_transcription = random.choice(transcription_options)
            self.display_interface.update_transcription(new_transcription)
            
            # Add to timeline if it contains specific keywords
            if any(kw in new_transcription.lower() for kw in ["vital", "hr", "bp", "gcs"]):
                self.timeline.add_event_from_text(new_transcription)
                
        # Add significant events occasionally
        if random.random() < 0.2:
            event_options = [
                {
                    "time": datetime.now().strftime("%H:%M"),
                    "description": "IV fluids started - normal saline"
                },
                {
                    "time": datetime.now().strftime("%H:%M"),
                    "description": f"Patient vitals: HR {hr_new}, trending {'up' if hr_new > hr_base else 'down'}"
                },
                {
                    "time": datetime.now().strftime("%H:%M"),
                    "description": "Pain management - additional analgesia administered"
                },
                {
                    "time": datetime.now().strftime("%H:%M"),
                    "description": "Re-assessing all injuries and interventions"
                },
                {
                    "time": datetime.now().strftime("%H:%M"),
                    "description": "Patient airway remains patent"
                }
            ]
            
            new_event = random.choice(event_options)
            self.display_interface.add_significant_event(new_event)
            
        # Update the display interface card data occasionally
        if random.random() < 0.1:
            updated_vitals = f"HR {hr_new}, BP {sbp_new}/{dbp_new}, RR {rr_new}, SpO2 {spo2_new}%"
            self.display_interface.update_card_data({
                "vital_signs": updated_vitals
            })
            
            # Add a treatment or intervention occasionally
            if random.random() < 0.3:
                treatment_options = [
                    "Pressure dressing reinforced",
                    "Second IV established - 18G left AC",
                    "Fluid bolus initiated - 500ml NS",
                    "Warmed blankets applied - preventing hypothermia",
                    "Oxygen applied via nasal cannula at 2L/min",
                    "C-spine precautions maintained",
                    "Bandages checked and reinforced",
                    "Pneumatic splint applied to stabilize injury"
                ]
                
                treatment = random.choice(treatment_options)
                current_treatments = self.display_interface.card_data.get("treatment_given", "")
                
                if treatment not in current_treatments:
                    updated_treatment = f"{current_treatments}, {treatment}"
                    self.display_interface.update_card_data({
                        "treatment_given": updated_treatment
                    })
                    
                    # Add to timeline
                    self.timeline.add_event_from_text(treatment)
            
    def handle_input(self):
        """Handle user input events."""
        for event in pygame.event.get():
            # Window close button
            if event.type == pygame.QUIT:
                self.running = False
                
            # Key presses
            elif event.type == pygame.KEYDOWN:
                # Escape exits the demo
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    
                # Toggle display mode with Tab key
                elif event.key == pygame.K_TAB:
                    if self.display_interface:
                        self.display_interface.toggle_display_mode()
                        
                # Toggle compact mode for vital signs with 'v'
                elif event.key == pygame.K_v:
                    if self.vital_signs:
                        self.vital_signs.toggle_compact_mode()
                        
                # Toggle compact mode for timeline with 't'
                elif event.key == pygame.K_t:
                    if self.timeline:
                        self.timeline.toggle_compact_mode()
                        
                # Generate test event with 'e'
                elif event.key == pygame.K_e:
                    self._generate_test_event()
                        
            # Mouse clicks
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle timeline clicks
                if self.timeline:
                    self.timeline.handle_click(event.pos)
                    
    def _generate_test_event(self):
        """Generate a test medical event for demonstration."""
        event_options = [
            {
                "type": "critical",
                "title": "Critical Finding",
                "description": "Patient developing signs of hypovolemic shock."
            },
            {
                "type": "treatment",
                "title": "Treatment Applied",
                "description": "Applied hemostatic dressing to secondary wound."
            },
            {
                "type": "medication",
                "title": "Medication Given",
                "description": "Administered tranexamic acid 1g IV."
            },
            {
                "type": "assessment",
                "title": "Reassessment",
                "description": "MARCH assessment repeated. No new injuries found."
            },
            {
                "type": "transport",
                "title": "Transport Update",
                "description": "MEDEVAC arriving in approximately 5 minutes."
            }
        ]
        
        # Select a random event
        new_event = random.choice(event_options)
        
        # Add to timeline
        self.timeline.add_event_from_dict(new_event)
        
        # Add to display interface as significant event
        self.display_interface.add_significant_event({
            "time": datetime.now().strftime("%H:%M"),
            "description": new_event["description"]
        })
        
        # Add to transcription
        self.display_interface.update_transcription(
            f"{new_event['title']}: {new_event['description']}"
        )
        
    def run(self):
        """Run the main application loop."""
        # Initialize all components
        if not self.initialize_components():
            print("Failed to initialize components")
            return False
            
        # Start the display interface thread
        self.display_interface.start()
        
        # Main loop
        self.running = True
        try:
            while self.running and not self.stop_event.is_set():
                # Handle input
                self.handle_input()
                
                # Generate random test data if enabled
                if self.generate_test_data:
                    self.generate_random_data()
                
                # Clear the screen
                self.screen.fill((0, 0, 0))
                
                # Draw vital signs component
                if self.vital_signs:
                    self.vital_signs.draw()
                    
                # Draw timeline component
                if self.timeline:
                    self.timeline.draw()
                
                # Update the display
                pygame.display.flip()
                
                # Cap the frame rate
                self.clock.tick(self.fps)
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Clean up
            self.cleanup()
            
        return True
            
    def cleanup(self):
        """Clean up resources and exit gracefully."""
        print("Cleaning up...")
        
        # Stop the display interface thread
        if self.display_interface:
            self.display_interface.stop()
            
        # Quit pygame
        pygame.quit()
        
        print("Cleanup complete")
        

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TCCC.ai Enhanced Display Demo")
    
    parser.add_argument("--fullscreen", action="store_true", 
                        help="Run in fullscreen mode")
    parser.add_argument("--width", type=int, default=None,
                        help="Custom display width")
    parser.add_argument("--height", type=int, default=None,
                        help="Custom display height")
    parser.add_argument("--no-test-data", action="store_true",
                        help="Disable random test data generation")
    
    return parser.parse_args()
    

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set custom resolution if provided
    resolution = None
    if args.width and args.height:
        resolution = (args.width, args.height)
    
    # Create and run the demo
    demo = EnhancedDisplayDemo(
        fullscreen=args.fullscreen,
        resolution=resolution,
        test_data=not args.no_test_data
    )
    
    # Run the demo
    demo.run()