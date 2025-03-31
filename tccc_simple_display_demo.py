#!/usr/bin/env python3
"""
TCCC.ai Simple Display Demo
--------------------------
A simple demo that cycles through mock data on the display
until touched or a key is pressed.
"""

import os
import sys
import time
import math
import pygame
import random
from datetime import datetime
from pygame.locals import *

# Set environment variables for WaveShare display
os.environ["TCCC_ENABLE_DISPLAY"] = "1"
os.environ["TCCC_DISPLAY_RESOLUTION"] = "1280x800"

# Import the DisplayInterface class
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.tccc.display.display_interface import DisplayInterface

class SimpleTCCCDisplayDemo:
    """Simple demo of dynamic display updates with auto loop"""
    
    def __init__(self):
        """Initialize the demo with mock data"""
        # Display setup
        self.display = DisplayInterface(width=1280, height=800, fullscreen=False)
        self.display.initialize()
        
        # Data that will be cycled through
        self.mock_transcriptions = [
            "Starting patient assessment...",
            "Patient has a gunshot wound to the left leg.",
            "Applying tourniquet two inches above the wound site.",
            "Checking for other injuries... patient also has shrapnel wounds to right arm.",
            "Administering 10mg morphine IV at 14:35.",
            "Checking vital signs: HR 110, BP 90/60, RR 22.",
            "Patient stabilized, preparing for evacuation.",
            "Requesting MEDEVAC, priority urgent.",
            "Estimated time to evacuation: 10 minutes.",
            "Continuing to monitor vital signs, BP now 100/70.",
        ]
        
        self.mock_events = [
            {"time": "14:30", "description": "Initial assessment started"},
            {"time": "14:32", "description": "GSW identified - left thigh, arterial bleeding"},
            {"time": "14:33", "description": "Tourniquet applied to left thigh"},
            {"time": "14:35", "description": "Shrapnel wounds identified - right arm"},
            {"time": "14:36", "description": "Morphine 10mg administered IV"},
            {"time": "14:38", "description": "Vital signs: HR 110, BP 90/60, RR 22"},
            {"time": "14:40", "description": "Patient stabilized for evacuation"},
            {"time": "14:42", "description": "MEDEVAC requested - priority urgent"},
            {"time": "14:45", "description": "Vital signs: HR 105, BP 100/70, RR 20"},
            {"time": "14:47", "description": "ETA for evacuation: 10 minutes"},
        ]
        
        self.mock_card_data = {
            "name": "John Doe",
            "rank": "SGT",
            "unit": "1st Battalion, 3rd Marines",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": "14:30",
            "mechanism_of_injury": "GSW + Shrapnel",
            "injuries": "GSW left thigh with arterial bleeding, shrapnel wounds to right arm",
            "vital_signs": "HR 105, BP 100/70, RR 20, O2 94%",
            "treatment_given": "Tourniquet to left thigh at 14:33, pressure dressing to right arm",
            "medications": "Morphine 10mg IV at 14:36",
            "evacuation_priority": "Urgent"
        }
        
        # Runtime variables
        self.current_data_index = 0
        self.update_interval = 2.0  # seconds between updates
        self.last_update_time = time.time()
        self.running = True
        self.view_mode = "live"
        
    def start(self):
        """Start the demo with auto-updating data"""
        # Add initial card data
        self.display.update_card_data(self.mock_card_data)
        
        # Custom loop instead of using display.start()
        # This allows us to control data updates and handle input
        pygame.init()
        self.display.screen = pygame.display.set_mode((self.display.width, self.display.height))
        pygame.display.set_caption("TCCC.ai Simple Display Demo")
        
        # Load assets and prepare display
        self.display._load_fonts()
        self.display._load_assets()
        
        clock = pygame.time.Clock()
        
        print("Display demo running. Press any key or touch screen to exit.")
        print("Press 'T' to toggle between live view and card view.")
        
        # Main loop
        while self.running:
            self.check_input()
            if not self.running:
                break
                
            self.update_display_data()
            
            # Clear screen and draw appropriate view
            self.display.screen.fill((0, 0, 0))
            
            if self.view_mode == "live":
                self.display._draw_live_screen()
            else:
                self.display._draw_card_screen()
            
            # Update display
            pygame.display.flip()
            
            # Cap frame rate
            clock.tick(30)
        
        # Clean up
        pygame.quit()
        print("Display demo ended")
    
    def check_input(self):
        """Check for user input to exit or toggle views"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_t:
                    # Toggle between live and card views
                    self.view_mode = "card" if self.view_mode == "live" else "live"
                    self.display.set_display_mode(self.view_mode)
                else:
                    # Any other key exits
                    self.running = False
            elif event.type == MOUSEBUTTONDOWN:
                # Any screen touch exits
                self.running = False
    
    def update_display_data(self):
        """Update the display with the next piece of mock data"""
        current_time = time.time()
        
        # Only update at specified intervals
        if current_time - self.last_update_time >= self.update_interval:
            # Update with next piece of data if available
            if self.current_data_index < len(self.mock_transcriptions):
                # Add the next transcription
                self.display.update_transcription(
                    self.mock_transcriptions[self.current_data_index]
                )
                
                # Add the next event if available
                if self.current_data_index < len(self.mock_events):
                    self.display.add_significant_event(
                        self.mock_events[self.current_data_index]
                    )
                
                # Update card with progressive data
                if self.current_data_index > 0 and self.current_data_index % 3 == 0:
                    # Update vital signs every 3 steps
                    vitals = [
                        "HR 110, BP 90/60, RR 22",
                        "HR 108, BP 95/65, RR 21",
                        "HR 105, BP 100/70, RR 20",
                        "HR 100, BP 105/75, RR 18"
                    ]
                    vital_index = min((self.current_data_index // 3), len(vitals) - 1)
                    
                    self.display.update_card_data({
                        "vital_signs": vitals[vital_index]
                    })
                
                # Increment for next update
                self.current_data_index += 1
            else:
                # Start over when we've gone through all data
                self.current_data_index = 0
            
            # Update the last update time
            self.last_update_time = current_time

if __name__ == "__main__":
    demo = SimpleTCCCDisplayDemo()
    demo.start()