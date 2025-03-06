#!/usr/bin/env python3
"""
Headless Display Test for TCCC.ai
----------------------------------------
Tests the display interface in a headless environment using the dummy video driver
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeadlessTest")

# Force the use of dummy drivers for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["DISPLAY"] = ":0"  # Fake display for X11

# Enable debug mode for more output
os.environ["SDL_DEBUG"] = "1"

# Import pygame
import pygame
from pygame.locals import *

# Initialize pygame
pygame.init()

# Patch pygame for headless mode
# Monkey patch pygame.display.set_mode to work in headless mode
original_set_mode = pygame.display.set_mode
def dummy_set_mode(size=(800, 600), flags=0, depth=0, display=0, vsync=0):
    logger.info(f"Creating dummy display surface {size}")
    return pygame.Surface(size)
pygame.display.set_mode = dummy_set_mode

# Create dummy font handling
class DummyFont:
    def __init__(self, *args, **kwargs):
        self.size_multiplier = 0.5
        
    def render(self, text, antialias, color, background=None):
        # Create a dummy surface for the text
        text_width = int(len(text) * 10 * self.size_multiplier)
        text_height = int(20 * self.size_multiplier)
        surface = pygame.Surface((text_width, text_height))
        surface.fill((0, 0, 0))  # Black background
        return surface
        
    def size(self, text):
        return (int(len(text) * 10 * self.size_multiplier), int(20 * self.size_multiplier))

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import display components after patching
from src.tccc.display.display_interface import DisplayInterface

def main():
    """Run a basic test of the display interface in headless mode"""
    print("Starting Headless Display Test...")
    
    # Create a display with smaller dimensions for testing
    display = DisplayInterface(width=800, height=480, fullscreen=False)
    
    # Monkey patch the display's initialize function for headless mode
    original_initialize = display.initialize
    
    # Also patch the font loading function
    original_load_fonts = display._load_fonts
    def patched_load_fonts():
        logger.info("Loading dummy fonts")
        # Create dummy fonts
        small_font = DummyFont()
        small_font.size_multiplier = 0.6
        medium_font = DummyFont()
        medium_font.size_multiplier = 0.8
        large_font = DummyFont()
        large_font.size_multiplier = 1.0
        
        display.fonts = {
            'small': small_font,
            'medium': medium_font,
            'large': large_font,
            'bold_small': small_font,
            'bold_medium': medium_font,
            'bold_large': large_font,
        }
        return True
    display._load_fonts = patched_load_fonts
    
    def patched_initialize():
        logger.info("Using headless display initialization")
        display.initialized = True
        # Create basic required objects
        display.screen = pygame.Surface((display.width, display.height))
        display.clock = pygame.time.Clock()
        display._load_fonts()
        return True
    display.initialize = patched_initialize
    
    print("Initializing display in headless mode...")
    if not display.initialize():
        print("Failed to initialize display")
        return 1
    
    print("Display initialized in headless mode")
    
    # Add some test data
    print("Adding test data...")
    display.update_transcription("Patient has a gunshot wound to the left leg.")
    display.add_significant_event("Gunshot wound identified - left thigh")
    display.update_transcription("I'm applying a tourniquet now.")
    display.add_significant_event("Tourniquet applied to left thigh")
    
    # Add card data
    display.update_card_data({
        "name": "John Doe",
        "rank": "SGT",
        "unit": "1st Battalion, 3rd Marines",
        "mechanism_of_injury": "GSW",
        "injuries": "GSW left thigh, arterial bleeding",
        "treatment_given": "Tourniquet applied to left thigh at 14:32",
        "vital_signs": "HR 110, BP 100/60, RR 22",
        "medications": "Morphine 10mg IV at 14:35",
        "evacuation_priority": "Urgent"
    })
    
    # Toggle display modes
    print("Testing display mode switching...")
    print("Current mode:", display.display_mode)
    display.toggle_display_mode()
    print("Toggled to mode:", display.display_mode)
    display.toggle_display_mode()
    print("Toggled back to mode:", display.display_mode)
    
    # Test that the lock works
    with display.lock:
        display.card_data["test"] = "Lock is working"
    print("Lock test passed")
    
    # Test data access
    print(f"Transcription entries: {len(display.transcription)}")
    print(f"Event entries: {len(display.significant_events)}")
    print(f"Card data entries: {len(display.card_data)}")
    
    print("Headless display test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())