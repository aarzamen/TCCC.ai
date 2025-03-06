#!/usr/bin/env python3
"""
Basic test for the display interface
"""

import os
import sys
import time

# Set dummy display driver for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tccc.display.display_interface import DisplayInterface

def main():
    """Run a basic test of the display interface"""
    print("Creating display interface...")
    display = DisplayInterface(width=800, height=480, fullscreen=False)
    
    print("Initializing display...")
    if not display.initialize():
        print("Failed to initialize display")
        return 1
    
    print("Display initialized successfully")
    
    print("Starting display...")
    display.start()
    print("Display started")
    
    print("Adding test data...")
    display.update_transcription("Test transcription 1")
    display.update_transcription("Test transcription 2")
    display.add_significant_event("Test event 1")
    display.add_significant_event("Test event 2")
    
    print("Updating card data...")
    display.update_card_data({
        "name": "Test Patient",
        "rank": "SGT",
        "unit": "Test Unit",
        "mechanism_of_injury": "Test Injury",
        "injuries": "Test injuries"
    })
    
    print("Toggling display mode...")
    display.toggle_display_mode()
    time.sleep(1)
    display.toggle_display_mode()
    
    print("Stopping display...")
    display.stop()
    print("Display stopped")
    
    print("Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())