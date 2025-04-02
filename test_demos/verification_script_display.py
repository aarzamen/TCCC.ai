#!/usr/bin/env python3
"""
Display Interface Verification Script
------------------------------------
Tests the display interface for the TCCC.ai system.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayVerification")

try:
    from tccc.display.display_interface import DisplayInterface
except ImportError:
    logger.error("Failed to import DisplayInterface")
    logger.info("Make sure you have activated the virtual environment and installed the package")
    sys.exit(1)

def main():
    """Run verification tests for display interface"""
    logger.info("Starting display interface verification")
    
    # Check if running on Jetson with display
    if not os.environ.get("DISPLAY"):
        logger.warning("No display detected. Running in headless mode might cause issues.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Verification cancelled")
            return
    
    # Create and initialize display
    display = DisplayInterface(width=800, height=480, fullscreen=False)
    
    try:
        # Start display
        logger.info("Starting display interface")
        display.start()
        
        # Test basic functionality
        logger.info("Testing transcription updates")
        display.update_transcription("Display verification test started")
        time.sleep(1)
        
        display.update_transcription("Testing transcription display functionality")
        time.sleep(1)
        
        logger.info("Testing significant event updates")
        display.add_significant_event("Verification test started")
        time.sleep(1)
        
        display.add_significant_event("Testing event tracking")
        time.sleep(1)
        
        logger.info("Testing TCCC card data updates")
        display.update_card_data({
            "name": "Test Patient",
            "rank": "SGT",
            "unit": "Test Unit",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "mechanism_of_injury": "Test Injury",
            "injuries": "Test injury description for display verification",
            "vital_signs": "HR 75, BP 120/80, RR 18",
            "treatment_given": "Test treatment for display verification",
            "medications": "No medications administered",
            "evacuation_priority": "Routine"
        })
        
        # Test display mode toggling
        logger.info("Testing display mode toggle")
        time.sleep(3)
        display.toggle_display_mode()  # Switch to card view
        
        logger.info("Showing TCCC Card view for 5 seconds")
        time.sleep(5)
        
        display.toggle_display_mode()  # Switch back to live view
        
        # Final verification messages
        display.update_transcription("All display tests completed successfully")
        display.add_significant_event("Display verification passed")
        
        logger.info("Display verification completed successfully")
        logger.info("Press Ctrl+C to exit or wait 10 seconds for automatic exit")
        
        # Wait for manual exit or timeout
        timeout = time.time() + 10  # 10 seconds from now
        while time.time() < timeout:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
    except Exception as e:
        logger.error(f"Error during display verification: {e}")
    finally:
        # Clean up
        logger.info("Stopping display interface")
        display.stop()

if __name__ == "__main__":
    main()