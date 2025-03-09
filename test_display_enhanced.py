#!/usr/bin/env python3
"""
Enhanced Display Test for TCCC.ai
--------------------------------
Tests the enhanced display interface with all features:
- Live view with three-column layout
- Card view with proper formatting
- Touch input support
- Animation transitions
- Performance monitoring
"""

import os
import sys
import time
import yaml
import argparse
from datetime import datetime

import pygame
from pygame.locals import *

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import display components
from src.tccc.display.display_interface import DisplayInterface

def load_config(config_path):
    """Load display configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TCCC Display Enhanced Test')
    
    # Display options
    parser.add_argument('--width', type=int, help='Display width')
    parser.add_argument('--height', type=int, help='Display height')
    parser.add_argument('--fullscreen', action='store_true', help='Use fullscreen mode')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/display.yaml', 
                       help='Path to display configuration file')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--show-fps', action='store_true', help='Show FPS counter')
    parser.add_argument('--show-touch', action='store_true', help='Show touch points')
    parser.add_argument('--software', action='store_true', help='Use software rendering (no OpenGL)')
    
    # Test options
    parser.add_argument('--duration', type=int, default=60, 
                       help='Test duration in seconds (0 for indefinite)')
    parser.add_argument('--update-interval', type=float, default=1.0,
                       help='Interval between data updates (seconds)')
    
    return parser.parse_args()

def create_test_data():
    """Create sample data for testing"""
    transcriptions = [
        "Starting patient assessment now.",
        "Patient has a gunshot wound to the left thigh with arterial bleeding.",
        "Applying a tourniquet 2 inches above the wound site.",
        "Tourniquet applied at 14:32 local time.",
        "Checking for other injuries... patient also has shrapnel wounds to right arm.",
        "Administering 10mg morphine IV for pain management.",
        "Vital signs: HR 110, BP 90/60, RR 22, O2 94%.",
        "Preparing patient for evacuation."
    ]
    
    events = [
        {"time": "14:30", "description": "Initial assessment started"},
        {"time": "14:31", "description": "GSW identified - left thigh, arterial bleeding"},
        {"time": "14:32", "description": "Tourniquet applied to left thigh"},
        {"time": "14:33", "description": "Shrapnel wounds identified - right arm"},
        {"time": "14:34", "description": "Pressure bandage applied to right arm"},
        {"time": "14:35", "description": "Morphine 10mg administered IV"},
        {"time": "14:37", "description": "Patient stabilized, ready for evacuation"},
        {"time": "14:40", "description": "Evacuation team notified, ETA 10 minutes"}
    ]
    
    card_data = {
        "name": "John Doe",
        "rank": "SGT",
        "unit": "1st Battalion, 3rd Marines",
        "service_id": "12345",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": "14:30",
        "mechanism_of_injury": "GSW + Shrapnel",
        "injuries": "GSW left thigh with arterial bleeding, shrapnel wounds to right arm",
        "vital_signs": "HR 110, BP 90/60, RR 22, O2 94%",
        "treatment_given": "Tourniquet to left thigh at 14:32, pressure dressing to right arm",
        "medications": "Morphine 10mg IV at 14:35",
        "evacuation_priority": "Urgent"
    }
    
    return transcriptions, events, card_data

def add_dynamic_data(display, transcriptions, events, interval=1.0):
    """Add additional dynamic data at regular intervals"""
    # New transcription phrases
    new_transcriptions = [
        "Checking tourniquet effectiveness, still controlling bleeding.",
        "Updating vital signs: BP is now 100/65, slight improvement.",
        "Patient reporting pain level of 6 out of 10.",
        "Administering additional fluids via IV.",
        "Evacuation ETA updated to 5 minutes.",
        "Re-assessing shrapnel wounds, no additional bleeding noted.",
        "Confirming patient identity and unit information.",
        "Recording medication dosage and time in notes."
    ]
    
    # New events
    new_events = [
        {"time": datetime.now().strftime("%H:%M"), "description": "Vital signs updated: HR 105, BP 100/65"},
        {"time": datetime.now().strftime("%H:%M"), "description": "Pain assessment: 6/10"},
        {"time": datetime.now().strftime("%H:%M"), "description": "IV fluids administered: 500ml"},
        {"time": datetime.now().strftime("%H:%M"), "description": "Evacuation ETA updated: 5 minutes"},
        {"time": datetime.now().strftime("%H:%M"), "description": "Secondary assessment completed"},
        {"time": datetime.now().strftime("%H:%M"), "description": "Tourniquet effectiveness confirmed"}
    ]
    
    # Select a random item from each list
    import random
    
    if random.random() < 0.7:  # 70% chance to add transcription
        new_transcription = random.choice(new_transcriptions)
        display.update_transcription(new_transcription)
        print(f"Added transcription: {new_transcription[:30]}...")
    
    if random.random() < 0.5:  # 50% chance to add event
        new_event = random.choice(new_events)
        display.add_significant_event(new_event)
        print(f"Added event: {new_event['description'][:30]}...")
    
    # Update card data occasionally
    if random.random() < 0.3:  # 30% chance to update card
        # Update vital signs
        hr = random.randint(100, 115)
        bp_sys = random.randint(90, 100)
        bp_dia = random.randint(60, 70)
        rr = random.randint(20, 24)
        o2 = random.randint(93, 98)
        
        vital_signs = f"HR {hr}, BP {bp_sys}/{bp_dia}, RR {rr}, O2 {o2}%"
        display.update_card_data({"vital_signs": vital_signs})
        print(f"Updated card vital signs: {vital_signs}")

def main():
    """Run the enhanced display test"""
    args = parse_arguments()
    
    print("===== TCCC Enhanced Display Test =====")
    print(f"Display: {args.width}x{args.height}, Fullscreen: {args.fullscreen}")
    
    # Load configuration
    config = None
    if os.path.exists(args.config):
        config = load_config(args.config)
    
    # Apply command line overrides to config
    if config is None:
        config = {}
    
    if args.width or args.height:
        if 'display' not in config:
            config['display'] = {}
        if args.width:
            config['display']['width'] = args.width
        if args.height:
            config['display']['height'] = args.height
        config['display']['fullscreen'] = args.fullscreen
    
    # Add debug settings from command line
    if args.debug or args.show_fps or args.show_touch:
        if 'advanced' not in config:
            config['advanced'] = {}
        config['advanced']['debug_mode'] = args.debug
        config['advanced']['show_touch_points'] = args.show_touch
        
        if 'performance' not in config:
            config['performance'] = {}
        config['performance']['show_fps'] = args.show_fps
        
    # Set environment variable for software rendering if requested
    if args.software:
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Suppress pygame welcome message
        os.environ['SDL_RENDERER_DRIVER'] = 'software'
        os.environ['SDL_VIDEO_GL_DRIVER'] = ''  # Disable OpenGL
        print("Using software rendering mode (no OpenGL)")
    
    # Create display interface with configuration
    width = args.width or (config.get('display', {}).get('width', 1280))
    height = args.height or (config.get('display', {}).get('height', 720))
    fullscreen = args.fullscreen or (config.get('display', {}).get('fullscreen', False))
    
    print(f"Initializing display: {width}x{height}, Fullscreen: {fullscreen}")
    display = DisplayInterface(width=width, height=height, fullscreen=fullscreen, config=config)
    
    # Initialize display
    if not display.initialize():
        print("Failed to initialize display")
        return 1
    
    print("Display initialized successfully")
    
    # Start display
    display.start()
    print("Display started")
    
    # Load test data
    print("Loading test data...")
    transcriptions, events, card_data = create_test_data()
    
    # Add initial test data
    for transcription in transcriptions:
        display.update_transcription(transcription)
    
    for event in events:
        display.add_significant_event(event)
    
    display.update_card_data(card_data)
    
    # Main test loop
    print(f"Running test for {args.duration} seconds (0 = indefinite)...")
    print("Press 'T' to toggle between live and card views")
    print("Press 'ESC' or close the window to exit")
    
    start_time = time.time()
    last_update = start_time
    running = True
    
    try:
        while running:
            # Check for exit conditions
            current_time = time.time()
            elapsed = current_time - start_time
            
            if args.duration > 0 and elapsed > args.duration:
                print(f"Test completed after {elapsed:.1f} seconds")
                running = False
                break
            
            # Add dynamic data at regular intervals
            if current_time - last_update > args.update_interval:
                add_dynamic_data(display, transcriptions, events, args.update_interval)
                last_update = current_time
                
            # Small sleep to avoid high CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        # Stop display
        print("Stopping display...")
        display.stop()
        print("Display stopped")
    
    print("Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())