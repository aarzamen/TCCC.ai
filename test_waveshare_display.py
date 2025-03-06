#!/usr/bin/env python3
"""
WaveShare 6.25" Display Test for TCCC.ai
---------------------------------------
Tests the display interface with the WaveShare 6.25" portrait display (720x1560)
"""

import os
import sys
import time
from datetime import datetime

# Set the environment variables to simulate the WaveShare display
os.environ["TCCC_ENABLE_DISPLAY"] = "1"
os.environ["TCCC_DISPLAY_RESOLUTION"] = "1560x720"  # Landscape orientation
os.environ["TCCC_DISPLAY_TYPE"] = "waveshare_6_25"

# Set dummy display driver for headless testing (comment these out when running on actual hardware)
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
from pygame.locals import *

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import display components
from src.tccc.display.display_interface import DisplayInterface

def draw_performance_metrics(screen, clock, start_time, frame_count):
    """Draw performance metrics on the screen"""
    if frame_count % 30 == 0:  # Update every 30 frames
        # Calculate FPS
        fps = clock.get_fps()
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Display information
        font = pygame.font.SysFont('courier', 16)
        
        fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 0))
        runtime_text = font.render(f"Runtime: {runtime:.1f}s", True, (255, 255, 0))
        
        # Draw semi-transparent background for metrics
        metrics_bg = pygame.Surface((200, 40), pygame.SRCALPHA)
        metrics_bg.fill((0, 0, 0, 128))  # Semi-transparent black
        screen.blit(metrics_bg, (10, 10))
        
        # Draw metrics
        screen.blit(fps_text, (15, 15))
        screen.blit(runtime_text, (15, 30))

def main():
    """Run a comprehensive test of the WaveShare display interface"""
    print("Starting WaveShare 6.25\" Display Test...")
    
    # Initialize display with WaveShare dimensions
    display = DisplayInterface(width=720, height=1560, fullscreen=False)
    
    print("Initializing display...")
    if not display.initialize():
        print("Failed to initialize display")
        return 1
    
    print("Display initialized successfully")
    
    # Start display manually instead of using display.start()
    # This allows us to directly control the loop for testing
    display.screen = pygame.display.set_mode((display.width, display.height))
    pygame.display.set_caption("TCCC.ai WaveShare Display Test")
    
    # Load fonts for display
    display.fonts = {
        'small': pygame.font.SysFont('Arial', 22),
        'medium': pygame.font.SysFont('Arial', 28),
        'large': pygame.font.SysFont('Arial', 36),
        'bold_small': pygame.font.SysFont('Arial', 22, bold=True),
        'bold_medium': pygame.font.SysFont('Arial', 28, bold=True),
        'bold_large': pygame.font.SysFont('Arial', 36, bold=True)
    }
    
    # Load logo if available
    display.avatar = None
    avatar_path = os.path.join(os.path.dirname(__file__), "images", "blue_logo.png")
    if os.path.exists(avatar_path):
        try:
            display.avatar = pygame.image.load(avatar_path)
            display.avatar = pygame.transform.scale(display.avatar, (80, 80))
        except Exception as e:
            print(f"Failed to load avatar image: {e}")
    
    # Create test data
    test_data = [
        "Patient has a gunshot wound to the left thigh with severe arterial bleeding.",
        "Applying a tourniquet now, 2 inches above the wound site.",
        "Tourniquet applied at 14:32 local time.",
        "Checking for other injuries... patient also has shrapnel wounds to right arm.",
        "Administering 10mg morphine IV for pain management.",
        "Vital signs: HR 110, BP 90/60, RR 22, O2 94%."
    ]
    
    test_events = [
        {"time": "14:30", "description": "Initial assessment started"},
        {"time": "14:31", "description": "GSW identified - left thigh, arterial bleeding"},
        {"time": "14:32", "description": "Tourniquet applied to left thigh"},
        {"time": "14:33", "description": "Shrapnel wounds identified - right arm"},
        {"time": "14:35", "description": "Morphine 10mg administered IV"},
        {"time": "14:37", "description": "Patient stabilized, ready for evacuation"}
    ]
    
    test_card_data = {
        "name": "John Doe",
        "rank": "SGT",
        "unit": "1st Battalion, 3rd Marines",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": "14:30",
        "mechanism_of_injury": "GSW + Shrapnel",
        "injuries": "GSW left thigh with arterial bleeding, shrapnel wounds to right arm",
        "vital_signs": "HR 110, BP 90/60, RR 22, O2 94%",
        "treatment_given": "Tourniquet to left thigh at 14:32, pressure dressing to right arm",
        "medications": "Morphine 10mg IV at 14:35",
        "evacuation_priority": "Urgent"
    }
    
    # Add test data to display
    for item in test_data:
        display.update_transcription(item)
    
    for event in test_events:
        display.add_significant_event(event)
    
    display.update_card_data(test_card_data)
    
    # Main display loop
    clock = pygame.time.Clock()
    running = True
    mode = "live"  # Start in live view
    start_time = time.time()
    frame_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_t:
                    # Toggle between live and card views
                    mode = "card" if mode == "live" else "live"
                elif event.key == K_SPACE:
                    # Add a new transcription on space
                    display.update_transcription(f"New transcription at {datetime.now().strftime('%H:%M:%S')}")
        
        # Clear screen
        display.screen.fill((0, 0, 0))
        
        # Draw appropriate view
        if mode == "live":
            display._draw_live_screen()
        else:
            display._draw_card_screen()
        
        # Draw performance metrics
        draw_performance_metrics(display.screen, clock, start_time, frame_count)
        
        # Update display
        pygame.display.flip()
        
        # Increment frame counter
        frame_count += 1
        
        # Cap at 30 FPS to save resources
        clock.tick(30)
    
    # Clean up
    pygame.quit()
    print("Display test completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())