#!/usr/bin/env python3
"""
Display Interface Verification Script - Enhanced
-----------------------------------------------
Tests the enhanced display interface for the TCCC.ai system.
Includes vital signs visualization and performance verification.
"""

import os
import sys
import time
import logging
import pygame
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedDisplayVerification")

# Add project root to path
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir))

# Try to import display components
try:
    from src.tccc.display.display_interface import DisplayInterface
    from src.tccc.display.display_config import get_display_config
    from src.tccc.display.visualization.vital_signs import VitalSignsMonitor
    from src.tccc.display.visualization.timeline import TimelineVisualization, TimelineEvent
except ImportError as e:
    logger.error(f"Failed to import display components: {e}")
    logger.info("Make sure you have activated the virtual environment and installed the package")
    sys.exit(1)

def verify_display_config():
    """Verify display configuration detection"""
    logger.info("Verifying display configuration...")
    
    # Get display configuration
    try:
        config = get_display_config()
        width, height = config.get_display_dimensions()
        profile = config.display_profile
        is_jetson = config.is_jetson_device()
        
        logger.info(f"Display configuration detected: {width}x{height}")
        logger.info(f"Display profile: {profile}")
        logger.info(f"Jetson device: {'Yes' if is_jetson else 'No'}")
        
        return True
    except Exception as e:
        logger.error(f"Error verifying display configuration: {e}")
        return False

def verify_display_interface():
    """Verify display interface functionality"""
    logger.info("Verifying display interface...")
    
    # Create display interface
    try:
        # Get configuration
        config = get_display_config()
        width, height = config.get_display_dimensions()
        
        # Create display with test size
        display = DisplayInterface(width=800, height=480, fullscreen=False)
        
        # Initialize display
        if not display.initialize():
            logger.error("Failed to initialize display")
            return False
        
        logger.info("Display interface initialized successfully")
        
        # Start display
        display.start()
        logger.info("Display interface started")
        
        # Test basic functionality
        display.update_transcription("Display verification test started")
        time.sleep(0.5)
        
        display.add_significant_event("Verification test event")
        time.sleep(0.5)
        
        display.update_card_data({
            "name": "Test Patient",
            "rank": "SGT",
            "unit": "Test Unit",
            "injuries": "Test injury for verification"
        })
        time.sleep(0.5)
        
        # Toggle display mode
        display.toggle_display_mode()
        time.sleep(0.5)
        display.toggle_display_mode()
        
        # Stop display
        display.stop()
        logger.info("Display interface stopped")
        
        return True
    except Exception as e:
        logger.error(f"Error verifying display interface: {e}")
        return False

def verify_vital_signs_monitor():
    """Verify vital signs monitor functionality"""
    logger.info("Verifying vital signs monitor...")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Create a test screen
        screen = pygame.display.set_mode((800, 600))
        
        # Create vital signs monitor
        vital_rect = pygame.Rect(50, 50, 700, 500)
        vitals = VitalSignsMonitor(screen, vital_rect)
        
        # Test updating vital signs
        vitals.update_vital("hr", 72)
        vitals.update_vital("sbp", 120)
        vitals.update_vital("dbp", 80)
        vitals.update_vital("rr", 16)
        vitals.update_vital("spo2", 98)
        vitals.update_vital("temp", 37.2)
        
        # Test drawing
        screen.fill((0, 0, 0))
        vitals.draw()
        pygame.display.flip()
        
        # Test compact mode
        vitals.toggle_compact_mode()
        screen.fill((0, 0, 0))
        vitals.draw()
        pygame.display.flip()
        
        # Test active vital setting
        vitals.set_active_vital("hr")
        vitals.toggle_compact_mode()  # Back to standard mode
        screen.fill((0, 0, 0))
        vitals.draw()
        pygame.display.flip()
        
        # Test parsing from text
        vitals.parse_vitals_from_text("HR is 80, BP 120/80, RR 18")
        
        # Clean up
        pygame.quit()
        
        logger.info("Vital signs monitor verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error verifying vital signs monitor: {e}")
        return False

def verify_display_performance():
    """Verify display performance"""
    logger.info("Verifying display performance...")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Get configuration
        config = get_display_config()
        width, height = config.get_display_dimensions()
        
        # Create test screen
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()
        
        # Track FPS
        fps_values = []
        
        # Run for short time to measure performance
        for i in range(100):
            # Clear screen
            screen.fill((0, 0, 0))
            
            # Draw test content
            pygame.draw.rect(screen, (100, 100, 200), pygame.Rect(100, 100, 600, 400))
            
            # Draw some text
            font = pygame.font.Font(None, 36)
            text = font.render(f"Performance Test Frame {i}", True, (255, 255, 255))
            screen.blit(text, (150, 150))
            
            # Draw 10 circles
            for j in range(10):
                pygame.draw.circle(screen, (255, 0, 0), (150 + j * 50, 250), 20)
            
            # Update display
            pygame.display.flip()
            
            # Track FPS
            fps_values.append(clock.get_fps())
            
            # Cap framerate
            clock.tick(60)
        
        # Clean up
        pygame.quit()
        
        # Calculate average FPS
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
        
        logger.info(f"Display performance: {avg_fps:.1f} FPS average")
        
        # Check if performance is acceptable
        if avg_fps >= 30:
            logger.info("Display performance is good (>= 30 FPS)")
            return True
        elif avg_fps >= 15:
            logger.warning("Display performance is marginal (15-30 FPS)")
            return True
        else:
            logger.error("Display performance is poor (< 15 FPS)")
            return False
    except Exception as e:
        logger.error(f"Error verifying display performance: {e}")
        return False

def verify_timeline():
    """Verify timeline visualization functionality"""
    logger.info("Verifying timeline visualization...")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Create a test screen
        screen = pygame.display.set_mode((800, 600))
        
        # Create timeline visualization
        timeline_rect = pygame.Rect(50, 50, 700, 500)
        timeline = TimelineVisualization(screen, timeline_rect)
        
        # Test adding events
        now = datetime.now()
        
        # Add test events of different types
        timeline.add_event(TimelineEvent(
            event_type="critical",
            title="Critical Injury",
            description="Arterial bleeding from left leg."
        ))
        
        timeline.add_event(TimelineEvent(
            event_type="treatment",
            title="Tourniquet Applied",
            description="CAT tourniquet applied to left leg."
        ))
        
        timeline.add_event(TimelineEvent(
            event_type="vital",
            title="Vital Signs",
            description="HR 125, BP 90/60, RR 22, SpO2 94%"
        ))
        
        # Test adding event from text
        timeline.add_event_from_text("Administering 10mg morphine IV for pain management")
        
        # Test drawing
        screen.fill((0, 0, 0))
        timeline.draw()
        pygame.display.flip()
        
        # Test filtering
        timeline.set_filter("critical")
        screen.fill((0, 0, 0))
        timeline.draw()
        pygame.display.flip()
        
        # Test compact mode
        timeline.toggle_compact_mode()
        screen.fill((0, 0, 0))
        timeline.draw()
        pygame.display.flip()
        
        # Clean up
        pygame.quit()
        
        logger.info("Timeline visualization verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error verifying timeline visualization: {e}")
        return False

def verify_integrated_components():
    """Verify integration of all display components"""
    logger.info("Verifying integrated display components...")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Create a test screen
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()
        
        # Create display interface in background thread
        display = DisplayInterface(width=800, height=600, fullscreen=False)
        if not display.initialize():
            logger.error("Failed to initialize display interface")
            return False
        
        display.start()
        
        # Create vital signs monitor
        vitals_rect = pygame.Rect(20, 80, 240, 200)
        vitals = VitalSignsMonitor(screen, vitals_rect)
        
        # Create timeline
        timeline_rect = pygame.Rect(20, 300, 760, 280)
        timeline = TimelineVisualization(screen, timeline_rect)
        
        # Add test data
        display.update_transcription("Integration test started")
        
        vitals.update_vital("hr", 88)
        vitals.update_vital("sbp", 125)
        vitals.update_vital("dbp", 78)
        
        timeline.add_event_from_text("Patient responsive to verbal stimuli")
        
        # Run for short time to test integration
        start_time = time.time()
        while time.time() - start_time < 3.0:  # Run for 3 seconds
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                
            # Clear screen
            screen.fill((0, 0, 0))
            
            # Draw components
            vitals.draw()
            timeline.draw()
            
            # Update display
            pygame.display.flip()
            
            # Cap framerate
            clock.tick(30)
        
        # Clean up
        display.stop()
        pygame.quit()
        
        logger.info("Integrated display components verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error verifying integrated display components: {e}")
        return False

def main():
    """Run verification tests for enhanced display interface"""
    logger.info("Starting enhanced display interface verification")
    
    # Check if running on Jetson with display
    if not os.environ.get("DISPLAY"):
        logger.warning("No display detected. Running in headless mode might cause issues.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Verification cancelled")
            return
    
    # Run verification tests
    config_result = verify_display_config()
    interface_result = verify_display_interface()
    vitals_result = verify_vital_signs_monitor()
    timeline_result = verify_timeline()
    integrated_result = verify_integrated_components()
    performance_result = verify_display_performance()
    
    # Print summary
    print("\n" + "=" * 60)
    print("ENHANCED DISPLAY VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Configuration Detection: {'PASS' if config_result else 'FAIL'}")
    print(f"Display Interface: {'PASS' if interface_result else 'FAIL'}")
    print(f"Vital Signs Monitor: {'PASS' if vitals_result else 'FAIL'}")
    print(f"Timeline Visualization: {'PASS' if timeline_result else 'FAIL'}")
    print(f"Integrated Components: {'PASS' if integrated_result else 'FAIL'}")
    print(f"Display Performance: {'PASS' if performance_result else 'FAIL'}")
    print("=" * 60)
    
    # Overall result
    overall_result = all([
        config_result, 
        interface_result, 
        vitals_result, 
        timeline_result,
        integrated_result,
        performance_result
    ])
    print(f"Overall Result: {'PASS' if overall_result else 'FAIL'}")
    
    return 0 if overall_result else 1

if __name__ == "__main__":
    sys.exit(main())