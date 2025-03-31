#!/usr/bin/env python3
"""
TCCC.ai Display Demo
------------------
Demonstration of the enhanced display capabilities for the TCCC project.
Features real-time vital signs visualization and medical data interface.

This demo showcases:
1. Auto-detected resolution for Waveshare display
2. Optimized rendering for Jetson Nano hardware
3. Real-time vital signs visualization with trends
4. Event timeline for medical interventions
5. Information-dense yet readable medical interface
"""

import os
import sys
import time
import argparse
import logging
import threading
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TCCCDisplayDemo")

# Set environment variables for WaveShare display
os.environ["TCCC_ENABLE_DISPLAY"] = "1"
os.environ["TCCC_DISPLAY_RESOLUTION"] = "1280x720"

# Add project root to path
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir))

# Import pygame for display
try:
    import pygame
    from pygame.locals import *
except ImportError:
    logger.error("pygame not installed. Installing pygame...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        import pygame
        from pygame.locals import *
    except Exception as e:
        logger.error(f"Failed to install pygame: {e}")
        sys.exit(1)

# Import TCCC display components
try:
    from src.tccc.display.display_interface import DisplayInterface
    from src.tccc.display.display_config import get_display_config
    from src.tccc.display.visualization.vital_signs import VitalSignsMonitor
except ImportError as e:
    logger.error(f"Failed to import TCCC display components: {e}")
    sys.exit(1)

class TCCCDisplayDemo:
    """
    Enhanced display demo for TCCC project featuring real-time
    vital signs visualization and medical data display.
    """
    
    def __init__(self, fullscreen=False, demo_mode="all"):
        """
        Initialize the display demo.
        
        Args:
            fullscreen: Whether to run in fullscreen mode
            demo_mode: Which demo to run ("all", "vitals", "events", "card")
        """
        self.fullscreen = fullscreen
        self.demo_mode = demo_mode
        
        # Load configuration
        self.config = get_display_config()
        
        # Runtime state
        self.running = False
        self.screen = None
        self.clock = None
        self.display_interface = None
        self.vital_signs = None
        
        # Demo data
        self.demo_data = {
            "transcription": [
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
            ],
            "events": [
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
            ],
            "vitals": [
                {"time": "14:30", "hr": 125, "bp": "130/85", "rr": 25, "spo2": 92, "temp": 36.8},
                {"time": "14:35", "hr": 120, "bp": "125/80", "rr": 24, "spo2": 93, "temp": 36.9},
                {"time": "14:40", "hr": 115, "bp": "120/75", "rr": 23, "spo2": 94, "temp": 37.0},
                {"time": "14:45", "hr": 110, "bp": "115/75", "rr": 22, "spo2": 95, "temp": 37.0},
                {"time": "14:50", "hr": 105, "bp": "110/70", "rr": 20, "spo2": 96, "temp": 37.1},
                {"time": "14:55", "hr": 100, "bp": "110/70", "rr": 18, "spo2": 97, "temp": 37.0},
                {"time": "15:00", "hr": 95, "bp": "115/75", "rr": 18, "spo2": 98, "temp": 36.9},
                {"time": "15:05", "hr": 90, "bp": "120/80", "rr": 16, "spo2": 98, "temp": 36.8},
            ],
            "card_data": {
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
        }
        
        # Demo control
        self.current_data_index = 0
        self.update_interval = 2.0  # seconds between updates
        self.last_update = time.time()
        self.current_view = "live"  # "live" or "card"
        self.show_vital_signs = True
        
        # Performance metrics
        self.fps_values = []
        self.frame_times = []
        
    def initialize(self):
        """Initialize the display demo components."""
        logger.info("Initializing TCCC Display Demo...")
        
        # Initialize pygame
        pygame.init()
        
        # Get display configuration
        display_width, display_height = self.config.get_display_dimensions()
        
        # Create screen with appropriate flags
        display_flags = pygame.DOUBLEBUF
        if self.fullscreen:
            display_flags |= pygame.FULLSCREEN
        
        # Create the screen
        logger.info(f"Creating display with resolution {display_width}x{display_height}")
        self.screen = pygame.display.set_mode(
            (display_width, display_height),
            display_flags
        )
        pygame.display.set_caption("TCCC.ai Enhanced Display Demo")
        
        # Create clock for frame rate control
        self.clock = pygame.time.Clock()
        
        # Initialize display interface
        self.display_interface = DisplayInterface(
            width=display_width,
            height=display_height,
            fullscreen=self.fullscreen
        )
        
        # Initialize display with default configuration
        if not self.display_interface.initialize():
            logger.error("Failed to initialize display interface")
            return False
        
        # Create vital signs component
        vital_rect = pygame.Rect(
            int(display_width * 0.68),  # Place in right column
            100,  # Below header
            int(display_width * 0.3),  # Width
            int(display_height * 0.4)   # Height
        )
        self.vital_signs = VitalSignsMonitor(self.screen, vital_rect)
        self.vital_signs.compact_mode = True  # Start in compact mode
        
        # Initialize with demo data
        self._populate_initial_data()
        
        # Mark as initialized
        logger.info("TCCC Display Demo initialized successfully")
        return True
    
    def _populate_initial_data(self):
        """Populate initial demo data."""
        # Add initial card data
        self.display_interface.update_card_data(self.demo_data["card_data"])
        
        # Add initial demo transcriptions and events
        for i in range(min(3, len(self.demo_data["transcription"]))):
            self.display_interface.update_transcription(self.demo_data["transcription"][i])
            
        for i in range(min(3, len(self.demo_data["events"]))):
            self.display_interface.add_significant_event(self.demo_data["events"][i])
            
        # Set initial vital signs
        initial_vitals = self.demo_data["vitals"][0]
        self.vital_signs.update_vital("hr", initial_vitals["hr"])
        
        # Parse BP into systolic and diastolic
        if "/" in initial_vitals["bp"]:
            sbp, dbp = map(int, initial_vitals["bp"].split("/"))
            self.vital_signs.update_vital("sbp", sbp)
            self.vital_signs.update_vital("dbp", dbp)
            
        self.vital_signs.update_vital("rr", initial_vitals["rr"])
        self.vital_signs.update_vital("spo2", initial_vitals["spo2"])
        self.vital_signs.update_vital("temp", initial_vitals["temp"])
    
    def start(self):
        """Start the display demo."""
        if not self.display_interface:
            logger.error("Display interface not initialized")
            return False
        
        # Mark as running
        self.running = True
        logger.info("TCCC Display Demo started")
        
        # Main loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Error in demo: {e}")
        finally:
            self.stop()
            
        return True
    
    def _main_loop(self):
        """Main display loop."""
        while self.running:
            # Process events
            self._process_events()
            
            # Update demo data
            self._update_demo_data()
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Draw display interface
            if self.current_view == "live":
                self.display_interface._draw_live_screen()
                
                # Draw vital signs on top if enabled
                if self.show_vital_signs:
                    self.vital_signs.draw()
            else:
                self.display_interface._draw_card_screen()
            
            # Draw performance metrics if enabled
            if self.config.get_config()["performance"].get("show_fps", False):
                self._draw_performance_metrics()
            
            # Flip the display
            pygame.display.flip()
            
            # Update performance metrics
            frame_time = self.clock.tick(self.config.get_fps_limit())
            self.frame_times.append(frame_time)
            self.fps_values.append(self.clock.get_fps())
            
            # Limit history
            if len(self.fps_values) > 100:
                self.fps_values.pop(0)
                self.frame_times.pop(0)
    
    def _process_events(self):
        """Process input events."""
        for event in pygame.event.get():
            # Handle exit events
            if event.type == QUIT:
                self.running = False
                
            # Keyboard input
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_t or event.key == K_TAB:
                    # Toggle between views
                    self.current_view = "card" if self.current_view == "live" else "live"
                    self.display_interface.set_display_mode(self.current_view)
                elif event.key == K_v:
                    # Toggle vital signs display
                    self.show_vital_signs = not self.show_vital_signs
                elif event.key == K_c:
                    # Toggle compact mode for vital signs
                    if self.vital_signs:
                        self.vital_signs.toggle_compact_mode()
                elif event.key == K_1:
                    # Set active vital to heart rate
                    if self.vital_signs:
                        self.vital_signs.set_active_vital("hr")
                elif event.key == K_2:
                    # Set active vital to blood pressure
                    if self.vital_signs:
                        self.vital_signs.set_active_vital("sbp")
                elif event.key == K_3:
                    # Set active vital to respiratory rate
                    if self.vital_signs:
                        self.vital_signs.set_active_vital("rr")
                elif event.key == K_4:
                    # Set active vital to SpO2
                    if self.vital_signs:
                        self.vital_signs.set_active_vital("spo2")
                elif event.key == K_5:
                    # Set active vital to temperature
                    if self.vital_signs:
                        self.vital_signs.set_active_vital("temp")
                elif event.key == K_f:
                    # Toggle FPS display
                    config = self.config.get_config()
                    show_fps = not config["performance"].get("show_fps", False)
                    config["performance"]["show_fps"] = show_fps
    
    def _update_demo_data(self):
        """Update the demo data periodically."""
        current_time = time.time()
        
        # Only update at specified intervals
        if current_time - self.last_update >= self.update_interval:
            # Reset update timer
            self.last_update = current_time
            
            # Update with next transcription if available
            if self.current_data_index < len(self.demo_data["transcription"]):
                self.display_interface.update_transcription(
                    self.demo_data["transcription"][self.current_data_index]
                )
                
            # Add next event if available
            if self.current_data_index < len(self.demo_data["events"]):
                self.display_interface.add_significant_event(
                    self.demo_data["events"][self.current_data_index]
                )
                
            # Update vital signs if available
            if self.current_data_index < len(self.demo_data["vitals"]):
                vital_data = self.demo_data["vitals"][self.current_data_index]
                
                if self.vital_signs:
                    self.vital_signs.update_vital("hr", vital_data["hr"])
                    
                    # Parse BP into systolic and diastolic
                    if "/" in vital_data["bp"]:
                        sbp, dbp = map(int, vital_data["bp"].split("/"))
                        self.vital_signs.update_vital("sbp", sbp)
                        self.vital_signs.update_vital("dbp", dbp)
                        
                    self.vital_signs.update_vital("rr", vital_data["rr"])
                    self.vital_signs.update_vital("spo2", vital_data["spo2"])
                    self.vital_signs.update_vital("temp", vital_data["temp"])
                    
                    # Update vital signs animation
                    self.vital_signs.update_animation()
                
                # Also update card data with latest vitals
                self.display_interface.update_card_data({
                    "vital_signs": f"HR {vital_data['hr']}, BP {vital_data['bp']}, " + 
                                  f"RR {vital_data['rr']}, SpO2 {vital_data['spo2']}%"
                })
            
            # Increment data index, looping back to start if needed
            self.current_data_index = (self.current_data_index + 1) % len(self.demo_data["transcription"])
    
    def _draw_performance_metrics(self):
        """Draw performance metrics for debugging."""
        # Skip if no FPS values
        if not self.fps_values:
            return
            
        # Calculate average FPS and frame time
        avg_fps = sum(self.fps_values) / len(self.fps_values)
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        # Create surface for metrics
        metrics_surface = pygame.Surface((200, 80), pygame.SRCALPHA)
        metrics_surface.fill((0, 0, 0, 180))  # Semi-transparent background
        
        # Create font for metrics
        font = pygame.font.Font(None, 24)
        
        # Draw metrics
        fps_text = font.render(f"FPS: {avg_fps:.1f}", True, (255, 255, 0))
        frame_time_text = font.render(f"Frame time: {avg_frame_time:.1f} ms", True, (255, 255, 0))
        res_text = font.render(f"Resolution: {self.screen.get_width()}x{self.screen.get_height()}", True, (255, 255, 0))
        
        metrics_surface.blit(fps_text, (10, 10))
        metrics_surface.blit(frame_time_text, (10, 30))
        metrics_surface.blit(res_text, (10, 50))
        
        # Draw on screen in top-right corner
        self.screen.blit(metrics_surface, (self.screen.get_width() - 210, 10))
    
    def stop(self):
        """Stop the display demo."""
        logger.info("Stopping TCCC Display Demo...")
        
        # Clean up components
        if self.display_interface:
            self.display_interface.stop()
            
        # Clean up pygame
        pygame.quit()
        
        logger.info("TCCC Display Demo stopped")
        return True


def print_instructions():
    """Print demo instructions for the user."""
    print("\n" + "#"*80)
    print("# TCCC.ai ENHANCED DISPLAY DEMO #".center(80) + " #")
    print("#"*80)
    
    print("\nThis demo showcases the enhanced display capabilities for the TCCC project.")
    print("It features automatic resolution detection, vital signs visualization, and more.")
    
    print("\nKEY CONTROLS:")
    print("  ESC      - Exit demo")
    print("  TAB / T  - Toggle between live view and TCCC card view")
    print("  V        - Toggle vital signs visualization")
    print("  C        - Toggle compact mode for vital signs")
    print("  1-5      - Select active vital sign (HR, BP, RR, SpO2, Temp)")
    print("  F        - Toggle FPS display\n")
    
    print("The demo will cycle through a scenario with real-time vital signs and events.")
    print("Watch how the interface adapts to show critical medical information clearly.\n")

def main():
    """Main entry point for the display demo."""
    parser = argparse.ArgumentParser(description="TCCC.ai Enhanced Display Demo")
    parser.add_argument("--fullscreen", "-f", action="store_true", help="Run in fullscreen mode")
    parser.add_argument("--demo", "-d", choices=["all", "vitals", "events", "card"], 
                       default="all", help="Choose which demo to run")
    args = parser.parse_args()
    
    # Print instructions
    print_instructions()
    
    # Create and initialize the demo
    demo = TCCCDisplayDemo(fullscreen=args.fullscreen, demo_mode=args.demo)
    if not demo.initialize():
        logger.error("Failed to initialize TCCC Display Demo")
        return 1
        
    # Start the demo
    print("\nStarting demo... Press ESC to exit.\n")
    demo.start()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())