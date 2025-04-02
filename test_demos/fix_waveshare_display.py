#!/usr/bin/env python3
"""
TCCC.ai WaveShare Display Fix
----------------------------
This script fixes the WaveShare display integration with the TCCC system.
It properly configures environment variables and tests the display functionality.
"""

import os
import sys
import time
import argparse
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayFix")

def check_environment():
    """Check the environment and dependencies"""
    logger.info("Checking environment and dependencies...")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check if running on Linux
    if sys.platform != "linux":
        logger.warning(f"Not running on Linux (platform: {sys.platform})")
    
    # Check for required modules
    try:
        import pygame
        logger.info(f"Found pygame version: {pygame.version.ver}")
    except ImportError:
        logger.error("pygame not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
            import pygame
            logger.info(f"Installed pygame version: {pygame.version.ver}")
        except Exception as e:
            logger.error(f"Failed to install pygame: {e}")
            print("Please install pygame manually: pip install pygame")
            return False
    
    # Check for YAML support (used for config)
    try:
        import yaml
        logger.info("PyYAML is available")
    except ImportError:
        logger.warning("PyYAML not installed, will use default values")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
            import yaml
            logger.info("Installed PyYAML successfully")
        except Exception as e:
            logger.warning(f"Failed to install PyYAML: {e}")
    
    # Check for display environment
    try:
        display_var = os.environ.get('DISPLAY')
        wayland_var = os.environ.get('WAYLAND_DISPLAY')
        
        if display_var:
            logger.info(f"DISPLAY environment variable: {display_var}")
        elif wayland_var:
            logger.info(f"WAYLAND_DISPLAY environment variable: {wayland_var}")
        else:
            logger.warning("No display environment variables found. Are you running in a GUI environment?")
    except Exception as e:
        logger.warning(f"Error checking display environment: {e}")
    
    # Check if the config file exists
    config_path = Path(__file__).parent / "config" / "display.yaml"
    if config_path.exists():
        logger.info(f"Found display configuration at {config_path}")
    else:
        logger.warning(f"No display configuration found at {config_path}")
    
    return True

def detect_hardware():
    """Detect hardware configuration"""
    logger.info("Detecting hardware configuration...")
    
    # Check for Jetson hardware
    is_jetson = os.path.exists('/etc/nv_tegra_release')
    if is_jetson:
        logger.info("Detected NVIDIA Jetson hardware")
    
    # Check for Raspberry Pi
    is_raspberry_pi = False
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                is_raspberry_pi = True
                logger.info(f"Detected Raspberry Pi: {model}")
    
    # Try to detect connected displays using xrandr
    try:
        result = subprocess.run(['xrandr'], capture_output=True, text=True)
        if result.returncode == 0:
            # Look for connected displays and resolutions
            displays = []
            for line in result.stdout.splitlines():
                if ' connected' in line:
                    displays.append(line.split(' ')[0])
                    logger.info(f"Found connected display: {line}")
            
            if displays:
                logger.info(f"Connected displays: {', '.join(displays)}")
            else:
                logger.warning("No connected displays detected with xrandr")
                
            # Check specifically for WaveShare resolution
            if '1560x720' in result.stdout or '720x1560' in result.stdout:
                logger.info("Detected WaveShare display resolution")
            else:
                logger.info("WaveShare display resolution not detected")
        else:
            logger.warning("xrandr command failed, cannot detect displays")
    except Exception as e:
        logger.warning(f"Error using xrandr: {e}")
    
    # Get display information using pygame
    try:
        import pygame
        pygame.init()
        pygame.display.init()
        
        # Get info about the current display
        info = pygame.display.Info()
        logger.info(f"Display driver: {pygame.display.get_driver()}")
        logger.info(f"Current resolution: {info.current_w}x{info.current_h}")
        
        pygame.display.quit()
    except Exception as e:
        logger.warning(f"Error getting display info with pygame: {e}")
    
    # Create hardware info dictionary
    hardware_info = {
        'is_jetson': is_jetson,
        'is_raspberry_pi': is_raspberry_pi,
        'displays': displays if 'displays' in locals() else [],
    }
    
    return hardware_info

def setup_environment_variables():
    """Set up environment variables for the display"""
    logger.info("Setting up display environment variables...")
    
    # Set environment variables for WaveShare display
    os.environ["TCCC_ENABLE_DISPLAY"] = "1"  # Enable display
    os.environ["TCCC_DISPLAY_RESOLUTION"] = "1560x720"  # Landscape orientation
    os.environ["TCCC_DISPLAY_TYPE"] = "waveshare_6_25"  # Display type
    
    # Enable hardware acceleration if available
    try:
        import pygame
        if pygame.display.get_driver() in ["kmsdrm", "wayland", "x11"]:
            logger.info("Enabling hardware acceleration")
            os.environ["SDL_VIDEODRIVER"] = pygame.display.get_driver()
        else:
            logger.info(f"Using software rendering with {pygame.display.get_driver()}")
    except:
        logger.warning("Could not check pygame driver for hardware acceleration")
    
    logger.info("Environment variables set successfully")

def create_display_config():
    """Create or update the display configuration file"""
    logger.info("Updating display configuration file...")
    
    config_dir = Path(__file__).parent / "config"
    config_path = config_dir / "display.yaml"
    
    # Create config directory if it doesn't exist
    config_dir.mkdir(exist_ok=True)
    
    try:
        import yaml
        
        # Default configuration for WaveShare display
        default_config = {
            'display': {
                'width': 1560,
                'height': 720,
                'orientation': 'landscape',
                'fullscreen': True,
                'touch': {
                    'enabled': True,
                    'calibration_enabled': True,
                    'transformation_matrix': [0, 1, 0, -1, 0, 1, 0, 0, 1]
                }
            },
            'ui': {
                'theme': 'dark',
                'color_schemes': {
                    'dark': {
                        'background': [0, 0, 0],
                        'text': [255, 255, 255],
                        'header': [0, 0, 255],
                        'highlight': [255, 215, 0],
                        'alert': [255, 0, 0],
                        'success': [0, 200, 0]
                    }
                },
                'logo': 'images/blue_logo.png',
                'animations': {
                    'enabled': True,
                    'transition_speed_ms': 300
                }
            },
            'performance': {
                'target_fps': 30
            }
        }
        
        # Check if config file exists
        existing_config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f)
                logger.info("Loaded existing config file")
        
        # Merge configs, with default_config as the base
        import collections.abc
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        # Update default config with existing config values
        if existing_config:
            config = update_dict(default_config, existing_config)
        else:
            config = default_config
        
        # Update specific values for WaveShare display
        config['display']['width'] = 1560
        config['display']['height'] = 720
        config['display']['orientation'] = 'landscape'
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Updated display configuration at {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating display configuration: {e}")
        return False

def test_display():
    """Run a basic display test"""
    logger.info("Running basic display test...")
    
    try:
        import pygame
        pygame.init()
        pygame.display.init()
        
        # Create window
        width, height = 1560, 720  # WaveShare dimensions
        try:
            screen = pygame.display.set_mode((width, height))
        except pygame.error:
            # Fallback to current display size
            info = pygame.display.Info()
            width, height = info.current_w, info.current_h
            screen = pygame.display.set_mode((width, height))
        
        pygame.display.set_caption("WaveShare Display Fix")
        
        # Set up colors and fonts
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        
        try:
            font_large = pygame.font.SysFont('Arial', 48)
            font_medium = pygame.font.SysFont('Arial', 36)
            font_small = pygame.font.SysFont('Arial', 24)
        except:
            # Fallback to default font
            font_large = pygame.font.Font(None, 48)
            font_medium = pygame.font.Font(None, 36)
            font_small = pygame.font.Font(None, 24)
        
        # Fill screen
        screen.fill(BLACK)
        
        # Draw header
        pygame.draw.rect(screen, BLUE, (0, 0, width, 80))
        header_text = font_large.render("TCCC.ai Display Fix", True, WHITE)
        screen.blit(header_text, (width//2 - header_text.get_width()//2, 20))
        
        # Draw info text
        info_text = [
            f"Display Driver: {pygame.display.get_driver()}",
            f"Resolution: {width}x{height}",
            f"Touch: Use mouse or touchscreen to click/touch the blocks below",
            f"Press ESC to exit, spacebar to clear touch points"
        ]
        
        for i, text in enumerate(info_text):
            rendered_text = font_medium.render(text, True, WHITE)
            screen.blit(rendered_text, (width//2 - rendered_text.get_width()//2, 100 + i*40))
        
        # Create a series of test blocks
        blocks = []
        block_colors = [RED, GREEN, BLUE, (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        block_width = 200
        block_height = 100
        margin = 20
        total_width = (block_width + margin) * 3 - margin
        
        start_x = (width - total_width) // 2
        start_y = 300
        
        for i in range(6):
            row = i // 3
            col = i % 3
            x = start_x + col * (block_width + margin)
            y = start_y + row * (block_height + margin)
            
            blocks.append({
                'rect': pygame.Rect(x, y, block_width, block_height),
                'color': block_colors[i],
                'text': f"Block {i+1}",
                'clicked': False
            })
        
        # Draw blocks
        for block in blocks:
            pygame.draw.rect(screen, block['color'], block['rect'])
            text = font_medium.render(block['text'], True, WHITE)
            screen.blit(text, (block['rect'].centerx - text.get_width()//2, 
                               block['rect'].centery - text.get_height()//2))
        
        pygame.display.flip()
        
        # Variables for tracking touch/click points
        touch_points = []
        MAX_POINTS = 10
        
        # Main loop
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Clear touch points
                        touch_points = []
                        # Reset block clicked states
                        for block in blocks:
                            block['clicked'] = False
                
                # Handle touch events (pygame 2.0+)
                elif hasattr(pygame, 'FINGERDOWN') and event.type == pygame.FINGERDOWN:
                    # Convert normalized position to screen coordinates
                    x = event.x * width
                    y = event.y * height
                    touch_points.append((x, y, time.time()))
                    
                    # Keep only recent points
                    if len(touch_points) > MAX_POINTS:
                        touch_points.pop(0)
                    
                    # Check if a block was clicked
                    for block in blocks:
                        if block['rect'].collidepoint(x, y):
                            block['clicked'] = True
                            print(f"Block {block['text']} touched")
                
                # Handle mouse clicks (for non-touch testing)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    touch_points.append((pos[0], pos[1], time.time()))
                    
                    # Keep only recent points
                    if len(touch_points) > MAX_POINTS:
                        touch_points.pop(0)
                    
                    # Check if a block was clicked
                    for block in blocks:
                        if block['rect'].collidepoint(pos):
                            block['clicked'] = True
                            print(f"Block {block['text']} clicked")
            
            # Redraw blocks (with highlight if clicked)
            for block in blocks:
                color = block['color']
                if block['clicked']:
                    # Lighten color for highlight
                    color = tuple(min(255, c + 100) for c in color)
                
                pygame.draw.rect(screen, color, block['rect'])
                text = font_medium.render(block['text'], True, WHITE)
                screen.blit(text, (block['rect'].centerx - text.get_width()//2, 
                                   block['rect'].centery - text.get_height()//2))
            
            # Draw touch points
            for i, (x, y, t) in enumerate(touch_points):
                # Age affects the circle size (newer = larger)
                age = time.time() - t
                radius = max(5, 20 - int(age * 5))
                
                # Draw circle
                pygame.draw.circle(screen, WHITE, (int(x), int(y)), radius, 2)
                
                # Draw index number
                index_text = font_small.render(str(i+1), True, WHITE)
                screen.blit(index_text, (int(x) - index_text.get_width()//2, 
                                         int(y) - index_text.get_height()//2))
            
            # Draw instructions at bottom
            bottom_text = font_medium.render("Press ESC to exit, SPACE to clear touch points", True, WHITE)
            screen.blit(bottom_text, (width//2 - bottom_text.get_width()//2, height - 50))
            
            pygame.display.flip()
            clock.tick(30)  # Cap at 30 FPS
        
        pygame.quit()
        return True
    except Exception as e:
        logger.error(f"Error in display test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tccc_display():
    """Test the actual TCCC display functionality"""
    logger.info("Running TCCC display interface test...")
    
    try:
        # Import the display interface
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from src.tccc.display.display_interface import DisplayInterface
        
        # Create a display interface with WaveShare dimensions
        display = DisplayInterface(width=1560, height=720, fullscreen=False)
        
        # Initialize the display
        if not display.initialize():
            logger.error("Failed to initialize TCCC display interface")
            return False
        
        logger.info("Display initialized successfully")
        
        # Start the display
        display.start()
        
        # Add some test data
        display.update_transcription("Testing WaveShare display integration...")
        time.sleep(1)
        
        display.update_transcription("Adding sample medical data")
        display.add_significant_event("TCCC display test started")
        time.sleep(1)
        
        display.update_transcription("Patient has a simulated injury for testing")
        display.add_significant_event("Simulated injury identified")
        time.sleep(1)
        
        display.update_card_data({
            "name": "Test Patient",
            "rank": "SPC",
            "unit": "Test Unit",
            "mechanism_of_injury": "Simulated",
            "injuries": "Test injury for display verification",
            "treatment_given": "Simulated treatment",
            "vital_signs": "HR 80, BP 120/80, RR 16, O2 99%",
            "medications": "None",
            "evacuation_priority": "Routine"
        })
        
        # Leave display running for 10 seconds
        logger.info("Display running with test data. Will exit in 10 seconds...")
        for i in range(10, 0, -1):
            logger.info(f"Exiting in {i} seconds...")
            # Add more data as we count down
            if i == 7:
                display.toggle_display_mode()  # Switch to card view
            elif i == 4:
                display.toggle_display_mode()  # Switch back to live view
            
            # Add a countdown message
            display.update_transcription(f"Test will complete in {i} seconds...")
            display.add_significant_event({"time": time.strftime("%H:%M:%S"), 
                                          "description": f"Countdown: {i} seconds remaining"})
            
            time.sleep(1)
        
        # Stop the display
        display.stop()
        logger.info("TCCC display test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in TCCC display test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TCCC WaveShare Display Fix")
    parser.add_argument('--env-only', action='store_true', 
                      help='Only check/set environment variables, no display test')
    parser.add_argument('--test-only', action='store_true',
                      help='Only run the display test')
    parser.add_argument('--full-test', action='store_true',
                      help='Run the full TCCC display interface test')
    args = parser.parse_args()
    
    print("=== TCCC.ai WaveShare Display Fix ===")
    
    success = True
    
    # Check environment and dependencies
    if not args.test_only:
        if not check_environment():
            print("❌ Environment check failed. Please resolve issues before continuing.")
            return False
    
    # Detect hardware if needed
    if not args.test_only:
        hardware_info = detect_hardware()
        
        # Set up environment variables
        setup_environment_variables()
        
        # Create/update display configuration
        if not create_display_config():
            print("⚠️ Failed to update display configuration")
            success = False
    
    # Run display test if requested or by default
    if args.full_test:
        if not test_tccc_display():
            print("❌ TCCC display test failed")
            success = False
    elif not args.env_only:
        if not test_display():
            print("❌ Display test failed")
            success = False
    
    if success:
        print("\n✅ WaveShare display fix completed successfully!")
        print("\nYou can now use the following commands:")
        print("  - Run a simple display test: python test_waveshare_display.py")
        print("  - Run the full TCCC system with display: python run_system.py --with-display")
    else:
        print("\n⚠️ WaveShare display fix completed with warnings or errors.")
        print("Please check the log output above for details.")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())