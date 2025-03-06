#!/usr/bin/env python3
"""
TCCC.ai Display Interface
------------------------
Enhanced display interface for WaveShare 6.25" touchscreen showing:
1. Transcribed text from STT engine 
2. Significant events parsed by LLM
3. TCCC Casualty Card (DD Form 1380) when care is complete

This implementation provides full hardware support for the WaveShare display,
with proper touch calibration, hardware acceleration, and adaptive UI based
on orientation and display capabilities.
"""

import os
import sys
import time
import threading
import logging
import json
import shutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayInterface")

# Try to import pygame, install if needed
try:
    import pygame
    from pygame.locals import *
except ImportError:
    logger.error("pygame not installed. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        import pygame
        from pygame.locals import *
        logger.info("pygame installed successfully")
    except Exception as e:
        logger.error(f"Failed to install pygame: {e}")
        raise

# Try to import other optional dependencies
try:
    import yaml
except ImportError:
    logger.warning("PyYAML not installed, will use fallback configuration")
    yaml = None

# Default colors - will be updated from config if available
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
GOLD = (255, 215, 0)

# Font sizes adjusted for WaveShare 6.25" display - will be updated from config
FONT_SMALL = 22
FONT_MEDIUM = 28
FONT_LARGE = 36

# Card fields from DD Form 1380 TCCC Card
CARD_FIELDS = [
    "name", "rank", "unit", 
    "date", "time", "mechanism_of_injury",
    "injuries", "vital_signs", "treatment_given",
    "medications", "evacuation_priority"
]

# Performance monitoring stats
PERF_STATS = {
    'fps': 0,
    'frame_time': 0,
    'frame_count': 0,
    'start_time': 0,
}

class DisplayInterface:
    """Enhanced display interface for TCCC.ai with WaveShare hardware support"""
    
    def __init__(self, width=1560, height=720, fullscreen=False, config_path=None):
        """
        Initialize the display interface with improved hardware support
        
        Args:
            width: Screen width (default 1560 for WaveShare 6.25" Display in landscape)
            height: Screen height (default 720 for WaveShare 6.25" Display in landscape)
            fullscreen: Whether to display in fullscreen mode
            config_path: Optional path to configuration file
        """
        # Display dimensions
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        
        # Custom config path if provided
        self.config_path = config_path
        
        # Flag to track if we're in portrait or landscape orientation
        self.portrait = height > width
        
        # Runtime state
        self.active = False
        self.initialized = False
        self.display_thread = None
        self.screen = None
        self.clock = None
        
        # Hardware details
        self.is_jetson = False
        self.is_waveshare = False
        self.display_driver = None
        self.has_touch = False
        self.touch_device_id = None
        self.auto_detect = True
        
        # Content to display
        self.transcription = []
        self.significant_events = []
        self.card_data = {}
        self.display_mode = "live"  # 'live' or 'card'
        self.last_update = time.time()
        
        # UI settings - will be updated from config
        self.theme = "dark"
        self.colors = {
            "background": BLACK,
            "text": WHITE,
            "header": BLUE,
            "highlight": GOLD,
            "alert": RED,
            "success": GREEN,
        }
        self.max_transcription_items = 10
        self.max_event_items = 8
        self.logo_path = None
        self.alt_logo_path = None
        self.column_widths = {
            "transcription": 0.38,
            "events": 0.34,
            "card_preview": 0.28,
        }
        
        # Animation settings
        self.animations_enabled = True
        self.transition_speed = 300  # ms
        self.fade_in = True
        self.smooth_scroll = True
        
        # Performance settings
        self.target_fps = 30
        self.fps_limit_battery = 15
        self.power_save_mode = False
        self.show_fps = False
        
        # Debug settings
        self.debug_mode = False
        self.show_touch_points = False
        
        # Touch input tracking
        self.touch_regions = []
        self.touch_points = []
        self.touch_sensitivity = 1.0
        self.touch_enabled = True
        self.touch_calibration_enabled = True
        self.touch_device_name = "WaveShare Touchscreen"
        self.touch_transformation_matrix = [0, 1, 0, -1, 0, 1, 0, 0, 1]
        
        # Thread lock for thread-safe updates
        self.lock = threading.Lock()
        
        # Load config first thing
        self.load_config()
        
    def load_config(self):
        """Load enhanced display configuration from file with hardware support"""
        try:
            if yaml is None:
                logger.warning("PyYAML not available, using default configuration")
                return False
                
            # Use custom config path if provided, otherwise use default
            if self.config_path:
                config_path = Path(self.config_path)
            else:
                config_path = Path(__file__).parent.parent.parent.parent / "config" / "display.yaml"
            
            if not config_path.exists():
                logger.warning(f"Display config file not found at {config_path}, using defaults")
                return False
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Display settings
            if 'display' in config:
                display_config = config['display']
                self.width = display_config.get('width', self.width)
                self.height = display_config.get('height', self.height)
                self.fullscreen = display_config.get('fullscreen', self.fullscreen)
                
                # Orientation
                orientation = display_config.get('orientation', 'landscape')
                self.portrait = orientation.lower() == 'portrait'
                
                # Touch settings
                touch_config = display_config.get('touch', {})
                self.touch_enabled = touch_config.get('enabled', True)
                self.touch_device_name = touch_config.get('device', "WaveShare Touchscreen")
                self.touch_calibration_enabled = touch_config.get('calibration_enabled', True)
                
                # Touch transformation matrix if specified
                if 'transformation_matrix' in touch_config:
                    self.touch_transformation_matrix = touch_config['transformation_matrix']
                
                # Touch sensitivity
                self.touch_sensitivity = touch_config.get('sensitivity', 1.0)
                
                # Touch regions if specified
                if 'regions' in touch_config:
                    # We'll process these when defining touch regions
                    self.touch_region_config = touch_config.get('regions', {})
            
            # UI settings
            if 'ui' in config:
                ui_config = config['ui']
                font_scale = ui_config.get('font_scale', 1.0)
                
                # Update font sizes if specified
                global FONT_SMALL, FONT_MEDIUM, FONT_LARGE
                FONT_SMALL = ui_config.get('small_font_size', FONT_SMALL)
                FONT_MEDIUM = ui_config.get('medium_font_size', FONT_MEDIUM)
                FONT_LARGE = ui_config.get('large_font_size', FONT_LARGE)
                
                # Scale fonts if needed
                if font_scale != 1.0:
                    FONT_SMALL = int(FONT_SMALL * font_scale)
                    FONT_MEDIUM = int(FONT_MEDIUM * font_scale)
                    FONT_LARGE = int(FONT_LARGE * font_scale)
                
                # Theme and color scheme
                self.theme = ui_config.get('theme', 'dark')
                
                # Load color scheme if specified
                if 'color_schemes' in ui_config and self.theme in ui_config['color_schemes']:
                    color_scheme = ui_config['color_schemes'][self.theme]
                    for key, value in color_scheme.items():
                        if key in self.colors:
                            self.colors[key] = tuple(value)
                
                # Logo paths
                self.logo_path = ui_config.get('logo', 'images/blue_logo.png')
                self.alt_logo_path = ui_config.get('alt_logo', 'images/green_logo.png')
                
                # Maximum items
                self.max_transcription_items = ui_config.get('max_transcription_items', self.max_transcription_items)
                self.max_event_items = ui_config.get('max_event_items', self.max_event_items)
                
                # Animation settings
                if 'animations' in ui_config:
                    animation_config = ui_config['animations']
                    self.animations_enabled = animation_config.get('enabled', True)
                    self.transition_speed = animation_config.get('transition_speed_ms', 300)
                    self.fade_in = animation_config.get('fade_in', True)
                    self.smooth_scroll = animation_config.get('scroll_smooth', True)
                
                # Layout settings
                if 'layout' in ui_config:
                    layout_config = ui_config['layout']
                    # Column width percentages
                    if 'column_1_width' in layout_config:
                        self.column_widths['transcription'] = layout_config['column_1_width']
                    if 'column_2_width' in layout_config:
                        self.column_widths['events'] = layout_config['column_2_width']
                    # Calculate card preview width as remainder
                    self.column_widths['card_preview'] = 1.0 - self.column_widths['transcription'] - self.column_widths['events']
            
            # Hardware settings
            if 'hardware' in config:
                hardware_config = config['hardware']
                self.auto_detect = hardware_config.get('auto_detect', True)
                
                # WaveShare specific settings
                if 'waveshare' in hardware_config:
                    waveshare_config = hardware_config['waveshare']
                    # Store for hardware detection
                    self.waveshare_config = waveshare_config
                
                # Jetson hardware settings
                if 'jetson' in hardware_config:
                    jetson_config = hardware_config['jetson']
                    # Handle Jetson-specific optimization settings
                    if 'performance' in jetson_config:
                        perf_config = jetson_config['performance']
                        self.target_fps = perf_config.get('fps_limit_ac', 30)
                        self.fps_limit_battery = perf_config.get('fps_limit_battery', 15)
                    
                    # Power save mode
                    self.power_save_mode = jetson_config.get('power_save_mode', False)
            
            # Performance monitoring
            if 'performance' in config:
                perf_config = config['performance']
                self.show_fps = perf_config.get('show_fps', False)
                self.target_fps = perf_config.get('target_fps', self.target_fps)
            
            # Advanced settings
            if 'advanced' in config:
                adv_config = config['advanced']
                self.debug_mode = adv_config.get('debug_mode', False)
                self.show_touch_points = adv_config.get('show_touch_points', False)
                
                # SDL video/audio driver settings
                sdl_videodriver = adv_config.get('sdl_videodriver', '')
                sdl_audiodriver = adv_config.get('sdl_audiodriver', '')
                
                # Set environment variables if specified
                if sdl_videodriver:
                    os.environ["SDL_VIDEODRIVER"] = sdl_videodriver
                if sdl_audiodriver:
                    os.environ["SDL_AUDIODRIVER"] = sdl_audiodriver
            
            logger.info(f"Loaded display config: {self.width}x{self.height}, " 
                      f"orientation: {'portrait' if self.portrait else 'landscape'}, "
                      f"theme: {self.theme}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading display config: {e}")
            return False
    
    def detect_display(self):
        """
        Enhanced display hardware detection with WaveShare and Jetson optimizations
        
        This method detects display hardware and sets appropriate settings based on
        the detected hardware. It supports WaveShare displays and Jetson hardware
        with optimized configurations.
        
        Returns:
            bool: True if display was detected and configured successfully
        """
        if not self.auto_detect:
            logger.info("Auto detection disabled, using configured settings")
            return False
            
        try:
            detected_waveshare = False
            detected_display = False
            detected_jetson = False
            
            # First check for Jetson hardware
            detected_jetson = self._detect_jetson_hardware()
            
            # Check for WaveShare display
            detected_waveshare = self._detect_waveshare_display()
            
            # If neither WaveShare nor Jetson was detected, try generic display detection
            if not detected_waveshare:
                detected_display = self._detect_generic_display()
            
            # Return success if any display was detected
            return detected_waveshare or detected_display
            
        except Exception as e:
            logger.error(f"Error detecting display hardware: {e}")
            return False
    
    def _detect_jetson_hardware(self):
        """
        Detect if running on NVIDIA Jetson hardware and configure optimizations
        
        Returns:
            bool: True if Jetson hardware was detected
        """
        try:
            # Method 1: Check via Jetson-specific system files
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    if any(jetson_name in model for jetson_name in ["jetson", "xavier", "nano", "orin"]):
                        self.is_jetson = True
                        logger.info(f"Detected Jetson hardware: {model.strip()}")
                        
                        # Apply Jetson-specific optimizations
                        self._apply_jetson_optimizations()
                        return True
            
            # Method 2: Check via our utility module if available
            try:
                from tccc.utils.jetson_integration import is_jetson_platform
                if is_jetson_platform():
                    self.is_jetson = True
                    logger.info("Detected Jetson hardware via utility module")
                    
                    # Apply Jetson-specific optimizations
                    self._apply_jetson_optimizations()
                    return True
            except ImportError:
                logger.debug("Jetson integration module not available")
            
            # Method 3: Check via CUDA device name
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0).lower()
                    if any(name in device_name for name in ["tegra", "nvidia", "jetson", "xavier", "orin"]):
                        self.is_jetson = True
                        logger.info(f"Detected Jetson hardware via CUDA: {device_name}")
                        
                        # Apply Jetson-specific optimizations
                        self._apply_jetson_optimizations()
                        return True
            except (ImportError, Exception) as e:
                logger.debug(f"CUDA detection not available: {e}")
            
            return False
        except Exception as e:
            logger.error(f"Error detecting Jetson hardware: {e}")
            return False
            
    def _apply_jetson_optimizations(self):
        """Apply Jetson-specific optimizations for display performance"""
        try:
            # Set optimal video driver for Jetson
            os.environ["SDL_VIDEODRIVER"] = "kmsdrm"
            
            # Adjust performance settings
            if self.power_save_mode:
                # Lower FPS target when in power save mode
                self.target_fps = self.fps_limit_battery
                logger.info(f"Power save mode enabled, limiting FPS to {self.target_fps}")
                
            logger.info("Applied Jetson display optimizations")
        except Exception as e:
            logger.error(f"Error applying Jetson optimizations: {e}")
            
    def _detect_waveshare_display(self):
        """
        Detect WaveShare display hardware via multiple methods
        
        Returns:
            bool: True if WaveShare display was detected
        """
        try:
            import subprocess
            
            # Method 1: Check environment variables
            if "DISPLAY_TYPE" in os.environ:
                display_type = os.environ["DISPLAY_TYPE"].lower()
                if "waveshare" in display_type:
                    logger.info("Detected WaveShare display from environment variable")
                    
                    # Set WaveShare-specific settings
                    if "6.25" in display_type:
                        self.width = 1560
                        self.height = 720
                        self.is_waveshare = True
                        logger.info("Using WaveShare 6.25\" display settings (1560x720)")
                        return True
            
            # Method 2: Check for WaveShare configuration on X11
            if os.path.exists("/etc/X11/xorg.conf.d"):
                # Look for display configuration files
                for config_file in os.listdir("/etc/X11/xorg.conf.d"):
                    if "waveshare" in config_file.lower() or "display" in config_file.lower():
                        with open(f"/etc/X11/xorg.conf.d/{config_file}", "r") as f:
                            content = f.read().lower()
                            if "waveshare" in content:
                                logger.info("Detected WaveShare display from X11 configuration")
                                self.is_waveshare = True
                                
                                # Try to extract resolution from configuration
                                import re
                                res_match = re.search(r'modes\s+[\"\']([\d]+x[\d]+)[\"\']\s*$', content, re.MULTILINE)
                                if res_match:
                                    resolution = res_match.group(1)
                                    width, height = map(int, resolution.split("x"))
                                    self.width = width
                                    self.height = height
                                    logger.info(f"Using resolution from config: {width}x{height}")
                                
                                return True
            
            # Method 3: Check for specific resolution with xrandr
            try:
                # The WaveShare 6.25" has a distinctive resolution of 1560x720 or 720x1560
                output = subprocess.check_output(["xrandr"]).decode()
                
                if "1560x720" in output or "720x1560" in output:
                    logger.info("Detected WaveShare 6.25\" display from resolution")
                    self.is_waveshare = True
                    
                    # Ensure correct orientation
                    if "1560x720" in output:
                        self.width = 1560
                        self.height = 720
                        self.portrait = False
                    else:
                        self.width = 720
                        self.height = 1560
                        self.portrait = True
                        
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("xrandr not available, skipping resolution detection")
            
            return False
        except Exception as e:
            logger.error(f"Error detecting WaveShare display: {e}")
            return False
            
    def _detect_generic_display(self):
        """
        Detect generic display hardware and set resolution
        
        Returns:
            bool: True if display was detected
        """
        try:
            import subprocess
            
            # Try to detect using xrandr if available
            try:
                output = subprocess.check_output(["xrandr"]).decode()
                
                # Check for connected displays
                for line in output.splitlines():
                    if " connected " in line:
                        # Extract resolution
                        for resolution_line in output.splitlines()[output.splitlines().index(line)+1:]:
                            if "*" in resolution_line:  # Current mode
                                resolution = resolution_line.strip().split()[0]
                                width, height = map(int, resolution.split("x"))
                                
                                # Update dimensions
                                if width != self.width or height != self.height:
                                    logger.info(f"Detected display resolution: {width}x{height}")
                                    self.width = width
                                    self.height = height
                                    self.portrait = height > width
                                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("xrandr not available, skipping resolution detection")
            
            # Check SDL display info directly
            try:
                # Initialize pygame display subsystem if not already done
                if not pygame.display.get_init():
                    pygame.display.init()
                
                # Get display info
                info = pygame.display.Info()
                if info.current_w > 0 and info.current_h > 0:
                    logger.info(f"Detected display via pygame: {info.current_w}x{info.current_h}")
                    self.width = info.current_w
                    self.height = info.current_h
                    self.portrait = info.current_h > info.current_w
                    return True
            except Exception as e:
                logger.debug(f"Failed to get pygame display info: {e}")
            
            # Check environment variable
            if "DISPLAY_RESOLUTION" in os.environ:
                try:
                    resolution = os.environ["DISPLAY_RESOLUTION"]
                    width, height = map(int, resolution.split("x"))
                    logger.info(f"Using display resolution from environment: {width}x{height}")
                    self.width = width
                    self.height = height
                    self.portrait = height > width
                    return True
                except Exception as e:
                    logger.warning(f"Invalid DISPLAY_RESOLUTION format: {e}")
            
            # No specific display detected
            logger.info("No specific display hardware detected, using defaults")
            return False
        except Exception as e:
            logger.error(f"Error detecting generic display: {e}")
            return False
    
    def setup_touch_input(self):
        """
        Enhanced touch input setup with hardware calibration
        
        Sets up touch input for the display with proper calibration,
        transformation matrix, and device mapping for WaveShare hardware.
        
        Returns:
            bool: True if touch input was set up successfully
        """
        if not self.touch_enabled:
            logger.info("Touch input disabled in config")
            return False
            
        try:
            import subprocess
            
            # Skip if pygame doesn't support touch
            if not hasattr(pygame, 'FINGERDOWN'):
                logger.warning("pygame does not support touch events (old version?)")
                return False
                
            # Enable touch events for pygame
            os.environ["SDL_HINT_TOUCH_MOUSE_EVENTS"] = "1"
            
            # Try to detect touch device
            touch_device_id = self._detect_touch_device()
            
            # Set up touch calibration if device found
            if touch_device_id:
                self.touch_device_id = touch_device_id
                self.has_touch = True
                
                # Try to calibrate touchscreen if enabled
                if self.touch_calibration_enabled:
                    success = self._calibrate_touch_device(touch_device_id)
                    if success:
                        logger.info(f"Touch calibration successful for device ID {touch_device_id}")
                    else:
                        logger.warning("Touch calibration failed, touch input may not be accurate")
                
                return True
            else:
                logger.warning(f"Touch device '{self.touch_device_name}' not found")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up touch input: {e}")
            return False
            
    def _detect_touch_device(self):
        """
        Detect touch input device using multiple methods
        
        Returns:
            str: Device ID if found, None otherwise
        """
        try:
            import subprocess
            
            device_id = None
            
            # Method 1: Check via xinput if available
            try:
                # Check if xinput is available
                if subprocess.call(["which", "xinput"], stdout=subprocess.PIPE) == 0:
                    # List input devices
                    output = subprocess.check_output(["xinput", "list"]).decode()
                    
                    # Look for touchscreen device
                    for line in output.splitlines():
                        if self.touch_device_name.lower() in line.lower():
                            # Extract the ID
                            import re
                            match = re.search(r'id=(\d+)', line)
                            if match:
                                device_id = match.group(1)
                                logger.info(f"Detected touch device: {self.touch_device_name} (ID: {device_id})")
                                return device_id
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("xinput not available, trying other touch detection methods")
            
            # Method 2: Check via /dev/input for touch devices
            try:
                if os.path.exists("/dev/input"):
                    # Check input device info
                    import glob
                    for device in glob.glob("/dev/input/event*"):
                        try:
                            # Try to get device info using libevdev if available
                            output = subprocess.check_output(["evtest", "--info", device], stderr=subprocess.DEVNULL).decode()
                            if "touch" in output.lower() or "screen" in output.lower():
                                # Extract device number
                                device_num = device.split("event")[-1]
                                logger.info(f"Detected touch device via evtest: {device} (ID: {device_num})")
                                return device_num
                        except (subprocess.SubprocessError, FileNotFoundError):
                            pass
            except Exception as e:
                logger.debug(f"Error checking input devices: {e}")
            
            # Method 3: Check if WaveShare was detected
            if self.is_waveshare:
                # WaveShare displays have known touch device names
                logger.info("WaveShare display detected, assuming touch input is available")
                # Set touch enabled for pygame but return None for device ID
                self.has_touch = True
                return "auto"
            
            return None
        except Exception as e:
            logger.error(f"Error detecting touch device: {e}")
            return None
            
    def _calibrate_touch_device(self, device_id):
        """
        Calibrate touch device with proper transformation matrix
        
        Args:
            device_id: ID of touch device to calibrate
            
        Returns:
            bool: True if calibration was successful
        """
        try:
            import subprocess
            
            # Skip if device_id is "auto" (pygame will handle it)
            if device_id == "auto":
                logger.info("Using automatic touch handling")
                return True
                
            # Try to calibrate using xinput if available
            try:
                # Check if xinput is available
                if subprocess.call(["which", "xinput"], stdout=subprocess.PIPE) == 0:
                    # Apply transformation matrix for orientation
                    matrix_str = " ".join(str(v) for v in self.touch_transformation_matrix)
                    
                    # Apply transformation matrix
                    subprocess.call(["xinput", "set-prop", device_id, 
                                    "--type=float", "Coordinate Transformation Matrix", 
                                    *matrix_str.split()])
                    
                    # Map touch to correct output
                    # Find the right output name
                    output = subprocess.check_output(["xrandr"]).decode()
                    output_name = None
                    
                    for line in output.splitlines():
                        if " connected" in line:
                            output_name = line.split()[0]
                            break
                    
                    if output_name:
                        subprocess.call(["xinput", "map-to-output", device_id, output_name])
                        logger.info(f"Touch input mapped to output {output_name}")
                    
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("xinput not available, skipping touch calibration")
            
            # Touch handling will fall back to pygame's built-in support
            logger.info("Using pygame's built-in touch input handling")
            return True
            
        except Exception as e:
            logger.error(f"Error calibrating touch device: {e}")
            return False
    
    def initialize(self):
        """
        Initialize display interface with hardware detection and optimization
        
        This method initializes pygame, detects and configures hardware,
        sets up the display with proper settings, and loads resources.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.initialized:
            logger.info("Display already initialized")
            return True
            
        try:
            # Configuration already loaded in __init__
            
            # Detect display hardware
            if self.auto_detect:
                self.detect_display()
            
            # Initialize pygame with optimal subsystems
            pygame.init()
            pygame.display.set_caption("TCCC.ai Field Assistant")
            
            # Create clock for FPS control
            self.clock = pygame.time.Clock()
            
            # Set up performance monitoring
            global PERF_STATS
            PERF_STATS['start_time'] = time.time()
            PERF_STATS['frame_count'] = 0
            
            # Set up display
            display_flags = 0
            if self.fullscreen:
                display_flags |= pygame.FULLSCREEN
                
            # Add hardware acceleration if available
            if pygame.display.get_driver() in ("kmsdrm", "wayland", "x11"):
                display_flags |= pygame.HWSURFACE | pygame.DOUBLEBUF
            
            # Add any additional flags based on hardware
            if self.is_jetson:
                # Add additional flags for Jetson optimizations
                logger.info("Adding Jetson-specific display flags")
                # Keep buffer size reasonable for Jetson
                display_flags |= pygame.RESIZABLE  # Allow window resize
                # Allow dynamic resolution adjustment based on performance
                
            # Create the screen with detected settings
            logger.info(f"Creating display with resolution {self.width}x{self.height}")
            self.screen = pygame.display.set_mode(
                (self.width, self.height), 
                display_flags
            )
            
            # Record the driver being used
            self.display_driver = pygame.display.get_driver()
            logger.info(f"Using display driver: {self.display_driver}")
            
            # Set up touch input
            self.setup_touch_input()
                
            # Load fonts with fallbacks
            self._load_fonts()
            
            # Load assets
            self._load_assets()
            
            # Mark as initialized
            self.initialized = True
            
            logger.info(f"Display interface initialized successfully: {self.width}x{self.height}, "
                       f"driver: {self.display_driver}, "
                       f"touch: {'enabled' if self.has_touch else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            return False
            
    def _load_fonts(self):
        """Load fonts with proper fallbacks for different platforms"""
        try:
            # Default font names with platform-specific fallbacks
            font_names = ['Arial', 'DejaVuSans', 'FreeSans', 'Liberation Sans']
            
            # Try to load each font in order of preference
            font_loaded = False
            for font_name in font_names:
                try:
                    self.fonts = {
                        'small': pygame.font.SysFont(font_name, FONT_SMALL),
                        'medium': pygame.font.SysFont(font_name, FONT_MEDIUM),
                        'large': pygame.font.SysFont(font_name, FONT_LARGE),
                        'bold_small': pygame.font.SysFont(font_name, FONT_SMALL, bold=True),
                        'bold_medium': pygame.font.SysFont(font_name, FONT_MEDIUM, bold=True),
                        'bold_large': pygame.font.SysFont(font_name, FONT_LARGE, bold=True)
                    }
                    logger.info(f"Loaded font '{font_name}'")
                    font_loaded = True
                    break
                except Exception:
                    continue
            
            # Fall back to default font if none of the specified fonts could be loaded
            if not font_loaded:
                logger.warning("Could not load specified fonts, using default font")
                self.fonts = {
                    'small': pygame.font.Font(None, FONT_SMALL),
                    'medium': pygame.font.Font(None, FONT_MEDIUM),
                    'large': pygame.font.Font(None, FONT_LARGE),
                    'bold_small': pygame.font.Font(None, FONT_SMALL),
                    'bold_medium': pygame.font.Font(None, FONT_MEDIUM),
                    'bold_large': pygame.font.Font(None, FONT_LARGE),
                }
        except Exception as e:
            logger.error(f"Error loading fonts: {e}")
            # Create emergency fallback fonts
            self.fonts = {
                'small': pygame.font.Font(None, FONT_SMALL),
                'medium': pygame.font.Font(None, FONT_MEDIUM),
                'large': pygame.font.Font(None, FONT_LARGE),
                'bold_small': pygame.font.Font(None, FONT_SMALL),
                'bold_medium': pygame.font.Font(None, FONT_MEDIUM),
                'bold_large': pygame.font.Font(None, FONT_LARGE),
            }
            
    def _load_assets(self):
        """Load images and other assets for the display"""
        try:
            # Load logos
            self.avatar = None
            self.alt_avatar = None
            
            # Main logo
            logo_path = Path(__file__).parent.parent.parent.parent
            if self.logo_path:
                main_logo_path = logo_path / self.logo_path
                if main_logo_path.exists():
                    try:
                        self.avatar = pygame.image.load(str(main_logo_path))
                        # Scale to a reasonable size for the display
                        self.avatar = pygame.transform.scale(self.avatar, (80, 80))
                        logger.info(f"Loaded main logo from {main_logo_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load main logo: {e}")
            
            # Alternate logo
            if self.alt_logo_path:
                alt_logo_path = logo_path / self.alt_logo_path
                if alt_logo_path.exists():
                    try:
                        self.alt_avatar = pygame.image.load(str(alt_logo_path))
                        # Scale to a reasonable size for the display
                        self.alt_avatar = pygame.transform.scale(self.alt_avatar, (80, 80))
                        logger.info(f"Loaded alternate logo from {alt_logo_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load alternate logo: {e}")
        
        except Exception as e:
            logger.error(f"Error loading assets: {e}")
            # We'll continue without assets if they can't be loaded
            
    def start(self):
        """
        Start the display interface in a background thread
        
        Initializes the display if needed and runs the main loop in a separate thread
        """
        if self.active:
            logger.warning("Display interface already running")
            return
            
        if not self.initialized and not self.initialize():
            logger.error("Failed to initialize display")
            return
            
        self.active = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        logger.info("Display interface started")
        
    def stop(self):
        """
        Stop the display interface and clean up resources
        
        Signals the display loop to stop, waits for the thread to finish,
        and cleans up pygame resources.
        """
        self.active = False
        if self.display_thread and self.display_thread.is_alive():
            logger.debug("Waiting for display thread to finish...")
            self.display_thread.join(timeout=2.0)
            if self.display_thread.is_alive():
                logger.warning("Display thread did not terminate within timeout")
        
        # Clean up resources
        pygame.quit()
        self.initialized = False
        logger.info("Display interface stopped")
        
    def _display_loop(self):
        """
        Main display loop (runs in background thread)
        
        Handles events, updates display state, and renders the interface
        with hardware-optimized performance.
        """
        try:
            # Initialize frame timing
            global PERF_STATS
            PERF_STATS['start_time'] = time.time()
            PERF_STATS['frame_count'] = 0
            
            # Define touch regions for common actions
            self.define_touch_regions()
            
            # Main loop
            while self.active:
                # Start frame timing
                frame_start = time.time()
                
                # Handle input events
                self._handle_events()
                
                # Clear screen with theme background color
                self.screen.fill(self.colors["background"])
                
                # Draw the appropriate screen
                if self.display_mode == "live":
                    self._draw_live_screen()
                else:
                    self._draw_card_screen()
                
                # Draw debug info if enabled
                if self.debug_mode:
                    self._draw_debug_info()
                
                # Display performance information if enabled
                if self.show_fps:
                    self._draw_performance_info()
                
                # Update display
                pygame.display.flip()
                
                # Update performance metrics
                PERF_STATS['frame_count'] += 1
                frame_time = time.time() - frame_start
                PERF_STATS['frame_time'] = frame_time
                
                # Calculate current FPS (as rolling average)
                if PERF_STATS['frame_count'] > 10:  # Start calculating after 10 frames
                    elapsed = time.time() - PERF_STATS['start_time']
                    current_fps = PERF_STATS['frame_count'] / elapsed if elapsed > 0 else 0
                    # Smooth FPS calculation
                    if PERF_STATS['fps'] == 0:
                        PERF_STATS['fps'] = current_fps
                    else:
                        PERF_STATS['fps'] = 0.95 * PERF_STATS['fps'] + 0.05 * current_fps
                
                # Cap framerate based on platform and settings
                target_fps = self.target_fps
                if self.is_jetson and self.power_save_mode:
                    target_fps = self.fps_limit_battery
                
                # Use clock to cap framerate
                self.clock.tick(target_fps)
                
        except Exception as e:
            logger.error(f"Error in display loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Ensure pygame is properly shut down on exit
            try:
                pygame.quit()
            except Exception:
                pass
            
    def _handle_events(self):
        """
        Handle pygame events including touch input
        
        Processes keyboard, mouse, touch, and system events with support
        for both traditional input and touchscreen interaction.
        """
        for event in pygame.event.get():
            # Handle quit events
            if event.type == QUIT:
                self.active = False
                return
                
            # Keyboard input
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.active = False
                    return
                elif event.key == K_t or event.key == K_TAB:
                    # Toggle display mode between live and card
                    with self.lock:
                        self.display_mode = "card" if self.display_mode == "live" else "live"
                        # Regenerate touch regions for new mode
                        self.define_touch_regions()
                elif event.key == K_d and (event.mod & KMOD_CTRL):
                    # Toggle debug mode with Ctrl+D
                    self.debug_mode = not self.debug_mode
                    logger.info(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
                elif event.key == K_f and (event.mod & KMOD_CTRL):
                    # Toggle FPS display with Ctrl+F
                    self.show_fps = not self.show_fps
                        
            # Touch events (pygame 2.0+)
            elif hasattr(pygame, 'FINGERDOWN') and event.type == pygame.FINGERDOWN:
                # Convert normalized touch position to screen coordinates
                x = event.x * self.width
                y = event.y * self.height
                
                # Apply touch sensitivity
                if self.touch_sensitivity != 1.0:
                    # Adjust coordinates toward the center based on sensitivity
                    center_x = self.width / 2
                    center_y = self.height / 2
                    x = center_x + (x - center_x) * self.touch_sensitivity
                    y = center_y + (y - center_y) * self.touch_sensitivity
                
                # Add to touch points for debugging
                if self.show_touch_points:
                    self.touch_points.append((x, y, time.time()))
                    # Keep only recent points
                    if len(self.touch_points) > 10:
                        self.touch_points.pop(0)
                
                # Check defined touch regions
                self._check_touch_regions(x, y)
                
            # Mouse input (also handles touch input on some platforms)
            elif event.type == MOUSEBUTTONDOWN:
                # Get mouse position
                x, y = event.pos
                
                # Add to touch points for debugging
                if self.show_touch_points:
                    self.touch_points.append((x, y, time.time()))
                    # Keep only recent points
                    if len(self.touch_points) > 10:
                        self.touch_points.pop(0)
                
                # Check defined touch regions
                self._check_touch_regions(x, y)
                
    def define_touch_regions(self):
        """
        Define interactive regions on the screen with enhanced configuration support
        
        Creates touchable regions with associated actions based on current display mode,
        screen dimensions, and configuration settings. Supports dynamic region definition
        from configuration file.
        """
        # Clear existing regions
        self.touch_regions = []
        
        # Add footer region for toggling between views (always present)
        self.touch_regions.append({
            'name': 'toggle_view',
            'rect': pygame.Rect(0, self.height - 50, self.width, 50),
            'action': self.toggle_display_mode,
            'highlight_color': GRAY,
            'visible': True
        })
        
        # Add more regions based on the display mode
        if self.display_mode == 'live':
            # Quick button to view card
            self.touch_regions.append({
                'name': 'show_card',
                'rect': pygame.Rect(self.width - 100, 60, 100, 40),
                'action': lambda: self.set_display_mode('card'),
                'highlight_color': BLUE,
                'visible': True
            })
            
            # Scrollable regions for transcription and events
            # Calculate column widths based on settings
            total_width = self.width
            column_1_width = int(total_width * self.column_widths['transcription'])
            column_2_width = int(total_width * self.column_widths['events'])
            
            # Add transcription scroll region
            self.touch_regions.append({
                'name': 'transcription_scroll',
                'rect': pygame.Rect(0, 110, column_1_width, self.height - 160),
                'action': self._handle_transcription_scroll,
                'highlight_color': None,  # No highlight
                'visible': False  # Invisible interaction region
            })
            
            # Add events scroll region
            self.touch_regions.append({
                'name': 'events_scroll',
                'rect': pygame.Rect(column_1_width, 110, column_2_width, self.height - 160),
                'action': self._handle_events_scroll,
                'highlight_color': None,
                'visible': False
            })
        else:
            # Card view regions
            self.touch_regions.append({
                'name': 'show_live',
                'rect': pygame.Rect(self.width - 100, 60, 100, 40),
                'action': lambda: self.set_display_mode('live'),
                'highlight_color': RED,
                'visible': True
            })
            
            # Add region for anatomical diagram interaction
            diagram_section_width = int(self.width * 0.35)
            self.touch_regions.append({
                'name': 'anatomical_diagram',
                'rect': pygame.Rect(20, 100, diagram_section_width - 40, self.height - 160),
                'action': self._handle_diagram_interaction,
                'highlight_color': None,
                'visible': False
            })
        
        # Add any regions defined in config
        if hasattr(self, 'touch_region_config'):
            for region_name, region_config in self.touch_region_config.items():
                if not region_config.get('enabled', True):
                    continue
                    
                # Get rect from config
                if 'rect' in region_config:
                    rect = region_config['rect']
                    # Handle special values: -1 = full width/height, negative = from right/bottom
                    x, y, w, h = rect
                    if x < 0:
                        x = self.width + x if x < 0 else x  # negative x = from right edge
                    if y < 0:
                        y = self.height + y  # negative y = from bottom edge
                    if w < 0:
                        w = self.width - x  # negative width = full remaining width
                    if h < 0:
                        h = self.height - y  # negative height = full remaining height
                        
                    # Create rectangle
                    rect_obj = pygame.Rect(x, y, w, h)
                    
                    # Add to touch regions with default action (for now)
                    self.touch_regions.append({
                        'name': region_name,
                        'rect': rect_obj,
                        'action': lambda: logger.info(f"Touch region '{region_name}' activated"),
                        'highlight_color': None,
                        'visible': False
                    })
        
        logger.debug(f"Defined {len(self.touch_regions)} touch regions for mode '{self.display_mode}'")
    
    def _handle_transcription_scroll(self):
        """Handle scrolling in the transcription region"""
        # To be implemented for scrolling functionality
        logger.debug("Transcription scroll area touched")
    
    def _handle_events_scroll(self):
        """Handle scrolling in the events region"""
        # To be implemented for scrolling functionality
        logger.debug("Events scroll area touched")
    
    def _handle_diagram_interaction(self):
        """Handle interaction with the anatomical diagram"""
        # To be implemented for anatomical diagram interaction
        logger.debug("Anatomical diagram touched")
            
    def _check_touch_regions(self, x, y):
        """
        Check if a touch/click is within defined interactive regions
        
        Args:
            x: X coordinate of touch/click
            y: Y coordinate of touch/click
            
        Returns:
            bool: True if a region was activated
        """
        for region in self.touch_regions:
            if region['rect'].collidepoint(x, y):
                # Call the associated action
                if 'action' in region and region['action']:
                    region['action']()
                    
                    # Visual feedback for touch if debug is enabled
                    if self.debug_mode and region.get('highlight_color'):
                        # Flash region briefly
                        original_surface = self.screen.copy()
                        pygame.draw.rect(self.screen, region['highlight_color'], region['rect'])
                        pygame.display.update(region['rect'])
                        pygame.time.wait(100)  # Brief flash
                        self.screen.blit(original_surface, (0, 0))
                        pygame.display.update(region['rect'])
                        
                return True
        return False
        
    def _draw_debug_info(self):
        """Draw debug information on the screen"""
        if not self.debug_mode:
            return
            
        # Draw touch regions with semi-transparent overlay
        for region in self.touch_regions:
            if region.get('visible', True):  # Only draw visible regions
                # Create semi-transparent surface
                s = pygame.Surface((region['rect'].width, region['rect'].height), pygame.SRCALPHA)
                s.fill((255, 255, 255, 50))  # Semi-transparent white
                self.screen.blit(s, region['rect'])
                
                # Draw region outline
                pygame.draw.rect(self.screen, (255, 255, 0), region['rect'], 1)
                
                # Draw region name
                name_text = self.fonts['small'].render(region['name'], True, (255, 255, 0))
                self.screen.blit(name_text, (region['rect'].x + 5, region['rect'].y + 5))
        
        # Draw touch points
        for i, (x, y, t) in enumerate(self.touch_points):
            # Calculate age of touch point
            age = time.time() - t
            if age > 3.0:  # Older than 3 seconds, don't show
                continue
                
            # Fade out based on age
            alpha = int(255 * (1.0 - age / 3.0))
            color = (255, 0, 0, alpha)
            
            # Draw circle at touch point
            radius = 20 - int(age * 5)  # Shrink over time
            if radius > 0:
                pygame.draw.circle(self.screen, color, (int(x), int(y)), radius, 2)
                
                # Draw label
                label = self.fonts['small'].render(f"{i+1}", True, (255, 255, 0))
                self.screen.blit(label, (int(x) - 5, int(y) - 10))
        
        # Draw device info
        info_text = (
            f"Display: {self.width}x{self.height} | "
            f"Driver: {self.display_driver} | "
            f"Touch: {'Yes' if self.has_touch else 'No'} | "
            f"FPS: {int(PERF_STATS['fps'])}"
        )
        info_surface = self.fonts['small'].render(info_text, True, (255, 255, 0))
        self.screen.blit(info_surface, (10, 10))
        
    def _draw_performance_info(self):
        """Draw performance metrics on the screen"""
        if not self.show_fps:
            return
            
        # Draw simple FPS counter in corner
        fps_text = f"{int(PERF_STATS['fps'])} FPS"
        fps_surface = self.fonts['small'].render(fps_text, True, (255, 255, 0))
        self.screen.blit(fps_surface, (self.width - fps_surface.get_width() - 10, 10))
    
    def _draw_live_screen(self):
        """Draw the live view with transcription and significant events (optimized for landscape display)"""
        # Draw header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(self.screen, BLUE, header_rect)
        
        # Draw avatar if available
        if self.avatar:
            # Resize avatar for the header
            avatar_size = 50
            avatar = pygame.transform.scale(self.avatar, (avatar_size, avatar_size))
            self.screen.blit(avatar, (10, 5))
            header_text = self.fonts['bold_large'].render("TCCC.ai FIELD ASSISTANT", True, WHITE)
            self.screen.blit(header_text, (avatar_size + 20, 15))
        else:
            header_text = self.fonts['bold_large'].render("TCCC.ai FIELD ASSISTANT", True, WHITE)
            self.screen.blit(header_text, (20, 15))
        
        # Draw current time
        current_time = datetime.now().strftime("%H:%M:%S")
        time_text = self.fonts['medium'].render(current_time, True, WHITE)
        self.screen.blit(time_text, (self.width - time_text.get_width() - 20, 20))
        
        # For landscape layout - use a three-column design
        # Calculate column widths
        total_width = self.width
        column_1_width = int(total_width * 0.38)  # 38% for transcription
        column_2_width = int(total_width * 0.34)  # 34% for events
        column_3_width = total_width - column_1_width - column_2_width  # Remainder for card preview
        
        # Column divider positions
        first_divider_x = column_1_width
        second_divider_x = first_divider_x + column_2_width
        
        # Draw vertical dividers between columns
        pygame.draw.line(self.screen, GRAY, (first_divider_x, 60), (first_divider_x, self.height - 50), 2)
        pygame.draw.line(self.screen, GRAY, (second_divider_x, 60), (second_divider_x, self.height - 50), 2)
        
        # Column titles
        transcription_title = self.fonts['bold_medium'].render("SPEECH TRANSCRIPTION", True, GOLD)
        events_title = self.fonts['bold_medium'].render("SIGNIFICANT EVENTS", True, GOLD)
        card_preview_title = self.fonts['bold_medium'].render("TCCC CARD PREVIEW", True, GOLD)
        
        # Position column titles
        self.screen.blit(transcription_title, (first_divider_x//2 - transcription_title.get_width()//2, 70))
        self.screen.blit(events_title, (first_divider_x + (column_2_width//2) - events_title.get_width()//2, 70))
        self.screen.blit(card_preview_title, (second_divider_x + (column_3_width//2) - card_preview_title.get_width()//2, 70))
        
        # Column 1: Transcription
        with self.lock:
            transcription_items = self.transcription[-self.max_transcription_items:] if self.transcription else []
        
        column_width = first_divider_x - 30  # Leave margin
        y_pos = 110
        
        for item in transcription_items:
            wrapped_lines = self._wrap_text(item, self.fonts['small'], column_width)
            for line in wrapped_lines:
                text = self.fonts['small'].render(line, True, WHITE)
                self.screen.blit(text, (15, y_pos))
                y_pos += 25
                
                # Stop if we're approaching the footer
                if y_pos > self.height - 60:
                    break
            
            # Add a small gap between transcription items
            y_pos += 5
            
            # Stop if we're approaching the footer
            if y_pos > self.height - 60:
                break
        
        # Column 2: Significant Events
        with self.lock:
            events = self.significant_events[-self.max_event_items:] if self.significant_events else []
        
        column_width = column_2_width - 30  # Leave margin
        y_pos = 110
        
        for event in events:
            # Draw timestamp if available
            if isinstance(event, dict) and 'time' in event:
                time_str = event.get('time', '')
                text = self.fonts['small'].render(time_str, True, GREEN)
                self.screen.blit(text, (first_divider_x + 15, y_pos))
                
                # Draw event description
                desc = event.get('description', '')
                wrapped_lines = self._wrap_text(desc, self.fonts['small'], column_width - 20)
                y_pos += 25
                for line in wrapped_lines:
                    text = self.fonts['small'].render(line, True, WHITE)
                    self.screen.blit(text, (first_divider_x + 25, y_pos))
                    y_pos += 25
                    
                    # Stop if we're approaching the footer
                    if y_pos > self.height - 60:
                        break
            else:
                # Simple string event
                wrapped_lines = self._wrap_text(str(event), self.fonts['small'], column_width)
                for line in wrapped_lines:
                    text = self.fonts['small'].render(line, True, WHITE)
                    self.screen.blit(text, (first_divider_x + 15, y_pos))
                    y_pos += 25
                    
                    # Stop if we're approaching the footer
                    if y_pos > self.height - 60:
                        break
            
            # Add a small gap between events
            y_pos += 10
            
            # Stop if we're approaching the footer
            if y_pos > self.height - 60:
                break
        
        # Column 3: Card Preview
        with self.lock:
            card_data = self.card_data.copy()
        
        column_width = column_3_width - 30  # Leave margin
        
        # Draw a compact anatomical figure at the top of the card preview
        figure_y = 120
        figure_x = second_divider_x + column_3_width // 2
        
        # Draw figure - keep it smaller for landscape mode
        # Head
        pygame.draw.circle(self.screen, WHITE, (figure_x, figure_y), 15, 2)
        # Body
        pygame.draw.line(self.screen, WHITE, (figure_x, figure_y + 15), (figure_x, figure_y + 60), 2)
        # Arms
        pygame.draw.line(self.screen, WHITE, (figure_x, figure_y + 25), (figure_x - 25, figure_y + 40), 2)
        pygame.draw.line(self.screen, WHITE, (figure_x, figure_y + 25), (figure_x + 25, figure_y + 40), 2)
        # Legs
        pygame.draw.line(self.screen, WHITE, (figure_x, figure_y + 60), (figure_x - 25, figure_y + 100), 2)
        pygame.draw.line(self.screen, WHITE, (figure_x, figure_y + 60), (figure_x + 25, figure_y + 100), 2)
        
        # Start card fields below the figure
        y_pos = figure_y + 120
        
        # Most important card fields for preview
        preview_fields = [
            ("Name:", card_data.get("name", "")),
            ("Rank:", card_data.get("rank", "")),
            ("Injuries:", card_data.get("injuries", "")),
            ("Treatment:", card_data.get("treatment_given", "")),
            ("Vital Signs:", card_data.get("vital_signs", ""))
        ]
        
        # Draw card fields
        for field_name, field_value in preview_fields:
            # Draw field name
            field_text = self.fonts['bold_small'].render(field_name, True, GOLD)
            self.screen.blit(field_text, (second_divider_x + 15, y_pos))
            
            # Draw field value
            if len(field_name) > 5:  # Multi-line fields like "Injuries:"
                # These fields can be multi-line, but keep them concise in preview
                max_lines = 2 if y_pos < self.height - 150 else 1
                wrapped_lines = self._wrap_text(field_value, self.fonts['small'], column_width - 20)
                for j, line in enumerate(wrapped_lines[:max_lines]):
                    value_text = self.fonts['small'].render(line, True, WHITE)
                    self.screen.blit(value_text, (second_divider_x + 15, y_pos + 25 + (j * 25)))
                y_pos += 25 * (max_lines + 1)
            else:
                # Single line for short fields
                value_text = self.fonts['small'].render(field_value, True, WHITE)
                self.screen.blit(value_text, (second_divider_x + 110, y_pos))
                y_pos += 25
            
            # Add space between fields
            y_pos += 5
            
            # Stop if we're approaching the footer
            if y_pos > self.height - 60:
                break
        
        # Footer with instructions
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(self.screen, GRAY, footer_rect)
        instruction_text = self.fonts['medium'].render("Press 'T' for TCCC Card View", True, BLACK)
        self.screen.blit(instruction_text, (self.width//2 - instruction_text.get_width()//2, self.height - 35))
    
    def _draw_card_screen(self):
        """Draw the TCCC Casualty Card (DD Form 1380) for landscape display"""
        # Draw card header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(self.screen, RED, header_rect)
        
        # Draw title in a single line for landscape display
        title_text = self.fonts['bold_large'].render("TCCC CASUALTY CARD (DD FORM 1380)", True, WHITE)
        subtitle_text = self.fonts['small'].render("MEDICAL RECORD - SUPPLEMENTAL MEDICAL DATA", True, WHITE)
        
        # Center align the text
        self.screen.blit(title_text, (self.width//2 - title_text.get_width()//2, 10))
        self.screen.blit(subtitle_text, (self.width//2 - subtitle_text.get_width()//2, 40))
        
        # Draw card data
        with self.lock:
            card_data = self.card_data.copy()
        
        # For landscape, use a two-section layout:
        # Left section (35% width): Anatomical diagram
        # Right section (65% width): Patient data in columns
        
        # Calculate section widths
        diagram_section_width = int(self.width * 0.35)
        data_section_width = self.width - diagram_section_width
        
        # Draw divider between diagram and data
        pygame.draw.line(self.screen, GRAY, (diagram_section_width, 60), (diagram_section_width, self.height - 50), 2)
        
        # Left section: Anatomical diagram
        # Draw section title
        diagram_title = self.fonts['bold_medium'].render("ANATOMICAL DIAGRAM", True, GOLD)
        self.screen.blit(diagram_title, (diagram_section_width//2 - diagram_title.get_width()//2, 70))
        
        # Draw anatomical diagram centered in left section
        diagram_x = diagram_section_width // 2
        diagram_y = self.height // 2 - 70  # Centered vertically, adjusted for header
        
        # Draw figure border
        diagram_rect = pygame.Rect(20, 100, diagram_section_width - 40, self.height - 160)
        pygame.draw.rect(self.screen, WHITE, diagram_rect, 2)
        
        # Draw larger stick figure for better visibility
        # Head
        head_radius = 30
        pygame.draw.circle(self.screen, WHITE, (diagram_x, diagram_y - 60), head_radius, 2)
        
        # Body
        body_length = 120
        pygame.draw.line(self.screen, WHITE, (diagram_x, diagram_y - 30), (diagram_x, diagram_y + body_length - 30), 3)
        
        # Arms
        arm_length = 70
        pygame.draw.line(self.screen, WHITE, (diagram_x, diagram_y), (diagram_x - arm_length, diagram_y + 20), 3)
        pygame.draw.line(self.screen, WHITE, (diagram_x, diagram_y), (diagram_x + arm_length, diagram_y + 20), 3)
        
        # Legs
        leg_length = 120
        pygame.draw.line(self.screen, WHITE, (diagram_x, diagram_y + body_length - 30), (diagram_x - 50, diagram_y + body_length + 90), 3)
        pygame.draw.line(self.screen, WHITE, (diagram_x, diagram_y + body_length - 30), (diagram_x + 50, diagram_y + body_length + 90), 3)
        
        # Right section: Patient data in a 2-column grid
        # Draw section title
        info_title = self.fonts['bold_medium'].render("PATIENT INFORMATION", True, GOLD)
        self.screen.blit(info_title, (diagram_section_width + (data_section_width//2) - info_title.get_width()//2, 70))
        
        # Calculate column widths for patient data (split into 2 columns)
        col_width = data_section_width // 2
        col1_x = diagram_section_width + 20  # Start of first column
        col2_x = diagram_section_width + col_width + 10  # Start of second column
        data_y_start = 110  # Start below the title
        
        # Organize fields into two columns for efficient use of space
        # Column 1: Patient identification and incident info
        col1_fields = [
            ("Name:", card_data.get("name", "")),
            ("Rank:", card_data.get("rank", "")),
            ("Unit:", card_data.get("unit", "")),
            ("Date:", card_data.get("date", datetime.now().strftime("%Y-%m-%d"))),
            ("Time:", card_data.get("time", datetime.now().strftime("%H:%M"))),
            ("MOI:", card_data.get("mechanism_of_injury", ""))
        ]
        
        # Column 2: Medical assessment and treatment
        col2_fields = [
            ("Injuries:", card_data.get("injuries", "")),
            ("Vital Signs:", card_data.get("vital_signs", "")),
            ("Treatment:", card_data.get("treatment_given", "")),
            ("Medications:", card_data.get("medications", "")),
            ("Evacuation:", card_data.get("evacuation_priority", ""))
        ]
        
        # Draw first column fields
        y_pos = data_y_start
        field_height = 30
        
        for field_name, field_value in col1_fields:
            # Draw field name
            field_text = self.fonts['bold_small'].render(field_name, True, GOLD)
            self.screen.blit(field_text, (col1_x, y_pos))
            
            # Draw field value
            value_x = col1_x + 70  # Offset for field name
            wrapped_lines = self._wrap_text(field_value, self.fonts['small'], col_width - 90)
            
            for j, line in enumerate(wrapped_lines[:2]):  # Limit to 2 lines per field
                value_text = self.fonts['small'].render(line, True, WHITE)
                self.screen.blit(value_text, (value_x, y_pos + j*25))
            
            # Move to next field
            y_pos += field_height + (0 if len(wrapped_lines) <= 1 else 20)
            
            # Stop if we're approaching the footer
            if y_pos > self.height - 60:
                break
        
        # Draw second column fields (medical info)
        y_pos = data_y_start
        
        for field_name, field_value in col2_fields:
            # Draw field name
            field_text = self.fonts['bold_small'].render(field_name, True, GOLD)
            self.screen.blit(field_text, (col2_x, y_pos))
            
            # Draw field value - multiline for medical fields
            value_x = col2_x + 110  # Wider offset for longer field names
            
            # Handle long fields specially
            if field_name in ["Injuries:", "Treatment:", "Vital Signs:"]:
                # These fields get more space - up to 4 lines
                wrapped_lines = self._wrap_text(field_value, self.fonts['small'], col_width - 130)
                # Start on same line for better space usage
                self.screen.blit(self.fonts['small'].render(wrapped_lines[0] if wrapped_lines else "", True, WHITE), 
                                (value_x, y_pos))
                
                # Remaining lines with indent
                for j, line in enumerate(wrapped_lines[1:4]):  # Show up to 3 more lines (4 total)
                    value_text = self.fonts['small'].render(line, True, WHITE)
                    self.screen.blit(value_text, (col2_x + 20, y_pos + (j+1)*25))
                
                # Move to next field with space for the lines we used
                line_count = min(4, len(wrapped_lines))
                y_pos += 25 * line_count
            else:
                # Standard fields get 2 lines
                wrapped_lines = self._wrap_text(field_value, self.fonts['small'], col_width - 130)
                for j, line in enumerate(wrapped_lines[:2]):  # Limit to 2 lines
                    value_text = self.fonts['small'].render(line, True, WHITE)
                    self.screen.blit(value_text, (value_x, y_pos + j*25))
                
                # Move to next field
                y_pos += field_height + (0 if len(wrapped_lines) <= 1 else 20)
            
            # Stop if we're approaching the footer
            if y_pos > self.height - 60:
                break
        
        # Footer with instructions
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(self.screen, GRAY, footer_rect)
        instruction_text = self.fonts['medium'].render("Press 'T' for Live View", True, BLACK)
        self.screen.blit(instruction_text, (self.width//2 - instruction_text.get_width()//2, self.height - 35))
    
    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within a given width"""
        if not text:
            return [""]
            
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            test_width = font.size(test_line)[0]
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                if not current_line:  # If the word itself is too long
                    lines.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines
    
    def update_transcription(self, text):
        """Update the transcription display with new text"""
        with self.lock:
            self.transcription.append(text)
            # Keep only the latest items
            if len(self.transcription) > self.max_transcription_items * 2:
                self.transcription = self.transcription[-self.max_transcription_items:]
            self.last_update = time.time()
    
    def add_significant_event(self, event):
        """
        Add a significant event to the display
        
        Args:
            event: String or dict with 'time' and 'description' keys
        """
        # If it's a string, convert to a dict with timestamp
        if isinstance(event, str):
            event = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'description': event
            }
        
        with self.lock:
            self.significant_events.append(event)
            # Keep only the latest items
            if len(self.significant_events) > self.max_event_items * 2:
                self.significant_events = self.significant_events[-self.max_event_items:]
            self.last_update = time.time()
    
    def update_card_data(self, data):
        """
        Update the TCCC Casualty Card data
        
        Args:
            data: Dictionary with card field values
        """
        with self.lock:
            self.card_data.update(data)
            self.last_update = time.time()
    
    def set_display_mode(self, mode):
        """
        Set the display mode
        
        Args:
            mode: Either 'live' or 'card'
        """
        if mode not in ['live', 'card']:
            logger.warning(f"Invalid display mode: {mode}")
            return
            
        with self.lock:
            self.display_mode = mode
    
    def toggle_display_mode(self):
        """Toggle between live view and card view"""
        with self.lock:
            self.display_mode = "card" if self.display_mode == "live" else "live"


# Standalone example usage
if __name__ == "__main__":
    # Example usage
    display = DisplayInterface(width=800, height=480, fullscreen=False)
    display.start()
    
    try:
        # Add some sample data
        display.update_transcription("Starting recording...")
        time.sleep(1)
        
        display.update_transcription("Patient has a gunshot wound to the left leg.")
        display.add_significant_event("Gunshot wound identified - left thigh")
        time.sleep(1)
        
        display.update_transcription("I'm applying a tourniquet now.")
        display.add_significant_event("Tourniquet applied to left thigh")
        time.sleep(1)
        
        display.update_transcription("Checking for other injuries...")
        time.sleep(1)
        
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
        
        # After showing sample data for 5 seconds, toggle to card view
        time.sleep(5)
        display.toggle_display_mode()
        
        # Wait for 20 seconds before exiting
        time.sleep(20)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        display.stop()