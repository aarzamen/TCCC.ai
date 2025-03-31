#!/usr/bin/env python3
"""
TCCC.ai Display Configuration Manager
------------------------------------
Manages display configuration profiles for different hardware setups.
Provides automatic detection of displays and optimization for Jetson Nano.
"""

import os
import sys
import logging
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayConfig")

class DisplayConfig:
    """
    Display Configuration Manager for TCCC.ai
    
    Handles display profiles, hardware detection, and optimization settings
    for various display types including the Waveshare display.
    """
    
    # Default configurations for different displays
    DEFAULT_CONFIGS = {
        "waveshare": {
            "width": 1280,
            "height": 720,
            "fullscreen": True,
            "touch_enabled": True,
            "fps_limit": 30,
            "performance_mode": "balanced"
        },
        "jetson_hdmi": {
            "width": 1920,
            "height": 1080,
            "fullscreen": True,
            "touch_enabled": False,
            "fps_limit": 30,
            "performance_mode": "balanced"
        },
        "desktop": {
            "width": 1280, 
            "height": 720,
            "fullscreen": False,
            "touch_enabled": False,
            "fps_limit": 60,
            "performance_mode": "quality"
        },
        "headless": {
            "width": 800,
            "height": 600,
            "fullscreen": False,
            "touch_enabled": False,
            "fps_limit": 15,
            "performance_mode": "minimal"
        }
    }
    
    # Color themes
    THEMES = {
        "default": {
            "background": (0, 0, 0),           # Black
            "text": (255, 255, 255),           # White
            "header": (0, 0, 255),             # Blue
            "highlight": (255, 215, 0),        # Gold
            "alert": (255, 0, 0),              # Red
            "success": (0, 200, 0),            # Green
        },
        "high_contrast": {
            "background": (0, 0, 0),           # Black
            "text": (255, 255, 255),           # White
            "header": (0, 120, 255),           # Brighter blue
            "highlight": (255, 255, 0),        # Yellow
            "alert": (255, 50, 50),            # Brighter red
            "success": (50, 255, 50),          # Brighter green
        },
        "night_mode": {
            "background": (0, 0, 0),           # Black
            "text": (180, 180, 180),           # Light gray
            "header": (0, 60, 120),            # Darker blue
            "highlight": (180, 150, 0),        # Darker gold
            "alert": (180, 0, 0),              # Darker red
            "success": (0, 120, 0),            # Darker green
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the display configuration manager.
        
        Args:
            config_path: Optional path to custom configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.display_profile = "desktop"  # Default profile
        self.is_jetson = self._detect_jetson()
        self.dimensions = (800, 480)  # Default fallback dimensions
        
        # Try to detect display hardware
        self.detected_resolution = self._detect_display_resolution()
        
        # Load configuration
        self._load_config()
        
    def _detect_jetson(self) -> bool:
        """
        Detect if running on NVIDIA Jetson hardware.
        
        Returns:
            bool: True if running on Jetson, False otherwise
        """
        # Check for Jetson-specific files
        jetson_paths = [
            '/etc/nv_tegra_release',
            '/proc/device-tree/model'
        ]
        
        for path in jetson_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        if 'jetson' in content.lower() or 'tegra' in content.lower():
                            logger.info("Detected Jetson hardware")
                            return True
                except:
                    pass
        
        # Check in platform info
        uname_info = platform.uname()
        if 'aarch64' in uname_info.machine.lower() and 'nvidia' in uname_info.version.lower():
            logger.info("Detected Jetson hardware from platform info")
            return True
            
        return False
    
    def _detect_display_resolution(self) -> Tuple[int, int]:
        """
        Attempt to detect the current display resolution.
        
        Returns:
            Tuple[int, int]: (width, height) of the detected display
        """
        # Check for environment variable first
        if "TCCC_DISPLAY_RESOLUTION" in os.environ:
            resolution_str = os.environ["TCCC_DISPLAY_RESOLUTION"]
            try:
                width, height = map(int, resolution_str.split('x'))
                logger.info(f"Display resolution from environment: {width}x{height}")
                return (width, height)
            except Exception as e:
                logger.warning(f"Failed to parse TCCC_DISPLAY_RESOLUTION: {e}")
        
        # Try to detect using pygame
        try:
            import pygame
            pygame.display.init()
            info = pygame.display.Info()
            width, height = info.current_w, info.current_h
            pygame.display.quit()
            
            if width > 0 and height > 0:
                logger.info(f"Detected display resolution: {width}x{height}")
                return (width, height)
        except Exception as e:
            logger.warning(f"Failed to detect display resolution with pygame: {e}")
        
        # Use Jetson-specific detection if on Jetson
        if self.is_jetson:
            # Common Waveshare resolutions
            if os.path.exists('/dev/fb0'):
                try:
                    import fcntl
                    import array
                    import struct
                    
                    # FBIOGET_VSCREENINFO constant
                    FBIOGET_VSCREENINFO = 0x4600
                    
                    with open('/dev/fb0', 'rb') as fb:
                        fix_info = array.array('c', ['\0'] * 68)
                        fcntl.ioctl(fb.fileno(), FBIOGET_VSCREENINFO, fix_info)
                        width, height = struct.unpack('II', fix_info[0:8])
                        
                        if width > 0 and height > 0:
                            logger.info(f"Detected framebuffer resolution: {width}x{height}")
                            return (width, height)
                except Exception as e:
                    logger.warning(f"Failed to detect framebuffer resolution: {e}")
        
        # Default fallback based on common profile
        if self.is_jetson:
            logger.info("Using default Jetson resolution: 1280x720")
            return (1280, 720)  # Default for Waveshare display
        else:
            logger.info("Using default desktop resolution: 1280x720")
            return (1280, 720)  # Default for desktop
    
    def _detect_display_profile(self) -> str:
        """
        Detect which display profile to use based on hardware and resolution.
        
        Returns:
            str: Name of the display profile to use
        """
        # Check if headless mode
        if not os.environ.get("DISPLAY") and not os.path.exists('/dev/fb0'):
            logger.info("Running in headless mode")
            return "headless"
        
        # Check for Jetson with Waveshare display
        if self.is_jetson:
            # Waveshare common resolutions
            waveshare_resolutions = [
                (1280, 720),   # 6.25" display
                (1560, 720),   # 7" display
                (800, 480),    # 5" display
            ]
            
            for res in waveshare_resolutions:
                if abs(res[0] - self.detected_resolution[0]) < 50 and abs(res[1] - self.detected_resolution[1]) < 50:
                    logger.info(f"Detected Waveshare display: {res[0]}x{res[1]}")
                    return "waveshare"
            
            # If not Waveshare but still Jetson, assume HDMI display
            logger.info("Using Jetson HDMI profile")
            return "jetson_hdmi"
        
        # Default to desktop for non-Jetson hardware
        logger.info("Using desktop display profile")
        return "desktop"
    
    def _load_config(self):
        """Load display configuration from file or use defaults."""
        config = {}
        
        # Try to load from config file
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        
        # Detect display profile if not explicitly set
        self.display_profile = config.get('profile', self._detect_display_profile())
        
        # Load profile configuration
        profile_config = self.DEFAULT_CONFIGS.get(self.display_profile, self.DEFAULT_CONFIGS["desktop"])
        
        # Set dimensions from detection if not specified
        if self.detected_resolution != (0, 0):
            profile_config["width"] = self.detected_resolution[0]
            profile_config["height"] = self.detected_resolution[1]
        
        # Apply custom config on top of profile defaults
        display_config = config.get('display', {})
        for key, value in display_config.items():
            profile_config[key] = value
        
        # Load theme
        theme_name = config.get('theme', 'default')
        theme = self.THEMES.get(theme_name, self.THEMES['default'])
        
        # Apply custom theme colors if provided
        custom_colors = config.get('colors', {})
        for key, value in custom_colors.items():
            theme[key] = value
        
        # Set final configuration
        self.config = {
            'profile': self.display_profile,
            'display': profile_config,
            'theme': theme_name,
            'colors': theme,
            'animation': config.get('animation', {
                'enabled': True,
                'transition_speed': 300,
                'fade_in': True,
                'scroll_smooth': True,
            }),
            'performance': config.get('performance', {
                'fps_limit': profile_config.get('fps_limit', 30),
                'optimization_level': profile_config.get('performance_mode', 'balanced'),
                'show_fps': False,
                'show_memory': False,
            }),
            'layout': config.get('layout', {
                'column_1_width': 0.38,
                'column_2_width': 0.34,
                'column_3_width': 0.28,
            }),
            'advanced': config.get('advanced', {
                'debug_mode': False,
                'touch_debug': False,
                'auto_detect': True,
            }),
            'fonts': config.get('fonts', {
                'small': 22,
                'medium': 28,
                'large': 36,
            }),
        }
        
        # Store dimensions for easy access
        self.dimensions = (profile_config["width"], profile_config["height"])
        
        logger.info(f"Active display profile: {self.display_profile}")
        logger.info(f"Display dimensions: {self.dimensions[0]}x{self.dimensions[1]}")
        
    def save_config(self, path: Optional[str] = None):
        """
        Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration to (defaults to config_path)
        """
        save_path = path or self.config_path
        if not save_path:
            logger.warning("No path specified for saving configuration")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create a simplified version for saving
            save_config = {
                'profile': self.display_profile,
                'display': self.config['display'],
                'theme': self.config['theme'],
                'animation': self.config['animation'],
                'performance': self.config['performance'],
                'layout': self.config['layout'],
                'advanced': self.config['advanced'],
                'fonts': self.config['fonts'],
            }
            
            with open(save_path, 'w') as f:
                json.dump(save_config, f, indent=2)
                
            logger.info(f"Configuration saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration.
        
        Returns:
            Dict[str, Any]: Current configuration
        """
        return self.config
    
    def get_display_dimensions(self) -> Tuple[int, int]:
        """
        Get the display dimensions.
        
        Returns:
            Tuple[int, int]: (width, height) of the display
        """
        return self.dimensions
    
    def get_theme_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get the theme colors.
        
        Returns:
            Dict[str, Tuple[int, int, int]]: Theme colors
        """
        return self.config['colors']
    
    def is_jetson_device(self) -> bool:
        """
        Check if running on Jetson device.
        
        Returns:
            bool: True if running on Jetson device
        """
        return self.is_jetson
    
    def get_optimization_level(self) -> str:
        """
        Get the optimization level.
        
        Returns:
            str: Optimization level ('minimal', 'balanced', 'quality')
        """
        return self.config['performance']['optimization_level']
    
    def get_fps_limit(self) -> int:
        """
        Get the FPS limit.
        
        Returns:
            int: FPS limit
        """
        return self.config['performance']['fps_limit']
    
    def is_touch_enabled(self) -> bool:
        """
        Check if touch input is enabled.
        
        Returns:
            bool: True if touch input is enabled
        """
        return self.config['display'].get('touch_enabled', False)
    
    def is_fullscreen(self) -> bool:
        """
        Check if fullscreen mode is enabled.
        
        Returns:
            bool: True if fullscreen mode is enabled
        """
        return self.config['display'].get('fullscreen', False)


# Create a global configuration instance
def get_display_config(config_path: Optional[str] = None) -> DisplayConfig:
    """
    Get or create the display configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        DisplayConfig: Display configuration instance
    """
    global _display_config_instance
    
    if '_display_config_instance' not in globals():
        _display_config_instance = DisplayConfig(config_path)
        
    return _display_config_instance


# Example usage
if __name__ == "__main__":
    # Print current display configuration
    config = get_display_config()
    print(f"Display Profile: {config.display_profile}")
    print(f"Resolution: {config.dimensions[0]}x{config.dimensions[1]}")
    print(f"Running on Jetson: {config.is_jetson}")
    print(f"Theme: {config.config['theme']}")
    
    # Print full configuration as JSON
    print("\nFull Configuration:")
    import json
    print(json.dumps(config.get_config(), indent=2))