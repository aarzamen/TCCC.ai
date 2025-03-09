#!/usr/bin/env python3
"""
TCCC.ai Display Interface
------------------------
Standard display interface for HDMI output showing:
1. Transcribed text from STT engine 
2. Significant events parsed by LLM
3. TCCC Casualty Card (DD Form 1380) when care is complete

This implementation uses standard HDMI output and supports any display connected
to the system. It works seamlessly with any HDMI monitor or display, including
the WaveShare display via standard HDMI connection.
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayInterface")

# Import pygame for display
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

# Default colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
GOLD = (255, 215, 0)

# Font sizes
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

# Performance monitoring
PERF_STATS = {
    'fps': 0,
    'frame_time': 0,
    'frame_count': 0,
    'start_time': 0,
}

class DisplayInterface:
    """Standard display interface for TCCC.ai using HDMI output"""
    
    def __init__(self, width=1280, height=720, fullscreen=False, config=None):
        """
        Initialize the display interface for standard HDMI output
        
        Args:
            width: Screen width (default 1280 for standard HD resolution)
            height: Screen height (default 720 for standard HD resolution)
            fullscreen: Whether to display in fullscreen mode
            config: Optional configuration dictionary
        """
        # Display dimensions
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        
        # Runtime state
        self.active = False
        self.initialized = False
        self.display_thread = None
        self.screen = None
        self.clock = None
        self.display_driver = None
        
        # Configuration
        self.auto_detect = True
        self.is_jetson = self._check_if_jetson()
        self.has_touch = False
        self.logo_path = "images/blue_logo.png"
        self.alt_logo_path = "images/green_logo.png"
        
        # Content to display
        self.transcription = []
        self.significant_events = []
        self.card_data = {}
        self.display_mode = "live"  # 'live' or 'card'
        self.last_update = time.time()
        
        # UI settings
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
        self.column_widths = {
            "transcription": 0.38,
            "events": 0.34,
            "card_preview": 0.28,
        }
        
        # Touch input
        self.touch_enabled = False
        self.touch_regions = {}
        self.touch_sensitivity = 1.0
        
        # Animation settings
        self.animations_enabled = True
        self.transition_speed = 300  # ms
        self.fade_in = True
        self.scroll_smooth = True
        
        # Performance settings
        self.target_fps = 30
        
        # Debug settings
        self.debug_mode = False
        self.show_touch_points = False
        self.show_fps = False
        self.show_memory_usage = False
        
        # Apply configuration if provided
        if config:
            self._apply_config(config)
        
        # Thread lock for thread-safe updates
        self.lock = threading.Lock()
    
    def _apply_config(self, config):
        """Apply configuration settings from a dictionary"""
        try:
            # Apply display settings
            if 'display' in config:
                display_config = config['display']
                if 'width' in display_config:
                    self.width = display_config['width']
                if 'height' in display_config:
                    self.height = display_config['height']
                if 'fullscreen' in display_config:
                    self.fullscreen = display_config['fullscreen']
                if 'auto_detect' in display_config:
                    self.auto_detect = display_config['auto_detect']
                    
                # Apply touch settings
                if 'touch' in display_config:
                    touch_config = display_config['touch']
                    if 'enabled' in touch_config:
                        self.touch_enabled = touch_config['enabled']
                    if 'sensitivity' in touch_config:
                        self.touch_sensitivity = touch_config['sensitivity']
            
            # Apply UI settings
            if 'ui' in config:
                ui_config = config['ui']
                if 'theme' in ui_config:
                    theme = ui_config['theme']
                    if theme in ['dark', 'light'] and 'color_schemes' in ui_config:
                        self.colors = ui_config['color_schemes'][theme]
                
                # Apply animation settings
                if 'animations' in ui_config:
                    animations_config = ui_config['animations']
                    if 'enabled' in animations_config:
                        self.animations_enabled = animations_config['enabled']
                    if 'transition_speed_ms' in animations_config:
                        self.transition_speed = animations_config['transition_speed_ms']
                    if 'fade_in' in animations_config:
                        self.fade_in = animations_config['fade_in']
                    if 'scroll_smooth' in animations_config:
                        self.scroll_smooth = animations_config['scroll_smooth']
                
                # Apply layout settings
                if 'layout' in ui_config and 'column_1_width' in ui_config['layout']:
                    self.column_widths["transcription"] = ui_config['layout']['column_1_width']
                    self.column_widths["events"] = ui_config['layout']['column_2_width']
                    self.column_widths["card_preview"] = 1.0 - ui_config['layout']['column_1_width'] - ui_config['layout']['column_2_width']
                
                # Apply logo settings
                if 'logo' in ui_config:
                    self.logo_path = ui_config['logo']
                if 'alt_logo' in ui_config:
                    self.alt_logo_path = ui_config['alt_logo']
            
            # Apply performance settings
            if 'performance' in config:
                perf_config = config['performance']
                if 'target_fps' in perf_config:
                    self.target_fps = perf_config['target_fps']
                if 'show_fps' in perf_config:
                    self.show_fps = perf_config['show_fps']
            
            # Apply debug settings
            if 'advanced' in config:
                adv_config = config['advanced']
                if 'debug_mode' in adv_config:
                    self.debug_mode = adv_config['debug_mode']
                if 'show_touch_points' in adv_config:
                    self.show_touch_points = adv_config['show_touch_points']
            
            logger.info(f"Applied configuration settings from config dictionary")
            return True
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
            return False
        
    def _check_if_jetson(self):
        """Check if running on NVIDIA Jetson hardware"""
        try:
            # Check for Jetson-specific file
            return os.path.exists('/etc/nv_tegra_release')
        except:
            return False
            
    def detect_display(self):
        """Detect display hardware and adjust settings accordingly"""
        try:
            # Initialize display module
            pygame.display.init()
            
            # Get display info
            info = pygame.display.Info()
            detected_width = info.current_w
            detected_height = info.current_h
            
            # Check if we have valid dimensions
            if detected_width > 0 and detected_height > 0:
                logger.info(f"Detected display: {detected_width}x{detected_height}")
                
                # Check if this looks like a WaveShare display
                if (detected_width == 1560 and detected_height == 720) or \
                   (detected_width == 720 and detected_height == 1560):
                    logger.info("Detected WaveShare 6.25 inch display")
                    self.width = detected_width
                    self.height = detected_height
                # Otherwise use detected resolution
                else:
                    self.width = detected_width
                    self.height = detected_height
                
            # Detect touch capability
            try:
                # Check if any input devices are available
                pygame.joystick.init()
                
                # Look for touch devices
                if hasattr(pygame, 'touch'):
                    pygame.touch.init()
                    num_touch_devices = pygame.touch.get_num_devices()
                    if num_touch_devices > 0:
                        logger.info(f"Detected {num_touch_devices} touch devices")
                        self.has_touch = True
                        self.touch_enabled = True
            except Exception as e:
                logger.warning(f"Error detecting touch devices: {e}")
                self.has_touch = False
                
            return True
        except Exception as e:
            logger.error(f"Display detection error: {e}")
            return False
            
    def setup_touch_input(self):
        """Set up touch input regions and handlers"""
        if not self.has_touch:
            logger.info("Touch input not available, skipping setup")
            return
            
        try:
            # Define touch regions based on screen layout
            self.touch_regions = {
                # Bottom bar for view toggle
                "toggle_view": pygame.Rect(0, self.height - 50, self.width, 50),
                
                # Card button in right panel
                "card_button": pygame.Rect(self.width - 150, 605, 150, 30),
                
                # Transcription scroll area
                "transcription_scroll": pygame.Rect(20, 100, 
                                               int(self.width * self.column_widths["transcription"]), 
                                               self.height - 160),
                                               
                # Events scroll area
                "events_scroll": pygame.Rect(int(self.width * self.column_widths["transcription"]) + 20, 
                                        100, 
                                        int(self.width * self.column_widths["events"]),
                                        self.height - 160),
                                        
                # Mode buttons (monitoring, documentation, reference)
                "mode_monitoring": pygame.Rect(40, 650, 120, 40),
                "mode_documentation": pygame.Rect(180, 650, 150, 40),
                "mode_reference": pygame.Rect(350, 650, 120, 40),
                
                # Action buttons (keyboard, save, emergency)
                "action_keyboard": pygame.Rect(650, 650, 120, 40),
                "action_save": pygame.Rect(800, 650, 120, 40),
                "action_emergency": pygame.Rect(1250, 650, 270, 40)
            }
            
            logger.info(f"Touch input setup complete with {len(self.touch_regions)} regions")
            return True
        except Exception as e:
            logger.error(f"Error setting up touch input: {e}")
            self.has_touch = False
            return False
        
    def initialize(self):
        """
        Initialize the display interface for standard HDMI output
        
        Returns:
            bool: True if initialization was successful
        """
        if self.initialized:
            logger.info("Display already initialized")
            return True
            
        try:
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
            self.icons = {}
            
            # Find project root - walk up from current file to find project root
            logo_path = Path(__file__).parent.parent.parent.parent
            
            # Main logo
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
            
            # Create status icons (if we can't load them)
            self.icons["audio_active"] = self._create_text_surface("🔊", self.fonts['medium'], WHITE)
            self.icons["battery"] = self._create_text_surface("🔋", self.fonts['medium'], WHITE)
            self.icons["clock"] = self._create_text_surface("🕒", self.fonts['medium'], WHITE)
            self.icons["warning"] = self._create_text_surface("⚠️", self.fonts['medium'], RED)
            self.icons["check"] = self._create_text_surface("✓", self.fonts['medium'], GREEN)
            self.icons["medical"] = self._create_text_surface("⚕️", self.fonts['medium'], WHITE)
            self.icons["vitals"] = self._create_text_surface("♥️", self.fonts['medium'], WHITE)
            self.icons["note"] = self._create_text_surface("📝", self.fonts['medium'], WHITE)
            self.icons["keyboard"] = self._create_text_surface("⌨️", self.fonts['medium'], WHITE)
            self.icons["save"] = self._create_text_surface("📋", self.fonts['medium'], WHITE)
            self.icons["emergency"] = self._create_text_surface("🚨", self.fonts['medium'], RED)
            
            logger.info(f"Created {len(self.icons)} icons for display")
        
        except Exception as e:
            logger.error(f"Error loading assets: {e}")
            # We'll continue without assets if they can't be loaded
            
    def _create_text_surface(self, text, font, color):
        """Create a surface with text rendered on it"""
        try:
            return font.render(text, True, color)
        except Exception as e:
            logger.error(f"Error creating text surface: {e}")
            # Return a small colored rectangle as fallback
            surface = pygame.Surface((30, 30))
            surface.fill(color)
            return surface
            
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
        
    def _handle_touch_input(self, pos):
        """
        Handle touch input at the given position
        
        Args:
            pos: tuple (x, y) of touch position
        """
        # Check each touch region
        for region_name, rect in self.touch_regions.items():
            if rect.collidepoint(pos):
                logger.info(f"Touch detected in region: {region_name}")
                
                # Handle different regions
                if region_name == "toggle_view":
                    with self.lock:
                        self.display_mode = "card" if self.display_mode == "live" else "live"
                    return True
                
                elif region_name == "card_button":
                    with self.lock:
                        self.display_mode = "card"
                    return True
                
                elif region_name == "transcription_scroll":
                    # Would implement scrolling here
                    return True
                
                elif region_name == "events_scroll":
                    # Would implement scrolling here
                    return True
                
                elif region_name.startswith("mode_"):
                    # Handle mode buttons - live view is "monitoring"
                    if region_name == "mode_monitoring":
                        with self.lock:
                            self.display_mode = "live"
                    elif region_name == "mode_documentation":
                        # Would implement documentation view here
                        logger.info("Documentation mode selected - not yet implemented")
                    elif region_name == "mode_reference":
                        # Would implement reference view here
                        logger.info("Reference mode selected - not yet implemented")
                    return True
                
                elif region_name.startswith("action_"):
                    # Handle action buttons
                    if region_name == "action_keyboard":
                        logger.info("Keyboard requested - not yet implemented")
                    elif region_name == "action_save":
                        logger.info("Save requested - not yet implemented")
                    elif region_name == "action_emergency":
                        logger.info("Emergency action requested - not yet implemented")
                    return True
                    
        return False
    
    def _display_loop(self):
        """
        Main display loop (runs in background thread)
        """
        try:
            # Initialize frame timing
            global PERF_STATS
            PERF_STATS['start_time'] = time.time()
            PERF_STATS['frame_count'] = 0
            
            # Animation tracking
            last_mode = self.display_mode
            transition_start = 0
            in_transition = False
            transition_progress = 0.0
            
            # Touch tracking
            touch_start_pos = None
            touch_current_pos = None
            is_dragging = False
            
            # Main loop
            while self.active:
                # Handle input events
                for event in pygame.event.get():
                    # Handle quit events
                    if event.type == QUIT:
                        self.active = False
                        break
                        
                    # Keyboard input
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            self.active = False
                            break
                        elif event.key == K_t or event.key == K_TAB:
                            # Toggle display mode
                            with self.lock:
                                last_mode = self.display_mode
                                self.display_mode = "card" if self.display_mode == "live" else "live"
                                
                                # Start transition if animations enabled
                                if self.animations_enabled:
                                    in_transition = True
                                    transition_start = time.time()
                                    transition_progress = 0.0
                                    
                    # Mouse/touch input handling
                    elif event.type == MOUSEBUTTONDOWN:
                        # Track starting position for potential drag
                        touch_start_pos = event.pos
                        touch_current_pos = event.pos
                        is_dragging = False
                        
                    elif event.type == MOUSEMOTION and touch_start_pos is not None:
                        # Update current position and check if dragging
                        touch_current_pos = event.pos
                        # Calculate drag distance
                        dx = touch_current_pos[0] - touch_start_pos[0]
                        dy = touch_current_pos[1] - touch_start_pos[1]
                        distance = (dx**2 + dy**2)**0.5
                        
                        # If moved more than threshold, consider it a drag
                        if distance > 10:
                            is_dragging = True
                            
                    elif event.type == MOUSEBUTTONUP:
                        # If not dragging, treat as a touch/click
                        if not is_dragging and touch_start_pos is not None:
                            self._handle_touch_input(event.pos)
                            
                        # Reset touch tracking
                        touch_start_pos = None
                        touch_current_pos = None
                        is_dragging = False
                
                # Update transition animation
                if in_transition and self.animations_enabled:
                    current_time = time.time()
                    elapsed = (current_time - transition_start) * 1000  # ms
                    transition_progress = min(1.0, elapsed / self.transition_speed)
                    
                    if transition_progress >= 1.0:
                        in_transition = False
                
                # Clear screen
                self.screen.fill(self.colors["background"])
                
                # Draw the appropriate screen
                if in_transition and self.animations_enabled:
                    # Crossfade between screens during transition
                    if last_mode == "live" and self.display_mode == "card":
                        # Render both screens to separate surfaces
                        live_surface = pygame.Surface((self.width, self.height))
                        live_surface.fill(self.colors["background"])
                        self._draw_live_screen(surface=live_surface)
                        
                        card_surface = pygame.Surface((self.width, self.height))
                        card_surface.fill(self.colors["background"])
                        self._draw_card_screen(surface=card_surface)
                        
                        # Apply crossfade
                        live_surface.set_alpha(int(255 * (1 - transition_progress)))
                        card_surface.set_alpha(int(255 * transition_progress))
                        
                        # Draw both surfaces
                        self.screen.blit(live_surface, (0, 0))
                        self.screen.blit(card_surface, (0, 0))
                    else:
                        # Similar for card -> live transition
                        card_surface = pygame.Surface((self.width, self.height))
                        card_surface.fill(self.colors["background"])
                        self._draw_card_screen(surface=card_surface)
                        
                        live_surface = pygame.Surface((self.width, self.height))
                        live_surface.fill(self.colors["background"])
                        self._draw_live_screen(surface=live_surface)
                        
                        # Apply crossfade
                        card_surface.set_alpha(int(255 * (1 - transition_progress)))
                        live_surface.set_alpha(int(255 * transition_progress))
                        
                        # Draw both surfaces
                        self.screen.blit(card_surface, (0, 0))
                        self.screen.blit(live_surface, (0, 0))
                else:
                    # Normal rendering without transition
                    if self.display_mode == "live":
                        self._draw_live_screen()
                    else:
                        self._draw_card_screen()
                
                # Draw touch debug overlay if needed
                if self.debug_mode:
                    # Draw touch indicator if touch is active
                    if self.show_touch_points and touch_current_pos is not None:
                        pygame.draw.circle(self.screen, RED, touch_current_pos, 15, 2)
                        
                        # Draw region outlines for debugging
                        for region_name, rect in self.touch_regions.items():
                            pygame.draw.rect(self.screen, (0, 255, 0), rect, 1)
                    
                    # Draw performance metrics
                    if self.show_fps:
                        self._draw_performance_metrics()
                
                # Update display
                pygame.display.flip()
                
                # Update performance metrics
                PERF_STATS['frame_count'] += 1
                
                # Cap framerate
                self.clock.tick(self.target_fps)
    
    def _draw_performance_metrics(self):
        """Draw performance metrics for debugging"""
        # Create performance stats surface with semi-transparent background
        metrics_surface = pygame.Surface((300, 100), pygame.SRCALPHA)
        metrics_surface.fill((0, 0, 0, 180))  # Semi-transparent black
        
        # Calculate FPS
        fps = self.clock.get_fps()
        frame_time = 1000.0 / fps if fps > 0 else 0
        
        # Calculate runtime
        runtime = time.time() - PERF_STATS['start_time']
        
        # Calculate memory usage (placeholder - would need psutil for actual usage)
        memory_usage = "N/A"
        if hasattr(self, 'show_memory_usage') and self.show_memory_usage:
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_usage = f"{memory_mb:.1f} MB"
            except:
                memory_usage = "N/A"
        
        # Render metrics text
        fps_text = self.fonts['small'].render(f"FPS: {fps:.1f} ({frame_time:.1f}ms/frame)", True, (255, 255, 0))
        runtime_text = self.fonts['small'].render(f"Runtime: {runtime:.1f}s", True, (255, 255, 0))
        frames_text = self.fonts['small'].render(f"Frames: {PERF_STATS['frame_count']}", True, (255, 255, 0))
        memory_text = self.fonts['small'].render(f"Memory: {memory_usage}", True, (255, 255, 0))
        
        # Position text on metrics surface
        metrics_surface.blit(fps_text, (10, 10))
        metrics_surface.blit(runtime_text, (10, 30))
        metrics_surface.blit(frames_text, (10, 50))
        metrics_surface.blit(memory_text, (10, 70))
        
        # Position metrics surface on screen
        self.screen.blit(metrics_surface, (10, 70))
            
    
    def _draw_live_screen(self, surface=None):
        """
        Draw the live view with transcription, significant events, and card preview
        
        Args:
            surface: Optional surface to draw on (for animation). If None, draws on self.screen
        """
        # Use provided surface or default to screen
        draw_surface = surface if surface is not None else self.screen
        
        # Draw modern clean header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(draw_surface, BLUE, header_rect)
        
        # Draw logo if available
        if hasattr(self, 'avatar') and self.avatar is not None:
            draw_surface.blit(self.avatar, (10, 5))
            header_x = 100  # Offset to account for logo
        else:
            header_x = 20
        
        # Draw title with bold font
        header_text = self.fonts['bold_large'].render("TCCC.ai FIELD ASSISTANT", True, WHITE)
        draw_surface.blit(header_text, (self.width//2 - header_text.get_width()//2, 15))
        
        # Draw status indicators - audio status, battery, time
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Render status icons and text
        audio_text = self.fonts['medium'].render("AUDIO ACTIVE", True, WHITE)
        battery_text = self.fonts['medium'].render("85%", True, WHITE)
        time_text = self.fonts['medium'].render(current_time, True, WHITE)
        session_text = self.fonts['medium'].render("SESSION: 00:12:45", True, WHITE)
        
        # Position status indicators
        if hasattr(self, 'icons'):
            # Place icons with text
            if "audio_active" in self.icons:
                draw_surface.blit(self.icons["audio_active"], (self.width - 480, 20))
                draw_surface.blit(audio_text, (self.width - 450, 20))
            
            if "battery" in self.icons:
                draw_surface.blit(self.icons["battery"], (self.width - 350, 20))
                draw_surface.blit(battery_text, (self.width - 320, 20))
            
            if "clock" in self.icons:
                draw_surface.blit(self.icons["clock"], (self.width - 250, 20))
        
        # Place time text
        draw_surface.blit(time_text, (self.width - time_text.get_width() - 20, 20))
        
        # Three-column layout as shown in mockup
        total_width = self.width
        column_1_width = int(total_width * self.column_widths["transcription"])  # 38% for transcription
        column_2_width = int(total_width * self.column_widths["events"])  # 34% for events
        column_3_width = total_width - column_1_width - column_2_width  # 28% for card preview
        
        # Column divider positions
        first_divider_x = column_1_width
        second_divider_x = first_divider_x + column_2_width
        
        # Draw vertical dividers between columns
        pygame.draw.line(draw_surface, GRAY, (first_divider_x, 60), (first_divider_x, self.height - 50), 2)
        pygame.draw.line(draw_surface, GRAY, (second_divider_x, 60), (second_divider_x, self.height - 50), 2)
        
        # Column titles
        transcription_title = self.fonts['bold_medium'].render("SPEECH TRANSCRIPTION", True, GOLD)
        events_title = self.fonts['bold_medium'].render("SIGNIFICANT EVENTS", True, GOLD)
        card_preview_title = self.fonts['bold_medium'].render("TCCC CARD PREVIEW", True, GOLD)
        
        # Position column titles
        draw_surface.blit(transcription_title, (first_divider_x//2 - transcription_title.get_width()//2, 70))
        draw_surface.blit(events_title, (first_divider_x + (column_2_width//2) - events_title.get_width()//2, 70))
        draw_surface.blit(card_preview_title, (second_divider_x + (column_3_width//2) - card_preview_title.get_width()//2, 70))
        
        # Column 1: Transcription - clean, readable text display
        with self.lock:
            transcription_items = self.transcription[-self.max_transcription_items:] if self.transcription else []
        
        column_width = first_divider_x - 30  # Leave margin for readability
        y_pos = 110
        
        # Draw text area background for better readability
        transcription_bg = pygame.Rect(10, 100, column_width, self.height - 160)
        pygame.draw.rect(draw_surface, (30, 30, 40), transcription_bg)
        pygame.draw.rect(draw_surface, GRAY, transcription_bg, 1)  # Border
        
        # Display transcriptions with highlighted speaker turns
        for i, item in enumerate(transcription_items):
            wrapped_lines = self._wrap_text(item, self.fonts['small'], column_width - 20)
            
            # Add speaker indicator for clarity
            if i % 2 == 0:  # Alternate for speaker turns
                speaker_indicator = "▶ "
            else:
                speaker_indicator = "◀ "
                
            for j, line in enumerate(wrapped_lines):
                # First line of each item gets speaker indicator
                if j == 0:
                    text = self.fonts['small'].render(speaker_indicator + line, True, WHITE)
                else:
                    text = self.fonts['small'].render("  " + line, True, WHITE)  # Indent continuation lines
                    
                draw_surface.blit(text, (20, y_pos))
                y_pos += 25
                
                # Stop if approaching footer
                if y_pos > self.height - 70:
                    break
            
            # Add gap between items
            y_pos += 10
            
            # Stop if approaching footer
            if y_pos > self.height - 70:
                break
        
        # Column 2: Significant Events - clear, hierarchical display
        with self.lock:
            events = self.significant_events[-self.max_event_items:] if self.significant_events else []
        
        column_width = column_2_width - 30  # Leave margin
        y_pos = 110
        
        # Event area background
        events_bg = pygame.Rect(first_divider_x + 10, 100, column_width - 20, self.height - 160)
        pygame.draw.rect(draw_surface, (30, 30, 40), events_bg)
        pygame.draw.rect(draw_surface, GRAY, events_bg, 1)  # Border
        
        for event in events:
            # Use visual indicators for event severity/type
            indicator = "■ "  # Default
            icon = None
            
            if isinstance(event, dict):
                time_str = event.get('time', datetime.now().strftime("%H:%M"))
                desc = event.get('description', '')
                
                # Visual indicator based on content keywords
                lower_desc = desc.lower()
                if any(word in lower_desc for word in ["critical", "severe", "bleeding", "airway", "breathing"]):
                    indicator = "⚠ "  # Warning indicator for critical events
                    icon = "warning" if hasattr(self, 'icons') and "warning" in self.icons else None
                elif any(word in lower_desc for word in ["applied", "administered", "tourniquet", "bandage"]):
                    indicator = "✓ "  # Check for treatment given
                    icon = "check" if hasattr(self, 'icons') and "check" in self.icons else None
                elif any(word in lower_desc for word in ["vitals", "heart rate", "blood pressure"]):
                    icon = "vitals" if hasattr(self, 'icons') and "vitals" in self.icons else None
                elif any(word in lower_desc for word in ["assessment"]):
                    icon = "note" if hasattr(self, 'icons') and "note" in self.icons else None
                elif any(word in lower_desc for word in ["identified", "found"]):
                    icon = "medical" if hasattr(self, 'icons') and "medical" in self.icons else None
                
                # Time with highlight color
                time_text = self.fonts['small'].render(time_str, True, GREEN)
                draw_surface.blit(time_text, (first_divider_x + 20, y_pos))
                
                # Icon if available
                if icon and hasattr(self, 'icons') and icon in self.icons:
                    draw_surface.blit(self.icons[icon], (first_divider_x + column_width - 40, y_pos))
                
                # Description with indicator
                wrapped_lines = self._wrap_text(desc, self.fonts['small'], column_width - 60)
                y_pos += 25
                
                for j, line in enumerate(wrapped_lines):
                    prefix = indicator if j == 0 else "  "  # Only first line gets indicator
                    text = self.fonts['small'].render(prefix + line, True, WHITE)
                    draw_surface.blit(text, (first_divider_x + 20, y_pos))
                    y_pos += 25
                    
                    if y_pos > self.height - 70:
                        break
            else:
                # Simple string event
                wrapped_lines = self._wrap_text(str(event), self.fonts['small'], column_width - 40)
                
                for j, line in enumerate(wrapped_lines):
                    prefix = indicator if j == 0 else "  "
                    text = self.fonts['small'].render(prefix + line, True, WHITE)
                    draw_surface.blit(text, (first_divider_x + 20, y_pos))
                    y_pos += 25
                    
                    if y_pos > self.height - 70:
                        break
            
            # Gap between events
            y_pos += 10
            
            if y_pos > self.height - 70:
                break
                
        # Column 3: TCCC Card Preview
        with self.lock:
            card_data = self.card_data.copy()
        
        column_width = column_3_width - 30  # Leave margin
        y_pos = 120
        
        # Card preview background
        card_bg = pygame.Rect(second_divider_x + 10, 100, column_width - 20, self.height - 160)
        pygame.draw.rect(draw_surface, (30, 30, 40), card_bg)
        pygame.draw.rect(draw_surface, GRAY, card_bg, 1)  # Border
        
        # Patient icon/placeholder and basic info
        patient_icon_rect = pygame.Rect(second_divider_x + 30, y_pos, 50, 60)
        pygame.draw.rect(draw_surface, (50, 50, 70), patient_icon_rect, 0, 5)
        
        # Patient icon (simple stick figure)
        pygame.draw.circle(draw_surface, WHITE, (second_divider_x + 55, y_pos + 20), 15, 1)  # Head
        pygame.draw.line(draw_surface, WHITE, (second_divider_x + 55, y_pos + 35), (second_divider_x + 55, y_pos + 70), 1)  # Body
        pygame.draw.line(draw_surface, WHITE, (second_divider_x + 55, y_pos + 45), (second_divider_x + 35, y_pos + 55), 1)  # Left arm
        pygame.draw.line(draw_surface, WHITE, (second_divider_x + 55, y_pos + 45), (second_divider_x + 75, y_pos + 55), 1)  # Right arm
        pygame.draw.line(draw_surface, WHITE, (second_divider_x + 55, y_pos + 70), (second_divider_x + 40, y_pos + 85), 1)  # Left leg
        pygame.draw.line(draw_surface, WHITE, (second_divider_x + 55, y_pos + 70), (second_divider_x + 70, y_pos + 85), 1)  # Right leg
        
        # Patient info
        name = card_data.get("name", "UNKNOWN")
        rank = card_data.get("rank", "")
        id_num = "#JS4321"  # Example ID
        
        # Draw patient info
        name_text = self.fonts['bold_small'].render(f"{rank} {name}", True, WHITE)
        id_text = self.fonts['small'].render(f"ID: {id_num}", True, GRAY)
        
        draw_surface.blit(name_text, (second_divider_x + 90, y_pos + 5))
        draw_surface.blit(id_text, (second_divider_x + 90, y_pos + 30))
        
        # Primary injuries section
        y_pos += 100
        injury_title = self.fonts['bold_small'].render("PRIMARY INJURIES:", True, WHITE)
        draw_surface.blit(injury_title, (second_divider_x + 30, y_pos))
        
        # Injuries from card data
        injuries = card_data.get("injuries", "No injuries recorded")
        injury_parts = injuries.split(",")[:2]  # Show only first two injuries
        
        y_pos += 25
        for injury in injury_parts:
            injury_text = self.fonts['small'].render(f"• {injury.strip()}", True, GRAY)
            draw_surface.blit(injury_text, (second_divider_x + 50, y_pos))
            y_pos += 25
            
        # Interventions section
        y_pos += 15
        intervention_title = self.fonts['bold_small'].render("INTERVENTIONS:", True, WHITE)
        draw_surface.blit(intervention_title, (second_divider_x + 30, y_pos))
        
        # Treatment from card data
        treatment = card_data.get("treatment_given", "No treatment recorded")
        treatment_parts = treatment.split(",")[:2]  # Show only first two treatments
        
        y_pos += 25
        for part in treatment_parts:
            treatment_text = self.fonts['small'].render(f"• {part.strip()}", True, GRAY)
            draw_surface.blit(treatment_text, (second_divider_x + 50, y_pos))
            y_pos += 25
            
        # Vitals section
        y_pos += 15
        vitals_title = self.fonts['bold_small'].render("VITALS:", True, WHITE)
        draw_surface.blit(vitals_title, (second_divider_x + 30, y_pos))
        
        # Vitals from card data
        vitals = card_data.get("vital_signs", "No vitals recorded")
        
        # Format vitals nicely
        vitals_parts = vitals.split(",")
        vitals_formatted = []
        
        # Process vital signs for display with trends
        if len(vitals_parts) >= 2:
            # First line: HR and BP
            hr_part = next((p for p in vitals_parts if "HR" in p or "heart rate" in p.lower()), "")
            bp_part = next((p for p in vitals_parts if "BP" in p or "blood pressure" in p.lower()), "")
            
            if hr_part:
                try:
                    # Extract HR value and add trend indicator
                    hr_val = int(''.join(c for c in hr_part if c.isdigit()))
                    trend = " ↑" if hr_val > 100 else " ↓" if hr_val < 60 else " →"
                    hr_text = hr_part.strip() + trend
                    vitals_formatted.append(hr_text)
                except:
                    vitals_formatted.append(hr_part.strip())
                    
            if bp_part:
                vitals_formatted.append(bp_part.strip() + " ↓")  # Assuming low BP for example
            
            # Second line: RR and SpO2
            rr_part = next((p for p in vitals_parts if "RR" in p or "respiratory" in p.lower()), "")
            spo2_part = next((p for p in vitals_parts if "O2" in p or "SpO2" in p.lower() or "sat" in p.lower()), "")
            
            if rr_part:
                vitals_formatted.append(rr_part.strip() + " ↑")  # Assuming elevated RR
                
            if spo2_part:
                vitals_formatted.append(spo2_part.strip() + " →")  # Assuming normal SpO2
        else:
            # Just split the string if not enough parts to identify
            vitals_formatted = [p.strip() for p in vitals_parts]
        
        # Display vital signs
        y_pos += 25
        for i, vital in enumerate(vitals_formatted[:4]):  # Show up to 4 vital signs
            color = RED if "↑" in vital or "↓" in vital else WHITE
            vital_text = self.fonts['small'].render(vital, True, color)
            
            # Arrange in two columns if possible
            if i % 2 == 0:
                draw_surface.blit(vital_text, (second_divider_x + 50, y_pos))
            else:
                draw_surface.blit(vital_text, (second_divider_x + column_width//2, y_pos))
                y_pos += 25  # Move to next line after second column
                
        # Evacuation section
        y_pos += 40
        evac_title = self.fonts['bold_small'].render("EVACUATION PRIORITY:", True, WHITE)
        draw_surface.blit(evac_title, (second_divider_x + 30, y_pos))
        
        # Evacuation priority from card data
        evac = card_data.get("evacuation_priority", "").upper()
        
        # Color code based on priority
        if "URGENT" in evac:
            evac_color = RED
        elif "PRIORITY" in evac:
            evac_color = GOLD
        elif "ROUTINE" in evac:
            evac_color = GREEN
        else:
            evac_color = WHITE
            evac = "UNKNOWN"
            
        y_pos += 25
        evac_text = self.fonts['bold_small'].render(evac, True, evac_color)
        draw_surface.blit(evac_text, (second_divider_x + 50, y_pos))
            
        # View Full Card button
        button_rect = pygame.Rect(second_divider_x + column_width - 170, y_pos + 40, 150, 30)
        pygame.draw.rect(draw_surface, (50, 50, 70), button_rect, 0, 5)
        pygame.draw.rect(draw_surface, GRAY, button_rect, 1, 5)  # Border
        
        button_text = self.fonts['bold_small'].render("VIEW FULL CARD", True, WHITE)
        draw_surface.blit(button_text, (button_rect.centerx - button_text.get_width()//2, 
                                       button_rect.centery - button_text.get_height()//2))
        
        # Footer with mode buttons and status
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(draw_surface, (50, 50, 60), footer_rect)  # Darker footer
        
        # Mode buttons (monitoring, documentation, reference)
        mode_buttons = [
            {"text": "MONITORING", "x": 40, "width": 120, "active": True},
            {"text": "DOCUMENTATION", "x": 180, "width": 150, "active": False},
            {"text": "REFERENCE", "x": 350, "width": 120, "active": False}
        ]
        
        for button in mode_buttons:
            button_rect = pygame.Rect(button["x"], self.height - 40, button["width"], 30)
            # Active button gets brighter background
            bg_color = (31, 75, 121) if button["active"] else (29, 58, 93)
            pygame.draw.rect(draw_surface, bg_color, button_rect, 0, 5)
            pygame.draw.rect(draw_surface, GRAY, button_rect, 1, 5)  # Border
            
            button_text = self.fonts['bold_small'].render(button["text"], True, WHITE)
            draw_surface.blit(button_text, (button_rect.centerx - button_text.get_width()//2, 
                                           button_rect.centery - button_text.get_height()//2))
        
        # Action buttons
        action_buttons = [
            {"icon": "keyboard", "text": "KEYBOARD", "x": 650, "width": 120},
            {"icon": "save", "text": "SAVE CARD", "x": 800, "width": 120},
            {"icon": "emergency", "text": "EMERGENCY ACTION", "x": 1250, "width": 270, "color": (177, 30, 30)}
        ]
        
        for button in action_buttons:
            button_rect = pygame.Rect(button["x"], self.height - 40, button["width"], 30)
            bg_color = button.get("color", (29, 58, 93))
            pygame.draw.rect(draw_surface, bg_color, button_rect, 0, 5)
            pygame.draw.rect(draw_surface, GRAY, button_rect, 1, 5)  # Border
            
            text = button["text"]
            if "icon" in button and hasattr(self, 'icons') and button["icon"] in self.icons:
                # Draw icon + text
                icon_x = button_rect.x + 10
                text_x = icon_x + 25
                draw_surface.blit(self.icons[button["icon"]], (icon_x, button_rect.centery - 10))
                text_surface = self.fonts['bold_small'].render(text, True, WHITE)
                draw_surface.blit(text_surface, (text_x, button_rect.centery - text_surface.get_height()//2))
            else:
                # Text only
                button_text = self.fonts['bold_small'].render(text, True, WHITE)
                draw_surface.blit(button_text, (button_rect.centerx - button_text.get_width()//2, 
                                               button_rect.centery - button_text.get_height()//2))
    
    def _draw_card_screen(self, surface=None):
        """
        Draw the TCCC Casualty Card with modern clean design
        
        Args:
            surface: Optional surface to draw on (for animation). If None, draws on self.screen
        """
        # Use provided surface or default to screen
        draw_surface = surface if surface is not None else self.screen
        
        # Draw card header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(draw_surface, RED, header_rect)
        
        # Draw title with modern typography 
        title_text = self.fonts['bold_large'].render("TCCC CASUALTY CARD", True, WHITE)
        subtitle_text = self.fonts['small'].render("DD FORM 1380 - TACTICAL COMBAT CASUALTY CARE", True, WHITE)
        
        # Center align text
        draw_surface.blit(title_text, (self.width//2 - title_text.get_width()//2, 10))
        draw_surface.blit(subtitle_text, (self.width//2 - subtitle_text.get_width()//2, 40))
        
        # Card data
        with self.lock:
            card_data = self.card_data.copy()
        
        # Main card background for better readability
        card_bg = pygame.Rect(20, 70, self.width - 40, self.height - 130)
        pygame.draw.rect(draw_surface, (30, 30, 40), card_bg)
        pygame.draw.rect(draw_surface, (100, 100, 110), card_bg, 2, border_radius=5)  # Border with rounded corners
        
        # Create three distinct sections
        section_width = (self.width - 60) // 3
        
        # Section 1: Patient identity (left)
        identity_title = self.fonts['bold_medium'].render("PATIENT IDENTITY", True, GOLD)
        draw_surface.blit(identity_title, (40, 85))
        
        # Identity section background
        identity_bg = pygame.Rect(30, 110, section_width, self.height - 170)
        pygame.draw.rect(draw_surface, (40, 40, 50), identity_bg, border_radius=5)
        
        # Identity fields
        identity_fields = [
            ("Name:", card_data.get("name", "UNKNOWN")),
            ("Rank:", card_data.get("rank", "")),
            ("Unit:", card_data.get("unit", "")),
            ("Service ID:", card_data.get("service_id", "")),
            ("Date:", card_data.get("date", datetime.now().strftime("%Y-%m-%d"))),
            ("Time:", card_data.get("time", datetime.now().strftime("%H:%M")))
        ]
        
        # Draw identity fields
        y_pos = 125
        for field_name, field_value in identity_fields:
            # Field name
            name_text = self.fonts['bold_small'].render(field_name, True, GOLD)
            draw_surface.blit(name_text, (40, y_pos))
            
            # Field value with highlight for certain fields
            if field_name in ["Name:", "Rank:"]:
                # Use larger font for key fields
                value_text = self.fonts['medium'].render(field_value, True, WHITE)
            else:
                value_text = self.fonts['small'].render(field_value, True, WHITE)
                
            draw_surface.blit(value_text, (40, y_pos + 25))
            
            # Separator line
            pygame.draw.line(draw_surface, (60, 60, 70), 
                          (40, y_pos + 50), 
                          (section_width + 20, y_pos + 50), 1)
            
            y_pos += 60
        
        # Section 2: Injury & Treatment (middle)
        injury_title = self.fonts['bold_medium'].render("INJURY & TREATMENT", True, GOLD)
        draw_surface.blit(injury_title, (section_width + 50, 85))
        
        # Injury section background
        injury_bg = pygame.Rect(section_width + 40, 110, section_width, self.height - 170)
        pygame.draw.rect(draw_surface, (40, 40, 50), injury_bg, border_radius=5)
        
        # Injury fields
        injury_fields = [
            ("Mechanism:", card_data.get("mechanism_of_injury", "")),
            ("Injuries:", card_data.get("injuries", "")),
            ("Treatment:", card_data.get("treatment_given", "")),
            ("Medications:", card_data.get("medications", ""))
        ]
        
        # Draw injury fields - allow more space for text
        y_pos = 125
        for field_name, field_value in injury_fields:
            # Field name
            name_text = self.fonts['bold_small'].render(field_name, True, GOLD)
            draw_surface.blit(name_text, (section_width + 50, y_pos))
            
            # Multi-line field values
            wrapped_lines = self._wrap_text(field_value, self.fonts['small'], section_width - 20)
            line_y = y_pos + 25
            
            # Show up to 4 lines per field
            for i, line in enumerate(wrapped_lines[:4]):
                value_text = self.fonts['small'].render(line, True, WHITE)
                draw_surface.blit(value_text, (section_width + 50, line_y))
                line_y += 22
            
            # Separator line
            pygame.draw.line(draw_surface, (60, 60, 70), 
                          (section_width + 50, line_y + 5), 
                          (section_width * 2 + 30, line_y + 5), 1)
            
            y_pos = line_y + 15
        
        # Section 3: Vital Signs & Evacuation (right)
        vitals_title = self.fonts['bold_medium'].render("VITALS & EVACUATION", True, GOLD)
        draw_surface.blit(vitals_title, (section_width * 2 + 50, 85))
        
        # Vitals section background
        vitals_bg = pygame.Rect(section_width * 2 + 40, 110, section_width, self.height - 170)
        pygame.draw.rect(draw_surface, (40, 40, 50), vitals_bg, border_radius=5)
        
        # Vital signs with visual indicators
        y_pos = 125
        vitals_title = self.fonts['bold_small'].render("Vital Signs:", True, GOLD)
        draw_surface.blit(vitals_title, (section_width * 2 + 50, y_pos))
        
        # Parse and display vital signs with visual indicators
        vitals_str = card_data.get("vital_signs", "")
        vitals_parts = vitals_str.split(',')
        
        y_pos += 30
        for part in vitals_parts:
            part = part.strip()
            if part:
                # Highlight abnormal values in red
                if ('HR' in part and any(str(n) in part for n in range(120, 220))):
                    color = RED  # Elevated heart rate
                elif ('BP' in part and any(n in part for n in ['<90', '<100'])):
                    color = RED  # Low blood pressure
                elif ('RR' in part and any(str(n) in part for n in range(30, 100))):
                    color = RED  # Elevated respiratory rate
                else:
                    color = WHITE
                
                value_text = self.fonts['small'].render(part, True, color)
                draw_surface.blit(value_text, (section_width * 2 + 60, y_pos))
                y_pos += 25
        
        # Evacuation priority with color-coded visual
        y_pos += 20
        evac_title = self.fonts['bold_small'].render("Evacuation Priority:", True, GOLD)
        draw_surface.blit(evac_title, (section_width * 2 + 50, y_pos))
        
        # Color code based on evacuation priority
        evac_priority = card_data.get("evacuation_priority", "").lower()
        if "urgent" in evac_priority:
            priority_color = RED
            priority_text = "URGENT"
        elif "priority" in evac_priority:
            priority_color = (255, 165, 0)  # Orange
            priority_text = "PRIORITY"
        elif "routine" in evac_priority:
            priority_color = GREEN
            priority_text = "ROUTINE"
        else:
            priority_color = WHITE
            priority_text = evac_priority.upper()
        
        # Priority display with visual emphasis
        priority_box = pygame.Rect(section_width * 2 + 60, y_pos + 30, section_width - 80, 40)
        pygame.draw.rect(draw_surface, priority_color, priority_box, border_radius=5)
        pygame.draw.rect(draw_surface, WHITE, priority_box, 2, border_radius=5)
        
        # Priority text
        priority_text = self.fonts['bold_medium'].render(priority_text, True, BLACK if priority_color == GREEN else WHITE)
        draw_surface.blit(priority_text, 
                       (priority_box.centerx - priority_text.get_width()//2, 
                        priority_box.centery - priority_text.get_height()//2))
        
        # Footer with mode buttons and back button
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(draw_surface, (50, 50, 60), footer_rect)
        
        # Mode buttons (documentation, reference)
        mode_buttons = [
            {"text": "MONITORING", "x": 40, "width": 120, "active": False},
            {"text": "DOCUMENTATION", "x": 180, "width": 150, "active": True},
            {"text": "REFERENCE", "x": 350, "width": 120, "active": False}
        ]
        
        for button in mode_buttons:
            button_rect = pygame.Rect(button["x"], self.height - 40, button["width"], 30)
            # Active button gets brighter background
            bg_color = (31, 75, 121) if button["active"] else (29, 58, 93)
            pygame.draw.rect(draw_surface, bg_color, button_rect, 0, 5)
            pygame.draw.rect(draw_surface, GRAY, button_rect, 1, 5)  # Border
            
            button_text = self.fonts['bold_small'].render(button["text"], True, WHITE)
            draw_surface.blit(button_text, (button_rect.centerx - button_text.get_width()//2, 
                                          button_rect.centery - button_text.get_height()//2))
        
        # Action buttons
        action_buttons = [
            {"icon": "save", "text": "SAVE CARD", "x": 800, "width": 120},
            {"icon": "keyboard", "text": "PRINT", "x": 950, "width": 120}
        ]
        
        for button in action_buttons:
            button_rect = pygame.Rect(button["x"], self.height - 40, button["width"], 30)
            bg_color = button.get("color", (29, 58, 93))
            pygame.draw.rect(draw_surface, bg_color, button_rect, 0, 5)
            pygame.draw.rect(draw_surface, GRAY, button_rect, 1, 5)  # Border
            
            text = button["text"]
            if "icon" in button and hasattr(self, 'icons') and button["icon"] in self.icons:
                # Draw icon + text
                icon_x = button_rect.x + 10
                text_x = icon_x + 25
                draw_surface.blit(self.icons[button["icon"]], (icon_x, button_rect.centery - 10))
                text_surface = self.fonts['bold_small'].render(text, True, WHITE)
                draw_surface.blit(text_surface, (text_x, button_rect.centery - text_surface.get_height()//2))
            else:
                # Text only
                button_text = self.fonts['bold_small'].render(text, True, WHITE)
                draw_surface.blit(button_text, (button_rect.centerx - button_text.get_width()//2, 
                                              button_rect.centery - button_text.get_height()//2))
        
        # Return to live view button
        button_rect = pygame.Rect(self.width//2 - 130, self.height - 40, 260, 30)
        pygame.draw.rect(draw_surface, (70, 70, 80), button_rect, border_radius=15)
        pygame.draw.rect(draw_surface, (100, 100, 110), button_rect, 1, border_radius=15)
        
        back_text = self.fonts['medium'].render("Return to Live View (T)", True, WHITE)
        draw_surface.blit(back_text, (self.width//2 - back_text.get_width()//2, self.height - 35))
    
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