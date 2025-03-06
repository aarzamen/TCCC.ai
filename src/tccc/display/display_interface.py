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
from typing import Dict, List, Optional

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
    
    def __init__(self, width=1280, height=720, fullscreen=False):
        """
        Initialize the display interface for standard HDMI output
        
        Args:
            width: Screen width (default 1280 for standard HD resolution)
            height: Screen height (default 720 for standard HD resolution)
            fullscreen: Whether to display in fullscreen mode
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
        
        # Performance settings
        self.target_fps = 30
        
        # Thread lock for thread-safe updates
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize the display with standard HDMI settings"""
        if self.initialized:
            logger.info("Display already initialized")
            return True
            
        try:
            # Initialize pygame
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
                
            # Create the screen
            logger.info(f"Creating display with resolution {self.width}x{self.height}")
            self.screen = pygame.display.set_mode(
                (self.width, self.height), 
                display_flags
            )
            
            # Load fonts
            self._load_fonts()
            
            # Mark as initialized
            self.initialized = True
            
            logger.info(f"Display interface initialized successfully: {self.width}x{self.height}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize display: {e}")
            return False
    
    def _load_fonts(self):
        """Load fonts with proper fallbacks for different platforms"""
        try:
            # Default font or use system font
            self.fonts = {
                'small': pygame.font.SysFont(None, FONT_SMALL),
                'medium': pygame.font.SysFont(None, FONT_MEDIUM),
                'large': pygame.font.SysFont(None, FONT_LARGE),
                'bold_small': pygame.font.SysFont(None, FONT_SMALL, bold=True),
                'bold_medium': pygame.font.SysFont(None, FONT_MEDIUM, bold=True),
                'bold_large': pygame.font.SysFont(None, FONT_LARGE, bold=True)
            }
            logger.info("Loaded system fonts")
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
        """
        try:
            # Initialize frame timing
            global PERF_STATS
            PERF_STATS['start_time'] = time.time()
            PERF_STATS['frame_count'] = 0
            
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
                                self.display_mode = "card" if self.display_mode == "live" else "live"
                
                # Clear screen
                self.screen.fill(self.colors["background"])
                
                # Draw the appropriate screen
                if self.display_mode == "live":
                    self._draw_live_screen()
                else:
                    self._draw_card_screen()
                
                # Update display
                pygame.display.flip()
                
                # Update performance metrics
                PERF_STATS['frame_count'] += 1
                
                # Cap framerate
                self.clock.tick(self.target_fps)
                
        except Exception as e:
            logger.error(f"Error in display loop: {e}")
        finally:
            # Ensure pygame is properly shut down on exit
            try:
                pygame.quit()
            except Exception:
                pass
            
    
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