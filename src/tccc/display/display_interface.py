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
        """Draw the live view with transcription and significant events (single screen clean UI)"""
        # Draw modern clean header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(self.screen, BLUE, header_rect)
        
        # Draw title with bold font
        header_text = self.fonts['bold_large'].render("TCCC.ai FIELD ASSISTANT", True, WHITE)
        self.screen.blit(header_text, (self.width//2 - header_text.get_width()//2, 15))
        
        # Draw current time
        current_time = datetime.now().strftime("%H:%M:%S")
        time_text = self.fonts['medium'].render(current_time, True, WHITE)
        self.screen.blit(time_text, (self.width - time_text.get_width() - 20, 20))
        
        # Two-column layout for clarity
        total_width = self.width
        column_1_width = int(total_width * 0.6)  # 60% for transcription
        column_2_width = total_width - column_1_width  # 40% for events
        
        # Column divider position
        divider_x = column_1_width
        
        # Draw vertical divider between columns
        pygame.draw.line(self.screen, GRAY, (divider_x, 60), (divider_x, self.height - 50), 2)
        
        # Column titles - cleaner look
        transcription_title = self.fonts['bold_medium'].render("LIVE TRANSCRIPTION", True, GOLD)
        events_title = self.fonts['bold_medium'].render("CRITICAL EVENTS", True, GOLD)
        
        # Position column titles
        self.screen.blit(transcription_title, (divider_x//2 - transcription_title.get_width()//2, 70))
        self.screen.blit(events_title, (divider_x + (column_2_width//2) - events_title.get_width()//2, 70))
        
        # Column 1: Transcription - clean, readable text display
        with self.lock:
            transcription_items = self.transcription[-self.max_transcription_items:] if self.transcription else []
        
        column_width = divider_x - 30  # Leave margin for readability
        y_pos = 110
        
        # Draw text area background for better readability
        transcription_bg = pygame.Rect(10, 100, column_width, self.height - 160)
        pygame.draw.rect(self.screen, (30, 30, 40), transcription_bg)
        pygame.draw.rect(self.screen, GRAY, transcription_bg, 1)  # Border
        
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
                    
                self.screen.blit(text, (20, y_pos))
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
        events_bg = pygame.Rect(divider_x + 10, 100, column_width - 20, self.height - 160)
        pygame.draw.rect(self.screen, (30, 30, 40), events_bg)
        pygame.draw.rect(self.screen, GRAY, events_bg, 1)  # Border
        
        for event in events:
            # Use visual indicators for event severity/type
            indicator = "■ "  # Default
            
            if isinstance(event, dict):
                time_str = event.get('time', datetime.now().strftime("%H:%M"))
                desc = event.get('description', '')
                
                # Visual indicator based on content keywords
                lower_desc = desc.lower()
                if any(word in lower_desc for word in ["critical", "severe", "bleeding", "airway", "breathing"]):
                    indicator = "⚠ "  # Warning indicator for critical events
                elif any(word in lower_desc for word in ["applied", "administered", "tourniquet", "bandage"]):
                    indicator = "✓ "  # Check for treatment given
                
                # Time with highlight color
                time_text = self.fonts['small'].render(time_str, True, GREEN)
                self.screen.blit(time_text, (divider_x + 20, y_pos))
                
                # Description with indicator
                wrapped_lines = self._wrap_text(desc, self.fonts['small'], column_width - 40)
                y_pos += 25
                
                for j, line in enumerate(wrapped_lines):
                    prefix = indicator if j == 0 else "  "  # Only first line gets indicator
                    text = self.fonts['small'].render(prefix + line, True, WHITE)
                    self.screen.blit(text, (divider_x + 20, y_pos))
                    y_pos += 25
                    
                    if y_pos > self.height - 70:
                        break
            else:
                # Simple string event
                wrapped_lines = self._wrap_text(str(event), self.fonts['small'], column_width - 40)
                
                for j, line in enumerate(wrapped_lines):
                    prefix = indicator if j == 0 else "  "
                    text = self.fonts['small'].render(prefix + line, True, WHITE)
                    self.screen.blit(text, (divider_x + 20, y_pos))
                    y_pos += 25
                    
                    if y_pos > self.height - 70:
                        break
            
            # Gap between events
            y_pos += 10
            
            if y_pos > self.height - 70:
                break
        
        # Modern footer with status indicators and instructions
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(self.screen, (50, 50, 60), footer_rect)  # Darker footer
        
        # Status indicators
        status_text = self.fonts['bold_small'].render("● RECORDING", True, GREEN)
        self.screen.blit(status_text, (20, self.height - 35))
        
        # Center button prompt
        button_rect = pygame.Rect(self.width//2 - 130, self.height - 40, 260, 30)
        pygame.draw.rect(self.screen, (70, 70, 80), button_rect, border_radius=15)
        pygame.draw.rect(self.screen, (100, 100, 110), button_rect, 1, border_radius=15)  # Button border
        
        instruction_text = self.fonts['medium'].render("Press 'T' for TCCC Card", True, WHITE)
        self.screen.blit(instruction_text, (self.width//2 - instruction_text.get_width()//2, self.height - 35))
    
    def _draw_card_screen(self):
        """Draw the TCCC Casualty Card with modern clean design"""
        # Draw card header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(self.screen, RED, header_rect)
        
        # Draw title with modern typography 
        title_text = self.fonts['bold_large'].render("TCCC CASUALTY CARD", True, WHITE)
        subtitle_text = self.fonts['small'].render("DD FORM 1380 - TACTICAL COMBAT CASUALTY CARE", True, WHITE)
        
        # Center align text
        self.screen.blit(title_text, (self.width//2 - title_text.get_width()//2, 10))
        self.screen.blit(subtitle_text, (self.width//2 - subtitle_text.get_width()//2, 40))
        
        # Card data
        with self.lock:
            card_data = self.card_data.copy()
        
        # Main card background for better readability
        card_bg = pygame.Rect(20, 70, self.width - 40, self.height - 130)
        pygame.draw.rect(self.screen, (30, 30, 40), card_bg)
        pygame.draw.rect(self.screen, (100, 100, 110), card_bg, 2, border_radius=5)  # Border with rounded corners
        
        # Create three distinct sections
        section_width = (self.width - 60) // 3
        
        # Section 1: Patient identity (left)
        identity_title = self.fonts['bold_medium'].render("PATIENT IDENTITY", True, GOLD)
        self.screen.blit(identity_title, (40, 85))
        
        # Identity section background
        identity_bg = pygame.Rect(30, 110, section_width, self.height - 170)
        pygame.draw.rect(self.screen, (40, 40, 50), identity_bg, border_radius=5)
        
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
            self.screen.blit(name_text, (40, y_pos))
            
            # Field value with highlight for certain fields
            if field_name in ["Name:", "Rank:"]:
                # Use larger font for key fields
                value_text = self.fonts['medium'].render(field_value, True, WHITE)
            else:
                value_text = self.fonts['small'].render(field_value, True, WHITE)
                
            self.screen.blit(value_text, (40, y_pos + 25))
            
            # Separator line
            pygame.draw.line(self.screen, (60, 60, 70), 
                          (40, y_pos + 50), 
                          (section_width + 20, y_pos + 50), 1)
            
            y_pos += 60
        
        # Section 2: Injury & Treatment (middle)
        injury_title = self.fonts['bold_medium'].render("INJURY & TREATMENT", True, GOLD)
        self.screen.blit(injury_title, (section_width + 50, 85))
        
        # Injury section background
        injury_bg = pygame.Rect(section_width + 40, 110, section_width, self.height - 170)
        pygame.draw.rect(self.screen, (40, 40, 50), injury_bg, border_radius=5)
        
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
            self.screen.blit(name_text, (section_width + 50, y_pos))
            
            # Multi-line field values
            wrapped_lines = self._wrap_text(field_value, self.fonts['small'], section_width - 20)
            line_y = y_pos + 25
            
            # Show up to 4 lines per field
            for i, line in enumerate(wrapped_lines[:4]):
                value_text = self.fonts['small'].render(line, True, WHITE)
                self.screen.blit(value_text, (section_width + 50, line_y))
                line_y += 22
            
            # Separator line
            pygame.draw.line(self.screen, (60, 60, 70), 
                          (section_width + 50, line_y + 5), 
                          (section_width * 2 + 30, line_y + 5), 1)
            
            y_pos = line_y + 15
        
        # Section 3: Vital Signs & Evacuation (right)
        vitals_title = self.fonts['bold_medium'].render("VITALS & EVACUATION", True, GOLD)
        self.screen.blit(vitals_title, (section_width * 2 + 50, 85))
        
        # Vitals section background
        vitals_bg = pygame.Rect(section_width * 2 + 40, 110, section_width, self.height - 170)
        pygame.draw.rect(self.screen, (40, 40, 50), vitals_bg, border_radius=5)
        
        # Vital signs with visual indicators
        y_pos = 125
        vitals_title = self.fonts['bold_small'].render("Vital Signs:", True, GOLD)
        self.screen.blit(vitals_title, (section_width * 2 + 50, y_pos))
        
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
                self.screen.blit(value_text, (section_width * 2 + 60, y_pos))
                y_pos += 25
        
        # Evacuation priority with color-coded visual
        y_pos += 20
        evac_title = self.fonts['bold_small'].render("Evacuation Priority:", True, GOLD)
        self.screen.blit(evac_title, (section_width * 2 + 50, y_pos))
        
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
        pygame.draw.rect(self.screen, priority_color, priority_box, border_radius=5)
        pygame.draw.rect(self.screen, WHITE, priority_box, 2, border_radius=5)
        
        # Priority text
        priority_text = self.fonts['bold_medium'].render(priority_text, True, BLACK if priority_color == GREEN else WHITE)
        self.screen.blit(priority_text, 
                       (priority_box.centerx - priority_text.get_width()//2, 
                        priority_box.centery - priority_text.get_height()//2))
        
        # Footer with button to return to live view
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(self.screen, (50, 50, 60), footer_rect)
        
        # Back button
        button_rect = pygame.Rect(self.width//2 - 130, self.height - 40, 260, 30)
        pygame.draw.rect(self.screen, (70, 70, 80), button_rect, border_radius=15)
        pygame.draw.rect(self.screen, (100, 100, 110), button_rect, 1, border_radius=15)
        
        back_text = self.fonts['medium'].render("Return to Live View (T)", True, WHITE)
        self.screen.blit(back_text, (self.width//2 - back_text.get_width()//2, self.height - 35))
    
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