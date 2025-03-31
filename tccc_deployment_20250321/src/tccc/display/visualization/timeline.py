#!/usr/bin/env python3
"""
TCCC.ai Timeline Visualization
----------------------------
Provides visualization for medical events over time including
treatments, assessments, and significant findings.
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Timeline")

try:
    import pygame
    from pygame.locals import *
except ImportError:
    logger.error("pygame not installed. Install it with: pip install pygame")
    raise

# Import display configuration manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from display_config import get_display_config


class TimelineEvent:
    """Represents a single event on the timeline."""
    
    def __init__(self, 
                timestamp: Optional[datetime] = None,
                event_type: str = "info",
                title: str = "",
                description: str = "",
                icon: Optional[str] = None,
                source: Optional[str] = None,
                color: Optional[Tuple[int, int, int]] = None):
        """
        Initialize a timeline event.
        
        Args:
            timestamp: Time when the event occurred (default: now)
            event_type: Type of event (info, warning, critical, treatment, etc.)
            title: Short title for the event
            description: Detailed description of the event
            icon: Optional icon identifier for visual representation
            source: Source of the event (e.g., "audio", "sensor", "manual")
            color: Optional custom color for the event
        """
        self.timestamp = timestamp or datetime.now()
        self.event_type = event_type
        self.title = title
        self.description = description
        self.icon = icon
        self.source = source
        self.custom_color = color
        
        # Time since event occurred (set on creation for initial sorting)
        self.time_ago = (datetime.now() - self.timestamp).total_seconds()
        
    def update_time_ago(self):
        """Update the time ago field based on current time."""
        self.time_ago = (datetime.now() - self.timestamp).total_seconds()
        
    def format_time_ago(self) -> str:
        """Format time ago in a human-readable format."""
        seconds = int(self.time_ago)
        
        if seconds < 60:
            return f"{seconds}s ago"
        elif seconds < 3600:
            return f"{seconds // 60}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"
        else:
            return f"{seconds // 86400}d {(seconds % 86400) // 3600}h ago"
            
    def get_color(self, theme_colors: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Get the color for this event based on type and theme."""
        if self.custom_color:
            return self.custom_color
            
        # Default colors based on event type
        type_colors = {
            "info": theme_colors["text"],
            "warning": (255, 165, 0),  # Orange
            "critical": theme_colors["alert"],
            "treatment": theme_colors["success"],
            "assessment": theme_colors["highlight"],
            "vital": (0, 191, 255),  # Deep sky blue
            "medication": (138, 43, 226),  # BlueViolet
            "transport": (65, 105, 225),  # RoyalBlue
        }
        
        return type_colors.get(self.event_type, theme_colors["text"])
        
    def get_icon_name(self) -> str:
        """Get the appropriate icon name based on event type."""
        if self.icon:
            return self.icon
            
        # Default icons based on event type
        type_icons = {
            "info": "info",
            "warning": "warning",
            "critical": "critical",
            "treatment": "treatment",
            "assessment": "assessment",
            "vital": "vitals",
            "medication": "medication",
            "transport": "transport",
        }
        
        return type_icons.get(self.event_type, "info")
        

class TimelineVisualization:
    """
    Timeline visualization for medical events.
    Shows chronological view of significant events.
    """
    
    # Event type definitions for categorization
    EVENT_TYPES = {
        "info": {
            "name": "Information",
            "icon": "info",
            "priority": 0
        },
        "warning": {
            "name": "Warning",
            "icon": "warning",
            "priority": 2
        },
        "critical": {
            "name": "Critical",
            "icon": "critical",
            "priority": 3
        },
        "treatment": {
            "name": "Treatment",
            "icon": "treatment",
            "priority": 1
        },
        "assessment": {
            "name": "Assessment",
            "icon": "assessment",
            "priority": 0
        },
        "vital": {
            "name": "Vital Signs",
            "icon": "vitals",
            "priority": 1
        },
        "medication": {
            "name": "Medication",
            "icon": "medication",
            "priority": 1
        },
        "transport": {
            "name": "Transport",
            "icon": "transport",
            "priority": 2
        }
    }
    
    def __init__(self, 
                surface: Optional[pygame.Surface] = None,
                rect: Optional[pygame.Rect] = None,
                config: Optional[Dict] = None,
                max_events: int = 100):
        """
        Initialize the timeline visualization.
        
        Args:
            surface: Pygame surface to draw on
            rect: Rectangle defining the area to draw in
            config: Optional configuration dictionary
            max_events: Maximum number of events to store
        """
        self.surface = surface
        self.rect = rect
        self.config = config or get_display_config().get_config()
        self.max_events = max_events
        
        # Visual settings
        self.theme_colors = self.config["colors"]
        self.fonts = self._load_fonts()
        
        # Events storage
        self.events = deque(maxlen=max_events)
        
        # Display settings
        self.compact_mode = False
        self.filter_type = None  # When set, only show events of this type
        self.show_icons = True
        self.show_details = True
        self.current_page = 0
        self.events_per_page = 10
        
        # Animation settings
        self.animation_active = False
        self.animation_progress = 0.0
        self.last_animation_time = time.time()
        self.animation_speed = 1.0  # seconds for full animation
        
        # Interaction
        self.selected_event_index = -1
        
        # Create test events if empty
        if not self.events:
            self._create_test_events()
            
    def _load_fonts(self) -> Dict[str, pygame.font.Font]:
        """
        Load fonts for timeline display.
        
        Returns:
            Dict[str, pygame.font.Font]: Dictionary of font objects
        """
        fonts = {}
        try:
            # Try to load standard fonts first
            font_names = ['Arial', 'DejaVuSans', 'FreeSans', 'Liberation Sans']
            
            font_loaded = False
            for font_name in font_names:
                try:
                    fonts = {
                        'title': pygame.font.SysFont(font_name, 28, bold=True),
                        'event_title': pygame.font.SysFont(font_name, 22, bold=True),
                        'event_time': pygame.font.SysFont(font_name, 20),
                        'event_desc': pygame.font.SysFont(font_name, 18),
                        'label': pygame.font.SysFont(font_name, 20, bold=True),
                        'button': pygame.font.SysFont(font_name, 16),
                    }
                    font_loaded = True
                    break
                except Exception:
                    continue
            
            # Fall back to default font if none of the specified fonts could be loaded
            if not font_loaded:
                fonts = {
                    'title': pygame.font.Font(None, 28),
                    'event_title': pygame.font.Font(None, 22),
                    'event_time': pygame.font.Font(None, 20),
                    'event_desc': pygame.font.Font(None, 18),
                    'label': pygame.font.Font(None, 20),
                    'button': pygame.font.Font(None, 16),
                }
                
        except Exception as e:
            logger.error(f"Error loading fonts: {e}")
            # Emergency fallback
            fonts = {
                'title': pygame.font.Font(None, 28),
                'event_title': pygame.font.Font(None, 22),
                'event_time': pygame.font.Font(None, 20),
                'event_desc': pygame.font.Font(None, 18),
                'label': pygame.font.Font(None, 20),
                'button': pygame.font.Font(None, 16),
            }
            
        return fonts
        
    def set_surface(self, surface: pygame.Surface, rect: pygame.Rect):
        """
        Set the surface and rectangle to draw on.
        
        Args:
            surface: Pygame surface to draw on
            rect: Rectangle defining the area to draw in
        """
        self.surface = surface
        self.rect = rect
        
    def add_event(self, event: TimelineEvent):
        """
        Add an event to the timeline.
        
        Args:
            event: TimelineEvent to add
        """
        # Update time ago for proper sorting
        event.update_time_ago()
        
        # Add to the timeline
        self.events.appendleft(event)
        
        # Mark animation for this new event
        self.animation_active = True
        self.animation_progress = 0.0
        self.last_animation_time = time.time()
        
    def add_event_from_dict(self, event_dict: Dict[str, Any]):
        """
        Add an event from a dictionary.
        
        Args:
            event_dict: Dictionary containing event information
        """
        # Parse timestamp if provided as string
        timestamp = event_dict.get('timestamp', None)
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
            
        # Create and add event
        event = TimelineEvent(
            timestamp=timestamp,
            event_type=event_dict.get('type', 'info'),
            title=event_dict.get('title', ''),
            description=event_dict.get('description', ''),
            icon=event_dict.get('icon', None),
            source=event_dict.get('source', None),
            color=event_dict.get('color', None)
        )
        
        self.add_event(event)
        
    def add_event_from_text(self, text: str, event_type: str = 'info'):
        """
        Create and add an event from a text description.
        Automatically categorizes based on content keywords.
        
        Args:
            text: Text description of the event
            event_type: Default event type if not auto-detected
        """
        # Try to determine event type from text content
        lower_text = text.lower()
        
        # Determine event type from content
        if any(word in lower_text for word in ["bleeding", "airway", "breathing", "critical", "severe"]):
            detected_type = "critical"
            title = "Critical Finding"
        elif any(word in lower_text for word in ["tourniquet", "bandage", "applied", "dressed", "treatment"]):
            detected_type = "treatment" 
            title = "Treatment Applied"
        elif any(word in lower_text for word in ["assess", "assessment", "examined", "inspection"]):
            detected_type = "assessment"
            title = "Patient Assessment"
        elif any(word in lower_text for word in ["heart rate", "blood pressure", "respiration", "temperature", "vitals"]):
            detected_type = "vital"
            title = "Vital Signs"
        elif any(word in lower_text for word in ["medication", "administered", "dose", "drug", "morphine", "fentanyl"]):
            detected_type = "medication"
            title = "Medication Given"
        elif any(word in lower_text for word in ["evacuate", "transport", "move", "medevac", "evacuation"]):
            detected_type = "transport"
            title = "Transport/Evacuation"
        elif any(word in lower_text for word in ["warning", "caution", "monitor", "watch for"]):
            detected_type = "warning"
            title = "Warning"
        else:
            detected_type = event_type
            title = "Information"
            
        # Create and add the event
        event = TimelineEvent(
            timestamp=datetime.now(),
            event_type=detected_type,
            title=title,
            description=text
        )
        
        self.add_event(event)
        
    def update_events(self):
        """Update all events' time ago field and animations."""
        # Update time for all events
        for event in self.events:
            event.update_time_ago()
            
        # Update animation
        if self.animation_active:
            current_time = time.time()
            elapsed = current_time - self.last_animation_time
            self.animation_progress += elapsed / self.animation_speed
            
            if self.animation_progress >= 1.0:
                self.animation_active = False
                self.animation_progress = 0.0
                
            self.last_animation_time = current_time
            
    def clear_events(self):
        """Clear all events from the timeline."""
        self.events.clear()
        self.selected_event_index = -1
        self.current_page = 0
        
    def set_filter(self, event_type: Optional[str] = None):
        """
        Set a filter to show only events of a specific type.
        
        Args:
            event_type: Type of event to filter for, or None to show all
        """
        self.filter_type = event_type
        
    def draw_timeline_view(self):
        """Draw the main timeline visualization."""
        if not self.surface or not self.rect:
            return
            
        # Get display area
        rect = self.rect
        
        # Background
        pygame.draw.rect(self.surface, (20, 20, 30), rect, border_radius=5)
        pygame.draw.rect(self.surface, (50, 50, 60), rect, 1, border_radius=5)
        
        # Title and filter indicator
        title_text = "EVENT TIMELINE"
        if self.filter_type:
            title_text += f" - {self.EVENT_TYPES[self.filter_type]['name']} Events"
            
        title = self.fonts['title'].render(title_text, True, self.theme_colors["highlight"])
        self.surface.blit(title, (rect.x + 15, rect.y + 10))
        
        # Filter buttons
        self._draw_filter_buttons(rect.x + rect.width - 200, rect.y + 15, 180, 25)
        
        # Timeline area
        timeline_rect = pygame.Rect(rect.x + 10, rect.y + 50, 
                                  rect.width - 20, rect.height - 60)
                                  
        # Update time for all events
        self.update_events()
        
        # Get filtered events
        filtered_events = list(self.events)
        if self.filter_type:
            filtered_events = [e for e in filtered_events if e.event_type == self.filter_type]
            
        # Calculate pagination
        total_pages = max(1, (len(filtered_events) + self.events_per_page - 1) // self.events_per_page)
        start_idx = self.current_page * self.events_per_page
        page_events = filtered_events[start_idx:start_idx + self.events_per_page]
        
        # Draw events
        if not page_events:
            # No events message
            no_events_text = "No events to display"
            no_events = self.fonts['event_title'].render(no_events_text, True, self.theme_colors["text"])
            self.surface.blit(no_events, (timeline_rect.centerx - no_events.get_width()//2, 
                                        timeline_rect.centery - no_events.get_height()//2))
        else:
            # Draw events
            self._draw_events(timeline_rect, page_events)
            
        # Draw pagination controls
        if total_pages > 1:
            self._draw_pagination(timeline_rect.centerx, timeline_rect.bottom - 30, 
                                total_pages, self.current_page)
                                
    def _draw_events(self, rect: pygame.Rect, events: List[TimelineEvent]):
        """
        Draw events in the timeline view.
        
        Args:
            rect: Rectangle to draw in
            events: List of events to display
        """
        # Event layout settings
        item_height = 75 if self.show_details else 40
        spacing = 10
        max_visible = min(len(events), (rect.height - 20) // (item_height + spacing))
        
        # Draw each visible event
        for i, event in enumerate(events[:max_visible]):
            item_y = rect.y + (i * (item_height + spacing))
            
            # Event background with subtle gradient
            item_rect = pygame.Rect(rect.x, item_y, rect.width, item_height)
            
            # Animation for new events
            if i == 0 and self.animation_active:
                # Slide-in animation for newest event
                offset = int((1.0 - self.animation_progress) * item_rect.width * 0.3)
                item_rect.x += offset
                item_rect.width -= offset
                
            # Selected event highlight
            if i == self.selected_event_index:
                pygame.draw.rect(self.surface, (40, 45, 60), item_rect, border_radius=5)
                pygame.draw.rect(self.surface, event.get_color(self.theme_colors), 
                               item_rect, 2, border_radius=5)
            else:
                pygame.draw.rect(self.surface, (30, 35, 50), item_rect, border_radius=5)
                pygame.draw.rect(self.surface, (60, 65, 75), item_rect, 1, border_radius=5)
                
                # Colored indicator for event type
                indicator_rect = pygame.Rect(item_rect.x, item_rect.y, 5, item_rect.height)
                pygame.draw.rect(self.surface, event.get_color(self.theme_colors), 
                               indicator_rect, border_radius=2)
            
            # Time indicator
            time_text = event.format_time_ago()
            time_width = 80
            time_surface = self.fonts['event_time'].render(time_text, True, (200, 200, 200))
            self.surface.blit(time_surface, (item_rect.right - time_width - 5, item_rect.y + 8))
            
            # Icon if available
            icon_size = 24
            icon_x = item_rect.x + 15
            
            # Get default icon for event type if not explicitly set
            icon_name = event.get_icon_name()
            
            # Title with icon spacing
            title_x = icon_x + icon_size + 10 if self.show_icons else item_rect.x + 15
            title_width = item_rect.width - title_x - time_width - 15
            
            # Truncate title if needed
            title_text = event.title
            title_surface = self.fonts['event_title'].render(title_text, True, 
                                                          event.get_color(self.theme_colors))
            
            while title_surface.get_width() > title_width and title_text:
                title_text = title_text[:-5] + "..."
                title_surface = self.fonts['event_title'].render(title_text, True, 
                                                              event.get_color(self.theme_colors))
                
            self.surface.blit(title_surface, (title_x, item_rect.y + 8))
            
            # Description if enabled
            if self.show_details and event.description:
                desc_text = event.description
                max_desc_width = item_rect.width - 30
                
                # Split description into lines if needed
                desc_surfaces = []
                
                while desc_text and len(desc_surfaces) < 2:  # Limit to 2 lines
                    desc_surface = self.fonts['event_desc'].render(desc_text, True, self.theme_colors["text"])
                    
                    if desc_surface.get_width() <= max_desc_width:
                        desc_surfaces.append(desc_surface)
                        break
                        
                    # Truncate and try again
                    chars_that_fit = len(desc_text)
                    while chars_that_fit > 0:
                        test_text = desc_text[:chars_that_fit]
                        test_surface = self.fonts['event_desc'].render(test_text, True, 
                                                                     self.theme_colors["text"])
                        if test_surface.get_width() <= max_desc_width:
                            desc_surfaces.append(test_surface)
                            desc_text = desc_text[chars_that_fit:].strip()
                            break
                        chars_that_fit -= 1
                
                # Render the description lines
                for j, surface in enumerate(desc_surfaces):
                    self.surface.blit(surface, (title_x, item_rect.y + 38 + j * 18))
                    
                # If there's more text, add ellipsis
                if desc_text and len(desc_surfaces) == 2:
                    ellipsis = self.fonts['event_desc'].render("...", True, self.theme_colors["text"])
                    self.surface.blit(ellipsis, (title_x + desc_surfaces[-1].get_width() + 5, 
                                               item_rect.y + 38 + (len(desc_surfaces) - 1) * 18))
                    
    def _draw_filter_buttons(self, x: int, y: int, width: int, height: int):
        """
        Draw filter buttons for event types.
        
        Args:
            x: X position
            y: Y position
            width: Width of button area
            height: Height of buttons
        """
        # Filter label
        filter_label = self.fonts['label'].render("Filter:", True, self.theme_colors["text"])
        self.surface.blit(filter_label, (x, y))
        
        # Button for 'All' events
        all_btn_rect = pygame.Rect(x + 70, y, 45, height)
        btn_color = (40, 45, 60) if self.filter_type is None else (30, 30, 40)
        pygame.draw.rect(self.surface, btn_color, all_btn_rect, border_radius=3)
        pygame.draw.rect(self.surface, (60, 65, 75), all_btn_rect, 1, border_radius=3)
        
        all_text = self.fonts['button'].render("All", True, self.theme_colors["text"])
        self.surface.blit(all_text, (all_btn_rect.centerx - all_text.get_width()//2, 
                                   all_btn_rect.centery - all_text.get_height()//2))
                                   
        # Button for critical events
        critical_btn_rect = pygame.Rect(x + 125, y, 70, height)
        btn_color = (40, 45, 60) if self.filter_type == "critical" else (30, 30, 40)
        pygame.draw.rect(self.surface, btn_color, critical_btn_rect, border_radius=3)
        pygame.draw.rect(self.surface, (60, 65, 75), critical_btn_rect, 1, border_radius=3)
        
        critical_text = self.fonts['button'].render("Critical", True, self.theme_colors["alert"])
        self.surface.blit(critical_text, (critical_btn_rect.centerx - critical_text.get_width()//2, 
                                        critical_btn_rect.centery - critical_text.get_height()//2))
                                        
    def _draw_pagination(self, center_x: int, y: int, total_pages: int, current_page: int):
        """
        Draw pagination controls.
        
        Args:
            center_x: X center position for controls
            y: Y position
            total_pages: Total number of pages
            current_page: Current page (0-based)
        """
        # Draw page indicator
        page_text = f"Page {current_page + 1} of {total_pages}"
        page_surface = self.fonts['button'].render(page_text, True, self.theme_colors["text"])
        self.surface.blit(page_surface, (center_x - page_surface.get_width()//2, y))
        
        # Previous button
        if current_page > 0:
            prev_text = "◀ Previous"
            prev_surface = self.fonts['button'].render(prev_text, True, self.theme_colors["text"])
            prev_rect = pygame.Rect(center_x - 150, y, 100, 25)
            pygame.draw.rect(self.surface, (30, 30, 40), prev_rect, border_radius=3)
            pygame.draw.rect(self.surface, (60, 65, 75), prev_rect, 1, border_radius=3)
            self.surface.blit(prev_surface, (prev_rect.centerx - prev_surface.get_width()//2, 
                                           prev_rect.centery - prev_surface.get_height()//2))
            
        # Next button
        if current_page < total_pages - 1:
            next_text = "Next ▶"
            next_surface = self.fonts['button'].render(next_text, True, self.theme_colors["text"])
            next_rect = pygame.Rect(center_x + 50, y, 100, 25)
            pygame.draw.rect(self.surface, (30, 30, 40), next_rect, border_radius=3)
            pygame.draw.rect(self.surface, (60, 65, 75), next_rect, 1, border_radius=3)
            self.surface.blit(next_surface, (next_rect.centerx - next_surface.get_width()//2, 
                                           next_rect.centery - next_surface.get_height()//2))
                                           
    def draw_compact_view(self):
        """Draw the timeline in compact view mode."""
        if not self.surface or not self.rect:
            return
            
        rect = self.rect
        
        # Background
        pygame.draw.rect(self.surface, (20, 20, 30), rect, border_radius=5)
        pygame.draw.rect(self.surface, (50, 50, 60), rect, 1, border_radius=5)
        
        # Title
        title = self.fonts['label'].render("EVENT TIMELINE", True, self.theme_colors["highlight"])
        self.surface.blit(title, (rect.x + 10, rect.y + 5))
        
        # Update all events
        self.update_events()
        
        # Get filtered events
        filtered_events = list(self.events)
        if self.filter_type:
            filtered_events = [e for e in filtered_events if e.event_type == self.filter_type]
            
        # Determine number of visible events
        max_visible = 5
        
        # Calculate item dimensions
        item_spacing = 5
        item_width = (rect.width - (max_visible + 1) * item_spacing) // max_visible
        item_height = rect.height - 35
        
        # Draw event blocks
        for i, event in enumerate(filtered_events[:max_visible]):
            item_x = rect.x + (i * (item_width + item_spacing)) + item_spacing
            item_y = rect.y + 30
            
            # Event background
            item_rect = pygame.Rect(item_x, item_y, item_width, item_height)
            pygame.draw.rect(self.surface, (30, 35, 50), item_rect, border_radius=3)
            
            # Colored top for event type
            top_rect = pygame.Rect(item_x, item_y, item_width, 5)
            pygame.draw.rect(self.surface, event.get_color(self.theme_colors), top_rect, border_radius=2)
            
            # Time indicator
            time_text = event.format_time_ago()
            time_surface = self.fonts['button'].render(time_text, True, (180, 180, 180))
            self.surface.blit(time_surface, (item_x + 5, item_y + 10))
            
            # Title (truncated)
            title_width = item_width - 10
            title_text = event.title
            while len(title_text) > 3:
                title_surface = self.fonts['event_desc'].render(title_text, True, 
                                                           event.get_color(self.theme_colors))
                if title_surface.get_width() <= title_width:
                    break
                title_text = title_text[:-1]
                
            if title_text != event.title:
                title_text = title_text[:-3] + "..."
                
            title_surface = self.fonts['event_desc'].render(title_text, True, 
                                                       event.get_color(self.theme_colors))
            self.surface.blit(title_surface, (item_x + 5, item_y + 30))
            
    def draw(self):
        """Draw the timeline visualization."""
        if self.compact_mode:
            self.draw_compact_view()
        else:
            self.draw_timeline_view()
            
    def handle_click(self, pos: Tuple[int, int]) -> bool:
        """
        Handle mouse click at the given position.
        
        Args:
            pos: (x, y) position of the click
            
        Returns:
            bool: True if the click was handled
        """
        if not self.rect:
            return False
            
        # Check if click is within our area
        if not self.rect.collidepoint(pos):
            return False
            
        if self.compact_mode:
            # In compact mode, toggle to full mode on click
            self.compact_mode = False
            return True
            
        # Process click in full mode
        
        # Filter buttons
        filter_buttons_rect = pygame.Rect(
            self.rect.x + self.rect.width - 200,
            self.rect.y + 15,
            180, 25
        )
        
        if filter_buttons_rect.collidepoint(pos):
            # All button
            all_btn_rect = pygame.Rect(filter_buttons_rect.x + 70, filter_buttons_rect.y, 45, 25)
            if all_btn_rect.collidepoint(pos):
                self.filter_type = None
                return True
                
            # Critical button
            critical_btn_rect = pygame.Rect(filter_buttons_rect.x + 125, filter_buttons_rect.y, 70, 25)
            if critical_btn_rect.collidepoint(pos):
                self.filter_type = "critical"
                return True
                
        # Timeline area
        timeline_rect = pygame.Rect(
            self.rect.x + 10,
            self.rect.y + 50,
            self.rect.width - 20,
            self.rect.height - 60
        )
        
        if timeline_rect.collidepoint(pos):
            # Pagination controls
            pagination_y = timeline_rect.bottom - 30
            
            # Get filtered events
            filtered_events = list(self.events)
            if self.filter_type:
                filtered_events = [e for e in filtered_events if e.event_type == self.filter_type]
                
            total_pages = max(1, (len(filtered_events) + self.events_per_page - 1) // self.events_per_page)
            
            # Previous button
            if self.current_page > 0:
                prev_rect = pygame.Rect(timeline_rect.centerx - 150, pagination_y, 100, 25)
                if prev_rect.collidepoint(pos):
                    self.current_page -= 1
                    return True
                    
            # Next button
            if self.current_page < total_pages - 1:
                next_rect = pygame.Rect(timeline_rect.centerx + 50, pagination_y, 100, 25)
                if next_rect.collidepoint(pos):
                    self.current_page += 1
                    return True
                    
            # Check for event item clicks
            item_height = 75 if self.show_details else 40
            spacing = 10
            
            # Get events for current page
            start_idx = self.current_page * self.events_per_page
            page_events = filtered_events[start_idx:start_idx + self.events_per_page]
            
            for i, event in enumerate(page_events):
                item_y = timeline_rect.y + (i * (item_height + spacing))
                item_rect = pygame.Rect(timeline_rect.x, item_y, timeline_rect.width, item_height)
                
                if item_rect.collidepoint(pos):
                    # Toggle selection
                    if self.selected_event_index == i:
                        self.selected_event_index = -1
                    else:
                        self.selected_event_index = i
                    return True
                    
        return False
        
    def toggle_compact_mode(self):
        """Toggle between compact and full visualization modes."""
        self.compact_mode = not self.compact_mode
        self.selected_event_index = -1  # Clear selection on mode change
        
    def next_page(self):
        """Go to the next page of events."""
        # Get filtered events
        filtered_events = list(self.events)
        if self.filter_type:
            filtered_events = [e for e in filtered_events if e.event_type == self.filter_type]
            
        total_pages = max(1, (len(filtered_events) + self.events_per_page - 1) // self.events_per_page)
        
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.selected_event_index = -1  # Clear selection on page change
            
    def prev_page(self):
        """Go to the previous page of events."""
        if self.current_page > 0:
            self.current_page -= 1
            self.selected_event_index = -1  # Clear selection on page change
            
    def _create_test_events(self):
        """Create sample events for testing/preview."""
        # Sample events with different types
        now = datetime.now()
        
        self.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=2),
            event_type="critical",
            title="Critical Bleeding",
            description="Patient has arterial bleeding from left leg wound."
        ))
        
        self.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=3, seconds=30),
            event_type="assessment",
            title="Initial Assessment",
            description="MARCH assessment complete. Multiple injuries identified."
        ))
        
        self.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=5),
            event_type="treatment",
            title="Tourniquet Applied",
            description="CAT tourniquet applied to left leg, 5cm above wound."
        ))
        
        self.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=6, seconds=15),
            event_type="vital",
            title="Vital Signs",
            description="HR 125, BP 90/60, RR 22, SpO2 94%"
        ))
        
        self.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=8),
            event_type="medication",
            title="Pain Management",
            description="10mg morphine administered IV"
        ))
        
        self.add_event(TimelineEvent(
            timestamp=now - timedelta(minutes=15),
            event_type="transport",
            title="MEDEVAC Requested",
            description="Priority evacuation requested. ETA 10 minutes."
        ))
        

# Example usage
if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    
    # Create a test window
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Timeline Visualization Test")
    
    # Create timeline
    timeline_rect = pygame.Rect(50, 50, 700, 500)
    timeline = TimelineVisualization(screen, timeline_rect)
    
    # Add some test events
    timeline.add_event_from_text("Massive hemorrhage from left thigh, arterial bleeding observed.")
    timeline.add_event_from_text("Applied combat gauze and pressure dressing to wound.")
    timeline.add_event_from_text("Vital signs: BP 110/70, HR 100, RR 18, SpO2 95%")
    
    # Create a clock for consistent frame rate
    clock = pygame.time.Clock()
    
    # Main loop
    running = True
    compact_mode = False
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    compact_mode = not compact_mode
                    timeline.toggle_compact_mode()
                elif event.key == pygame.K_a:
                    # Add a test event
                    timeline.add_event_from_text("New event at " + datetime.now().strftime("%H:%M:%S"))
                elif event.key == pygame.K_f:
                    # Toggle filter
                    if timeline.filter_type is None:
                        timeline.set_filter("critical")
                    else:
                        timeline.set_filter(None)
                elif event.key == pygame.K_LEFT:
                    timeline.prev_page()
                elif event.key == pygame.K_RIGHT:
                    timeline.next_page()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                timeline.handle_click(event.pos)
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw timeline
        timeline.draw()
        
        # Update display
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(30)
    
    # Clean up
    pygame.quit()
    sys.exit()