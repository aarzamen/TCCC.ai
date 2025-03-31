#!/usr/bin/env python3
"""
TCCC.ai Vital Signs Visualization
--------------------------------
Provides visualization components for medical vital signs such as
heart rate, blood pressure, respiration rate, and SpO2.
"""

import os
import sys
import time
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VitalSigns")

try:
    import pygame
    from pygame.locals import *
except ImportError:
    logger.error("pygame not installed. Install it with: pip install pygame")
    raise

# Import display configuration manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from display_config import get_display_config


class VitalSignsMonitor:
    """
    Visual monitor for vital signs with real-time graph display.
    Supports standard vital signs: HR, BP, RR, SpO2, Temp.
    """
    
    # Vital sign normal ranges and colors
    VITAL_RANGES = {
        "hr": {
            "name": "Heart Rate",
            "unit": "bpm",
            "low": 60,
            "normal_low": 60,
            "normal_high": 100,
            "high": 120,
            "critical_low": 40,
            "critical_high": 150,
            "default": 80,
            "min_graph": 30,
            "max_graph": 180,
        },
        "sbp": {
            "name": "Systolic BP",
            "unit": "mmHg",
            "low": 90,
            "normal_low": 100,
            "normal_high": 140,
            "high": 160,
            "critical_low": 70,
            "critical_high": 200,
            "default": 120,
            "min_graph": 60,
            "max_graph": 220,
        },
        "dbp": {
            "name": "Diastolic BP",
            "unit": "mmHg",
            "low": 60,
            "normal_low": 60,
            "normal_high": 90,
            "high": 100,
            "critical_low": 40,
            "critical_high": 120,
            "default": 80,
            "min_graph": 30,
            "max_graph": 130,
        },
        "rr": {
            "name": "Resp Rate",
            "unit": "bpm",
            "low": 12,
            "normal_low": 12,
            "normal_high": 20,
            "high": 24,
            "critical_low": 8,
            "critical_high": 30,
            "default": 16,
            "min_graph": 6,
            "max_graph": 40,
        },
        "spo2": {
            "name": "SpO2",
            "unit": "%",
            "low": 92,
            "normal_low": 95,
            "normal_high": 100,
            "high": 100,
            "critical_low": 85,
            "critical_high": 100,
            "default": 98,
            "min_graph": 80,
            "max_graph": 100,
        },
        "temp": {
            "name": "Temperature",
            "unit": "°C",
            "low": 36.0,
            "normal_low": 36.5,
            "normal_high": 37.5,
            "high": 38.0,
            "critical_low": 35.0,
            "critical_high": 40.0,
            "default": 37.0,
            "min_graph": 34.0,
            "max_graph": 41.0,
        }
    }
    
    def __init__(self, 
                surface: Optional[pygame.Surface] = None,
                rect: Optional[pygame.Rect] = None,
                config: Optional[Dict] = None,
                history_length: int = 60):
        """
        Initialize the vital signs monitor.
        
        Args:
            surface: Pygame surface to draw on
            rect: Rectangle defining the area to draw in
            config: Optional configuration dictionary
            history_length: Number of data points to keep in history
        """
        self.surface = surface
        self.rect = rect
        self.config = config or get_display_config().get_config()
        self.history_length = history_length
        
        # Visual settings
        self.theme_colors = self.config["colors"]
        self.fonts = self._load_fonts()
        
        # Initialize vital sign data and history
        self.vital_data = {
            vital: {
                "current": info["default"],
                "history": deque([info["default"]] * history_length, maxlen=history_length),
                "last_update": time.time(),
                "trend": 0,  # -1: decreasing, 0: stable, 1: increasing
                "alarm": False
            } for vital, info in self.VITAL_RANGES.items()
        }
        
        # Blood pressure needs special handling
        self.bp_text = f"{self.vital_data['sbp']['current']}/{self.vital_data['dbp']['current']}"
        
        # Visual display settings
        self.show_graph = True
        self.show_alarms = True
        self.active_vital = "hr"  # Default selected vital for detailed view
        self.compact_mode = False
        
        # Animation settings
        self.animation_speed = 5  # Points per update
        self.last_animation = time.time()
        self.animation_interval = 0.1  # 100ms between animations
        
        # Graph animation state
        self.graph_position = 0
        
    def _load_fonts(self) -> Dict[str, pygame.font.Font]:
        """
        Load fonts for vital signs display.
        
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
                        'vital_large': pygame.font.SysFont(font_name, 36, bold=True),
                        'vital_medium': pygame.font.SysFont(font_name, 28, bold=True),
                        'vital_small': pygame.font.SysFont(font_name, 22),
                        'unit': pygame.font.SysFont(font_name, 18),
                        'label': pygame.font.SysFont(font_name, 20, bold=True),
                    }
                    font_loaded = True
                    break
                except Exception:
                    continue
            
            # Fall back to default font if none of the specified fonts could be loaded
            if not font_loaded:
                fonts = {
                    'vital_large': pygame.font.Font(None, 36),
                    'vital_medium': pygame.font.Font(None, 28),
                    'vital_small': pygame.font.Font(None, 22),
                    'unit': pygame.font.Font(None, 18),
                    'label': pygame.font.Font(None, 20),
                }
                
        except Exception as e:
            logger.error(f"Error loading fonts: {e}")
            # Emergency fallback
            fonts = {
                'vital_large': pygame.font.Font(None, 36),
                'vital_medium': pygame.font.Font(None, 28),
                'vital_small': pygame.font.Font(None, 22),
                'unit': pygame.font.Font(None, 18),
                'label': pygame.font.Font(None, 20),
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
        
    def update_vital(self, vital_type: str, value: float):
        """
        Update a vital sign value.
        
        Args:
            vital_type: Type of vital sign (hr, sbp, dbp, rr, spo2, temp)
            value: New value for the vital sign
        """
        if vital_type not in self.vital_data:
            logger.warning(f"Unknown vital sign type: {vital_type}")
            return
            
        # Get previous value
        previous = self.vital_data[vital_type]["current"]
        
        # Update current value and time
        self.vital_data[vital_type]["current"] = value
        self.vital_data[vital_type]["last_update"] = time.time()
        
        # Update history
        self.vital_data[vital_type]["history"].append(value)
        
        # Calculate trend
        if value > previous + 1:
            self.vital_data[vital_type]["trend"] = 1
        elif value < previous - 1:
            self.vital_data[vital_type]["trend"] = -1
        else:
            self.vital_data[vital_type]["trend"] = 0
            
        # Check alarm conditions
        ranges = self.VITAL_RANGES[vital_type]
        if value <= ranges["critical_low"] or value >= ranges["critical_high"]:
            self.vital_data[vital_type]["alarm"] = True
        else:
            self.vital_data[vital_type]["alarm"] = False
            
        # Special handling for blood pressure
        if vital_type in ["sbp", "dbp"]:
            self.bp_text = f"{self.vital_data['sbp']['current']}/{self.vital_data['dbp']['current']}"
        
    def update_from_entity(self, entity: Dict[str, Any]):
        """
        Update vital signs from a medical entity dictionary.
        
        Args:
            entity: Entity dictionary with vital sign information
        """
        # Skip if not a vital sign entity
        if entity.get("type") != "vital_sign":
            return
            
        # Get the vital sign value
        value_str = entity.get("value", "")
        if not value_str:
            return
            
        # Get the name to determine the type
        name = entity.get("name", "").lower()
        
        # Parse different vital types
        if "heart rate" in name or "pulse" in name or "hr" in name:
            try:
                value = int(value_str.strip())
                self.update_vital("hr", value)
            except ValueError:
                pass
                
        elif "blood pressure" in name or "bp" in name:
            # Try to parse blood pressure format (e.g., "120/80")
            try:
                if "/" in value_str:
                    sbp, dbp = value_str.split("/")
                    self.update_vital("sbp", int(sbp.strip()))
                    self.update_vital("dbp", int(dbp.strip()))
            except ValueError:
                pass
                
        elif "respiratory rate" in name or "rr" in name:
            try:
                value = int(value_str.strip())
                self.update_vital("rr", value)
            except ValueError:
                pass
                
        elif "o2" in name or "oxygen" in name or "spo2" in name or "sat" in name:
            try:
                # Remove % sign if present
                value = int(value_str.strip().replace("%", ""))
                self.update_vital("spo2", value)
            except ValueError:
                pass
                
        elif "temp" in name or "temperature" in name:
            try:
                value = float(value_str.strip().replace("°C", "").replace("C", ""))
                self.update_vital("temp", value)
            except ValueError:
                pass
        
    def get_value_color(self, vital_type: str, value: float) -> Tuple[int, int, int]:
        """
        Get the appropriate color for a vital sign value based on its range.
        
        Args:
            vital_type: Type of vital sign
            value: Current value
            
        Returns:
            Tuple[int, int, int]: RGB color
        """
        if vital_type not in self.VITAL_RANGES:
            return self.theme_colors["text"]
            
        ranges = self.VITAL_RANGES[vital_type]
        
        # Critical conditions
        if value <= ranges["critical_low"] or value >= ranges["critical_high"]:
            return self.theme_colors["alert"]
            
        # Warning conditions
        if value < ranges["low"] or value > ranges["high"]:
            return (255, 165, 0)  # Orange
            
        # Normal range
        if ranges["normal_low"] <= value <= ranges["normal_high"]:
            return self.theme_colors["success"]
            
        # Borderline but not critical
        return (255, 255, 0)  # Yellow
    
    def draw_compact(self):
        """Draw vital signs in compact mode (minimal screen space)."""
        if not self.surface or not self.rect:
            return
            
        rect = self.rect
        x, y = rect.x, rect.y
        width, height = rect.width, rect.height
        
        # Background
        pygame.draw.rect(self.surface, (20, 20, 30), rect, border_radius=5)
        pygame.draw.rect(self.surface, (50, 50, 60), rect, 1, border_radius=5)
        
        # Title
        title = self.fonts['label'].render("VITAL SIGNS", True, self.theme_colors["highlight"])
        self.surface.blit(title, (x + 10, y + 5))
        
        # Calculate layout
        vital_height = height - 30
        item_width = width // 3
        item_height = vital_height // 2
        
        # Draw vital boxes arranged in a grid
        vitals_to_display = [
            {"type": "hr", "position": (0, 0)},
            {"type": "bp", "position": (0, 1)},
            {"type": "rr", "position": (1, 0)},
            {"type": "spo2", "position": (1, 1)},
            {"type": "temp", "position": (2, 0)},
        ]
        
        for vital_info in vitals_to_display:
            vital_type = vital_info["type"]
            col, row = vital_info["position"]
            
            item_x = x + (col * item_width) + 5
            item_y = y + 30 + (row * item_height)
            
            # Draw background with rounded corners
            item_rect = pygame.Rect(item_x, item_y, item_width - 10, item_height - 10)
            pygame.draw.rect(self.surface, (30, 30, 40), item_rect, border_radius=3)
            
            # Handle BP special case
            if vital_type == "bp":
                # Get colors for systolic and diastolic
                sbp_color = self.get_value_color("sbp", self.vital_data["sbp"]["current"])
                dbp_color = self.get_value_color("dbp", self.vital_data["dbp"]["current"])
                
                # Label
                bp_label = self.fonts['label'].render("BP", True, self.theme_colors["text"])
                self.surface.blit(bp_label, (item_x + 5, item_y + 3))
                
                # Value
                bp_value = self.fonts['vital_medium'].render(self.bp_text, True, sbp_color)
                self.surface.blit(bp_value, (item_x + 5, item_y + item_height // 2 - 10))
                
                # Unit
                bp_unit = self.fonts['unit'].render("mmHg", True, self.theme_colors["text"])
                self.surface.blit(bp_unit, (item_x + 5, item_y + item_height - 20))
                
                # Draw trends
                if self.vital_data["sbp"]["trend"] > 0:
                    trend_text = "↑"
                elif self.vital_data["sbp"]["trend"] < 0:
                    trend_text = "↓"
                else:
                    trend_text = "→"
                    
                trend = self.fonts['vital_small'].render(trend_text, True, sbp_color)
                self.surface.blit(trend, (item_x + item_width - 30, item_y + 5))
                
            else:
                # Regular vital signs
                if vital_type == "temp":
                    value = self.vital_data[vital_type]["current"]
                    value_color = self.get_value_color(vital_type, value)
                    value_text = f"{value:.1f}"
                    name = "TEMP"
                    unit = "°C"
                else:
                    vital_range = self.VITAL_RANGES.get(vital_type, None)
                    if not vital_range:
                        continue
                        
                    value = self.vital_data[vital_type]["current"]
                    value_color = self.get_value_color(vital_type, value)
                    value_text = str(int(value))
                    name = vital_range["name"].upper().split()[0]  # First word only
                    unit = vital_range["unit"]
                
                # Label
                label = self.fonts['label'].render(name, True, self.theme_colors["text"])
                self.surface.blit(label, (item_x + 5, item_y + 3))
                
                # Value
                value_surf = self.fonts['vital_medium'].render(value_text, True, value_color)
                self.surface.blit(value_surf, (item_x + 5, item_y + item_height // 2 - 10))
                
                # Unit
                unit_surf = self.fonts['unit'].render(unit, True, self.theme_colors["text"])
                self.surface.blit(unit_surf, (item_x + 5, item_y + item_height - 20))
                
                # Draw trend indicator
                if vital_type in self.vital_data:
                    if self.vital_data[vital_type]["trend"] > 0:
                        trend_text = "↑"
                    elif self.vital_data[vital_type]["trend"] < 0:
                        trend_text = "↓"
                    else:
                        trend_text = "→"
                        
                    trend = self.fonts['vital_small'].render(trend_text, True, value_color)
                    self.surface.blit(trend, (item_x + item_width - 30, item_y + 5))
        
    def draw_standard(self):
        """Draw vital signs in standard mode with graphs."""
        if not self.surface or not self.rect:
            return
            
        rect = self.rect
        x, y = rect.x, rect.y
        width, height = rect.height
        
        # Background
        pygame.draw.rect(self.surface, (20, 20, 30), rect, border_radius=5)
        pygame.draw.rect(self.surface, (50, 50, 60), rect, 1, border_radius=5)
        
        # Title
        title = self.fonts['label'].render("VITAL SIGNS MONITOR", True, self.theme_colors["highlight"])
        self.surface.blit(title, (x + 10, y + 5))
        
        # Calculate layout
        vitals_panel_width = min(350, rect.width // 3)
        graph_panel_width = rect.width - vitals_panel_width - 20
        
        vitals_panel_rect = pygame.Rect(x + 10, y + 30, vitals_panel_width, rect.height - 40)
        graph_panel_rect = pygame.Rect(x + vitals_panel_width + 20, y + 30, 
                                       graph_panel_width, rect.height - 40)
        
        # Draw vitals panel
        pygame.draw.rect(self.surface, (30, 30, 40), vitals_panel_rect, border_radius=3)
        
        # Calculate vital sign item height
        item_height = vitals_panel_rect.height // 5
        
        # Draw each vital sign
        vitals_to_display = ["hr", "bp_combined", "rr", "spo2", "temp"]
        for i, vital_type in enumerate(vitals_to_display):
            item_y = vitals_panel_rect.y + (i * item_height) + 5
            
            if vital_type == "bp_combined":
                # Blood pressure is a special case handling both systolic and diastolic
                bp_rect = pygame.Rect(vitals_panel_rect.x + 5, item_y, 
                                     vitals_panel_rect.width - 10, item_height - 10)
                if self.active_vital in ["sbp", "dbp"]:
                    pygame.draw.rect(self.surface, (40, 40, 60), bp_rect, border_radius=3)
                
                # Label
                bp_label = self.fonts['label'].render("BLOOD PRESSURE", True, self.theme_colors["text"])
                self.surface.blit(bp_label, (bp_rect.x + 5, bp_rect.y + 5))
                
                # Value with systolic and diastolic colors
                sbp_color = self.get_value_color("sbp", self.vital_data["sbp"]["current"])
                bp_value = self.fonts['vital_large'].render(self.bp_text, True, sbp_color)
                self.surface.blit(bp_value, (bp_rect.x + 10, bp_rect.y + 30))
                
                # Unit
                bp_unit = self.fonts['unit'].render("mmHg", True, self.theme_colors["text"])
                self.surface.blit(bp_unit, (bp_rect.x + bp_value.get_width() + 15, bp_rect.y + 40))
                
                # Draw trend
                if self.vital_data["sbp"]["trend"] > 0:
                    trend_text = "↑"
                elif self.vital_data["sbp"]["trend"] < 0:
                    trend_text = "↓"
                else:
                    trend_text = "→"
                
                trend = self.fonts['vital_medium'].render(trend_text, True, sbp_color)
                self.surface.blit(trend, (bp_rect.right - 30, bp_rect.y + 25))
                
            else:
                # Standard vital signs
                vital_rect = pygame.Rect(vitals_panel_rect.x + 5, item_y, 
                                       vitals_panel_rect.width - 10, item_height - 10)
                
                # Highlight the active vital sign
                if vital_type == self.active_vital:
                    pygame.draw.rect(self.surface, (40, 40, 60), vital_rect, border_radius=3)
                
                # Get vital sign details
                if vital_type in self.VITAL_RANGES:
                    vital_range = self.VITAL_RANGES[vital_type]
                    value = self.vital_data[vital_type]["current"]
                    
                    # Format value based on type
                    if vital_type == "temp":
                        value_text = f"{value:.1f}"
                    else:
                        value_text = str(int(value))
                        
                    value_color = self.get_value_color(vital_type, value)
                    
                    # Label
                    label = self.fonts['label'].render(vital_range["name"].upper(), True, 
                                                     self.theme_colors["text"])
                    self.surface.blit(label, (vital_rect.x + 5, vital_rect.y + 5))
                    
                    # Value
                    value_surf = self.fonts['vital_large'].render(value_text, True, value_color)
                    self.surface.blit(value_surf, (vital_rect.x + 10, vital_rect.y + 30))
                    
                    # Unit
                    unit_surf = self.fonts['unit'].render(vital_range["unit"], True, 
                                                         self.theme_colors["text"])
                    self.surface.blit(unit_surf, 
                                    (vital_rect.x + value_surf.get_width() + 15, 
                                     vital_rect.y + 40))
                    
                    # Draw trend
                    if vital_type in self.vital_data:
                        if self.vital_data[vital_type]["trend"] > 0:
                            trend_text = "↑"
                        elif self.vital_data[vital_type]["trend"] < 0:
                            trend_text = "↓"
                        else:
                            trend_text = "→"
                            
                        trend = self.fonts['vital_medium'].render(trend_text, True, value_color)
                        self.surface.blit(trend, (vital_rect.right - 30, vital_rect.y + 25))
        
        # Draw graph panel if enabled
        if self.show_graph:
            pygame.draw.rect(self.surface, (30, 30, 40), graph_panel_rect, border_radius=3)
            
            # Draw graph for active vital sign
            if self.active_vital in self.vital_data:
                self._draw_vital_graph(graph_panel_rect, self.active_vital)
    
    def _draw_vital_graph(self, rect: pygame.Rect, vital_type: str):
        """
        Draw a graph for the specified vital sign.
        
        Args:
            rect: Rectangle to draw the graph in
            vital_type: Type of vital sign to graph
        """
        if vital_type not in self.VITAL_RANGES:
            return
            
        # Get vital sign data
        vital_range = self.VITAL_RANGES[vital_type]
        history = list(self.vital_data[vital_type]["history"])
        
        # Calculate graph dimensions
        padding = 20
        graph_rect = pygame.Rect(
            rect.x + padding,
            rect.y + padding,
            rect.width - (padding * 2),
            rect.height - (padding * 2) - 30  # Extra space at bottom for labels
        )
        
        # Draw graph title
        title = self.fonts['label'].render(f"{vital_range['name']} Trend", True, 
                                         self.theme_colors["highlight"])
        self.surface.blit(title, (graph_rect.x, rect.y + 10))
        
        # Draw graph background
        pygame.draw.rect(self.surface, (40, 40, 50), graph_rect, border_radius=3)
        
        # Calculate graph scale
        min_value = vital_range["min_graph"]
        max_value = vital_range["max_graph"]
        value_range = max_value - min_value
        
        # Draw horizontal gridlines and labels
        num_lines = 5
        for i in range(num_lines + 1):
            y_value = max_value - (i * (value_range / num_lines))
            y_pos = graph_rect.y + (i * (graph_rect.height / num_lines))
            
            # Draw line
            pygame.draw.line(
                self.surface,
                (60, 60, 70),
                (graph_rect.x, y_pos),
                (graph_rect.right, y_pos),
                1
            )
            
            # Draw label
            if vital_type == "temp":
                value_text = f"{y_value:.1f}"
            else:
                value_text = str(int(y_value))
                
            label = self.fonts['unit'].render(value_text, True, self.theme_colors["text"])
            self.surface.blit(label, (graph_rect.x - label.get_width() - 5, y_pos - 8))
            
        # Draw vertical gridlines for time
        num_vert_lines = 6
        for i in range(num_vert_lines + 1):
            x_pos = graph_rect.x + (i * (graph_rect.width / num_vert_lines))
            
            # Draw line
            pygame.draw.line(
                self.surface,
                (60, 60, 70),
                (x_pos, graph_rect.y),
                (x_pos, graph_rect.bottom),
                1
            )
            
            # Draw time label (e.g., "2m ago", "1m ago", "now")
            time_ago = num_vert_lines - i
            if time_ago == 0:
                time_text = "now"
            else:
                time_text = f"{time_ago}m ago"
                
            time_label = self.fonts['unit'].render(time_text, True, self.theme_colors["text"])
            self.surface.blit(time_label, 
                             (x_pos - time_label.get_width() // 2, 
                              graph_rect.bottom + 5))
        
        # Draw normal range background
        normal_low = vital_range["normal_low"]
        normal_high = vital_range["normal_high"]
        normal_low_y = graph_rect.y + graph_rect.height - \
                      ((normal_low - min_value) / value_range) * graph_rect.height
        normal_high_y = graph_rect.y + graph_rect.height - \
                       ((normal_high - min_value) / value_range) * graph_rect.height
        
        normal_rect = pygame.Rect(
            graph_rect.x,
            normal_high_y,
            graph_rect.width,
            normal_low_y - normal_high_y
        )
        
        # Draw normal range with transparent green
        normal_surface = pygame.Surface((normal_rect.width, normal_rect.height), pygame.SRCALPHA)
        normal_surface.fill((0, 128, 0, 40))  # Semi-transparent green
        self.surface.blit(normal_surface, (normal_rect.x, normal_rect.y))
        
        # Draw the trend line if we have enough points
        if len(history) > 1:
            points = []
            
            for i, value in enumerate(history):
                # Scale to fit graph
                x = graph_rect.x + (i * graph_rect.width / (len(history) - 1))
                y = graph_rect.y + graph_rect.height - \
                   ((value - min_value) / value_range) * graph_rect.height
                points.append((x, y))
            
            # Draw line with gradient color
            for i in range(len(points) - 1):
                start_value = history[i]
                end_value = history[i + 1]
                start_color = self.get_value_color(vital_type, start_value)
                end_color = self.get_value_color(vital_type, end_value)
                
                # Draw line segment
                pygame.draw.line(
                    self.surface,
                    end_color,  # Use end point color
                    points[i],
                    points[i + 1],
                    2
                )
            
            # Draw dots at each data point
            for i, point in enumerate(points):
                value = history[i]
                color = self.get_value_color(vital_type, value)
                pygame.draw.circle(self.surface, color, point, 3)
            
            # Draw current value with larger indicator
            current_value = history[-1]
            current_color = self.get_value_color(vital_type, current_value)
            pygame.draw.circle(self.surface, current_color, points[-1], 6)
            pygame.draw.circle(self.surface, (255, 255, 255), points[-1], 6, 1)
            
        # Draw critical thresholds
        critical_low = vital_range["critical_low"]
        critical_high = vital_range["critical_high"]
        
        critical_low_y = graph_rect.y + graph_rect.height - \
                        ((critical_low - min_value) / value_range) * graph_rect.height
        critical_high_y = graph_rect.y + graph_rect.height - \
                         ((critical_high - min_value) / value_range) * graph_rect.height
        
        # Draw critical lines
        pygame.draw.line(
            self.surface,
            self.theme_colors["alert"],
            (graph_rect.x, critical_low_y),
            (graph_rect.right, critical_low_y),
            1
        )
        
        pygame.draw.line(
            self.surface,
            self.theme_colors["alert"],
            (graph_rect.x, critical_high_y),
            (graph_rect.right, critical_high_y),
            1
        )
        
        # Draw current value text
        current_value = self.vital_data[vital_type]["current"]
        if vital_type == "temp":
            value_text = f"Current: {current_value:.1f}{vital_range['unit']}"
        else:
            value_text = f"Current: {int(current_value)}{vital_range['unit']}"
            
        value_label = self.fonts['label'].render(value_text, True, 
                                               self.get_value_color(vital_type, current_value))
        self.surface.blit(value_label, 
                         (graph_rect.right - value_label.get_width(), rect.y + 10))
    
    def draw(self):
        """Draw the vital signs monitor on the surface."""
        # Skip if no surface
        if not self.surface or not self.rect:
            return
            
        # Draw based on mode
        if self.compact_mode:
            self.draw_compact()
        else:
            self.draw_standard()
            
    def update_animation(self):
        """Update animation state for smooth transitions."""
        current_time = time.time()
        if current_time - self.last_animation >= self.animation_interval:
            self.last_animation = current_time
            self.graph_position = (self.graph_position + self.animation_speed) % self.history_length
            
    def set_active_vital(self, vital_type: str):
        """
        Set the active vital sign for detailed view.
        
        Args:
            vital_type: Type of vital sign
        """
        if vital_type in self.VITAL_RANGES or vital_type in ["sbp", "dbp"]:
            self.active_vital = vital_type
            
    def toggle_compact_mode(self):
        """Toggle between compact and standard display modes."""
        self.compact_mode = not self.compact_mode
        
    def parse_vitals_from_text(self, text: str):
        """
        Parse vital signs from a text string.
        
        Args:
            text: Text string containing vital sign information
        """
        text = text.lower()
        
        # Blood pressure: Look for patterns like "BP 120/80" or "blood pressure 120/80"
        bp_patterns = [
            r"bp\s*(?:is|of|at)?\s*(\d+)/(\d+)",
            r"blood pressure\s*(?:is|of|at)?\s*(\d+)/(\d+)",
        ]
        
        for pattern in bp_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    sbp = int(match.group(1))
                    dbp = int(match.group(2))
                    self.update_vital("sbp", sbp)
                    self.update_vital("dbp", dbp)
                    break
                except (ValueError, IndexError):
                    pass
                    
        # Heart rate: Look for patterns like "HR 80" or "heart rate 80"
        hr_patterns = [
            r"hr\s*(?:is|of|at)?\s*(\d+)",
            r"heart rate\s*(?:is|of|at)?\s*(\d+)",
            r"pulse\s*(?:is|of|at)?\s*(\d+)",
        ]
        
        for pattern in hr_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    hr = int(match.group(1))
                    self.update_vital("hr", hr)
                    break
                except (ValueError, IndexError):
                    pass
                    
        # Respiratory rate: Look for patterns like "RR 18" or "respiratory rate 18"
        rr_patterns = [
            r"rr\s*(?:is|of|at)?\s*(\d+)",
            r"respiratory rate\s*(?:is|of|at)?\s*(\d+)",
            r"respirations\s*(?:is|of|at)?\s*(\d+)",
        ]
        
        for pattern in rr_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    rr = int(match.group(1))
                    self.update_vital("rr", rr)
                    break
                except (ValueError, IndexError):
                    pass
                    
        # Oxygen saturation: Look for patterns like "O2 sat 98%" or "SpO2 98%"
        spo2_patterns = [
            r"(?:o2|oxygen) sat\s*(?:is|of|at)?\s*(\d+)%?",
            r"spo2\s*(?:is|of|at)?\s*(\d+)%?",
            r"saturation\s*(?:is|of|at)?\s*(\d+)%?",
        ]
        
        for pattern in spo2_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    spo2 = int(match.group(1))
                    self.update_vital("spo2", spo2)
                    break
                except (ValueError, IndexError):
                    pass
                    
        # Temperature: Look for patterns like "temp 37.2" or "temperature 37.2°C"
        temp_patterns = [
            r"temp\s*(?:is|of|at)?\s*(\d+\.?\d*)(?:°c|c)?",
            r"temperature\s*(?:is|of|at)?\s*(\d+\.?\d*)(?:°c|c)?",
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    temp = float(match.group(1))
                    self.update_vital("temp", temp)
                    break
                except (ValueError, IndexError):
                    pass


# Example usage
if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    
    # Create a test window
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Vital Signs Test")
    
    # Create vital signs monitor
    vital_rect = pygame.Rect(50, 50, 700, 500)
    vitals = VitalSignsMonitor(screen, vital_rect)
    
    # Set some test values
    vitals.update_vital("hr", 72)
    vitals.update_vital("sbp", 120)
    vitals.update_vital("dbp", 80)
    vitals.update_vital("rr", 16)
    vitals.update_vital("spo2", 98)
    vitals.update_vital("temp", 37.2)
    
    # Create a clock for consistent frame rate
    clock = pygame.time.Clock()
    
    # Test with parsing from text
    test_inputs = [
        "BP is 125/82, HR 88, RR 20",
        "Patient vitals: heart rate 90, blood pressure 130/85",
        "Oxygen saturation 94%, temperature 38.2°C"
    ]
    
    input_index = 0
    last_input_time = time.time()
    
    # Main loop
    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    vitals.toggle_compact_mode()
                elif event.key == pygame.K_1:
                    vitals.set_active_vital("hr")
                elif event.key == pygame.K_2:
                    vitals.set_active_vital("sbp")
                elif event.key == pygame.K_3:
                    vitals.set_active_vital("rr")
                elif event.key == pygame.K_4:
                    vitals.set_active_vital("spo2")
                elif event.key == pygame.K_5:
                    vitals.set_active_vital("temp")
        
        # Test different inputs every 3 seconds
        current_time = time.time()
        if current_time - last_input_time >= 3.0:
            last_input_time = current_time
            input_text = test_inputs[input_index]
            vitals.parse_vitals_from_text(input_text)
            input_index = (input_index + 1) % len(test_inputs)
            
            # Randomly adjust heart rate for demo animation
            current_hr = vitals.vital_data["hr"]["current"]
            new_hr = current_hr + (2 * (0.5 - (time.time() % 1)))  # Slight oscillation
            vitals.update_vital("hr", new_hr)
        
        # Update animation
        vitals.update_animation()
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw vital signs
        vitals.draw()
        
        # Update display
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(30)
    
    # Clean up
    pygame.quit()
    sys.exit()