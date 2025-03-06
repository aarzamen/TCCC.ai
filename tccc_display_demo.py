#!/usr/bin/env python3
"""
TCCC.ai Display Demo for WaveShare 6.25" Display
-----------------------------------------------
Demonstrates the TCCC application interface on the WaveShare display
"""

import os
import sys
import time
import math
import datetime
import random
import pygame

class TCCCDisplayDemo:
    """Simple demo of the TCCC display interface"""
    
    def __init__(self, fullscreen=False):
        """Initialize the demo"""
        # Initialize pygame
        pygame.init()
        
        # Get available display info
        pygame.display.init()
        info = pygame.display.Info()
        
        # Set up display dimensions - detect WaveShare if possible
        self.width = 1560  # Default WaveShare 6.25" landscape
        self.height = 720
        
        # Try to detect actual display dimensions
        if info.current_w > 0 and info.current_h > 0:
            print(f"Detected display: {info.current_w}x{info.current_h}")
            self.width = info.current_w
            self.height = info.current_h
        
        # Set up display
        self.fullscreen = fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
            print(f"Using fullscreen mode: {self.width}x{self.height}")
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            print(f"Using windowed mode: {self.width}x{self.height}")
        
        pygame.display.set_caption("TCCC.ai Field Assistant")
        
        # Load fonts with fallbacks
        try:
            self.fonts = {
                'small': pygame.font.Font(None, 24),
                'medium': pygame.font.Font(None, 36),
                'large': pygame.font.Font(None, 48),
                'title': pygame.font.Font(None, 60),
            }
        except Exception as e:
            print(f"Error loading fonts: {e}")
            # Fallback to system fonts
            self.fonts = {
                'small': pygame.font.SysFont('Arial', 24),
                'medium': pygame.font.SysFont('Arial', 36),
                'large': pygame.font.SysFont('Arial', 48),
                'title': pygame.font.SysFont('Arial', 60),
            }
        
        # Set up colors
        self.colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'blue': (0, 0, 255),
            'light_blue': (100, 100, 255),
            'red': (255, 0, 0),
            'green': (0, 180, 0),
            'yellow': (255, 255, 0),
            'gray': (100, 100, 100),
            'dark_gray': (50, 50, 50),
        }
        
        # Load logo if available (fallback to drawing one)
        self.logo = None
        logo_paths = [
            os.path.join(os.path.dirname(__file__), "images/blue_logo.png"),
            os.path.join(os.path.dirname(__file__), "images/green_logo.png"),
        ]
        
        for path in logo_paths:
            if os.path.exists(path):
                try:
                    self.logo = pygame.image.load(path)
                    self.logo = pygame.transform.scale(self.logo, (80, 80))
                    print(f"Loaded logo from {path}")
                    break
                except Exception as e:
                    print(f"Failed to load logo from {path}: {e}")
        
        # Demo data for display
        self.transcription = [
            "Starting recording...",
            "Patient has a gunshot wound to the left thigh with arterial bleeding.",
            "Applying a tourniquet 2 inches above the wound site.",
            "Tourniquet applied at 14:32 local time.",
            "Checking for other injuries...",
            "Patient also has shrapnel wounds to right arm, applying pressure bandage.",
            "Need to administer morphine for pain management.",
            "Administering 10mg morphine IV.",
            "Vital signs: HR 110, BP 90/60, RR 22, O2 94%.",
            "Patient stabilized, preparing for evacuation."
        ]
        
        self.events = [
            {"time": "14:30", "description": "Initial assessment started"},
            {"time": "14:31", "description": "GSW identified - left thigh, arterial bleeding"},
            {"time": "14:32", "description": "Tourniquet applied to left thigh"},
            {"time": "14:33", "description": "Shrapnel wounds identified - right arm"},
            {"time": "14:34", "description": "Pressure bandage applied to right arm"},
            {"time": "14:35", "description": "Morphine 10mg administered IV"},
            {"time": "14:37", "description": "Patient stabilized, ready for evacuation"},
            {"time": "14:40", "description": "Evacuation team notified, ETA 10 minutes"},
        ]
        
        self.card_data = {
            "name": "John Doe",
            "rank": "SGT",
            "unit": "1st Battalion, 3rd Marines",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "time": "14:30",
            "mechanism_of_injury": "GSW + Shrapnel",
            "injuries": "GSW left thigh with arterial bleeding, shrapnel wounds to right arm",
            "vital_signs": "HR 110, BP 90/60, RR 22, O2 94%",
            "treatment_given": "Tourniquet to left thigh at 14:32, pressure dressing to right arm",
            "medications": "Morphine 10mg IV at 14:35",
            "evacuation_priority": "Urgent"
        }
        
        # Display state
        self.current_view = "live"  # 'live' or 'card'
        self.active = True
        self.clock = pygame.time.Clock()
        
    def draw_live_view(self):
        """Draw the live transcription view"""
        # Draw header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(self.screen, self.colors['blue'], header_rect)
        
        # Draw logo and title
        if self.logo:
            self.screen.blit(self.logo, (10, 5))
            title_x = 100
        else:
            # Draw a simple circle as logo placeholder
            pygame.draw.circle(self.screen, self.colors['light_blue'], (40, 30), 25)
            title_x = 80
            
        title_text = self.fonts['title'].render("TCCC.ai FIELD ASSISTANT", True, self.colors['white'])
        self.screen.blit(title_text, (title_x, 10))
        
        # Draw current time
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        time_text = self.fonts['medium'].render(current_time, True, self.colors['white'])
        self.screen.blit(time_text, (self.width - time_text.get_width() - 20, 20))
        
        # Draw column layout - split screen into three sections
        col1_width = int(self.width * 0.38)  # Transcription
        col2_width = int(self.width * 0.34)  # Events
        col3_width = self.width - col1_width - col2_width  # Card preview
        
        # Draw column dividers
        pygame.draw.line(self.screen, self.colors['gray'], 
                         (col1_width, 60), (col1_width, self.height - 50), 2)
        pygame.draw.line(self.screen, self.colors['gray'], 
                         (col1_width + col2_width, 60), (col1_width + col2_width, self.height - 50), 2)
        
        # Draw column headers
        col1_title = self.fonts['large'].render("SPEECH TRANSCRIPTION", True, self.colors['yellow'])
        col2_title = self.fonts['large'].render("SIGNIFICANT EVENTS", True, self.colors['yellow'])
        col3_title = self.fonts['large'].render("TCCC CARD PREVIEW", True, self.colors['yellow'])
        
        self.screen.blit(col1_title, (col1_width//2 - col1_title.get_width()//2, 70))
        self.screen.blit(col2_title, (col1_width + col2_width//2 - col2_title.get_width()//2, 70))
        self.screen.blit(col3_title, (col1_width + col2_width + col3_width//2 - col3_title.get_width()//2, 70))
        
        # Draw transcription (left column)
        y_pos = 120
        max_visible = 8
        for i, text in enumerate(self.transcription[-max_visible:]):
            # Add animation for the most recent item
            color = self.colors['white']
            if i == len(self.transcription[-max_visible:]) - 1:
                # Make the most recent transcript entry pulse
                pulse = int(128 + 127 * math.sin(time.time() * 3))
                color = (pulse, pulse, 255)
                
            # Wrap text to fit column
            words = text.split(' ')
            line = ""
            wrapped_lines = []
            for word in words:
                test_line = line + word + " "
                text_width = self.fonts['medium'].size(test_line)[0]
                if text_width < col1_width - 30:
                    line = test_line
                else:
                    wrapped_lines.append(line)
                    line = word + " "
            wrapped_lines.append(line)
            
            # Draw wrapped text
            for line in wrapped_lines:
                text_surface = self.fonts['medium'].render(line, True, color)
                self.screen.blit(text_surface, (15, y_pos))
                y_pos += 30
            
            # Add spacing between transcript items
            y_pos += 10
        
        # Draw events (middle column)
        y_pos = 120
        for event in self.events[-8:]:  # Show last 8 events
            # Draw timestamp
            time_text = self.fonts['small'].render(event['time'], True, self.colors['green'])
            self.screen.blit(time_text, (col1_width + 15, y_pos))
            
            # Draw description - wrap text if needed
            desc = event['description']
            words = desc.split(' ')
            line = ""
            wrapped_lines = []
            for word in words:
                test_line = line + word + " "
                text_width = self.fonts['medium'].size(test_line)[0]
                if text_width < col2_width - 30:
                    line = test_line
                else:
                    wrapped_lines.append(line)
                    line = word + " "
            wrapped_lines.append(line)
            
            # Draw wrapped description
            y_pos += 25
            for line in wrapped_lines:
                desc_text = self.fonts['medium'].render(line, True, self.colors['white'])
                self.screen.blit(desc_text, (col1_width + 30, y_pos))
                y_pos += 30
            
            # Add spacing between events
            y_pos += 15
        
        # Draw card preview (right column)
        card_x = col1_width + col2_width + 20
        card_y = 120
        
        # Draw stick figure
        figure_x = card_x + col3_width // 2
        figure_y = card_y + 50
        
        # Head
        pygame.draw.circle(self.screen, self.colors['white'], (figure_x, figure_y), 20, 2)
        # Body
        pygame.draw.line(self.screen, self.colors['white'], (figure_x, figure_y + 20), (figure_x, figure_y + 80), 2)
        # Arms
        pygame.draw.line(self.screen, self.colors['white'], (figure_x, figure_y + 40), (figure_x - 40, figure_y + 30), 2)
        pygame.draw.line(self.screen, self.colors['white'], (figure_x, figure_y + 40), (figure_x + 40, figure_y + 30), 2)
        # Legs
        pygame.draw.line(self.screen, self.colors['white'], (figure_x, figure_y + 80), (figure_x - 30, figure_y + 120), 2)
        pygame.draw.line(self.screen, self.colors['white'], (figure_x, figure_y + 80), (figure_x + 30, figure_y + 120), 2)
        
        # Draw injuries - red circles
        pygame.draw.circle(self.screen, self.colors['red'], (figure_x - 20, figure_y + 100), 8)  # Left thigh
        pygame.draw.circle(self.screen, self.colors['red'], (figure_x + 35, figure_y + 30), 5)  # Right arm
        
        # Draw card fields
        field_y = figure_y + 140
        fields = [
            ("Name:", self.card_data["name"]),
            ("Injuries:", self.card_data["injuries"]),
            ("Treatment:", self.card_data["treatment_given"])
        ]
        
        for label, value in fields:
            # Draw label
            label_text = self.fonts['medium'].render(label, True, self.colors['yellow'])
            self.screen.blit(label_text, (card_x, field_y))
            
            # Wrap value text
            words = value.split(' ')
            line = ""
            wrapped_lines = []
            max_width = col3_width - 30
            for word in words:
                test_line = line + word + " "
                text_width = self.fonts['small'].size(test_line)[0]
                if text_width < max_width:
                    line = test_line
                else:
                    wrapped_lines.append(line)
                    line = word + " "
            wrapped_lines.append(line)
            
            # Draw wrapped value
            field_y += 30
            for line in wrapped_lines:
                value_text = self.fonts['small'].render(line, True, self.colors['white'])
                self.screen.blit(value_text, (card_x + 20, field_y))
                field_y += 25
            
            field_y += 10
        
        # Draw footer
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(self.screen, self.colors['gray'], footer_rect)
        
        footer_text = self.fonts['medium'].render("Tap here or press 'T' for TCCC Card View", True, self.colors['black'])
        self.screen.blit(footer_text, (self.width//2 - footer_text.get_width()//2, self.height - 35))
    
    def draw_card_view(self):
        """Draw the TCCC casualty card view"""
        # Draw card header
        header_rect = pygame.Rect(0, 0, self.width, 60)
        pygame.draw.rect(self.screen, self.colors['red'], header_rect)
        
        title_text = self.fonts['title'].render("TCCC CASUALTY CARD (DD FORM 1380)", True, self.colors['white'])
        subtitle_text = self.fonts['small'].render("MEDICAL RECORD - SUPPLEMENTAL MEDICAL DATA", True, self.colors['white'])
        
        self.screen.blit(title_text, (self.width//2 - title_text.get_width()//2, 5))
        self.screen.blit(subtitle_text, (self.width//2 - subtitle_text.get_width()//2, 40))
        
        # Draw card in two sections
        diagram_width = int(self.width * 0.35)  # Left section for diagram
        data_width = self.width - diagram_width  # Right section for data
        
        # Draw divider
        pygame.draw.line(self.screen, self.colors['gray'], 
                         (diagram_width, 60), (diagram_width, self.height - 50), 2)
        
        # Left section: anatomical diagram
        diagram_title = self.fonts['large'].render("ANATOMICAL DIAGRAM", True, self.colors['yellow'])
        self.screen.blit(diagram_title, (diagram_width//2 - diagram_title.get_width()//2, 70))
        
        # Draw diagram panel
        diagram_rect = pygame.Rect(20, 100, diagram_width - 40, self.height - 160)
        pygame.draw.rect(self.screen, self.colors['dark_gray'], diagram_rect)
        pygame.draw.rect(self.screen, self.colors['white'], diagram_rect, 2)
        
        # Draw stick figure
        figure_x = diagram_width // 2
        figure_y = 180
        figure_scale = 3
        
        # Head
        pygame.draw.circle(self.screen, self.colors['white'], (figure_x, figure_y), 20 * figure_scale, 3)
        # Body
        pygame.draw.line(self.screen, self.colors['white'], 
                         (figure_x, figure_y + 20 * figure_scale), 
                         (figure_x, figure_y + 80 * figure_scale), 3)
        # Arms
        pygame.draw.line(self.screen, self.colors['white'], 
                         (figure_x, figure_y + 30 * figure_scale), 
                         (figure_x - 40 * figure_scale, figure_y + 25 * figure_scale), 3)
        pygame.draw.line(self.screen, self.colors['white'], 
                         (figure_x, figure_y + 30 * figure_scale), 
                         (figure_x + 40 * figure_scale, figure_y + 25 * figure_scale), 3)
        # Legs
        pygame.draw.line(self.screen, self.colors['white'], 
                         (figure_x, figure_y + 80 * figure_scale), 
                         (figure_x - 30 * figure_scale, figure_y + 130 * figure_scale), 3)
        pygame.draw.line(self.screen, self.colors['white'], 
                         (figure_x, figure_y + 80 * figure_scale), 
                         (figure_x + 30 * figure_scale, figure_y + 130 * figure_scale), 3)
        
        # Draw injuries - with animation
        pulse_size = math.sin(time.time() * 3) * 3
        # Left thigh wound (major)
        pygame.draw.circle(self.screen, self.colors['red'], 
                          (figure_x - 20 * figure_scale, figure_y + 100 * figure_scale), 
                          10 * figure_scale + pulse_size, 0)
        pygame.draw.circle(self.screen, self.colors['white'], 
                          (figure_x - 20 * figure_scale, figure_y + 100 * figure_scale), 
                          10 * figure_scale + pulse_size, 2)
        
        # Right arm wound (minor)
        pygame.draw.circle(self.screen, self.colors['red'], 
                          (figure_x + 35 * figure_scale, figure_y + 25 * figure_scale), 
                          5 * figure_scale + pulse_size, 0)
        pygame.draw.circle(self.screen, self.colors['white'], 
                          (figure_x + 35 * figure_scale, figure_y + 25 * figure_scale), 
                          5 * figure_scale + pulse_size, 2)
        
        # Shrapnel marks
        for i in range(3):
            x_offset = 30 + i * 8
            y_offset = 25 + i * 3
            pygame.draw.line(self.screen, self.colors['red'], 
                            (figure_x + x_offset * figure_scale, figure_y + y_offset * figure_scale - 5), 
                            (figure_x + x_offset * figure_scale + 5, figure_y + y_offset * figure_scale + 5), 2)
            pygame.draw.line(self.screen, self.colors['red'], 
                            (figure_x + x_offset * figure_scale + 5, figure_y + y_offset * figure_scale - 5), 
                            (figure_x + x_offset * figure_scale, figure_y + y_offset * figure_scale + 5), 2)
        
        # Right section: patient data
        info_title = self.fonts['large'].render("PATIENT INFORMATION", True, self.colors['yellow'])
        self.screen.blit(info_title, (diagram_width + (data_width//2) - info_title.get_width()//2, 70))
        
        # Split data section into two columns
        col_width = data_width // 2
        col1_x = diagram_width + 20
        col2_x = diagram_width + col_width + 10
        
        # Column 1: Basic info
        fields_col1 = [
            ("Name:", self.card_data["name"]),
            ("Rank:", self.card_data["rank"]),
            ("Unit:", self.card_data["unit"]),
            ("Date:", self.card_data["date"]),
            ("Time:", self.card_data["time"]),
            ("Mechanism:", self.card_data["mechanism_of_injury"])
        ]
        
        # Column 2: Medical info
        fields_col2 = [
            ("Injuries:", self.card_data["injuries"]),
            ("Vital Signs:", self.card_data["vital_signs"]),
            ("Treatment:", self.card_data["treatment_given"]),
            ("Medications:", self.card_data["medications"]),
            ("Evacuation:", self.card_data["evacuation_priority"])
        ]
        
        # Draw column 1
        y_pos = 120
        for label, value in fields_col1:
            # Draw label
            label_text = self.fonts['medium'].render(label, True, self.colors['yellow'])
            self.screen.blit(label_text, (col1_x, y_pos))
            
            # Draw value - single line for basic info
            value_text = self.fonts['medium'].render(value, True, self.colors['white'])
            self.screen.blit(value_text, (col1_x + 100, y_pos))
            
            y_pos += 45
        
        # Draw column 2
        y_pos = 120
        for label, value in fields_col2:
            # Draw label
            label_text = self.fonts['medium'].render(label, True, self.colors['yellow'])
            self.screen.blit(label_text, (col2_x, y_pos))
            
            # Wrap text for medical info fields
            words = value.split(' ')
            line = ""
            wrapped_lines = []
            max_width = col_width - 30
            for word in words:
                test_line = line + word + " "
                text_width = self.fonts['medium'].size(test_line)[0]
                if text_width < max_width:
                    line = test_line
                else:
                    wrapped_lines.append(line)
                    line = word + " "
            wrapped_lines.append(line)
            
            # Draw wrapped text
            value_y = y_pos + 35
            for line in wrapped_lines:
                value_text = self.fonts['medium'].render(line, True, self.colors['white'])
                self.screen.blit(value_text, (col2_x + 20, value_y))
                value_y += 35
            
            y_pos = value_y + 15
        
        # Draw footer
        footer_rect = pygame.Rect(0, self.height - 50, self.width, 50)
        pygame.draw.rect(self.screen, self.colors['gray'], footer_rect)
        
        footer_text = self.fonts['medium'].render("Tap here or press 'T' for Live View", True, self.colors['black'])
        self.screen.blit(footer_text, (self.width//2 - footer_text.get_width()//2, self.height - 35))
    
    def simulate_new_data(self):
        """Simulate new transcription and events data coming in"""
        # Only add new data occasionally to simulate real-time data
        if random.random() < 0.05:  # 5% chance each frame
            # Add a new transcription
            new_phrases = [
                "Patient's blood pressure is stabilizing at 100/70.",
                "Pulse rate decreasing to 90 BPM.",
                "Respiration is regular at 18 per minute.",
                "Evacuation transport is 5 minutes out.",
                "Preparing patient for transport.",
                "Rechecking tourniquet, still secure.",
                "Administering additional fluids via IV.",
                "Applying combat gauze to secondary wound.",
                "Splinting right arm to stabilize injury.",
                "Checking pupils, equal and reactive."
            ]
            self.transcription.append(random.choice(new_phrases))
            
            # Keep reasonable history length
            if len(self.transcription) > 20:
                self.transcription = self.transcription[-20:]
            
            # 30% chance to add a new event when transcription is added
            if random.random() < 0.3:
                # Generate current time plus some random minutes
                event_time = datetime.datetime.now() + datetime.timedelta(minutes=random.randint(1, 5))
                time_str = event_time.strftime("%H:%M")
                
                new_events = [
                    f"Vital signs updated: HR 90, BP 100/70",
                    f"IV fluids administered",
                    f"Transport ETA {random.randint(3, 10)} minutes",
                    f"Secondary assessment completed",
                    f"Splint applied to right arm",
                    f"Combat gauze applied to secondary wound",
                    f"Patient prepped for transport",
                    f"Tourniquet rechecked, condition stable"
                ]
                
                self.events.append({
                    "time": time_str,
                    "description": random.choice(new_events)
                })
                
                # Keep reasonable history length
                if len(self.events) > 15:
                    self.events = self.events[-15:]
    
    def run(self, duration=60):
        """
        Run the TCCC display demo
        
        Args:
            duration: How long to run in seconds (0 for indefinite)
        """
        start_time = time.time()
        running = True
        
        try:
            while running:
                # Check for exit conditions
                if duration > 0 and time.time() - start_time > duration:
                    running = False
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_t:
                            # Toggle between live and card views
                            self.current_view = "card" if self.current_view == "live" else "live"
                    
                    # Handle touch/click for view switching
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Check if click is in the footer area
                        if event.pos[1] > self.height - 50:
                            # Toggle view
                            self.current_view = "card" if self.current_view == "live" else "live"
                
                # Update simulation
                self.simulate_new_data()
                
                # Clear screen
                self.screen.fill(self.colors['black'])
                
                # Draw appropriate view
                if self.current_view == "live":
                    self.draw_live_view()
                else:
                    self.draw_card_view()
                
                # Update display
                pygame.display.flip()
                
                # Cap at 30 FPS
                self.clock.tick(30)
            
            print("TCCC display demo completed")
            return True
            
        except Exception as e:
            print(f"Error in TCCC display demo: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            pygame.quit()

if __name__ == "__main__":
    # Parse command line arguments
    fullscreen = "--fullscreen" in sys.argv
    
    # Parse duration if provided (0 means run indefinitely)
    duration = 0
    for arg in sys.argv:
        if arg.isdigit():
            duration = int(arg)
            break
    
    # Run the demo
    demo = TCCCDisplayDemo(fullscreen=fullscreen)
    demo.run(duration)