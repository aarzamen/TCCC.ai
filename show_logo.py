#!/usr/bin/env python3
"""
Display TCCC logo on the WaveShare screen
"""
import os
import sys
import pygame
import math
import time

def show_logo(fullscreen=True, duration=0):
    # Set display variable
    os.environ['DISPLAY'] = ':0'
    
    # Initialize pygame
    pygame.init()
    
    # Set up display
    display_info = pygame.display.Info()
    if display_info.current_w > 0 and display_info.current_h > 0:
        width, height = display_info.current_w, display_info.current_h
    else:
        # WaveShare 6.25" default resolution
        width, height = 1560, 720
    
    print(f"Using resolution: {width}x{height}")
    
    if fullscreen:
        screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((width, height))
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 50, 255)
    LIGHT_BLUE = (100, 150, 255)
    
    # Try to load logo from file first
    logo = None
    logo_paths = [
        os.path.join(os.path.dirname(__file__), "images", "blue_logo.png"),
        os.path.join(os.path.dirname(__file__), "images", "green_logo.png")
    ]
    
    for path in logo_paths:
        if os.path.exists(path):
            try:
                logo = pygame.image.load(path)
                # Scale to reasonable size
                logo_size = min(width, height) // 2
                logo = pygame.transform.scale(logo, (logo_size, logo_size))
                print(f"Loaded logo from {path}")
                break
            except Exception as e:
                print(f"Failed to load logo {path}: {e}")
    
    try:
        # Main display loop
        start_time = time.time()
        running = True
        
        while running:
            # Check if duration has passed
            if duration > 0 and time.time() - start_time > duration:
                running = False
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    running = False
            
            # Fill screen with black
            screen.fill(BLACK)
            
            # Common measurements
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            
            # If we have a logo image, display it
            if logo:
                logo_x = width // 2 - logo.get_width() // 2
                logo_y = height // 2 - logo.get_height() // 2
                screen.blit(logo, (logo_x, logo_y))
            else:
                # Draw a simple TCCC logo
                # Draw blue circle
                pygame.draw.circle(screen, BLUE, (center_x, center_y), radius, 0)
                
                # Draw white cross (medical symbol)
                cross_width = radius // 4
                pygame.draw.rect(screen, WHITE, 
                                (center_x - cross_width // 2, 
                                 center_y - radius // 1.5, 
                                 cross_width, 
                                 radius * 1.2), 0)
                
                pygame.draw.rect(screen, WHITE, 
                                (center_x - radius // 1.5, 
                                 center_y - cross_width // 2, 
                                 radius * 1.2, 
                                 cross_width), 0)
                
                # Draw pulsing highlight effect
                pulse = abs(math.sin(time.time() * 2)) * 30 + 10
                pygame.draw.circle(screen, LIGHT_BLUE, (center_x, center_y), radius + pulse, 3)
            
            # Draw TCCC.ai text
            try:
                font = pygame.font.SysFont('Arial', 72)
                title = font.render("TCCC.ai", True, WHITE)
                screen.blit(title, (width//2 - title.get_width()//2, 
                                    height//2 + radius + 40))
                
                # Draw smaller subtitle
                font_small = pygame.font.SysFont('Arial', 36)
                subtitle = font_small.render("Tactical Combat Casualty Care", True, WHITE)
                screen.blit(subtitle, (width//2 - subtitle.get_width()//2, 
                                       height//2 + radius + 120))
            except Exception as e:
                print(f"Error displaying text: {e}")
            
            # Update display
            pygame.display.flip()
            
            # Short delay to reduce CPU usage
            time.sleep(0.01)
        
    except Exception as e:
        print(f"Error displaying logo: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    fullscreen = "--windowed" not in sys.argv
    
    # Parse duration if provided (0 means display until key/click)
    duration = 0
    for arg in sys.argv:
        if arg.isdigit():
            duration = int(arg)
            break
    
    show_logo(fullscreen, duration)