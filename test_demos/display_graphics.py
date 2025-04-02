#!/usr/bin/env python3
"""
Graphics Demo for WaveShare 6.25" Display
-----------------------------------------
Displays graphics and animations on the WaveShare screen
This script works without needing any special setup or configuration
"""

import os
import sys
import time
import math
import random
import pygame

def run_demo(duration=30, fullscreen=False):
    """
    Run a graphics demo on the WaveShare screen
    
    Args:
        duration: How long to run the demo in seconds
        fullscreen: Whether to use fullscreen mode
    """
    # Initialize pygame
    pygame.init()
    
    # Get display info
    pygame.display.init()
    display_info = pygame.display.Info()
    
    # Use the WaveShare resolution or fall back to current display
    width, height = 1560, 720  # Default WaveShare 6.25" landscape
    
    # Check if a specific resolution was detected
    if display_info.current_w > 0 and display_info.current_h > 0:
        # If display has portrait dimensions, use portrait mode
        if display_info.current_h > display_info.current_w:
            width, height = 720, 1560  # Portrait mode
        else:
            width, height = display_info.current_w, display_info.current_h
            
    print(f"Using display resolution: {width}x{height}")
    
    # Set up display
    if fullscreen:
        screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
        print("Using fullscreen mode")
    else:
        screen = pygame.display.set_mode((width, height))
        print("Using windowed mode")
    
    pygame.display.set_caption("TCCC Graphics Demo")
    
    # Load font
    try:
        font = pygame.font.Font(None, 60)  # Default font, large size
        small_font = pygame.font.Font(None, 36)
    except Exception:
        print("Failed to load fonts, using fallback")
        # Create a fallback font if pygame's default font fails
        font = pygame.font.SysFont('Arial', 60)
        small_font = pygame.font.SysFont('Arial', 36)
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    
    # Demo setup
    start_time = time.time()
    running = True
    clock = pygame.time.Clock()
    fps = 30
    
    # Animation variables
    circle_radius = 50
    circle_x = width // 2
    circle_y = height // 2
    circle_speed_x = 5
    circle_speed_y = 3
    circle_color = RED
    
    # Text animation variables
    text = "TCCC.ai on WaveShare Display"
    text_y = height // 4
    text_color = WHITE
    
    # Demo loop
    try:
        while running and (time.time() - start_time < duration):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # Change colors with number keys
                    elif event.key == pygame.K_1:
                        circle_color = RED
                    elif event.key == pygame.K_2:
                        circle_color = GREEN
                    elif event.key == pygame.K_3:
                        circle_color = BLUE
                    elif event.key == pygame.K_4:
                        circle_color = YELLOW
                    elif event.key == pygame.K_5:
                        circle_color = CYAN
                    elif event.key == pygame.K_6:
                        circle_color = MAGENTA
                
                # Handle touch input or mouse clicks
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    circle_x, circle_y = pos
            
            # Clear screen
            screen.fill(BLACK)
            
            # Calculate current time
            current_time = time.time() - start_time
            time_left = duration - current_time
            
            # Update circle position
            circle_x += circle_speed_x
            circle_y += circle_speed_y
            
            # Bounce off edges
            if circle_x - circle_radius < 0 or circle_x + circle_radius > width:
                circle_speed_x = -circle_speed_x
                # Change color on bounce
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                circle_color = (r, g, b)
                
            if circle_y - circle_radius < 0 or circle_y + circle_radius > height:
                circle_speed_y = -circle_speed_y
                # Change color on bounce
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                circle_color = (r, g, b)
            
            # Draw a pulsing circle that changes size
            pulse = math.sin(current_time * 3) * 10
            pygame.draw.circle(screen, circle_color, (int(circle_x), int(circle_y)), int(circle_radius + pulse))
            
            # Draw diagonal crossing lines
            pygame.draw.line(screen, RED, (0, 0), (width, height), 3)
            pygame.draw.line(screen, BLUE, (width, 0), (0, height), 3)
            
            # Draw header text
            header_text = font.render("TCCC.ai Display Demo", True, WHITE)
            screen.blit(header_text, (width//2 - header_text.get_width()//2, 20))
            
            # Draw animated text
            text_color = (
                int(128 + 127 * math.sin(current_time * 2)),
                int(128 + 127 * math.sin(current_time * 2 + 2)),
                int(128 + 127 * math.sin(current_time * 2 + 4))
            )
            rendered_text = font.render(text, True, text_color)
            text_x = int(width // 2 - rendered_text.get_width() // 2 + math.sin(current_time) * 100)
            screen.blit(rendered_text, (text_x, text_y))
            
            # Draw time remaining
            time_text = small_font.render(f"Time remaining: {int(time_left)}s", True, WHITE)
            screen.blit(time_text, (width - time_text.get_width() - 20, height - time_text.get_height() - 20))
            
            # Draw instructions
            keys_text = small_font.render("Press ESC to exit, 1-6 to change colors, click/touch to move circle", True, WHITE)
            screen.blit(keys_text, (width//2 - keys_text.get_width()//2, height - 60))
            
            # Draw a sine wave
            wave_points = []
            for x in range(0, width, 5):
                y = int(height // 2 + math.sin(x / 50 + current_time * 5) * 50)
                wave_points.append((x, y))
            
            if len(wave_points) > 1:
                pygame.draw.lines(screen, GREEN, False, wave_points, 3)
            
            # Draw FPS counter
            fps_text = small_font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
            screen.blit(fps_text, (20, 20))
            
            # Update display
            pygame.display.flip()
            
            # Cap at target FPS
            clock.tick(fps)
        
        print("Graphics demo completed successfully")
        return True
    
    except Exception as e:
        print(f"Error in graphics demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    # Parse command line arguments
    fullscreen = "--fullscreen" in sys.argv
    
    # Parse duration if provided
    duration = 30  # Default
    for arg in sys.argv:
        if arg.isdigit():
            duration = int(arg)
            break
    
    run_demo(duration, fullscreen)