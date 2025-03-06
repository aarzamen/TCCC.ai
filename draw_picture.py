#!/usr/bin/env python3
"""
Draw a simple picture on the WaveShare screen
"""
import os
import sys
import pygame
import random
import math

def draw_picture(fullscreen=True):
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
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    
    # Draw the picture
    try:
        # Fill screen with black
        screen.fill(BLACK)
        
        # Draw a white frame around the edge
        pygame.draw.rect(screen, WHITE, (10, 10, width-20, height-20), 5)
        
        # Draw title text
        try:
            font = pygame.font.SysFont('Arial', 72)
            title = font.render("TCCC.ai", True, WHITE)
            screen.blit(title, (width//2 - title.get_width()//2, 30))
        except:
            # Fallback if font doesn't work
            pygame.draw.rect(screen, WHITE, (width//2 - 150, 30, 300, 60), 0)
        
        # Draw a red circle in the center
        center_x, center_y = width//2, height//2
        pygame.draw.circle(screen, RED, (center_x, center_y), 100, 0)
        
        # Draw a blue circle inside the red one
        pygame.draw.circle(screen, BLUE, (center_x, center_y), 70, 0)
        
        # Draw green circle in the center
        pygame.draw.circle(screen, GREEN, (center_x, center_y), 40, 0)
        
        # Draw yellow cross over everything
        pygame.draw.line(screen, YELLOW, (center_x - 200, center_y), (center_x + 200, center_y), 10)
        pygame.draw.line(screen, YELLOW, (center_x, center_y - 200), (center_x, center_y + 200), 10)
        
        # Draw some random squares
        for i in range(10):
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            pos_x = random.randint(50, width - 50)
            pos_y = random.randint(50, height - 50)
            size = random.randint(20, 60)
            
            pygame.draw.rect(screen, color, (pos_x, pos_y, size, size), 0)
        
        # Update display
        pygame.display.flip()
        
        # Wait for key press or click to exit
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    running = False
        
    except Exception as e:
        print(f"Error drawing picture: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    fullscreen = "--windowed" not in sys.argv
    draw_picture(fullscreen)