#!/usr/bin/env python3
"""
Simple WaveShare Display Test
This is a minimal pygame script to test the WaveShare display
"""

import os
import sys
import time
import pygame

# Set the display variable
os.environ['DISPLAY'] = ':0'

# Initialize pygame
pygame.init()

# Set up the display
width, height = 1560, 720  # WaveShare 6.25" in landscape mode
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple WaveShare Test")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Create a font
font = pygame.font.Font(None, 48)

# Main loop
running = True
start_time = time.time()

try:
    while running and time.time() - start_time < 30:  # Run for 30 seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
        # Clear the screen
        screen.fill(BLACK)
        
        # Draw a white rectangle in the middle
        pygame.draw.rect(screen, WHITE, (width//4, height//4, width//2, height//2))
        
        # Draw text
        text = font.render(f"WaveShare Display Test - {int(time.time() - start_time)}s", True, BLUE)
        screen.blit(text, (width//2 - text.get_width()//2, 50))
        
        # Draw a red line from top-left to bottom-right
        pygame.draw.line(screen, RED, (0, 0), (width, height), 5)
        
        # Draw a green line from top-right to bottom-left
        pygame.draw.line(screen, GREEN, (width, 0), (0, height), 5)
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        pygame.time.Clock().tick(30)
        
    print("Display test completed successfully")
except Exception as e:
    print(f"Error: {e}")
finally:
    pygame.quit()
    sys.exit(0)