#!/usr/bin/env python3
"""
Simple image display for WaveShare screen
"""
import os
import sys
import pygame

def show_image(image_path, fullscreen=True):
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
    
    # Load the image
    try:
        image = pygame.image.load(image_path)
        print(f"Loaded image: {image_path}")
        
        # Scale image to fit screen
        img_width, img_height = image.get_size()
        
        # Calculate scale factor to maintain aspect ratio
        width_ratio = width / img_width
        height_ratio = height / img_height
        scale_factor = min(width_ratio, height_ratio)
        
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        
        # Scale image
        image = pygame.transform.scale(image, (new_width, new_height))
        
        # Position image in center
        x = (width - new_width) // 2
        y = (height - new_height) // 2
        
        # Display image
        screen.fill((0, 0, 0))  # Black background
        screen.blit(image, (x, y))
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
        print(f"Error displaying image: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_image.py <image_path> [--windowed]")
        sys.exit(1)
        
    image_path = sys.argv[1]
    fullscreen = "--windowed" not in sys.argv
    
    show_image(image_path, fullscreen)