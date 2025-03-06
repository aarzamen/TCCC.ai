#!/usr/bin/env python3
"""
Simple Image Display for WaveShare 6.25" Display
------------------------------------------------
Displays images on the WaveShare screen with minimal dependencies
"""

import os
import sys
import time
import pygame

def display_image(image_path, fullscreen=False, duration=10):
    """
    Display an image on the WaveShare screen
    
    Args:
        image_path: Path to the image file to display
        fullscreen: Whether to use fullscreen mode
        duration: How long to display the image in seconds
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
    
    pygame.display.set_caption("TCCC Image Viewer")
    
    # Load image
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return False
            
        print(f"Loading image: {image_path}")
        image = pygame.image.load(image_path)
        
        # Scale image to fit screen while maintaining aspect ratio
        img_width, img_height = image.get_size()
        aspect_ratio = img_width / img_height
        
        # Calculate new dimensions that fit within the screen
        new_width = width
        new_height = int(new_width / aspect_ratio)
        
        if new_height > height:
            new_height = height
            new_width = int(new_height * aspect_ratio)
        
        # Scale the image
        image = pygame.transform.smoothscale(image, (new_width, new_height))
        
        # Calculate position to center the image
        pos_x = (width - new_width) // 2
        pos_y = (height - new_height) // 2
        
        print(f"Image scaled to {new_width}x{new_height} and centered")
        
        # Main loop
        start_time = time.time()
        running = True
        
        while running and (time.time() - start_time < duration):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Clear screen with black
            screen.fill((0, 0, 0))
            
            # Draw the image centered
            screen.blit(image, (pos_x, pos_y))
            
            # Update the display
            pygame.display.flip()
            
            # Short sleep to reduce CPU usage
            time.sleep(0.01)
        
        print("Image displayed successfully")
        return True
    
    except Exception as e:
        print(f"Error displaying image: {e}")
        return False
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python display_image.py <image_path> [duration] [--fullscreen]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    duration = 10  # Default duration
    fullscreen = False
    
    # Parse additional arguments
    for arg in sys.argv[2:]:
        if arg == "--fullscreen":
            fullscreen = True
        elif arg.isdigit():
            duration = int(arg)
    
    # Display the image
    success = display_image(image_path, fullscreen, duration)
    sys.exit(0 if success else 1)