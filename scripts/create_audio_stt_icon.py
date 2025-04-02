#!/usr/bin/env python3
"""
Create SVG icons for the TCCC Audio-to-Text System
"""

import os
import math
import random
from pathlib import Path

def create_standard_icon(output_path):
    """Create standard icon for audio-to-text system"""
    
    # Define colors
    primary_color = "#3498db"  # Blue
    secondary_color = "#2c3e50"  # Dark blue/grey
    accent_color = "#ecf0f1"  # Light grey
    
    # Create SVG content
    svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="512" height="512" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">

  <!-- Background Circle -->
  <circle cx="256" cy="256" r="250" fill="{primary_color}" />
  
  <!-- Sound Waves -->
  <g fill="none" stroke="{accent_color}" stroke-width="6">
    <circle cx="256" cy="256" r="100" />
    <circle cx="256" cy="256" r="150" />
    <circle cx="256" cy="256" r="200" />
  </g>
  
  <!-- Microphone -->
  <rect x="231" y="136" width="50" height="120" rx="25" fill="{secondary_color}" />
  <path d="M196 256 Q196 300, 256 300 Q316 300, 316 256" 
        stroke="{secondary_color}" stroke-width="12" fill="none" />
  <rect x="246" y="300" width="20" height="50" fill="{secondary_color}" />
  <rect x="206" y="350" width="100" height="10" rx="5" fill="{secondary_color}" />
  
  <!-- Text Elements -->
  <g fill="{accent_color}">
    <rect x="156" y="376" width="200" height="8" rx="4" />
    <rect x="176" y="396" width="160" height="8" rx="4" />
    <rect x="196" y="416" width="120" height="8" rx="4" />
  </g>

</svg>"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Created standard icon: {output_path}")

def create_battlefield_icon(output_path):
    """Create battlefield mode icon for audio-to-text system"""
    
    # Define colors
    primary_color = "#27ae60"  # Green
    secondary_color = "#2c3e50"  # Dark blue/grey
    accent_color = "#ecf0f1"  # Light grey
    highlight_color = "#f39c12"  # Orange
    
    # Create SVG content
    svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="512" height="512" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">

  <!-- Background Circle -->
  <circle cx="256" cy="256" r="250" fill="{primary_color}" />
  
  <!-- Sound Waves (Irregular) -->
  <g fill="none" stroke="{accent_color}" stroke-width="6">
    <path d="M256,156 Q356,181 356,256 Q356,331 256,356 Q156,331 156,256 Q156,181 256,156" />
    <path d="M256,106 Q406,146 406,256 Q406,366 256,406 Q106,366 106,256 Q106,146 256,106" />
  </g>
  
  <!-- Battle Elements -->
  <g fill="{highlight_color}">
    <circle cx="156" cy="176" r="15" />
    <circle cx="356" cy="176" r="15" />
    <circle cx="156" cy="336" r="15" />
    <circle cx="356" cy="336" r="15" />
  </g>
  
  <!-- Microphone (Military Style) -->
  <rect x="231" y="136" width="50" height="120" rx="10" fill="{secondary_color}" />
  <path d="M196 256 Q196 300, 256 300 Q316 300, 316 256" 
        stroke="{secondary_color}" stroke-width="12" fill="none" />
  <rect x="246" y="300" width="20" height="50" fill="{secondary_color}" />
  <rect x="206" y="350" width="100" height="10" rx="5" fill="{secondary_color}" />
  
  <!-- Text Elements with Filter Effect -->
  <defs>
    <filter id="noise" x="0" y="0" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="2" result="noise"/>
      <feDisplacementMap in="SourceGraphic" in2="noise" scale="5" xChannelSelector="R" yChannelSelector="G"/>
    </filter>
  </defs>
  
  <g fill="{accent_color}" filter="url(#noise)">
    <rect x="156" y="376" width="200" height="8" rx="4" />
    <rect x="176" y="396" width="160" height="8" rx="4" />
    <rect x="196" y="416" width="120" height="8" rx="4" />
  </g>

</svg>"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Created battlefield icon: {output_path}")

def create_advanced_icon(output_path):
    """Create advanced animated icon for audio-to-text system"""
    
    # Define colors
    primary_color = "#2ecc71"  # Green
    secondary_color = "#2c3e50"  # Dark blue/grey
    accent_color = "#ecf0f1"  # Light grey
    highlight_color = "#3498db"  # Blue
    
    # Create SVG content with animation
    svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="512" height="512" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">

  <!-- Definitions -->
  <defs>
    <!-- Pulse Animation -->
    <radialGradient id="pulse" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
      <stop offset="0%" stop-color="{primary_color}" stop-opacity="0.8">
        <animate attributeName="stop-opacity" values="0.8;0.2;0.8" dur="4s" repeatCount="indefinite" />
      </stop>
      <stop offset="100%" stop-color="{primary_color}" stop-opacity="0.2">
        <animate attributeName="stop-opacity" values="0.2;0.05;0.2" dur="4s" repeatCount="indefinite" />
      </stop>
    </radialGradient>
    
    <!-- Wave Animation -->
    <filter id="wave" x="-20%" y="-20%" width="140%" height="140%">
      <feTurbulence type="fractalNoise" baseFrequency="0.01" numOctaves="1" result="noise">
        <animate attributeName="baseFrequency" values="0.01;0.02;0.01" dur="10s" repeatCount="indefinite" />
      </feTurbulence>
      <feDisplacementMap in="SourceGraphic" in2="noise" scale="5" xChannelSelector="R" yChannelSelector="G" />
    </filter>
  </defs>

  <!-- Background Circle with Pulse Effect -->
  <circle cx="256" cy="256" r="250" fill="url(#pulse)" />
  
  <!-- Sound Waves with Animation -->
  <g fill="none" stroke="{accent_color}" stroke-width="3" opacity="0.7">
    <circle cx="256" cy="256" r="100">
      <animate attributeName="r" values="100;110;100" dur="2s" repeatCount="indefinite" />
      <animate attributeName="opacity" values="0.7;0.3;0.7" dur="2s" repeatCount="indefinite" />
    </circle>
    <circle cx="256" cy="256" r="150">
      <animate attributeName="r" values="150;160;150" dur="2s" begin="0.5s" repeatCount="indefinite" />
      <animate attributeName="opacity" values="0.7;0.3;0.7" dur="2s" begin="0.5s" repeatCount="indefinite" />
    </circle>
    <circle cx="256" cy="256" r="200">
      <animate attributeName="r" values="200;210;200" dur="2s" begin="1s" repeatCount="indefinite" />
      <animate attributeName="opacity" values="0.7;0.3;0.7" dur="2s" begin="1s" repeatCount="indefinite" />
    </circle>
  </g>
  
  <!-- Microphone -->
  <rect x="231" y="136" width="50" height="120" rx="25" fill="{secondary_color}" />
  <path d="M196 256 Q196 300, 256 300 Q316 300, 316 256" 
        stroke="{secondary_color}" stroke-width="12" fill="none" />
  <rect x="246" y="300" width="20" height="50" fill="{secondary_color}" />
  <rect x="206" y="350" width="100" height="10" rx="5" fill="{secondary_color}" />
  
  <!-- Text to Speech Effect -->
  <g fill="{highlight_color}" filter="url(#wave)">
    <rect x="156" y="376" width="200" height="8" rx="4" />
    <rect x="176" y="396" width="160" height="8" rx="4" />
    <rect x="196" y="416" width="120" height="8" rx="4" />
  </g>

</svg>"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Created advanced icon: {output_path}")

def main():
    """Create all icons for the TCCC Audio-to-Text System"""
    
    # Create images directory if it doesn't exist
    image_dir = Path("images")
    image_dir.mkdir(exist_ok=True)
    
    # Create icons
    create_standard_icon(image_dir / "audio_stt_blue.svg")
    create_battlefield_icon(image_dir / "audio_stt_battlefield.svg")
    create_advanced_icon(image_dir / "audio_stt_animated.svg")
    
    print(f"\nAll icons created successfully in {image_dir} directory")
    print("Use these icons with desktop shortcuts for the TCCC Audio-to-Text System")
    print("The animated icon (audio_stt_animated.svg) is best viewed in a modern web browser or SVG viewer")

if __name__ == "__main__":
    main()