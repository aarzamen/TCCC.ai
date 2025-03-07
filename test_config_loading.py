#!/usr/bin/env python3
"""Test script for loading Jetson optimizer configuration"""

import os
import sys
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "jetson_optimizer.yaml")

print(f"Testing config loading from: {CONFIG_PATH}")

try:
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)
        
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Successfully loaded config file!")
    print(f"Config contains {len(config)} top-level keys")
    
    # Print some key config values
    print("\nKey configuration values:")
    print(f"- Power mode: {config.get('power_mode', 'Not set')}")
    print(f"- CUDA enabled: {config.get('cuda_enabled', 'Not set')}")
    print(f"- Whisper model size: {config.get('models', {}).get('whisper', {}).get('model_size', 'Not set')}")
    print(f"- Whisper compute type: {config.get('models', {}).get('whisper', {}).get('compute_type', 'Not set')}")
    print(f"- LLM quantization: {config.get('models', {}).get('llm', {}).get('quantization', 'Not set')}")
    
    # Check profiles
    profiles = config.get('profiles', {})
    print(f"\nFound {len(profiles)} performance profiles:")
    for name, profile in profiles.items():
        print(f"- {name}: {profile.get('description', 'No description')}")
        
    print("\nConfig loading test completed successfully!")
    sys.exit(0)
    
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)