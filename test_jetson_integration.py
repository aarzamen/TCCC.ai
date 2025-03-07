#!/usr/bin/env python3
"""
Simple test script for Jetson Integration module.
This tests basic functionality of the Jetson integration classes.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import optimizer modules directly with error handling
try:
    from src.tccc.utils.jetson_optimizer import JetsonOptimizer
    print("Successfully imported JetsonOptimizer module")
except Exception as e:
    print(f"Error importing JetsonOptimizer: {e}")
    sys.exit(1)

# Test basic optimizer functionality
try:
    print("\nTesting JetsonOptimizer:")
    # Initialize with auto_setup=False to avoid potential blocking operations
    optimizer = JetsonOptimizer(auto_setup=False)
    print(f"- Running on Jetson platform: {optimizer.is_jetson}")
    print(f"- Optimizer initialized successfully")
    
    # Get optimal settings without full environment setup
    suggestions = optimizer.suggest_optimal_settings()
    print("\nRecommended settings:")
    print(f"- Whisper model: {suggestions['whisper_settings']['suggested_model']}")
    print(f"- Compute type: {suggestions['whisper_settings']['compute_type']}")
    print(f"- LLM quantization: {suggestions['llm_settings']['quantization']}")
    print(f"- Available memory: {suggestions['hardware_detected']['available_memory_gb']:.2f} GB")
except Exception as e:
    print(f"Error testing JetsonOptimizer: {e}")
    sys.exit(1)

# Test JetsonIntegration if available
try:
    from src.tccc.utils.jetson_integration import JetsonIntegration
    print("\nSuccessfully imported JetsonIntegration module")
    
    print("\nTesting JetsonIntegration:")
    # Initialize with auto_setup=False to avoid blocking operations
    integration = JetsonIntegration(auto_setup=False)
    print(f"- Running on Jetson platform: {integration.is_jetson}")
    
    # Test Whisper optimization
    whisper_opts = integration.optimize_faster_whisper("tiny.en")
    print("\nWhisper optimization for tiny.en:")
    for key, value in whisper_opts.items():
        print(f"- {key}: {value}")
    
    # Test LLM optimization
    llm_opts = integration.optimize_llm()
    print("\nLLM optimization:")
    for key, value in llm_opts.items():
        print(f"- {key}: {value}")
except Exception as e:
    print(f"Error testing JetsonIntegration: {e}")

print("\nAll basic tests completed successfully!")