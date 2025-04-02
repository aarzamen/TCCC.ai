#!/usr/bin/env python3
"""
Verification script for the TCCC.ai Jetson Optimizer module.

This script tests the Jetson optimization module to verify it correctly
detects hardware capabilities and suggests optimal settings.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import required packages
try:
    import numpy as np
    import torch
    import psutil
except ImportError as e:
    print(f"Error importing required package: {e}")
    print("Please make sure all required packages are installed:")
    print("  - numpy")
    print("  - torch")
    print("  - psutil")
    sys.exit(1)

# Import Jetson optimizer
jetson_optimizer_available = False
jetson_integration_available = False

try:
    from src.tccc.utils.jetson_optimizer import JetsonOptimizer
    jetson_optimizer_available = True
except ImportError as e:
    print(f"Warning: Couldn't import JetsonOptimizer: {e}")

try:
    from src.tccc.utils.jetson_integration import JetsonIntegration, initialize_jetson_optimizations
    jetson_integration_available = True
except ImportError as e:
    print(f"Warning: Couldn't import JetsonIntegration: {e}")

if not jetson_optimizer_available and not jetson_integration_available:
    print("Error: Neither Jetson optimizer modules could be imported.")
    print("Please make sure the modules are properly installed.")
    sys.exit(1)


def setup_argparse():
    parser = argparse.ArgumentParser(description="Verify Jetson optimizer functionality")
    parser.add_argument("--run-diagnostics", action="store_true", help="Run hardware diagnostics")
    parser.add_argument("--check-audio", action="store_true", help="Check audio device configuration")
    parser.add_argument("--check-whisper", action="store_true", help="Check Whisper model optimization")
    parser.add_argument("--check-llm", action="store_true", help="Check LLM model optimization")
    parser.add_argument("--monitor-resources", action="store_true", help="Monitor system resources")
    parser.add_argument("--apply-profile", choices=["emergency", "field", "training"], 
                      help="Apply a performance profile")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    
    return parser.parse_args()


def run_diagnostics():
    """Run hardware diagnostics"""
    print("\n=== Hardware Diagnostics ===")
    
    # Initialize the optimizer
    optimizer = JetsonOptimizer()
    suggestions = optimizer.suggest_optimal_settings()
    
    # Print hardware info
    print(f"Running on Jetson: {suggestions['hardware_detected']['is_jetson']}")
    print(f"CUDA available: {suggestions['hardware_detected']['has_cuda']}")
    print(f"Available memory: {suggestions['hardware_detected']['available_memory_gb']} GB")
    
    # Check CPU info
    print("\nCPU Information:")
    print(f"  CPU Count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1.0)}%")
    
    # Check memory
    mem = psutil.virtual_memory()
    print("\nMemory Information:")
    print(f"  Total: {mem.total / (1024**3):.2f} GB")
    print(f"  Available: {mem.available / (1024**3):.2f} GB")
    print(f"  Used: {mem.percent}%")
    
    # Check GPU if available
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        
        # Check memory
        if hasattr(torch.cuda, 'memory_allocated'):
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        if hasattr(torch.cuda, 'memory_reserved'):
            print(f"  Memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    
    # Print model recommendations
    print("\nRecommended Settings:")
    print(f"STT Engine: {suggestions['whisper_settings']['suggested_model']} model with {suggestions['whisper_settings']['compute_type']} precision")
    print(f"LLM: {suggestions['llm_settings']['quantization']} quantization with max {suggestions['llm_settings']['max_tokens']} tokens")
    print(f"Power Mode: {suggestions['general']['power_mode']}")
    print(f"Batch Size: {suggestions['general']['batch_size']}")
    
    return True


def check_audio_devices():
    """Check audio device configuration"""
    print("\n=== Audio Device Configuration ===")
    
    # Initialize the integration
    integration = JetsonIntegration()
    audio_devices = integration.setup_audio_devices()
    
    # Check for Razer Mini
    if audio_devices['razer_mini']['detected']:
        print(f"✅ Razer Seiren V3 Mini detected (card {audio_devices['razer_mini']['card_id']})")
        print(f"  Device ID: {audio_devices['razer_mini']['device_id']}")
        
        # Check environment variables
        print("\nEnvironment Variables:")
        for var in ["TCCC_AUDIO_CARD", "TCCC_AUDIO_DEVICE", "TCCC_AUDIO_RATE", "TCCC_AUDIO_FORMAT", "TCCC_AUDIO_CHANNELS"]:
            if var in os.environ:
                print(f"  {var}={os.environ[var]}")
            else:
                print(f"  {var} not set")
    else:
        print("❌ Razer Seiren V3 Mini not detected")
    
    # Check for Logitech Headset
    if audio_devices['logitech_headset']['detected']:
        print(f"\n✅ Logitech USB Headset detected (card {audio_devices['logitech_headset']['card_id']})")
        print(f"  Device ID: {audio_devices['logitech_headset']['device_id']}")
    else:
        print("\n❌ Logitech USB Headset not detected")
    
    return bool(audio_devices['razer_mini']['detected'] or audio_devices['logitech_headset']['detected'])


def check_whisper_optimization():
    """Check Whisper model optimization"""
    print("\n=== Whisper Model Optimization ===")
    
    # Initialize the integration
    integration = JetsonIntegration()
    
    # Get optimized whisper parameters for different models
    models = ["tiny.en", "base.en", "small.en"]
    
    for model in models:
        print(f"\nOptimizations for {model} model:")
        params = integration.optimize_faster_whisper(model)
        
        # Print key parameters
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    return True


def check_llm_optimization():
    """Check LLM model optimization"""
    print("\n=== LLM Model Optimization ===")
    
    # Initialize the integration
    integration = JetsonIntegration()
    
    # Get optimized LLM parameters
    params = integration.optimize_llm()
    
    # Print key parameters
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    return True


def monitor_resources(duration=10):
    """Monitor system resources"""
    print(f"\n=== Monitoring System Resources (for {duration} seconds) ===")
    
    # Initialize the integration
    integration = JetsonIntegration()
    
    # Start resource monitoring
    integration.start_resource_monitoring(interval=1.0)
    
    # Monitor for the specified duration
    for i in range(duration):
        stats = integration.get_resource_stats()
        print(f"\rCPU: {stats['cpu_percent']:.1f}% | Memory: {stats['memory_percent']:.1f}% | "
              f"GPU: {stats['gpu_percent']:.1f}% | Time: {i+1}/{duration}s", end="")
        time.sleep(1)
    
    print("\n\nAverages:")
    stats = integration.get_resource_stats()
    print(f"  CPU: {stats['averages']['cpu']:.1f}%")
    print(f"  Memory: {stats['averages']['memory']:.1f}%")
    print(f"  GPU: {stats['averages']['gpu']:.1f}%")
    
    # Stop monitoring
    integration.stop_resource_monitoring()
    
    return True


def apply_profile(profile_name):
    """Apply a performance profile"""
    print(f"\n=== Applying {profile_name} Profile ===")
    
    # Initialize the integration
    integration = JetsonIntegration()
    
    # Apply the profile
    success = integration.apply_profile(profile_name)
    
    if success:
        print(f"✅ Successfully applied '{profile_name}' profile")
        
        # Print profile details
        profile = integration.config.get("profiles", {}).get(profile_name, {})
        print("\nProfile settings:")
        for key, value in profile.items():
            if key != "description":
                print(f"  {key}: {value}")
    else:
        print(f"❌ Failed to apply '{profile_name}' profile")
    
    return success


def main():
    """Main function"""
    print("TCCC.ai Jetson Optimizer Verification")
    print("=====================================")
    
    args = setup_argparse()
    
    # If no options specified, show help
    if not any(vars(args).values()):
        print("Please specify verification options.")
        print("Run with --help for available options.")
        return False
    
    results = []
    
    # Run all checks or specific ones
    if args.all or args.run_diagnostics:
        results.append(("Diagnostics", run_diagnostics()))
    
    if args.all or args.check_audio:
        results.append(("Audio Device Configuration", check_audio_devices()))
    
    if args.all or args.check_whisper:
        results.append(("Whisper Optimization", check_whisper_optimization()))
    
    if args.all or args.check_llm:
        results.append(("LLM Optimization", check_llm_optimization()))
    
    if args.all or args.monitor_resources:
        results.append(("Resource Monitoring", monitor_resources()))
    
    if args.apply_profile:
        results.append((f"{args.apply_profile.capitalize()} Profile", apply_profile(args.apply_profile)))
    
    # Print summary
    print("\n=== Verification Summary ===")
    all_pass = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        all_pass = all_pass and success
    
    # Return overall result
    return all_pass


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)