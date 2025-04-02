#!/usr/bin/env python3
"""
Test script to verify TCCC hardware configuration:
- Waveshare display via HDMI
- Razer Seiren V3 Mini USB microphone
- HDMI audio output
"""

import os
import sys
import time
import subprocess
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

def test_microphone():
    """Test microphone input."""
    print_separator("Testing Microphone")
    
    try:
        import sounddevice as sd
        
        # List audio devices
        print("Available audio devices:")
        devices = sd.query_devices()
        
        # Find Razer microphone
        razer_device = None
        for i, device in enumerate(devices):
            if 'Seiren' in device['name'] or 'Razer' in device['name']:
                razer_device = i
                print(f"✅ Found Razer Seiren V3 Mini: Device {i}")
                print(f"   {device['name']}")
                break
        
        if razer_device is None:
            print("❌ Razer Seiren V3 Mini not found")
            print("Available input devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  {i}: {device['name']}")
            return False
        
        # Record a short sample to verify microphone works
        print("\nRecording 3 seconds of audio from Razer microphone...")
        fs = 16000  # Sample rate
        duration = 3  # seconds
        
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=razer_device)
        sd.wait()
        
        # Check if recording contains audio
        if len(recording) > 0:
            max_amplitude = np.max(np.abs(recording))
            print(f"Recording complete: Max amplitude = {max_amplitude:.6f}")
            
            if max_amplitude > 0.01:
                print("✅ Microphone is working (detected audio signal)")
            else:
                print("⚠️ Microphone may not be working (very low audio level)")
                
            # Save recording for verification
            try:
                import soundfile as sf
                filename = "test_mic.wav"
                sf.write(filename, recording, fs)
                print(f"Saved test recording to {filename}")
            except Exception as e:
                print(f"Could not save recording: {e}")
                
            return True
        else:
            print("❌ Failed to record audio")
            return False
            
    except Exception as e:
        print(f"Error testing microphone: {e}")
        return False

def test_display():
    """Test display output."""
    print_separator("Testing Display")
    
    try:
        # Check if running in X environment
        if 'DISPLAY' not in os.environ:
            print("❌ Not running in X environment, cannot test display")
            return False
            
        # Run xrandr to get display information
        result = subprocess.run(['xrandr'], capture_output=True, text=True)
        
        if result.returncode == 0:
            output = result.stdout
            print("Display information:")
            for line in output.splitlines():
                if 'connected' in line or ' connected' in line:
                    print(f"  {line}")
                elif 'x' in line and 'Hz' in line and '*' in line:
                    print(f"  Current mode: {line.strip()}")
            
            if 'HDMI' in output:
                print("✅ HDMI display detected")
                return True
            else:
                print("❌ No HDMI display detected")
                return False
        else:
            print("❌ Failed to run xrandr command")
            return False
            
    except Exception as e:
        print(f"Error testing display: {e}")
        return False

def test_audio_output():
    """Test audio output."""
    print_separator("Testing Audio Output")
    
    try:
        import sounddevice as sd
        
        # List audio devices
        print("Available audio output devices:")
        devices = sd.query_devices()
        
        # Find HDMI output
        hdmi_device = None
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                if 'HDMI' in device['name']:
                    hdmi_device = i
                    print(f"✅ Found HDMI audio output: Device {i}")
                    print(f"   {device['name']}")
                    break
        
        if hdmi_device is None:
            print("❌ HDMI audio output not found")
            print("Available output devices:")
            for i, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    print(f"  {i}: {device['name']}")
            return False
            
        # Play a test tone to verify audio output
        try:
            fs = 44100  # Sample rate
            duration = 1  # seconds
            
            # Generate a 440 Hz sine wave
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            tone = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            print("Playing test tone through HDMI output...")
            sd.play(tone, fs, device=hdmi_device)
            sd.wait()
            print("✅ Test tone playback complete")
            
            return True
        except Exception as e:
            print(f"Error playing test tone: {e}")
            return False
            
    except Exception as e:
        print(f"Error testing audio output: {e}")
        return False

def load_config():
    """Load device configuration."""
    print_separator("Loading Device Configuration")
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config", "device_config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            print("Device configuration loaded:")
            print(f"  Audio input: {config['audio']['input']['name']} (Device {config['audio']['input']['device']})")
            print(f"  Audio output: {config['audio']['output']['name']}")
            print(f"  Display: {config['display']['name']} ({config['display']['resolution']})")
            print(f"  Hardware platform: {config['hardware']['platform']}")
            
            return config
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return None
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def main():
    """Main test function."""
    print_separator("TCCC Hardware Setup Test")
    print("Testing hardware configuration for TCCC system\n")
    
    # Load configuration
    config = load_config()
    
    # Test all hardware components
    mic_result = test_microphone()
    display_result = test_display()
    audio_result = test_audio_output()
    
    # Display overall results
    print_separator("Test Results")
    print(f"Razer Seiren V3 Mini: {'✅ PASS' if mic_result else '❌ FAIL'}")
    print(f"Waveshare HDMI Display: {'✅ PASS' if display_result else '❌ FAIL'}")
    print(f"HDMI Audio Output: {'✅ PASS' if audio_result else '❌ FAIL'}")
    
    overall_pass = mic_result and display_result and audio_result
    print(f"\nOverall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    # Write test results to file
    try:
        with open("hardware_test_results.txt", "w") as f:
            f.write(f"TCCC Hardware Test Results\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Razer Seiren V3 Mini: {'PASS' if mic_result else 'FAIL'}\n")
            f.write(f"Waveshare HDMI Display: {'PASS' if display_result else 'FAIL'}\n")
            f.write(f"HDMI Audio Output: {'PASS' if audio_result else 'FAIL'}\n\n")
            f.write(f"Overall: {'PASS' if overall_pass else 'FAIL'}\n")
        
        print(f"Test results saved to hardware_test_results.txt")
    except Exception as e:
        print(f"Error saving test results: {e}")
    
    return 0 if overall_pass else 1

if __name__ == "__main__":
    sys.exit(main())
