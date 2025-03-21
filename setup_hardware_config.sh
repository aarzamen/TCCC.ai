#!/bin/bash
# TCCC Hardware Configuration Script
# This script configures the system for testing with the specified hardware:
# - Waveshare display via HDMI
# - Razer Seiren V3 Mini USB microphone
# - HDMI audio output

echo "===== TCCC Hardware Configuration ====="
echo "Setting up hardware for TCCC testing"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if the Razer Seiren V3 Mini is connected
check_razer_mic() {
    echo "Checking for Razer Seiren V3 Mini microphone..."
    
    if command_exists arecord; then
        FOUND_RAZER=$(arecord -l | grep -i "seiren" || echo "")
        if [ -n "$FOUND_RAZER" ]; then
            echo "✅ Razer Seiren V3 Mini detected"
            echo "$FOUND_RAZER"
            return 0
        else
            echo "❌ Razer Seiren V3 Mini not detected. Available audio devices:"
            arecord -l
            return 1
        fi
    else
        echo "❌ arecord command not found, cannot check audio devices"
        return 1
    fi
}

# Function to check for Waveshare display
check_display() {
    echo "Checking display configuration..."
    
    if command_exists xrandr; then
        DISPLAY_INFO=$(xrandr | grep -E "connected|[0-9]+x[0-9]+")
        echo "Display information:"
        echo "$DISPLAY_INFO"
        
        FOUND_HDMI=$(echo "$DISPLAY_INFO" | grep -i "HDMI" || echo "")
        if [ -n "$FOUND_HDMI" ]; then
            echo "✅ HDMI display detected"
            return 0
        else
            echo "❌ No HDMI display detected"
            return 1
        fi
    else
        echo "❌ xrandr command not found, cannot check display"
        return 1
    fi
}

# Function to configure HDMI audio output
configure_audio_output() {
    echo "Configuring HDMI audio output..."
    
    if command_exists pactl; then
        # List available sinks
        echo "Available audio output devices:"
        pactl list short sinks
        
        # Find HDMI sink
        HDMI_SINK=$(pactl list short sinks | grep -i "hdmi" | cut -f1 || echo "")
        
        if [ -n "$HDMI_SINK" ]; then
            echo "Setting default sink to HDMI output (sink $HDMI_SINK)"
            pactl set-default-sink "$HDMI_SINK"
            echo "✅ HDMI audio output configured"
            return 0
        else
            echo "❌ No HDMI audio output found"
            echo "Setting up fallback audio output..."
            return 1
        fi
    else
        echo "❌ pactl command not found, cannot configure audio output"
        return 1
    fi
}

# Function to configure Razer microphone as default
configure_razer_mic() {
    echo "Configuring Razer Seiren V3 Mini as default microphone..."
    
    if command_exists pactl; then
        # List available sources
        echo "Available audio input devices:"
        pactl list short sources
        
        # Find Razer microphone
        RAZER_SOURCE=$(pactl list short sources | grep -i "seiren\|razer" | cut -f1 || echo "")
        
        if [ -n "$RAZER_SOURCE" ]; then
            echo "Setting default source to Razer Seiren V3 Mini (source $RAZER_SOURCE)"
            pactl set-default-source "$RAZER_SOURCE"
            echo "✅ Razer Seiren V3 Mini configured as default microphone"
            return 0
        else
            echo "❌ Razer Seiren V3 Mini not found in audio sources"
            echo "Setting up fallback microphone..."
            return 1
        fi
    else
        echo "❌ pactl command not found, cannot configure microphone"
        return 1
    fi
}

# Function to create a test file with device configuration
create_device_config() {
    echo "Creating TCCC device configuration file..."
    
    cat > "/home/ama/tccc-project/config/device_config.yaml" << EOF
# TCCC Device Configuration
# Generated on: $(date)

audio:
  input:
    device: 0  # Razer Seiren V3 Mini
    name: "Razer Seiren V3 Mini"
    fallback_device: 1
  output:
    device: "hdmi"
    name: "HDMI Audio Output"
    fallback_device: "default"
  
display:
  type: "hdmi"
  name: "Waveshare HDMI Display"
  resolution: "1560x720"

hardware:
  platform: "jetson_nano"
  memory: "4GB"
  optimize_for_device: true
EOF

    echo "✅ Device configuration file created at: /home/ama/tccc-project/config/device_config.yaml"
}

# Function to update TCCC configuration to use the hardware
update_tccc_config() {
    echo "Updating TCCC configuration to use specified hardware..."
    
    # Check if config directory exists, create if not
    if [ ! -d "/home/ama/tccc-project/config" ]; then
        mkdir -p "/home/ama/tccc-project/config"
    fi
    
    # Create hardware configuration file
    create_device_config
    
    # Create test script to verify hardware setup
    cat > "/home/ama/tccc-project/test_hardware_setup.py" << EOF
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
EOF
    
    # Make test script executable
    chmod +x "/home/ama/tccc-project/test_hardware_setup.py"
    
    echo "✅ Created hardware test script at: /home/ama/tccc-project/test_hardware_setup.py"
}

# Main execution flow
echo "Initializing hardware configuration..."

# Create config directories if not exist
mkdir -p /home/ama/tccc-project/config 2>/dev/null

# Run all configuration functions
check_razer_mic
check_display
configure_audio_output
configure_razer_mic
update_tccc_config

echo ""
echo "Hardware configuration complete!"
echo "To test the hardware setup, run: ./test_hardware_setup.py"
echo ""

# Make this script executable
chmod +x "$0"

exit 0