# TCCC.ai Display Setup Guide

This guide provides comprehensive instructions for setting up and configuring the WaveShare 6.25" display (1560x720) with the TCCC.ai system. The setup supports multiple platforms including NVIDIA Jetson, Raspberry Pi, and standard Linux systems.

## Hardware Setup

### Display Specifications

- **Model**: WaveShare 6.25" IPS LCD Display
- **Resolution**: 1560x720 pixels (landscape) or 720x1560 pixels (portrait)
- **Interface**: HDMI + USB (touchscreen)
- **Power**: Powered via USB connection
- **Dimensions**: 165.4mm × 76.8mm × 4.4mm (LWH)
- **Viewing Angle**: 178° (H), 178° (V)
- **Touch Support**: Capacitive touch panel with 5-point multi-touch

### Physical Installation

1. **Connect the display**:
   - Connect the HDMI cable to your device's HDMI port
   - Connect the USB cable to one of your device's USB ports (for touchscreen functionality)
   - Connect the power cable to the USB power port if needed (some models may draw power from the USB data connection)

2. **Position the display**:
   - The display is designed with a hinged design similar to palmtop computers
   - The display can be folded down when not in use
   - For optimal viewing angle, position at approximately 120° from the base
   - For permanent installations, the display can be mounted using the mounting holes

## Software Configuration

### Automated Setup (Recommended)

The enhanced WaveShare display setup script handles the entire configuration process with automatic hardware detection:

```bash
# Run the setup script with superuser privileges
sudo ./setup_waveshare_display.sh
```

This script will:
1. Detect your platform (Jetson, Raspberry Pi, or standard Linux)
2. Install required dependencies
3. Check for and detect WaveShare display hardware
4. Configure display settings specific to your platform
5. Set up touchscreen calibration with auto-detection
6. Create startup scripts for automatic configuration at boot
7. Create the TCCC display configuration file
8. Generate a test script for verification
9. Add desktop launcher for easy testing

After running the script, reboot your system when prompted.

### Manual Configuration

If the automated setup doesn't work for your environment, you can follow these manual steps:

#### 1. Install Required Dependencies

For Debian/Ubuntu/Raspberry Pi OS:
```bash
sudo apt-get update
sudo apt-get install -y xinput x11-xserver-utils python3-pygame python3-yaml
```

For Fedora/RHEL:
```bash
sudo dnf install -y xorg-x11-server-utils xinput python3-pygame python3-pyyaml
```

For Arch Linux:
```bash
sudo pacman -Sy xorg-xinput xorg-xrandr python-pygame python-yaml
```

#### 2. Configure Display Resolution

Create a display configuration file:
```bash
sudo mkdir -p /etc/X11/xorg.conf.d/
sudo nano /etc/X11/xorg.conf.d/99-waveshare-display.conf
```

Add the following content for landscape mode:
```
Section "Monitor"
    Identifier "HDMI-0"
    Option "PreferredMode" "1560x720"
EndSection

Section "Screen"
    Identifier "Screen0"
    Monitor "HDMI-0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1560x720"
    EndSubSection
EndSection
```

For Raspberry Pi, you also need to modify `/boot/config.txt`:
```bash
sudo nano /boot/config.txt
```

Add the following lines:
```
# WaveShare 6.25" display configuration
display_rotate=1  # 1=90 degrees (landscape)
hdmi_group=2
hdmi_mode=87
hdmi_cvt=720 1560 60 6 0 0 0  # Width Height FPS
```

#### 3. Create Touchscreen Calibration Script

Detect the touchscreen device:
```bash
xinput list
```

Look for "WaveShare" or similar touchscreen device and note the ID.

Create a calibration script:
```bash
sudo nano /usr/local/bin/calibrate-waveshare-touch.sh
```

Add the following content (enhanced with auto-detection):
```bash
#!/bin/bash
# WaveShare Touch Calibration Script

# Try to detect the touchscreen device
DEVICE_ID=$(xinput list | grep -i "waveshare\|touch" | sed -n 's/.*id=\([0-9]*\).*/\1/p' | head -1)

if [ -z "$DEVICE_ID" ]; then
    echo "WaveShare touchscreen not found"
    exit 1
fi

# Detect orientation
if xrandr | grep -i "1560x720" >/dev/null; then
    echo "Using landscape orientation"
    # For landscape mode
    xinput set-prop "$DEVICE_ID" --type=float "Coordinate Transformation Matrix" 0 1 0 -1 0 1 0 0 1
else
    echo "Using portrait orientation"
    # For portrait mode
    xinput set-prop "$DEVICE_ID" --type=float "Coordinate Transformation Matrix" 1 0 0 0 1 0 0 0 1
fi

# Map touchscreen to correct output
OUTPUT=$(xrandr | grep " connected" | cut -d" " -f1 | head -1)
if [ -n "$OUTPUT" ]; then
    xinput map-to-output "$DEVICE_ID" "$OUTPUT"
    echo "Touchscreen mapped to $OUTPUT"
else
    echo "No display output found"
fi
```

Make it executable:
```bash
sudo chmod +x /usr/local/bin/calibrate-waveshare-touch.sh
```

#### 4. Add Calibration to Startup

```bash
sudo mkdir -p /etc/xdg/autostart/
sudo nano /etc/xdg/autostart/waveshare-touch-calibration.desktop
```

Add the following content:
```
[Desktop Entry]
Type=Application
Name=WaveShare Touch Calibration
Comment=Calibrate WaveShare touchscreen on startup
Exec=/usr/local/bin/calibrate-waveshare-touch.sh
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
```

#### 5. Create Display Configuration for TCCC

```bash
mkdir -p config
nano config/display.yaml
```

Add the configuration content (see "Display Configuration File" section below).

## Display Configuration File

The display configuration file (`config/display.yaml`) contains comprehensive settings for the display module:

```yaml
# Display Configuration for TCCC Project

# Display hardware settings
display:
  # Display dimensions (default is WaveShare 6.25" in landscape orientation)
  width: 1560
  height: 720
  
  # Display orientation
  orientation: landscape  # 'landscape' or 'portrait'
  
  # Fullscreen mode
  fullscreen: true
  
  # Touch input settings
  touch:
    enabled: true
    device: "WaveShare Touchscreen"
    calibration_enabled: true
    # Touch transformation matrix for correct mapping
    transformation_matrix: [0, 1, 0, -1, 0, 1, 0, 0, 1]
    # Touch sensitivity
    sensitivity: 1.0
    # Touch regions - defines interactive areas on screen
    regions:
      # Toggle between live and card view
      toggle_view:
        enabled: true
        rect: [0, -50, -1, 50]  # [x, y, width, height] (-1 means full width)
      # Quick buttons
      card_button:
        enabled: true
        rect: [-100, 60, 100, 40]  # Right side button for card view

# UI settings
ui:
  # Font settings
  font_scale: 1.0
  small_font_size: 22
  medium_font_size: 28
  large_font_size: 36
  
  # Color scheme
  theme: "dark"  # 'dark' or 'light'
  color_schemes:
    dark:
      background: [0, 0, 0]
      text: [255, 255, 255]
      header: [0, 0, 255]
      highlight: [255, 215, 0]
      alert: [255, 0, 0]
      success: [0, 200, 0]
    light:
      background: [240, 240, 240]
      text: [10, 10, 10]
      header: [50, 50, 200]
      highlight: [200, 160, 0]
      alert: [200, 0, 0]
      success: [0, 150, 0]
  
  # Maximum display items
  max_transcription_items: 10
  max_event_items: 8
  
  # Animation settings
  animations:
    enabled: true
    transition_speed_ms: 300
    fade_in: true
    scroll_smooth: true

# Hardware-specific settings
hardware:
  # Auto-detect display hardware
  auto_detect: true
  
  # WaveShare specific settings
  waveshare:
    model: "6.25_inch"
    rotation: 1  # 1=90° (landscape), 0=0° (portrait)
  
  # Jetson hardware settings
  jetson:
    use_hardware_acceleration: true
    # Power optimization
    power_save_mode: true
    # Performance settings
    performance:
      fps_limit_ac: 30  # FPS when on AC power
      fps_limit_battery: 15  # FPS when on battery
```

## Testing the Display

After setup, run the enhanced display test script to verify hardware functionality:

```bash
# Run the display test script with full screen mode
python test_waveshare_display.py --fullscreen

# Show only system information
python test_waveshare_display.py --info

# Run with specific resolution
python test_waveshare_display.py --width 1560 --height 720
```

### Test Script Features

The test script provides:
1. Auto-detection of display resolution and orientation
2. Touch input testing with visual feedback
3. System information display and diagnostics
4. Performance monitoring with FPS display
5. Debug mode for detailed hardware information
6. Display of touch points and regions

### Testing TCCC.ai with the Display

To run the TCCC.ai system with display support:

```bash
# Run with display support
python run_system.py --with-display
```

The system will:
1. Auto-detect the display hardware
2. Configure the appropriate resolution and orientation
3. Enable touch input if available
4. Present the TCCC.ai interface optimized for the display

## Troubleshooting

### Display Not Detected

1. **Check connections**:
   - Ensure HDMI cable is securely connected
   - Try a different HDMI cable
   - Check power supply to the display

2. **Check display configuration**:
   ```bash
   xrandr --query
   ```
   
   The WaveShare display should appear in the list with 1560x720 resolution (landscape) or 720x1560 (portrait).

3. **Manually set resolution**:
   ```bash
   # For landscape mode
   xrandr --newmode "1560x720_60.00" 95.75 1560 1640 1800 2040 720 723 733 750 -hsync +vsync
   xrandr --addmode HDMI-0 "1560x720_60.00"
   xrandr --output HDMI-0 --mode "1560x720_60.00"
   
   # For portrait mode
   xrandr --newmode "720x1560_60.00" 95.75 720 760 900 1040 1560 1563 1573 1590 -hsync +vsync
   xrandr --addmode HDMI-0 "720x1560_60.00"
   xrandr --output HDMI-0 --mode "720x1560_60.00"
   ```

4. **Check system logs**:
   ```bash
   dmesg | grep -i 'hdmi\|display'
   ```

### Touchscreen Not Working

1. **Check USB connection**:
   ```bash
   lsusb
   ```
   
   You should see "WaveShare", "Touch", "USBTOUCH", or similar device.

2. **Check if touchscreen is detected**:
   ```bash
   xinput list
   ```
   
   Look for "WaveShare Touchscreen" or similar in the list.

3. **Run calibration manually**:
   ```bash
   sudo /usr/local/bin/calibrate-waveshare-touch.sh
   ```

4. **Check touchscreen properties**:
   ```bash
   # Replace X with your device ID from xinput list
   xinput list-props X
   ```

5. **Test touch input separately**:
   ```bash
   # Install evtest for testing input devices
   sudo apt-get install evtest
   
   # Run the test (you'll need to select your touch device)
   sudo evtest
   ```

### Touch Alignment Issues

If touch input doesn't align with the display:

1. **Check orientation in config**:
   ```bash
   # Edit the configuration file
   nano config/display.yaml
   ```
   
   Ensure `orientation` matches your actual display setup.

2. **Manual calibration**:
   ```bash
   # Replace X with your device ID from xinput list
   
   # For landscape mode:
   xinput set-prop X --type=float "Coordinate Transformation Matrix" 0 1 0 -1 0 1 0 0 1
   
   # For portrait mode:
   xinput set-prop X --type=float "Coordinate Transformation Matrix" 1 0 0 0 1 0 0 0 1
   
   # For rotated portrait (180°):
   xinput set-prop X --type=float "Coordinate Transformation Matrix" -1 0 1 0 -1 1 0 0 1
   
   # For rotated landscape (180°):
   xinput set-prop X --type=float "Coordinate Transformation Matrix" 0 -1 1 1 0 0 0 0 1
   ```

3. **Recalibrate and map to output**:
   ```bash
   # Replace X with your device ID
   # Replace OUTPUT with your display name from xrandr (usually HDMI-0)
   xinput map-to-output X OUTPUT
   ```

### Performance Optimization

1. **For Jetson devices**:
   ```bash
   # Edit the configuration file
   nano config/display.yaml
   ```
   
   Enable hardware acceleration and adjust performance settings:
   ```yaml
   hardware:
     jetson:
       use_hardware_acceleration: true
       power_save_mode: true
       performance:
         fps_limit_ac: 30
         fps_limit_battery: 15
   ```

2. **Reduce animation effects**:
   ```yaml
   ui:
     animations:
       enabled: false
   ```

3. **Check CPU/GPU usage**:
   ```bash
   # For NVIDIA Jetson
   tegrastats
   
   # For general Linux
   htop
   ```

## Platform-Specific Notes

### NVIDIA Jetson

- Jetson devices benefit from hardware acceleration for display rendering
- The display module automatically detects Jetson hardware and applies optimizations
- For best performance, ensure `jetson_integration.py` utility is enabled
- Power management settings are particularly useful when running on battery

### Raspberry Pi

- On Raspberry Pi, the display rotation is configured in `/boot/config.txt`
- Make sure you have the latest firmware with `sudo rpi-update`
- Display performance may be improved by allocating more GPU memory:
  ```bash
  # Edit /boot/config.txt
  gpu_mem=128
  ```

### Standard Linux

- Most Linux distributions work well with the standard X11 configuration
- For systems using Wayland, you may need to use `wlr-randr` instead of `xrandr`
- Some desktop environments may override display settings; check your display settings in the system preferences

## Additional Display Options

If using a different display:

1. **HDMI Monitor**: 
   - Works out of the box, just connect and start with `--with-display`
   - The display module will auto-detect resolution

2. **Different WaveShare Display Models**: 
   - Edit `config/display.yaml` to match your display's resolution
   - Adjust the `waveshare.model` setting
   - For portrait orientation, set `orientation: portrait`

3. **Using Internal Display** (for laptops or dev kits):
   - Set `fullscreen: false` in the config to use in windowed mode
   - The display will auto-detect and use the available screen real estate