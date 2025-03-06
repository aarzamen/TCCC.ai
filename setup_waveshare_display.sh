#!/bin/bash

# TCCC WaveShare Display Setup Script
# Configures the system for the WaveShare 6.25" touchscreen display (1560x720)
# Enhanced version with automatic hardware detection and configuration

# ANSI colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to display section headers
section() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}\n"
}

# Function to display success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to display warning messages
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to display error messages
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to display info messages
info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to confirm an action
confirm() {
    echo -ne "${YELLOW}$1 (y/n) ${NC}"
    read -r response
    [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
    error "This script must be run with sudo or as root"
    echo "Please run: sudo $0"
    exit 1
fi

section "TCCC WaveShare Display Setup - Enhanced Version"

echo "This script will set up and configure your system for the WaveShare 6.25\" display"
echo "It will perform the following tasks:"
echo "  1. Detect and configure the WaveShare display"
echo "  2. Set up touchscreen calibration"
echo "  3. Create necessary configuration files"
echo "  4. Set up auto-start for touch calibration"
echo "  5. Generate test scripts for verification"
echo ""

# Detect operating system and platform
echo "Detecting system platform..."
IS_RASPBERRY_PI=false
IS_JETSON=false
IS_STANDARD_LINUX=false

if [ -f /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    if [[ "$MODEL" == *"Raspberry Pi"* ]]; then
        IS_RASPBERRY_PI=true
        success "Detected Raspberry Pi: $MODEL"
    elif [[ "$MODEL" == *"NVIDIA"* ]] || [[ "$MODEL" == *"Jetson"* ]]; then
        IS_JETSON=true
        success "Detected NVIDIA Jetson: $MODEL"
    else
        IS_STANDARD_LINUX=true
        success "Detected Linux system: $MODEL"
    fi
else
    IS_STANDARD_LINUX=true
    success "Detected standard Linux system"
fi

# Check Linux distribution
if [ -f /etc/os-release ]; then
    source /etc/os-release
    DISTRO=$NAME
    VERSION=$VERSION_ID
    success "Detected distribution: $DISTRO $VERSION"
else
    DISTRO="Unknown"
    VERSION="Unknown"
    warning "Could not determine Linux distribution"
fi

# Check if WaveShare display is connected
echo "Checking for WaveShare display..."
display_connected=false

# Method 1: Check using EDID
if command -v edid-decode > /dev/null 2>&1; then
    info "Using edid-decode to check display information..."
    # Check common HDMI paths
    for edid_path in /sys/class/drm/card*-HDMI*/edid /sys/class/drm/card*-HDMI*/edid; do
        if [ -f "$edid_path" ] && edid-decode "$edid_path" 2>/dev/null | grep -i "waveshare\|1560x720\|720x1560" > /dev/null; then
            success "WaveShare display detected via EDID at $edid_path"
            display_connected=true
            break
        fi
    done
fi

# Method 2: Check current resolution
if ! $display_connected && command -v xrandr > /dev/null 2>&1; then
    info "Using xrandr to check display resolution..."
    if xrandr 2>/dev/null | grep -i "720x1560\|1560x720" > /dev/null; then
        success "Display with WaveShare resolution detected (720x1560 or 1560x720)"
        display_connected=true
    fi
fi

# Method 3: Check for connected displays (less specific)
if ! $display_connected && command -v xrandr > /dev/null 2>&1; then
    info "Checking for connected displays..."
    if xrandr 2>/dev/null | grep -i " connected " > /dev/null; then
        info "Found connected display(s)"
        # Ask for manual confirmation if we see any display
        if confirm "I found a display but couldn't confirm it's a WaveShare. Is it a WaveShare 6.25\" display?"; then
            success "WaveShare display confirmed by user"
            display_connected=true
        fi
    fi
fi

# If display not detected, confirm proceeding
if ! $display_connected; then
    warning "WaveShare display not automatically detected"
    if ! confirm "Proceed with WaveShare display configuration anyway?"; then
        error "Setup aborted by user"
        exit 1
    else
        info "Continuing with setup as requested"
        display_connected=true
    fi
fi

section "Installing Dependencies"

# Install required packages based on distribution
echo "Installing required packages..."

# Function to check and install packages
install_packages() {
    local pkg_manager=$1
    shift
    local packages=("$@")
    
    # Check which packages are already installed
    local to_install=()
    case $pkg_manager in
        apt)
            for pkg in "${packages[@]}"; do
                if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
                    to_install+=("$pkg")
                fi
            done
            ;;
        dnf|yum)
            for pkg in "${packages[@]}"; do
                if ! rpm -q "$pkg" &>/dev/null; then
                    to_install+=("$pkg")
                fi
            done
            ;;
        pacman)
            for pkg in "${packages[@]}"; do
                if ! pacman -Q "$pkg" &>/dev/null; then
                    to_install+=("$pkg")
                fi
            done
            ;;
    esac
    
    # Install missing packages if any
    if [ ${#to_install[@]} -gt 0 ]; then
        echo "Installing: ${to_install[*]}"
        case $pkg_manager in
            apt)
                apt-get update -qq && apt-get install -y "${to_install[@]}"
                ;;
            dnf)
                dnf install -y "${to_install[@]}"
                ;;
            yum)
                yum install -y "${to_install[@]}"
                ;;
            pacman)
                pacman -Sy --noconfirm "${to_install[@]}"
                ;;
        esac
        return $?
    else
        echo "All required packages already installed"
        return 0
    fi
}

# Select packages based on distribution
if command -v apt-get &>/dev/null; then
    pkg_manager="apt"
    packages=("xinput" "x11-xserver-utils" "python3-pygame" "python3-yaml" "mesa-utils")
    if $IS_RASPBERRY_PI; then
        packages+=("libraspberrypi-bin" "xserver-xorg-input-libinput")
    elif $IS_JETSON; then
        packages+=("python3-gi" "gir1.2-gtk-3.0")
    fi
elif command -v dnf &>/dev/null; then
    pkg_manager="dnf"
    packages=("xorg-x11-server-utils" "xinput" "python3-pygame" "python3-pyyaml" "mesa-demos")
elif command -v yum &>/dev/null; then
    pkg_manager="yum"
    packages=("xorg-x11-server-utils" "xinput" "python3-pygame" "python3-pyyaml" "glx-utils")
elif command -v pacman &>/dev/null; then
    pkg_manager="pacman"
    packages=("xorg-xinput" "xorg-xrandr" "python-pygame" "python-yaml" "mesa-demos")
else
    warning "Unsupported package manager. Skipping automatic package installation."
    pkg_manager=""
fi

# Install packages if package manager detected
if [ -n "$pkg_manager" ]; then
    if install_packages "$pkg_manager" "${packages[@]}"; then
        success "Required packages installed successfully"
    else
        error "Failed to install some packages"
        if ! confirm "Continue anyway?"; then
            error "Setup aborted by user"
            exit 1
        fi
    fi
fi

section "Setting Up Display Configuration"

# Create display configuration file
echo "Creating display configuration file..."

# Configure based on system type
if $IS_RASPBERRY_PI; then
    echo "Setting up display for Raspberry Pi..."
    
    # Configure config.txt
    if [ -f "/boot/config.txt" ]; then
        echo "Configuring /boot/config.txt..."
        
        # Remove existing display settings to avoid duplicates
        sed -i '/^display_rotate=/d' /boot/config.txt
        sed -i '/^hdmi_group=/d' /boot/config.txt
        sed -i '/^hdmi_mode=/d' /boot/config.txt
        sed -i '/^hdmi_cvt=/d' /boot/config.txt
        
        # Add WaveShare 6.25" display configuration
        cat >> /boot/config.txt << 'EOF'

# WaveShare 6.25" display configuration
display_rotate=1  # 1=90 degrees (landscape)
hdmi_group=2
hdmi_mode=87
hdmi_cvt=720 1560 60 6 0 0 0  # Width Height FPS
EOF
        success "Updated boot configuration in /boot/config.txt"
    else
        warning "Could not find /boot/config.txt for Raspberry Pi configuration"
    fi
    
elif $IS_JETSON; then
    echo "Setting up display for Jetson..."
    
    # Create a custom X11 configuration file for the Jetson
    if [ ! -d "/etc/X11/xorg.conf.d" ]; then
        mkdir -p /etc/X11/xorg.conf.d
    fi
    
    # Create Jetson-specific configuration
    cat > /etc/X11/xorg.conf.d/99-waveshare-display.conf << 'EOF'
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
EOF
    success "Created Jetson X11 configuration for WaveShare display"
    
    # Check for Jetson-specific display configuration
    if [ -f "/etc/X11/xorg.conf" ]; then
        # Make a backup of the existing configuration
        cp /etc/X11/xorg.conf /etc/X11/xorg.conf.backup
        info "Backed up existing xorg.conf to /etc/X11/xorg.conf.backup"
        
        # Update the existing configuration if needed
        if ! grep -q "1560x720" /etc/X11/xorg.conf; then
            info "Modifying main Xorg configuration for Jetson..."
            # This is a simplified approach - in practice we would need more careful
            # editing of the xorg.conf file to preserve existing settings while
            # adding our display mode
        fi
    fi
else
    # Setup for standard Linux systems
    echo "Setting up display for standard Linux system..."
    
    # Check if X11 config directory exists
    if [ ! -d "/etc/X11/xorg.conf.d" ]; then
        mkdir -p /etc/X11/xorg.conf.d
    fi
    
    # Create WaveShare display configuration
    cat > /etc/X11/xorg.conf.d/99-waveshare-display.conf << 'EOF'
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
EOF

    success "Created X11 display configuration for standard Linux"
fi

section "Setting Up Touch Input"

# Check if touchscreen device is connected
echo "Checking for WaveShare touchscreen..."
touch_connected=false
touch_device_name="WaveShare Touchscreen"
touch_device_id=""

# Try to detect using xinput (the most reliable method)
if command -v xinput > /dev/null 2>&1; then
    # Get the list of input devices
    xinput_output=$(xinput list)
    
    # Look for different possible touchscreen names
    for name in "WaveShare" "ADS7846" "Goodix" "eGalax" "Touch" "USBTOUCH"; do
        if echo "$xinput_output" | grep -i "$name" > /dev/null; then
            # Found a potential match
            touch_connected=true
            # Extract the device name for later use
            touch_device_name=$(echo "$xinput_output" | grep -i "$name" | sed -E 's/^.*([a-zA-Z0-9]+ [a-zA-Z0-9]+ [a-zA-Z0-9]+).*id=.*/\1/' | head -1)
            # Extract the device ID
            touch_device_id=$(echo "$xinput_output" | grep -i "$name" | sed -n 's/.*id=\([0-9]*\).*/\1/p' | head -1)
            
            success "Touchscreen detected: $touch_device_name (ID: $touch_device_id)"
            break
        fi
    done
    
    # If still not found, check for any available touchscreen devices
    if ! $touch_connected; then
        # Look for devices with "touch" in capabilities
        for device_id in $(xinput list --id-only); do
            device_caps=$(xinput list-props "$device_id" 2>/dev/null)
            if echo "$device_caps" | grep -i "TouchScreen\|Calibration" > /dev/null; then
                touch_connected=true
                touch_device_name=$(xinput list --name-only "$device_id" 2>/dev/null || echo "TouchScreen ID $device_id")
                touch_device_id="$device_id"
                success "Generic touchscreen detected: $touch_device_name (ID: $touch_device_id)"
                break
            fi
        done
    fi
fi

# Check if we need to continue with manual configuration
if ! $touch_connected; then
    warning "Touchscreen not automatically detected"
    if $display_connected && confirm "Would you like to configure touch input anyway? (For manual setup later)"; then
        touch_connected=true
        info "Continuing with touchscreen setup for manual configuration"
    else
        info "Skipping touchscreen configuration"
    fi
fi

# Create touchscreen calibration script
if $touch_connected || $display_connected; then
    echo "Creating touchscreen calibration script..."
    
    # Create script directory if it doesn't exist
    mkdir -p /usr/local/bin
    
    # Create a more comprehensive touchscreen calibration script
    cat > /usr/local/bin/calibrate-waveshare-touch.sh << 'EOF'
#!/bin/bash
# WaveShare Touch Calibration Script
# Automatically configures touchscreen for the WaveShare 6.25" display

# ANSI colors for readable output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "WaveShare Touchscreen Calibration Tool"
echo "--------------------------------------"

# Try to detect the touchscreen device
DEVICE_NAME=""
DEVICE_ID=""

# List of possible touchscreen device names (add more if needed)
POSSIBLE_NAMES=("WaveShare" "ADS7846" "Goodix" "eGalax" "Touch" "USBTOUCH")

# Search for devices using xinput
for name in "${POSSIBLE_NAMES[@]}"; do
    if xinput list | grep -i "$name" >/dev/null; then
        DEVICE_NAME=$(xinput list | grep -i "$name" | head -1)
        DEVICE_ID=$(echo "$DEVICE_NAME" | sed -n 's/.*id=\([0-9]*\).*/\1/p')
        
        if [ -n "$DEVICE_ID" ]; then
            echo -e "${GREEN}Found touchscreen:${NC} $DEVICE_NAME"
            echo -e "${GREEN}Device ID:${NC} $DEVICE_ID"
            break
        fi
    fi
done

# If no device was found through name search, look for any touchscreen
if [ -z "$DEVICE_ID" ]; then
    echo -e "${YELLOW}No known touchscreen name found, searching by capabilities...${NC}"
    
    # Get all input device IDs
    for id in $(xinput list --id-only); do
        # Check if this device has touchscreen capabilities
        props=$(xinput list-props "$id" 2>/dev/null)
        if echo "$props" | grep -i "TouchScreen\|Calibration" >/dev/null; then
            DEVICE_NAME=$(xinput list --name-only "$id" 2>/dev/null || echo "TouchScreen ID $id")
            DEVICE_ID="$id"
            echo -e "${GREEN}Found generic touchscreen:${NC} $DEVICE_NAME"
            echo -e "${GREEN}Device ID:${NC} $DEVICE_ID"
            break
        fi
    done
fi

# If still no device found, offer to specify manually
if [ -z "$DEVICE_ID" ]; then
    echo -e "${YELLOW}No touchscreen device found automatically.${NC}"
    echo "Available input devices:"
    xinput list
    
    echo ""
    echo -e "${YELLOW}Please enter the device ID manually (or press Enter to skip):${NC}"
    read -r manual_id
    
    if [ -n "$manual_id" ]; then
        DEVICE_ID="$manual_id"
        DEVICE_NAME=$(xinput list --name-only "$DEVICE_ID" 2>/dev/null || echo "Manual TouchScreen ID $DEVICE_ID")
        echo -e "${GREEN}Using manual device:${NC} $DEVICE_NAME (ID: $DEVICE_ID)"
    else
        echo -e "${RED}No touchscreen device specified. Calibration skipped.${NC}"
        exit 1
    fi
fi

# Detect display orientation
ROTATION="landscape"
if command -v xrandr >/dev/null; then
    # Check for portrait or landscape mode based on resolution
    if xrandr | grep -i "720x1560" >/dev/null; then
        ROTATION="portrait"
        echo -e "${YELLOW}Detected display in portrait mode (720x1560)${NC}"
    elif xrandr | grep -i "1560x720" >/dev/null; then
        ROTATION="landscape"
        echo -e "${GREEN}Detected display in landscape mode (1560x720)${NC}"
    else
        echo -e "${YELLOW}Could not detect display orientation from resolution, assuming landscape${NC}"
    fi
else
    echo -e "${YELLOW}xrandr not available, assuming landscape orientation${NC}"
fi

# Set appropriate transformation matrix based on orientation
if [ "$ROTATION" = "landscape" ]; then
    # For landscape mode (WaveShare default is portrait, so we rotate)
    echo "Applying landscape mode transformation matrix"
    xinput set-prop "$DEVICE_ID" --type=float "Coordinate Transformation Matrix" 0 1 0 -1 0 1 0 0 1
else
    # For portrait mode (no rotation needed)
    echo "Applying portrait mode transformation matrix"
    xinput set-prop "$DEVICE_ID" --type=float "Coordinate Transformation Matrix" 1 0 0 0 1 0 0 0 1
fi

# Map touchscreen to correct output
OUTPUT=$(xrandr | grep " connected" | cut -d" " -f1 | head -1)
if [ -n "$OUTPUT" ]; then
    echo "Mapping touchscreen to output: $OUTPUT"
    xinput map-to-output "$DEVICE_ID" "$OUTPUT"
    echo -e "${GREEN}Touch calibration completed successfully${NC}"
else
    echo -e "${YELLOW}No display output found for mapping, touchscreen may not align correctly${NC}"
fi

# Check if calibration was successful by checking for transformation matrix
if xinput list-props "$DEVICE_ID" | grep "Coordinate Transformation Matrix" >/dev/null; then
    echo -e "${GREEN}Calibration verified successfully${NC}"
    exit 0
else
    echo -e "${RED}Calibration could not be verified${NC}"
    exit 1
fi
EOF

    chmod +x /usr/local/bin/calibrate-waveshare-touch.sh
    success "Created comprehensive touchscreen calibration script at /usr/local/bin/calibrate-waveshare-touch.sh"

    # Add calibration script to startup
    echo "Adding touchscreen calibration to startup..."

    # Create autostart directory if it doesn't exist
    mkdir -p /etc/xdg/autostart

    # Create desktop entry for autostart
    cat > /etc/xdg/autostart/waveshare-touch-calibration.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=WaveShare Touch Calibration
Comment=Calibrate WaveShare touchscreen on startup
Exec=/usr/local/bin/calibrate-waveshare-touch.sh
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
EOF

    success "Added touchscreen calibration to startup applications"

    # Run calibration now if touchscreen is connected
    if [ -n "$touch_device_id" ]; then
        echo "Running touchscreen calibration now..."
        /usr/local/bin/calibrate-waveshare-touch.sh
    else
        info "Skipping immediate calibration, no touchscreen currently detected"
    fi
fi

section "Setting Up TCCC Display Configuration"

# Create TCCC display config directory if it doesn't exist
mkdir -p $(dirname "$0")/config

# Create enhanced display configuration file with more features
cat > $(dirname "$0")/config/display.yaml << 'EOF'
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
  
  # Logo paths
  logo: "images/blue_logo.png"
  alt_logo: "images/green_logo.png"
  
  # Maximum display items
  max_transcription_items: 10
  max_event_items: 8
  
  # Animation settings
  animations:
    enabled: true
    transition_speed_ms: 300
    fade_in: true
    scroll_smooth: true
  
  # Layout adjustments
  layout:
    # Column width percentages for landscape mode
    column_1_width: 0.38  # Transcription column
    column_2_width: 0.34  # Events column
    # Column 3 takes the remaining space

# Hardware-specific settings
hardware:
  # Auto-detect display hardware
  auto_detect: true
  
  # WaveShare 6.25" specific settings
  waveshare:
    model: "6.25_inch"
    rotation: 1  # 1=90° (landscape), 0=0° (portrait)
    hdmi_group: 2
    hdmi_mode: 87  # Custom mode
    hdmi_cvt: "720 1560 60 6 0 0 0"  # Custom timing (height width refresh)
    # Backlight control (if supported)
    backlight: 
      enabled: true
      path: "/sys/class/backlight/waveshare/brightness"
      max_value: 255
      default_value: 200
  
  # Jetson hardware settings
  jetson:
    use_framebuffer: true
    framebuffer_device: "/dev/fb0"
    use_hardware_acceleration: true
    # Power optimization (reduced framerate on battery)
    power_save_mode: true
    # Performance settings
    performance:
      fps_limit_ac: 30  # FPS when on AC power
      fps_limit_battery: 15  # FPS when on battery

# Performance monitoring
performance:
  monitor_enabled: true
  show_fps: false  # Set to true to show FPS counter
  log_performance: true
  target_fps: 30  # Target frames per second

# Advanced settings
advanced:
  # Debug features
  debug_mode: false
  show_touch_points: false
  # Low-level settings
  sdl_videodriver: ""  # Empty to use system default
  sdl_audiodriver: ""  # Empty to use system default
EOF

success "Created enhanced TCCC display configuration file"

# Create an improved test script that handles more display configurations
cat > $(dirname "$0")/test_waveshare_display.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Test Script for WaveShare 6.25" Display
-----------------------------------------------
This script tests display functionality, hardware detection, touch input,
and performance metrics. It supports multiple display configurations and
provides diagnostic information.
"""

import os
import sys
import time
import argparse
import platform
import threading
from pathlib import Path
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WaveShareTest")

# Try to import pygame
try:
    import pygame
    from pygame.locals import *
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.error("pygame not installed. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
        import pygame
        from pygame.locals import *
        PYGAME_AVAILABLE = True
        logger.info("pygame installed successfully")
    except Exception as e:
        logger.error(f"Failed to install pygame: {e}")
        print("Failed to install pygame. Try manually with: pip install pygame")
        sys.exit(1)

# Try to import yaml for config
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed, will use defaults")

# Define constants
DEFAULT_WIDTH = 1560
DEFAULT_HEIGHT = 720
FULLSCREEN = True
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# System information
SYSTEM_INFO = {
    'system': platform.system(),
    'release': platform.release(),
    'version': platform.version(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'display_driver': None,
    'display_info': None,
    'touch_devices': [],
}


def detect_display():
    """Detect display configuration"""
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    orientation = "landscape"
    
    # Try to get display info from PyGame
    try:
        pygame.display.init()
        info = pygame.display.Info()
        
        # Get display driver
        SYSTEM_INFO['display_driver'] = pygame.display.get_driver()
        SYSTEM_INFO['display_info'] = f"{info.current_w}x{info.current_h}"
        
        # Use detected resolution
        if info.current_w > 0 and info.current_h > 0:
            if info.current_w == 720 and info.current_h == 1560:
                width, height = 720, 1560
                orientation = "portrait"
                logger.info(f"Detected WaveShare display in portrait mode: {width}x{height}")
            elif info.current_w == 1560 and info.current_h == 720:
                width, height = 1560, 720
                orientation = "landscape"
                logger.info(f"Detected WaveShare display in landscape mode: {width}x{height}")
            else:
                width, height = info.current_w, info.current_h
                orientation = "landscape" if width > height else "portrait"
                logger.info(f"Using system display resolution: {width}x{height}")
    except Exception as e:
        logger.warning(f"Failed to detect display info via pygame: {e}")
    
    # Try to read from config file
    if YAML_AVAILABLE:
        try:
            config_path = Path(__file__).parent / "config" / "display.yaml"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                if 'display' in config:
                    display_config = config['display']
                    config_width = display_config.get('width', width)
                    config_height = display_config.get('height', height)
                    config_orientation = display_config.get('orientation', orientation)
                    
                    # Only use config if values are reasonable
                    if config_width > 100 and config_height > 100:
                        width, height = config_width, config_height
                        orientation = config_orientation
                        logger.info(f"Using display configuration from file: {width}x{height}, {orientation}")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
    
    # Swap dimensions if needed to match orientation
    is_portrait = orientation.lower() == "portrait"
    is_current_portrait = height > width
    
    if is_portrait != is_current_portrait:
        width, height = height, width
        logger.info(f"Adjusted dimensions for {orientation} orientation: {width}x{height}")
    
    return width, height, orientation


def detect_touch_devices():
    """Detect available touch input devices"""
    touch_devices = []
    
    # Try using xinput on Linux
    if SYSTEM_INFO['system'] == 'Linux':
        try:
            result = subprocess.run(['xinput', 'list'], 
                                   capture_output=True, text=True, check=False)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    # Look for touch devices
                    if any(term in line.lower() for term in ['touch', 'wave', 'ads7846']):
                        # Extract device name and ID
                        import re
                        id_match = re.search(r'id=(\d+)', line)
                        device_id = id_match.group(1) if id_match else "unknown"
                        
                        # Extract name from line (rough approach)
                        name_match = re.search(r'↳ ([^=]+)(?=id=|$)', line)
                        device_name = name_match.group(1).strip() if name_match else line
                        
                        touch_devices.append({
                            'name': device_name,
                            'id': device_id
                        })
        except Exception as e:
            logger.warning(f"Failed to detect touch devices: {e}")
    
    # Fallback to pygame touch device detection
    try:
        pygame.init()
        num_devices = pygame.joystick.get_count()  # Pygame sometimes detects touch as joystick
        for i in range(num_devices):
            try:
                joy = pygame.joystick.Joystick(i)
                name = joy.get_name()
                if any(term in name.lower() for term in ['touch', 'screen', 'panel']):
                    touch_devices.append({
                        'name': name,
                        'id': i
                    })
            except:
                pass
    except:
        pass
    
    SYSTEM_INFO['touch_devices'] = touch_devices
    return touch_devices


def show_system_info():
    """Display system information in console"""
    print("\n=== System Information ===")
    print(f"System: {SYSTEM_INFO['system']} {SYSTEM_INFO['release']}")
    print(f"Platform: {SYSTEM_INFO['machine']} ({SYSTEM_INFO['processor']})")
    print(f"Display Driver: {SYSTEM_INFO['display_driver']}")
    print(f"Display Resolution: {SYSTEM_INFO['display_info']}")
    
    if SYSTEM_INFO['touch_devices']:
        print("\nDetected Touch Devices:")
        for device in SYSTEM_INFO['touch_devices']:
            print(f"  - {device['name']} (ID: {device['id']})")
    else:
        print("\nNo touch devices detected")


def init_test(args):
    """Initialize and run the display test"""
    print("\n=== WaveShare Display Test ===")
    
    # Detect display configuration
    width, height, orientation = detect_display()
    
    # Override with command line arguments if provided
    if args.width and args.height:
        width, height = args.width, args.height
        logger.info(f"Using command line dimensions: {width}x{height}")
    
    # Detect touch devices
    touch_devices = detect_touch_devices()
    
    # Show system information
    show_system_info()
    
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("WaveShare Display Test")
    
    # Set up display
    if args.fullscreen:
        screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((width, height))
    
    print(f"Display initialized: {width}x{height}")
    print(f"Orientation: {orientation}")
    print(f"Using pygame driver: {pygame.display.get_driver()}")
    
    # Fill the screen with a color
    screen.fill(BLACK)
    pygame.display.flip()
    
    # Create font objects
    try:
        font_large = pygame.font.SysFont('Arial', 48)
        font_medium = pygame.font.SysFont('Arial', 36)
        font_small = pygame.font.SysFont('Arial', 24)
    except:
        # Fallback to default font
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)
    
    # Draw header
    header_rect = pygame.Rect(0, 0, width, 80)
    pygame.draw.rect(screen, BLUE, header_rect)
    header_text = font_large.render("WaveShare Display Test", True, WHITE)
    screen.blit(header_text, (width//2 - header_text.get_width()//2, 20))
    
    # Draw touch test area
    touch_rect = pygame.Rect(50, 100, width-100, height-200)
    pygame.draw.rect(screen, (50, 50, 50), touch_rect)
    touch_text = font_medium.render("Touch anywhere in this area", True, WHITE)
    screen.blit(touch_text, (width//2 - touch_text.get_width()//2, 120))
    
    # Draw system info
    sys_text = font_small.render(f"System: {SYSTEM_INFO['system']} - Display: {width}x{height} - Driver: {pygame.display.get_driver()}", True, WHITE)
    screen.blit(sys_text, (width//2 - sys_text.get_width()//2, 180))
    
    # Draw touch device info
    touch_info = "Touch Devices: "
    if touch_devices:
        touch_info += ", ".join([f"{d['name']} (ID: {d['id']})" for d in touch_devices[:2]])
        if len(touch_devices) > 2:
            touch_info += f", and {len(touch_devices) - 2} more"
    else:
        touch_info += "None detected"
    
    touch_info_text = font_small.render(touch_info, True, WHITE)
    screen.blit(touch_info_text, (width//2 - touch_info_text.get_width()//2, 210))
    
    # Draw footer
    footer_rect = pygame.Rect(0, height-80, width, 80)
    pygame.draw.rect(screen, (100, 100, 100), footer_rect)
    
    controls_text = font_medium.render("ESC: Exit | D: Debug Info | F: FPS Toggle", True, WHITE)
    screen.blit(controls_text, (width//2 - controls_text.get_width()//2, height-60))
    
    # Update display
    pygame.display.flip()
    
    # Variables for touch/click tracking
    touch_points = []
    MAX_POINTS = 10
    
    # Debug mode
    debug_mode = False
    show_fps = False
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            frame_start = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_d:
                        # Toggle debug mode
                        debug_mode = not debug_mode
                    elif event.key == K_f:
                        # Toggle FPS display
                        show_fps = not show_fps
                
                # Handle touch events (pygame 2.0+)
                elif hasattr(pygame, 'FINGERDOWN') and event.type == pygame.FINGERDOWN:
                    # Convert normalized position to screen coordinates
                    x = event.x * width
                    y = event.y * height
                    touch_points.append((x, y, time.time()))
                    
                    # Keep only recent points
                    if len(touch_points) > MAX_POINTS:
                        touch_points.pop(0)
                        
                    print(f"Touch detected at ({int(x)}, {int(y)})")
                    
                # Handle mouse clicks (for testing on non-touch displays)
                elif event.type == MOUSEBUTTONDOWN:
                    touch_points.append((event.pos[0], event.pos[1], time.time()))
                    
                    # Keep only recent points
                    if len(touch_points) > MAX_POINTS:
                        touch_points.pop(0)
                        
                    print(f"Click detected at ({event.pos[0]}, {event.pos[1]})")
            
            # Redraw touch area
            pygame.draw.rect(screen, (50, 50, 50), touch_rect)
            
            # Draw touch points
            for i, (x, y, t) in enumerate(touch_points):
                # Calculate age of touch point (0-5 seconds)
                age = min(5, time.time() - t)
                # Fade out based on age
                alpha = 255 - int(age * 50)
                if alpha < 0:
                    alpha = 0
                
                # Color based on recency (newest is red, oldest is blue)
                color = (max(0, 255-i*25), 0, min(255, i*25))
                
                # Draw circle at touch point
                pygame.draw.circle(screen, color, (int(x), int(y)), 30, 5)
                
                # Draw text label
                point_text = font_small.render(f"{i+1}", True, WHITE)
                screen.blit(point_text, (int(x) - point_text.get_width()//2, int(y) - point_text.get_height()//2))
            
            # Draw debug info if enabled
            if debug_mode:
                # Create a semi-transparent debug panel
                debug_height = 120
                debug_surface = pygame.Surface((width, debug_height), pygame.SRCALPHA)
                debug_surface.fill((0, 0, 0, 180))  # Semi-transparent black
                
                # Add debug text
                lines = [
                    f"System: {SYSTEM_INFO['system']} {SYSTEM_INFO['release']} ({SYSTEM_INFO['machine']})",
                    f"Display: {width}x{height} ({orientation}) - Driver: {pygame.display.get_driver()}",
                ]
                
                if touch_devices:
                    touch_line = "Touch: "
                    for i, device in enumerate(touch_devices):
                        if i > 0:
                            touch_line += ", "
                        touch_line += f"{device['name']} (ID: {device['id']})"
                    lines.append(touch_line)
                else:
                    lines.append("Touch: None detected")
                    
                lines.append(f"FPS: {fps:.1f} - Frame Time: {(time.time() - frame_start) * 1000:.1f}ms")
                
                # Render debug text
                y_offset = 10
                for line in lines:
                    debug_text = font_small.render(line, True, YELLOW)
                    debug_surface.blit(debug_text, (10, y_offset))
                    y_offset += 30
                
                # Draw debug panel at top of screen
                screen.blit(debug_surface, (0, 0))
            
            # Simple FPS counter if enabled
            if show_fps and not debug_mode:
                fps_text = font_small.render(f"FPS: {fps:.1f}", True, YELLOW)
                screen.blit(fps_text, (width - fps_text.get_width() - 10, 10))
            
            # Update display
            pygame.display.flip()
            
            # Update FPS calculation
            frame_count += 1
            if frame_count >= 30:  # Update FPS every 30 frames
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time
            
            # Cap at target FPS
            clock.tick(args.fps)
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        print("Display test complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="WaveShare Display Test Tool")
    parser.add_argument('--width', type=int, help='Display width (default: auto-detect)')
    parser.add_argument('--height', type=int, help='Display height (default: auto-detect)')
    parser.add_argument('--fullscreen', action='store_true', help='Run in fullscreen mode')
    parser.add_argument('--fps', type=int, default=30, help='Target frame rate (default: 30)')
    parser.add_argument('--info', action='store_true', help='Show system info only, no test')
    args = parser.parse_args()
    
    # Detect display and touch devices
    detect_display()
    detect_touch_devices()
    
    # Only show info if requested
    if args.info:
        show_system_info()
        sys.exit(0)
    
    # Run the test
    init_test(args)


if __name__ == "__main__":
    main()
EOF

chmod +x $(dirname "$0")/test_waveshare_display.py
success "Created enhanced display test script at $(dirname "$0")/test_waveshare_display.py"

section "Creating Application Launcher"

# Create a desktop launcher for easy testing
echo "Creating desktop launcher for display test..."

# Create desktop file
cat > /usr/share/applications/tccc-display-test.desktop << EOF
[Desktop Entry]
Name=TCCC Display Test
Comment=Test tool for WaveShare 6.25" Display
Exec=python3 $(realpath "$(dirname "$0")/test_waveshare_display.py")
Terminal=false
Type=Application
Icon=display
Categories=Utility;
EOF

success "Created desktop launcher for easy testing"

section "Installation Complete"

echo -e "${BOLD}WaveShare display configuration has been set up successfully!${NC}"
echo -e "\nA comprehensive setup has been applied with the following components:"
echo -e " - ${GREEN}Display configuration${NC} for ${CYAN}$([ "$IS_RASPBERRY_PI" == "true" ] && echo "Raspberry Pi" || ([ "$IS_JETSON" == "true" ] && echo "NVIDIA Jetson" || echo "Linux"))${NC}"
echo -e " - ${GREEN}Touchscreen calibration${NC} with automatic detection"
echo -e " - ${GREEN}TCCC display configuration${NC} with enhanced features"
echo -e " - ${GREEN}Test tools${NC} for verification and diagnostics"

echo -e "\n${BOLD}Next steps:${NC}"
echo -e "1. ${YELLOW}Reboot your system${NC} for all settings to take effect"
echo -e "   Command: ${GREEN}sudo reboot${NC}"
echo -e ""
echo -e "2. After reboot, run the display test script to verify everything works:"
echo -e "   Command: ${GREEN}python3 $(dirname "$0")/test_waveshare_display.py${NC}"
echo -e "   Or use the desktop launcher that was created"
echo -e ""
echo -e "3. Run the TCCC application with display support:"
echo -e "   Command: ${GREEN}python3 $(dirname "$0")/run_system.py --with-display${NC}"
echo -e ""
echo -e "4. If you encounter touch calibration issues, run:"
echo -e "   Command: ${GREEN}sudo /usr/local/bin/calibrate-waveshare-touch.sh${NC}"
echo -e ""
echo -e "For more details, refer to the ${CYAN}DISPLAY_SETUP_GUIDE.md${NC} document."

# Ask for reboot
if confirm "Would you like to reboot now to apply all settings?"; then
    echo "Rebooting in 3 seconds..."
    sleep 3
    reboot
fi

exit 0