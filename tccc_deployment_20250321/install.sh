#!/bin/bash
#
# TCCC.ai Installation Script
#
# This script will install the TCCC.ai system on the target hardware.

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================\033[0m"
echo -e "${BLUE}   TCCC.ai Installation Script        \033[0m"
echo -e "${BLUE}=======================================\033[0m"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run this script as root or with sudo\033[0m"
    exit 1
fi

# Check system type for platform-specific setup
SYSTEM_TYPE="standard"

if [ -f /etc/nv_tegra_release ]; then
    SYSTEM_TYPE="jetson"
    echo -e "${GREEN}Detected NVIDIA Jetson platform\033[0m"
elif [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    SYSTEM_TYPE="raspberrypi"
    echo -e "${GREEN}Detected Raspberry Pi platform\033[0m"
else
    echo -e "${YELLOW}Detected standard Linux platform\033[0m"
fi

# Choose appropriate installation script
if [ "$SYSTEM_TYPE" = "jetson" ]; then
    echo -e "${GREEN}Running Jetson-specific setup...\033[0m"
    bash ./scripts/setup_jetson_mvp.sh
else
    echo -e "${GREEN}Running standard deployment...\033[0m"
    bash ./scripts/deployment_script.sh
fi

echo -e "${GREEN}Installation complete!\033[0m"
echo -e "${YELLOW}You can now start the system with:\033[0m"
echo -e "   ./start_tccc.sh"
