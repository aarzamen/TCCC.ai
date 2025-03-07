#!/bin/bash
#
# TCCC.ai Deployment Package Creator
#
# This script builds a deployment package for the TCCC.ai system that
# can be deployed to target hardware. It includes all necessary
# scripts, configurations, and instructions.

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   TCCC.ai Deployment Package Creator  ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Set the build directory
BUILD_DIR="tccc_deployment_$(date +%Y%m%d)"
ARCHIVE_NAME="${BUILD_DIR}.tar.gz"

# Check if we need to clean up previous build
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Previous build directory found: $BUILD_DIR${NC}"
    read -p "Do you want to remove it and create a fresh build? (y/n): " clean_build
    if [[ $clean_build =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing previous build directory...${NC}"
        rm -rf "$BUILD_DIR"
    else
        echo -e "${RED}Cannot proceed with existing build directory. Exiting.${NC}"
        exit 1
    fi
fi

# Create build directory structure
echo -e "${GREEN}Creating build directory structure...${NC}"
mkdir -p "$BUILD_DIR"/{scripts,config,models,data,docs}

# Run tests and verifications
echo -e "${GREEN}Running verifications to ensure system is ready for deployment...${NC}"
./run_all_verifications.sh --quick --output "$BUILD_DIR/verification_results.txt"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Some verifications failed. Review the results in $BUILD_DIR/verification_results.txt${NC}"
    read -p "Do you want to continue with the build anyway? (y/n): " continue_build
    if [[ ! $continue_build =~ ^[Yy]$ ]]; then
        echo -e "${RED}Build aborted due to verification failures.${NC}"
        exit 1
    fi
fi

# Copy deployment scripts
echo -e "${GREEN}Copying deployment scripts...${NC}"
cp deployment_script.sh "$BUILD_DIR/scripts/"
cp setup_jetson_mvp.sh "$BUILD_DIR/scripts/"
cp configure_razor_mini3.sh "$BUILD_DIR/scripts/"
cp run_all_verifications.sh "$BUILD_DIR/scripts/"
cp download_models.py "$BUILD_DIR/scripts/"
cp process_rag_documents.py "$BUILD_DIR/scripts/"
cp download_rag_documents.py "$BUILD_DIR/scripts/"
cp run_system.py "$BUILD_DIR/scripts/"

# Create package installation script
cat > "$BUILD_DIR/install.sh" << EOL
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

echo -e "\${BLUE}=======================================${NC}"
echo -e "\${BLUE}   TCCC.ai Installation Script        ${NC}"
echo -e "\${BLUE}=======================================${NC}"

# Check if running as root
if [ "\$EUID" -ne 0 ]; then
    echo -e "\${RED}Please run this script as root or with sudo${NC}"
    exit 1
fi

# Check system type for platform-specific setup
SYSTEM_TYPE="standard"

if [ -f /etc/nv_tegra_release ]; then
    SYSTEM_TYPE="jetson"
    echo -e "\${GREEN}Detected NVIDIA Jetson platform${NC}"
elif [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    SYSTEM_TYPE="raspberrypi"
    echo -e "\${GREEN}Detected Raspberry Pi platform${NC}"
else
    echo -e "\${YELLOW}Detected standard Linux platform${NC}"
fi

# Choose appropriate installation script
if [ "\$SYSTEM_TYPE" = "jetson" ]; then
    echo -e "\${GREEN}Running Jetson-specific setup...${NC}"
    bash ./scripts/setup_jetson_mvp.sh
else
    echo -e "\${GREEN}Running standard deployment...${NC}"
    bash ./scripts/deployment_script.sh
fi

echo -e "\${GREEN}Installation complete!${NC}"
echo -e "\${YELLOW}You can now start the system with:${NC}"
echo -e "   ./start_tccc.sh"
EOL

chmod +x "$BUILD_DIR/install.sh"

# Copy configuration files
echo -e "${GREEN}Copying configuration files...${NC}"
cp -r config/* "$BUILD_DIR/config/"

# Copy sample data
echo -e "${GREEN}Copying sample data...${NC}"
cp -r data/sample_documents "$BUILD_DIR/data/"

# Copy documentation
echo -e "${GREEN}Copying documentation...${NC}"
cp README.md "$BUILD_DIR/docs/"
cp TCCC_DEPLOYMENT_GUIDE.md "$BUILD_DIR/docs/" 2>/dev/null || echo -e "${YELLOW}Deployment guide not found, skipping...${NC}"
cp BATTLEFIELD_AUDIO_IMPROVEMENTS.md "$BUILD_DIR/docs/" 2>/dev/null || echo -e "${YELLOW}Audio improvements guide not found, skipping...${NC}"
cp DISPLAY_SETUP_GUIDE.md "$BUILD_DIR/docs/" 2>/dev/null || echo -e "${YELLOW}Display setup guide not found, skipping...${NC}"

# Generate requirements file with exact versions
echo -e "${GREEN}Generating requirements file with exact versions...${NC}"
pip freeze > "$BUILD_DIR/requirements.txt"

# Copy source code
echo -e "${GREEN}Copying source code...${NC}"
cp -r src "$BUILD_DIR/"
cp setup.py "$BUILD_DIR/"

# Create a verification script for after installation
cat > "$BUILD_DIR/verify_installation.sh" << EOL
#!/bin/bash
#
# TCCC.ai Installation Verification Script
#
# This script verifies that the TCCC.ai system was installed correctly.

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "\${GREEN}=======================================${NC}"
echo -e "\${GREEN}   TCCC.ai Installation Verification  ${NC}"
echo -e "\${GREEN}=======================================${NC}"

# Check Python installation
echo -e "\${YELLOW}Checking Python installation...${NC}"
if command -v python3 >/dev/null 2>&1; then
    python_version=\$(python3 --version)
    echo -e "\${GREEN}Python installed: \$python_version${NC}"
else
    echo -e "\${RED}Python 3 not found!${NC}"
    exit 1
fi

# Check virtual environment
echo -e "\${YELLOW}Checking virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "\${GREEN}Virtual environment found${NC}"
else
    echo -e "\${RED}Virtual environment not found!${NC}"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check TCCC package installation
echo -e "\${YELLOW}Checking TCCC package installation...${NC}"
if python3 -c "import tccc" 2>/dev/null; then
    echo -e "\${GREEN}TCCC package installed${NC}"
else
    echo -e "\${RED}TCCC package not installed!${NC}"
    exit 1
fi

# Check for required data files
echo -e "\${YELLOW}Checking for required data files...${NC}"
if [ -d "data/documents" ]; then
    echo -e "\${GREEN}Documents directory found${NC}"
else
    echo -e "\${YELLOW}Documents directory not found, please run download_rag_documents.py${NC}"
fi

# Run a basic system check
echo -e "\${YELLOW}Running basic system check...${NC}"
if [ -f "scripts/run_all_verifications.sh" ]; then
    bash scripts/run_all_verifications.sh --quick
else
    echo -e "\${RED}Verification script not found!${NC}"
    exit 1
fi

echo -e "\${GREEN}Installation verification completed!${NC}"
EOL

chmod +x "$BUILD_DIR/verify_installation.sh"

# Create the archive
echo -e "${GREEN}Creating deployment archive...${NC}"
tar -czf "$ARCHIVE_NAME" "$BUILD_DIR"

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}   Deployment Package Created!         ${NC}"
echo -e "${GREEN}=======================================${NC}"
echo -e "Package: ${YELLOW}$ARCHIVE_NAME${NC}"
echo -e "Size: ${YELLOW}$(du -h "$ARCHIVE_NAME" | cut -f1)${NC}"
echo -e "\nTo deploy, transfer this archive to the target system and run:"
echo -e "${BLUE}tar -xzf $ARCHIVE_NAME${NC}"
echo -e "${BLUE}cd $BUILD_DIR${NC}"
echo -e "${BLUE}sudo ./install.sh${NC}"