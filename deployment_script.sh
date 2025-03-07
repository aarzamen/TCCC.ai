#!/bin/bash

# TCCC.ai Deployment Script
# This script sets up the entire TCCC.ai system for deployment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}   TCCC.ai System Deployment Script   ${NC}"
echo -e "${GREEN}=======================================${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display progress
progress() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Function to display success
success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to display error
error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run this script as root or with sudo"
fi

INSTALL_DIR=$(pwd)
progress "Installation directory: $INSTALL_DIR"

# Step 1: Check system dependencies
progress "Checking system dependencies..."

if ! command_exists python3; then
    error "Python 3 is not installed. Please install Python 3.10 or later."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$(echo "$PYTHON_VERSION < 3.10" | bc)" -eq 1 ]; then
    error "Python 3.10 or later is required, found $PYTHON_VERSION"
fi
success "Python $PYTHON_VERSION is installed"

# Step 2: Install required system packages
progress "Installing system dependencies..."

apt-get update
apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    libatlas-base-dev \
    espeak-ng \
    git

success "System dependencies installed"

# Step 3: Create and activate virtual environment
progress "Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment by sourcing
source venv/bin/activate

# Step 4: Upgrade pip
progress "Upgrading pip..."
pip install --upgrade pip

# Step 5: Check if running on Jetson platform
progress "Checking if running on NVIDIA Jetson platform..."
JETSON_PLATFORM=false

if [ -f /etc/nv_tegra_release ]; then
    JETSON_PLATFORM=true
    progress "Detected NVIDIA Jetson platform, installing Jetson-specific dependencies..."
    
    # Install Jetson-specific packages
    apt-get install -y \
        nvidia-jetpack \
        nvidia-tensorrt \
        python3-libnvinfer \
        jtop
    
    pip install jetson-stats
fi

# Step 6: Install Python dependencies with NumPy 1.x to avoid compatibility issues
progress "Installing Python dependencies..."

# Downgrade numpy to 1.24.3 to avoid compatibility issues with transformers
pip install "numpy<2.0.0"

# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio flake8 black

# Step 7: Install the TCCC package in development mode
progress "Installing TCCC package in development mode..."
pip install -e .

# Step 8: Download required models
progress "Downloading required models..."

# Create directories for models if they don't exist
mkdir -p models/stt
mkdir -p models/llm
mkdir -p models/embeddings

# Download faster-whisper base model
if [ ! -d "models/stt/whisper-base" ]; then
    progress "Downloading faster-whisper base model..."
    python -c "from faster_whisper import WhisperModel; WhisperModel('base', download_root='models/stt/whisper-base')"
fi

# Download sentence transformer model for embeddings
if [ ! -d "models/embeddings/all-MiniLM-L12-v2" ]; then
    progress "Downloading sentence transformer model..."
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2', cache_folder='models/embeddings')"
fi

# Process RAG documents
progress "Processing RAG documents..."
python download_rag_documents.py
python process_rag_documents.py

# Step 9: Run verification script
progress "Running verification script..."
./run_all_verifications.sh || echo -e "${YELLOW}[WARNING]${NC} Some verifications failed, but continuing with deployment"

# Step 10: Install display-specific dependencies
progress "Installing display dependencies..."
if [ "$JETSON_PLATFORM" = true ]; then
    # Install display drivers and libraries for Jetson
    apt-get install -y \
        libsdl2-dev \
        libsdl2-ttf-dev \
        libsdl2-image-dev \
        python3-pygame
    
    pip install pygame
else
    # For non-Jetson platforms
    pip install pygame
fi

# Step 11: Create a desktop shortcut/service
progress "Creating system service for TCCC.ai..."

# Create systemd service file
cat > /etc/systemd/system/tccc.service << EOL
[Unit]
Description=TCCC.ai System Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/src/tccc/system/run_system.py
Restart=on-failure
Environment="DISPLAY=:0"
Environment="PYTHONPATH=$INSTALL_DIR"

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable tccc.service

# Step 12: Create startup script
progress "Creating startup script..."

cat > $INSTALL_DIR/start_tccc.sh << EOL
#!/bin/bash
# TCCC.ai Startup Script

# Activate virtual environment
source $INSTALL_DIR/venv/bin/activate

# Run the TCCC system with monitor support
python $INSTALL_DIR/src/tccc/system/run_system.py --with-display

# Deactivate virtual environment when done
deactivate
EOL

chmod +x $INSTALL_DIR/start_tccc.sh

# Step 13: Create test script
progress "Creating test script..."

cat > $INSTALL_DIR/test_tccc.sh << EOL
#!/bin/bash
# TCCC.ai Test Script

# Activate virtual environment
source $INSTALL_DIR/venv/bin/activate

# Test with sample audio
python $INSTALL_DIR/verification_script_system_enhanced.py

# Deactivate virtual environment when done
deactivate
EOL

chmod +x $INSTALL_DIR/test_tccc.sh

# Step 14: Final instructions
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}   TCCC.ai System Deployment Complete  ${NC}"
echo -e "${GREEN}=======================================${NC}"

echo -e "${YELLOW}To start the TCCC.ai system:${NC}"
echo -e "1. Run as a service: ${GREEN}systemctl start tccc.service${NC}"
echo -e "2. Run manually: ${GREEN}./start_tccc.sh${NC}"
echo -e "3. Run test: ${GREEN}./test_tccc.sh${NC}"

echo -e "\n${YELLOW}To view system logs:${NC}"
echo -e "${GREEN}journalctl -u tccc.service -f${NC}"

echo -e "\n${YELLOW}NOTE:${NC} For proper functioning with Waveshare display,"
echo -e "ensure the display is properly connected and configured."

echo -e "\n${GREEN}Deployment completed successfully!${NC}"