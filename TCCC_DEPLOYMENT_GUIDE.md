# TCCC.ai Deployment Guide

This guide provides step-by-step instructions for deploying the TCCC.ai system on both Jetson devices and standard Linux systems.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Automated Deployment](#automated-deployment)
3. [Manual Deployment](#manual-deployment)
4. [Display Configuration](#display-configuration)
5. [Testing the System](#testing-the-system)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

- **Processor**:
  - Jetson Orin Nano (preferred)
  - Any x86_64 system with 4+ cores
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 15GB available space (SSD recommended)
- **Display**: 
  - WaveShare 6.25" display (1560x720 resolution) recommended
  - Or any HDMI monitor

### Software Requirements

- **Operating System**: 
  - JetPack 5.1.1+ for Jetson devices
  - Ubuntu 20.04/22.04 for standard PCs
- **Python**: 3.10 or later
- **Dependencies**: 
  - CUDA drivers (for Jetson or NVIDIA GPUs)
  - Audio libraries (PortAudio, ALSA)
  - SDL libraries (for display)

## Automated Deployment

For a quick and complete setup, use the provided deployment script:

```bash
# Clone the repository if you haven't already
git clone https://github.com/tccc-ai/tccc-project.git
cd tccc-project

# Make the deployment script executable
chmod +x deployment_script.sh

# Run the deployment script with root privileges
sudo ./deployment_script.sh
```

This script performs the following actions:
1. Installs all system dependencies
2. Creates a Python virtual environment
3. Installs required Python packages
4. Downloads and sets up models
5. Configures display support
6. Sets up system service for auto-start
7. Creates convenient scripts for testing and running

## Manual Deployment

If you prefer to perform the steps manually or need to customize the installation:

### 1. Install System Dependencies

```bash
# Update package lists
sudo apt-get update

# Install required packages
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    libatlas-base-dev \
    espeak-ng \
    libsdl2-dev \
    libsdl2-ttf-dev \
    libsdl2-image-dev \
    python3-pygame \
    git
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install NumPy 1.x (critical for compatibility)
pip install "numpy<2.0.0"

# Install required packages
pip install -r requirements.txt

# Install the TCCC package in development mode
pip install -e .
```

### 3. Download Models

```bash
# Create model directories
mkdir -p models/stt
mkdir -p models/llm
mkdir -p models/embeddings

# Download faster-whisper base model
python -c "from faster_whisper import WhisperModel; WhisperModel('base', download_root='models/stt/whisper-base')"

# Download sentence transformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2', cache_folder='models/embeddings')"

# Process RAG documents
python download_rag_documents.py
python process_rag_documents.py
```

### 4. Create Startup Scripts

```bash
# Create startup script
cat > start_tccc.sh << 'EOF'
#!/bin/bash
# TCCC.ai Startup Script

# Activate virtual environment
source ./venv/bin/activate

# Run the TCCC system with monitor support
python run_system.py --with-display

# Deactivate virtual environment when done
deactivate
EOF

chmod +x start_tccc.sh

# Create test script
cat > test_tccc.sh << 'EOF'
#!/bin/bash
# TCCC.ai Test Script

# Activate virtual environment
source ./venv/bin/activate

# Test with sample audio
python verification_script_system_enhanced.py

# Deactivate virtual environment when done
deactivate
EOF

chmod +x test_tccc.sh
```

## Display Configuration

The TCCC.ai system is designed to work with the WaveShare 6.25" display in landscape orientation (1560x720). For detailed display setup instructions, please refer to the [Display Setup Guide](DISPLAY_SETUP_GUIDE.md).

### Quick Display Test

To test your display configuration:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the display test
python test_waveshare_display.py

# For fullscreen mode
python test_waveshare_display.py --fullscreen
```

## Testing the System

After deployment, you can test the TCCC.ai system:

```bash
# Run the test script
./test_tccc.sh

# Or run the system with display support
./start_tccc.sh
```

## Running the System

### Using the Systemd Service

If you used the automated deployment script, a systemd service was created:

```bash
# Start the TCCC.ai service
sudo systemctl start tccc.service

# Enable the service to start on boot
sudo systemctl enable tccc.service

# Check the service status
sudo systemctl status tccc.service

# View the logs
journalctl -u tccc.service -f
```

### Manual Execution

```bash
# Start with display support
./start_tccc.sh

# Or with more options
source venv/bin/activate
python run_system.py --with-display --use-microphone
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   # Check and install missing dependencies
   python check_dependencies.py
   ```

2. **NumPy Version Issues**:
   ```bash
   # Downgrade NumPy to version 1.x
   pip install "numpy<2.0.0"
   ```

3. **Display Not Working**:
   ```bash
   # Test display configuration
   python test_waveshare_display.py --dummy
   ```

4. **Audio Issues**:
   ```bash
   # List audio devices
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

5. **Verification Failures**:
   ```bash
   # Run individual module tests
   ./run_module_tests.sh
   ```

### Getting Help

If you encounter issues not covered in this guide:

1. Check the system logs:
   ```bash
   journalctl -u tccc.service -f
   ```

2. Look in the application logs:
   ```bash
   cat logs/tccc.system.system.log
   ```

3. Run with verbose logging:
   ```bash
   python run_system.py --with-display --debug
   ```

4. Visit the GitHub repository for more information and support.
