#!/bin/bash

# TCCC.ai Deployment Guide Generator
# This script creates a comprehensive deployment guide based on the deployment script

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  TCCC.ai Deployment Guide Generator  ${NC}"
echo -e "${GREEN}=======================================${NC}"

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

INSTALL_DIR=$(pwd)
progress "Creating deployment guide for TCCC.ai system..."

# Create the deployment guide
cat > $INSTALL_DIR/TCCC_DEPLOYMENT_GUIDE.md << EOL
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

\`\`\`bash
# Clone the repository if you haven't already
git clone https://github.com/tccc-ai/tccc-project.git
cd tccc-project

# Make the deployment script executable
chmod +x deployment_script.sh

# Run the deployment script with root privileges
sudo ./deployment_script.sh
\`\`\`

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

\`\`\`bash
# Update package lists
sudo apt-get update

# Install required packages
sudo apt-get install -y \\
    python3-pip \\
    python3-venv \\
    python3-dev \\
    portaudio19-dev \\
    ffmpeg \\
    libsndfile1 \\
    libasound2-dev \\
    libatlas-base-dev \\
    espeak-ng \\
    libsdl2-dev \\
    libsdl2-ttf-dev \\
    libsdl2-image-dev \\
    python3-pygame \\
    git
\`\`\`

### 2. Set Up Python Environment

\`\`\`bash
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
\`\`\`

### 3. Download Models

\`\`\`bash
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
\`\`\`

### 4. Create Startup Scripts

\`\`\`bash
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
\`\`\`

## Display Configuration

The TCCC.ai system is designed to work with the WaveShare 6.25" display in landscape orientation (1560x720). For detailed display setup instructions, please refer to the [Display Setup Guide](DISPLAY_SETUP_GUIDE.md).

### Quick Display Test

To test your display configuration:

\`\`\`bash
# Activate the virtual environment
source venv/bin/activate

# Run the display test
python test_waveshare_display.py

# For fullscreen mode
python test_waveshare_display.py --fullscreen
\`\`\`

## Testing the System

After deployment, you can test the TCCC.ai system:

\`\`\`bash
# Run the test script
./test_tccc.sh

# Or run the system with display support
./start_tccc.sh
\`\`\`

## Running the System

### Using the Systemd Service

If you used the automated deployment script, a systemd service was created:

\`\`\`bash
# Start the TCCC.ai service
sudo systemctl start tccc.service

# Enable the service to start on boot
sudo systemctl enable tccc.service

# Check the service status
sudo systemctl status tccc.service

# View the logs
journalctl -u tccc.service -f
\`\`\`

### Manual Execution

\`\`\`bash
# Start with display support
./start_tccc.sh

# Or with more options
source venv/bin/activate
python run_system.py --with-display --use-microphone
\`\`\`

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   \`\`\`bash
   # Check and install missing dependencies
   python check_dependencies.py
   \`\`\`

2. **NumPy Version Issues**:
   \`\`\`bash
   # Downgrade NumPy to version 1.x
   pip install "numpy<2.0.0"
   \`\`\`

3. **Display Not Working**:
   \`\`\`bash
   # Test display configuration
   python test_waveshare_display.py --dummy
   \`\`\`

4. **Audio Issues**:
   \`\`\`bash
   # List audio devices
   python -c "import sounddevice as sd; print(sd.query_devices())"
   \`\`\`

5. **Verification Failures**:
   \`\`\`bash
   # Run individual module tests
   ./run_module_tests.sh
   \`\`\`

### Getting Help

If you encounter issues not covered in this guide:

1. Check the system logs:
   \`\`\`bash
   journalctl -u tccc.service -f
   \`\`\`

2. Look in the application logs:
   \`\`\`bash
   cat logs/tccc.system.system.log
   \`\`\`

3. Run with verbose logging:
   \`\`\`bash
   python run_system.py --with-display --debug
   \`\`\`

4. Visit the GitHub repository for more information and support.
EOL

# Create a technical reference guide for integration
cat > $INSTALL_DIR/TCCC_INTEGRATION_REFERENCE.md << EOL
# TCCC.ai Technical Integration Reference

This technical reference provides information for integrating and customizing the TCCC.ai system.

## System Architecture

The TCCC.ai system consists of seven main modules:

1. **Audio Pipeline**: Handles audio capture and preprocessing
2. **STT Engine**: Transcribes speech to text using Whisper models
3. **Processing Core**: Coordinates system components and processing
4. **Document Library**: Provides RAG capabilities for medical information
5. **LLM Analysis**: Extracts medical events and generates reports
6. **Data Store**: Persists events, reports, and system state
7. **Display Interface**: Provides visual interface on connected displays

## Configuration

The system's behavior can be customized through configuration files in the \`config/\` directory:

- \`audio_pipeline.yaml\`: Audio capture and processing settings
- \`data_store.yaml\`: Data persistence and backup configuration
- \`document_library.yaml\`: RAG system and embedding model settings
- \`llm_analysis.yaml\`: LLM model and inference settings
- \`processing_core.yaml\`: Core system settings and plugins
- \`stt_engine.yaml\`: Speech recognition models and parameters
- \`jetson_optimizer.yaml\`: Jetson-specific optimizations

## Hardware Optimization

### Jetson-Specific Optimizations

For Jetson devices, the system uses several optimizations:

1. **Model Quantization**: INT8/FP16 precision for models
2. **TensorRT Acceleration**: GPU-accelerated inference
3. **Memory Management**: Configurable limits to prevent OOM errors
4. **Power Profiles**: Dynamic adjustment based on workload

Example configuration in \`jetson_optimizer.yaml\`:

\`\`\`yaml
jetson:
  enable_optimization: true
  power_mode: 15W
  memory_limit:
    stt_engine: 1.5G
    llm_analysis: 1G
  use_tensorrt: true
  quantization:
    whisper: int8
    phi2: int8_float16
\`\`\`

## Display Integration

The display system is designed to work with various displays, prioritizing the WaveShare 6.25" display.

### Display Modes

The system supports two main display modes:

1. **Live View**: Three-column layout showing transcription, events, and card preview
2. **TCCC Card View**: Detailed view of the TCCC casualty card with anatomical diagram

### Environment Variables

Display configuration can be controlled via environment variables:

- \`TCCC_ENABLE_DISPLAY\`: Enable/disable display (0 or 1)
- \`TCCC_DISPLAY_RESOLUTION\`: Set resolution (e.g., "1560x720")
- \`TCCC_DISPLAY_TYPE\`: Display type identifier (e.g., "waveshare_6_25")
- \`TCCC_DISPLAY_ORIENTATION\`: Set to "landscape" or "portrait"

## API Reference

For custom integrations, TCCC.ai provides several Python APIs:

### System API

\`\`\`python
from tccc.system.system import TCCCSystem

# Initialize system
system = TCCCSystem()
system.initialize()

# Process audio file
result = system.process_audio_file("/path/to/audio.wav")

# Get system status
status = system.get_status()
\`\`\`

### Display API

\`\`\`python
from tccc.display.display_interface import DisplayInterface

# Initialize display
display = DisplayInterface(width=1560, height=720)
display.initialize()

# Update with new data
display.update_transcription("New transcription text...")
display.add_significant_event({"time": "14:30", "description": "Event description"})
display.update_card_data(card_data_dict)

# Control display mode
display.set_mode("live")  # or "card"
\`\`\`

## Extending the System

### Custom Plugins

The Processing Core supports custom plugins:

1. Create a Python module in \`src/tccc/processing_core/plugins/\`
2. Implement the plugin interface
3. Register the plugin in \`config/processing_core.yaml\`

Example plugin:

\`\`\`python
class CustomExtractor:
    def __init__(self):
        self.name = "custom_extractor"
    
    def initialize(self, core):
        # Setup code
        return True
    
    def process(self, text):
        # Processing logic
        return {"entities": [...]}
    
    def shutdown(self):
        # Cleanup code
        pass
\`\`\`

### Custom Models

To use custom models:

1. Place model files in \`models/\` directory
2. Update configuration in corresponding YAML file
3. Implement model interface if needed

Example for custom STT model:

\`\`\`yaml
# config/stt_engine.yaml
model:
  provider: custom
  name: my-custom-model
  path: /path/to/model
  params:
    beam_size: 5
    temperature: 0.0
\`\`\`

## Monitoring and Debugging

For system monitoring:

1. **Logs**: Check \`logs/\` directory for detailed component logs
2. **Metrics**: Use \`resource_monitor\` for system resource tracking
3. **Verification**: Run verification scripts to validate components

\`\`\`bash
# Run all verifications
./run_all_verifications.sh

# Monitor resource usage
python -c "from tccc.utils.monitoring import print_resource_usage; print_resource_usage()"
\`\`\`
EOL

success "Deployment guide created: TCCC_DEPLOYMENT_GUIDE.md"
success "Technical reference created: TCCC_INTEGRATION_REFERENCE.md"
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  Deployment documentation complete!   ${NC}"
echo -e "${GREEN}=======================================${NC}"