#\!/bin/bash
# FullSubNet Setup Script for TCCC Project
# This script sets up the FullSubNet speech enhancement model

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== FullSubNet Setup for TCCC Project ==="
echo "This script will download and set up FullSubNet for Nvidia Jetson"

# Create necessary directories
mkdir -p fullsubnet
mkdir -p models

# Clone FullSubNet repository
if [ \! -d "fullsubnet/.git" ]; then
    echo "Cloning FullSubNet repository..."
    git clone https://github.com/Audio-WestlakeU/FullSubNet.git fullsubnet
    cd fullsubnet
    # Checkout a specific commit for stability
    git checkout b8a6de731cf74bc52adb47e3b6ba3193eaa7736e
    cd ..
else
    echo "FullSubNet repository already exists."
fi

# Install dependencies
echo "Installing dependencies..."
pip install torch==1.12.0 torchaudio==0.12.0
pip install soundfile librosa pesq pypesq
pip install -e fullsubnet/

# Download pre-trained model
if [ \! -f "models/fullsubnet_best_model_58epochs.pth" ]; then
    echo "Downloading pre-trained model..."
    mkdir -p models/temp
    
    # Download from release
    wget https://github.com/Audio-WestlakeU/FullSubNet/releases/download/v1.0.0/fullsubnet_best_model_58epochs.pth -O models/fullsubnet_best_model_58epochs.pth
    
    echo "Pre-trained model downloaded to models/"
else
    echo "Pre-trained model already exists."
fi

# Create YAML configuration for FullSubNet
cat > fullsubnet_config.yaml << 'EOFYAML'
fullsubnet:
  enabled: true
  model_path: "fullsubnet_integration/models/fullsubnet_best_model_58epochs.pth"
  use_gpu: true
  sample_rate: 16000
  batch_size: 1
  chunk_size: 16000  # 1 second of audio at 16kHz
  frame_length: 512
  frame_shift: 256
  n_fft: 512
  win_length: 512
  hop_length: 256
  normalized_input: true
  normalized_output: true
  gpu_acceleration: true
  fallback_to_cpu: true
  voiceprint_database_path: ""
EOFYAML

echo "FullSubNet setup complete\!"
echo "Next steps:"
echo "1. Test the setup using test_fullsubnet.py"
echo "2. Integrate with microphone_to_text.py using fullsubnet_enhancer.py"
