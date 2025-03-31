#!/bin/bash
#
# TCCC.ai Jetson Edge Deployment Script
# -------------------------------------
# This script sets up a minimal viable product (MVP) deployment 
# of the TCCC.ai system on Jetson Orin Nano hardware.

# Set text colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RESET="\033[0m"

echo -e "${BLUE}=========================================${RESET}"
echo -e "${BLUE}  TCCC.ai Jetson MVP Deployment Script   ${RESET}"
echo -e "${BLUE}=========================================${RESET}"
echo

# Function to check if running on a Jetson device
function check_jetson() {
    if grep -q "tegra\|nvidia" /proc/cpuinfo 2>/dev/null || nvidia-smi | grep -q "Orin" 2>/dev/null; then
        echo -e "${GREEN}[✓] Jetson Orin platform detected${RESET}"
        return 0
    else
        echo -e "${YELLOW}[!] Warning: Not running on a Jetson platform${RESET}"
        echo -e "    Some optimizations will not be available"
        echo -e "    Continue for development/testing purposes only"
        return 1
    fi
}

# Function to install system dependencies
function install_system_dependencies() {
    echo -e "${BLUE}[i] Installing system dependencies...${RESET}"
    
    # Check if apt is available (Linux)
    if command -v apt-get &> /dev/null; then
        echo -e "${BLUE}[i] Updating package lists...${RESET}"
        sudo apt-get update
        
        echo -e "${BLUE}[i] Installing required packages...${RESET}"
        sudo apt-get install -y \
            python3-pip \
            python3-dev \
            python3-setuptools \
            python3-venv \
            build-essential \
            libasound2-dev \
            portaudio19-dev \
            libsndfile1 \
            ffmpeg \
            sox \
            alsa-utils \
            python3-pygame \
            libsdl2-dev \
            libsdl2-ttf-dev \
            libsdl2-image-dev
            
        echo -e "${GREEN}[✓] System dependencies installed${RESET}"
    else
        echo -e "${RED}[✗] Unsupported platform. This script requires apt package manager${RESET}"
        echo -e "    Please install the following dependencies manually:"
        echo -e "    - Python 3.8+ with pip"
        echo -e "    - Development tools (build-essential)"
        echo -e "    - Audio libraries (portaudio, libsndfile, ffmpeg, sox, alsa-utils)"
        return 1
    fi
    
    return 0
}

# Function to set up Python environment
function setup_python_environment() {
    echo -e "${BLUE}[i] Setting up Python environment...${RESET}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo -e "${BLUE}[i] Creating virtual environment...${RESET}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo -e "${BLUE}[i] Activating virtual environment...${RESET}"
    source venv/bin/activate
    
    # Upgrade pip
    echo -e "${BLUE}[i] Upgrading pip...${RESET}"
    pip install --upgrade pip
    
    # Install core dependencies
    echo -e "${BLUE}[i] Installing core Python packages...${RESET}"
    pip install -e .
    
    echo -e "${GREEN}[✓] Python environment setup complete${RESET}"
    return 0
}

# Function to configure microphone
function configure_microphone() {
    echo -e "${BLUE}[i] Configuring microphone...${RESET}"
    
    # Run the existing microphone configuration script
    ./configure_razor_mini3.sh
    
    echo -e "${BLUE}[i] Microphone configuration completed${RESET}"
    return 0
}

# Function to download minimal models
function download_minimal_models() {
    echo -e "${BLUE}[i] Downloading minimal models for edge deployment...${RESET}"
    
    # Create models directory if it doesn't exist
    if [ ! -d "models" ]; then
        mkdir -p models/stt
        mkdir -p models/llm
        mkdir -p models/embeddings
    fi
    
    # Check if pip is available in the current environment
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}[✗] pip not found. Make sure the virtual environment is activated${RESET}"
        return 1
    fi
    
    # Install Hugging Face CLI for model download
    echo -e "${BLUE}[i] Installing Hugging Face hub CLI...${RESET}"
    pip install -q huggingface_hub
    
    # Download small STT model (Whisper tiny.en)
    echo -e "${BLUE}[i] Downloading Whisper tiny.en model...${RESET}"
    python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', download_root='models/stt')"
    
    # Download small LLM model
    echo -e "${BLUE}[i] Downloading small LLM model...${RESET}"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/phi-2', local_dir='models/llm/phi-2', ignore_patterns=['*.safetensors', '*.msgpack', '*.h5'])"
    
    # Download small embedding model
    echo -e "${BLUE}[i] Downloading small embedding model...${RESET}"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='models/embeddings/all-MiniLM-L6-v2')"
    
    echo -e "${GREEN}[✓] Downloaded minimal models for edge deployment${RESET}"
    return 0
}

# Function to create optimized configurations
function create_optimized_configs() {
    echo -e "${BLUE}[i] Creating optimized configurations for edge deployment...${RESET}"
    
    # Create config directory if it doesn't exist
    mkdir -p config
    
    # Create optimized Jetson config
    cat > config/jetson_mvp.yaml << EOF
# TCCC.ai Jetson MVP Configuration
# Optimized for minimal resource usage on edge devices

# System settings
system:
  log_level: INFO
  cache_dir: "data/cache"
  tmp_dir: "data/tmp"

# Audio settings for Jetson
audio_pipeline:
  sample_rate: 16000
  channels: 1
  bit_depth: 16
  chunk_size: 2048
  vad_enabled: true
  vad_threshold: 0.5
  vad_min_speech_duration_ms: 250
  battlefield_noise_filter: true
  enhanced_voice_isolation: true

# STT engine settings
stt_engine:
  model: "tiny.en"
  model_path: "models/stt"
  compute_type: "float16"
  device: "cuda"
  beam_size: 1
  language: "en"

# LLM settings
llm_analysis:
  model: "phi-2"
  model_path: "models/llm/phi-2"
  quantization: "int8"
  max_tokens: 256
  temperature: 0.3
  device: "cuda"

# Document library settings
document_library:
  embedding_model: "all-MiniLM-L6-v2"
  embedding_model_path: "models/embeddings/all-MiniLM-L6-v2"
  documents_path: "data/documents"
  vector_db_path: "data/document_index/document_index.faiss"
  cache_enabled: true
  cache_path: "data/query_cache"

# Power and resource management
power_management:
  mode: "balanced"
  max_memory_usage_gb: 6.0
  cpu_threads: 2
  enable_monitoring: true
  monitoring_interval_seconds: 10
EOF

    echo -e "${GREEN}[✓] Created optimized configurations${RESET}"
    return 0
}

# Function to setup RAG system
function setup_rag_system() {
    echo -e "${BLUE}[i] Setting up RAG system with sample documents...${RESET}"
    
    # Create necessary directories
    mkdir -p data/documents
    mkdir -p data/document_index
    
    # Copy sample documents to documents directory
    if [ -d "data/sample_documents" ]; then
        cp data/sample_documents/* data/documents/
        echo -e "${BLUE}[i] Copied sample documents to data/documents/${RESET}"
    else
        echo -e "${YELLOW}[!] Sample documents not found, skipping${RESET}"
    fi
    
    # Process documents to create vector index
    if [ -f "process_rag_documents.py" ]; then
        echo -e "${BLUE}[i] Processing documents for RAG system...${RESET}"
        python process_rag_documents.py
    else
        echo -e "${YELLOW}[!] Document processing script not found, skipping${RESET}"
    fi
    
    echo -e "${GREEN}[✓] RAG system setup complete${RESET}"
    return 0
}

# Function to verify installation
function verify_installation() {
    echo -e "${BLUE}[i] Verifying installation...${RESET}"
    
    # 1. Check if Python environment is working
    if ! python -c "import sys; print(f'Python {sys.version}')"; then
        echo -e "${RED}[✗] Python verification failed${RESET}"
        return 1
    fi
    
    # 2. Check if TCCC package is installed correctly
    if ! python -c "import tccc; print(f'TCCC package version: {tccc.__version__}')"; then
        echo -e "${RED}[✗] TCCC package verification failed${RESET}"
        return 1
    fi
    
    # 3. Run a basic verification script
    if [ -f "verification_script_system_enhanced.py" ]; then
        echo -e "${BLUE}[i] Running system verification...${RESET}"
        python verification_script_system_enhanced.py
    else
        echo -e "${YELLOW}[!] System verification script not found, skipping${RESET}"
    fi
    
    # 4. Verify Jetson optimizations
    if [ -f "src/tccc/utils/jetson_optimizer.py" ]; then
        echo -e "${BLUE}[i] Verifying Jetson optimizations...${RESET}"
        python -c "from tccc.utils.jetson_optimizer import JetsonOptimizer; optimizer = JetsonOptimizer(); print('Jetson optimizations OK')"
    else
        echo -e "${YELLOW}[!] Jetson optimizer module not found, skipping${RESET}"
    fi
    
    echo -e "${GREEN}[✓] Installation verification complete${RESET}"
    return 0
}

# Function to create a startup script
function create_startup_script() {
    echo -e "${BLUE}[i] Creating startup script...${RESET}"
    
    # Create startup script
    cat > start_tccc_mvp.sh << EOF
#!/bin/bash
#
# TCCC.ai MVP Startup Script
# --------------------------

# Source the virtual environment
source venv/bin/activate

# Source audio environment variables
if [ -f "~/tccc_audio_env.sh" ]; then
    source ~/tccc_audio_env.sh
fi

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Set power mode to balanced
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 1
fi

# Start TCCC system with display integration
echo "Starting TCCC.ai MVP system with display..."

# Check if HDMI display is connected
if xrandr 2>/dev/null | grep -q " connected"; then
    echo "HDMI display detected, enabling display interface"
    export TCCC_ENABLE_DISPLAY=1
    
    # Check for WaveShare 6.25" display specifically (in either orientation)
    if xrandr | grep -q "1560x720"; then
        echo "WaveShare 6.25\" display detected in landscape mode (1560x720)"
        export TCCC_DISPLAY_RESOLUTION="1560x720"
        export TCCC_DISPLAY_TYPE="waveshare_6_25"
    elif xrandr | grep -q "720x1560"; then
        echo "WaveShare 6.25\" display detected in portrait mode (720x1560)"
        echo "Note: Landscape mode (1560x720) is recommended for best experience"
        export TCCC_DISPLAY_RESOLUTION="720x1560"
        export TCCC_DISPLAY_TYPE="waveshare_6_25"
    else
        # Check general display resolution
        display_res=$(xrandr | grep ' connected' | head -n1 | grep -o '[0-9]\+x[0-9]\+' | head -n1)
        if [ -n "$display_res" ]; then
            echo "Display resolution: $display_res"
            export TCCC_DISPLAY_RESOLUTION=$display_res
        else
            echo "Using default display resolution: 720x1560 (WaveShare 6.25\")"
            export TCCC_DISPLAY_RESOLUTION="720x1560"
            export TCCC_DISPLAY_TYPE="waveshare_6_25"
        fi
    fi
else
    echo "No HDMI display detected, running in headless mode"
    export TCCC_ENABLE_DISPLAY=0
fi

# Start the system
python -m tccc.system.system --config config/jetson_mvp.yaml

# This script can be run with:
# ./start_tccc_mvp.sh
EOF

    chmod +x start_tccc_mvp.sh
    
    echo -e "${GREEN}[✓] Created startup script: start_tccc_mvp.sh${RESET}"
    return 0
}

# Main function
function main() {
    # Welcome message
    echo -e "${BLUE}Welcome to the TCCC.ai Jetson MVP deployment script${RESET}"
    echo -e "${YELLOW}This script will set up a minimal viable product deployment of TCCC.ai on Jetson hardware${RESET}"
    echo
    
    # 1. Check if running on Jetson
    check_jetson
    is_jetson=$?
    
    # 2. Ask user to confirm installation
    echo -ne "${YELLOW}Continue with installation? (y/n): ${RESET}"
    read -r continue_install
    if [[ ! $continue_install =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation aborted${RESET}"
        exit 1
    fi
    
    # 3. Install system dependencies
    if ! install_system_dependencies; then
        echo -e "${RED}[✗] Failed to install system dependencies${RESET}"
        echo -e "    Please fix the issues and run the script again"
        exit 1
    fi
    
    # 4. Set up Python environment
    if ! setup_python_environment; then
        echo -e "${RED}[✗] Failed to set up Python environment${RESET}"
        exit 1
    fi
    
    # 5. Configure microphone
    if ! configure_microphone; then
        echo -e "${YELLOW}[!] Microphone configuration may have issues${RESET}"
        echo -e "    You can run ./configure_razor_mini3.sh manually later if needed"
    fi
    
    # 6. Download minimal models
    if ! download_minimal_models; then
        echo -e "${RED}[✗] Failed to download minimal models${RESET}"
        echo -e "    You can try running this step manually: "
        echo -e "    source venv/bin/activate && python download_models.py"
    fi
    
    # 7. Create optimized configurations
    if ! create_optimized_configs; then
        echo -e "${RED}[✗] Failed to create optimized configurations${RESET}"
        exit 1
    fi
    
    # 8. Set up RAG system
    if ! setup_rag_system; then
        echo -e "${YELLOW}[!] RAG system setup may have issues${RESET}"
        echo -e "    You can try setting up the RAG system manually later"
    fi
    
    # 9. Create startup script
    if ! create_startup_script; then
        echo -e "${RED}[✗] Failed to create startup script${RESET}"
        exit 1
    fi
    
    # 10. Verify installation
    if ! verify_installation; then
        echo -e "${YELLOW}[!] Installation verification found issues${RESET}"
        echo -e "    You may need to troubleshoot these before using the system"
    fi
    
    # Installation complete
    echo
    echo -e "${GREEN}=========================================${RESET}"
    echo -e "${GREEN}  TCCC.ai MVP Installation Complete     ${RESET}"
    echo -e "${GREEN}=========================================${RESET}"
    echo
    echo -e "${BLUE}To start the TCCC.ai system:${RESET}"
    echo -e "    ./start_tccc_mvp.sh"
    echo
    echo -e "${BLUE}For troubleshooting:${RESET}"
    echo -e "    - Check logs in the logs/ directory"
    echo -e "    - Run verification scripts individually"
    echo -e "    - Configure microphone with ./configure_razor_mini3.sh"
    echo
}

# Run the main function
main "$@"