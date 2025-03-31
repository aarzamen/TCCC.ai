#!/bin/bash
#
# STT Model Preloader Startup Script
# 
# This script preloads speech-to-text models at system startup to reduce
# initialization time when the TCCC system is launched.
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (one level up from scripts/system)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Activate the virtual environment
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Activated virtual environment"
else
    echo "Warning: Virtual environment not found at $PROJECT_ROOT/venv"
fi

# Change to the project directory
cd "$PROJECT_ROOT" || { echo "Error: Failed to change to project directory"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Set date for log file
LOG_DATE=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/stt_preload_$LOG_DATE.log"

# Determine if we have CUDA available and which models to preload
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "CUDA is available, preloading models on GPU"
    DEVICE="cuda"
    COMPUTE_TYPE="float16"
    
    # Check GPU memory to determine which models to preload
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
    
    if [ "$GPU_MEM" -gt "8000" ]; then
        # More than 8GB GPU memory, preload multiple models
        MODELS="tiny.en base small"
    elif [ "$GPU_MEM" -gt "4000" ]; then
        # 4-8GB GPU memory, preload base and tiny models
        MODELS="tiny.en base"
    else
        # Less than 4GB GPU memory, preload only tiny model
        MODELS="tiny.en"
    fi
else
    echo "CUDA not available, preloading models on CPU"
    DEVICE="cpu"
    COMPUTE_TYPE="int8"
    MODELS="tiny.en"
fi

# Run the preloader script in the background
echo "Starting STT model preloader with models: $MODELS"
echo "Preloading models... See $LOG_FILE for details"
python preload_stt_models.py --models $MODELS --device $DEVICE --compute-type $COMPUTE_TYPE > "$LOG_FILE" 2>&1 &

# Write PID to file for management
echo $! > logs/preload_stt_pid.txt

# Exit successfully
exit 0