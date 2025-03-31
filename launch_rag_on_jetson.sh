#!/bin/bash
#
# Launch script for the TCCC RAG system on Jetson hardware
# This script performs optimization and starts the RAG query service
#

set -e

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"
CONFIG_DIR="$SCRIPT_DIR/config"
OPTIMIZED_CONFIG="$CONFIG_DIR/optimized_jetson_rag.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/rag_system_$(date +%Y%m%d-%H%M%S).log"

# Create directories if they don't exist
mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

cleanup() {
    log "Shutting down RAG system..."
    # Add cleanup code here if needed
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Log system info
log "Starting TCCC RAG System on Jetson"
log "System information:"
uname -a >> "$LOG_FILE"
free -h >> "$LOG_FILE"
nvidia-smi >> "$LOG_FILE" 2>&1 || log "NVIDIA SMI not available"

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    log "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    log "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if optimized config exists, run optimization if it doesn't
if [ ! -f "$OPTIMIZED_CONFIG" ]; then
    log "Optimized configuration not found. Running optimization..."
    python "$SCRIPT_DIR/optimize_rag_performance.py" --optimize-all
else
    log "Using existing optimized configuration."
fi

# Launch RAG query service
log "Launching RAG query service..."

# First check if Jetson memory is limited (< 8GB)
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 8000 ]; then
    log "Limited memory detected ($TOTAL_MEM MB). Using low-memory mode."
    # Add low memory flag
    MEMORY_FLAG="--low-memory"
else
    MEMORY_FLAG=""
fi

# Start the RAG explorer in the background
log "Starting RAG explorer interface..."
python "$SCRIPT_DIR/jetson_rag_explorer.py" --config "$OPTIMIZED_CONFIG" $MEMORY_FLAG &
EXPLORER_PID=$!

log "RAG explorer started with PID $EXPLORER_PID"
log "RAG system is now running. Press Ctrl+C to stop."

# Wait for the process to finish
wait $EXPLORER_PID

log "RAG system stopped."
exit 0