#!/bin/bash
#
# TCCC Audio-to-Text System Launcher
#
# This script provides a comprehensive launcher for the TCCC Audio-to-Text
# System with model caching and optimizations. It handles all command-line
# options and provides clear user feedback.
#

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Display banner
echo -e "${BOLD}"
echo "================================================================="
echo -e "${BLUE}                 TCCC AUDIO-TO-TEXT SYSTEM${NC}${BOLD}                "
echo "================================================================="
echo -e "${NC}"

# Check if the virtual environment exists and activate it
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment not found at $SCRIPT_DIR/venv${NC}"
    echo -e "  Creating temporary environment settings..."
fi

# Change to the script directory
cd "$SCRIPT_DIR" || { 
    echo -e "${RED}✗ Error: Failed to change to script directory${NC}"
    exit 1
}

# Create logs directory if it doesn't exist
mkdir -p logs
echo -e "${GREEN}✓ Log directory verified${NC}"

# Set date for log file
LOG_DATE=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/audio_stt_$LOG_DATE.log"
TRANSCRIPT_FILE="transcript_$LOG_DATE.txt"

# Default parameters
USE_MIC=true
INPUT_FILE=""
BATTLEFIELD_MODE=false
MODEL_SIZE="tiny.en"
DISPLAY_ENABLED=false
DEVICE_ID=0
HIGH_QUALITY=false
OPTIMIZE_MEMORY=false

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --file)
            USE_MIC=false
            shift
            ;;
        --input-file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --battlefield)
            BATTLEFIELD_MODE=true
            shift
            ;;
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --display)
            DISPLAY_ENABLED=true
            shift
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --high-quality)
            HIGH_QUALITY=true
            shift
            ;;
        --optimize-memory)
            OPTIMIZE_MEMORY=true
            shift
            ;;
        --help)
            echo -e "${BOLD}Usage:${NC}"
            echo "  $0 [options]"
            echo ""
            echo -e "${BOLD}Options:${NC}"
            echo "  --file                  Use file input instead of microphone"
            echo "  --input-file FILE       Path to input audio file (with --file)"
            echo "  --battlefield           Enable battlefield audio enhancement"
            echo "  --model SIZE            Model size (tiny.en, base, small, medium)"
            echo "  --display               Show transcription on display"
            echo "  --device-id ID          Microphone device ID (default: 0)"
            echo "  --high-quality          Use higher quality audio processing"
            echo "  --optimize-memory       Reduce memory usage for resource-constrained devices"
            echo "  --help                  Show this help message"
            echo ""
            echo -e "${BOLD}Examples:${NC}"
            echo "  $0                      # Standard mode with microphone"
            echo "  $0 --battlefield        # Battlefield enhancement mode"
            echo "  $0 --file --input-file test_speech.wav  # Process file"
            exit 0
            ;;
        *)
            echo -e "${RED}✗ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model preloading is running
echo -e "${BLUE}Checking model cache status...${NC}"
if ps aux | grep "[p]reload_stt_models.py" > /dev/null; then
    echo -e "${GREEN}✓ Model preloading is active - startup will be faster${NC}"
elif [ -f "models/cache_status.json" ]; then
    echo -e "${GREEN}✓ Model cache file found - using cached models${NC}"
else
    echo -e "${YELLOW}⚠ No active model preloading detected${NC}"
    echo -e "  First model load may take longer. Consider running:"
    echo -e "  ${BOLD}python preload_stt_models.py${NC} for faster startup"
fi

# Build command based on options
CMD="python audio_to_stt_optimized.py --output-file $TRANSCRIPT_FILE"

if [ "$USE_MIC" = false ]; then
    CMD="$CMD --file"
    if [ -n "$INPUT_FILE" ]; then
        CMD="$CMD --input-file $INPUT_FILE"
    fi
    echo -e "${BLUE}Mode: File Input${NC}"
    if [ -n "$INPUT_FILE" ]; then
        if [ -f "$INPUT_FILE" ]; then
            echo -e "${GREEN}✓ Input file found: $INPUT_FILE${NC}"
        else
            echo -e "${YELLOW}⚠ Input file not found: $INPUT_FILE${NC}"
            echo -e "  Will use default test file instead"
        fi
    fi
else
    CMD="$CMD --device-id $DEVICE_ID"
    echo -e "${BLUE}Mode: Microphone Input (Device ID: $DEVICE_ID)${NC}"
    # Check if microphone is available
    if command -v arecord &> /dev/null; then
        if arecord -l | grep "card $DEVICE_ID" > /dev/null; then
            echo -e "${GREEN}✓ Microphone detected on device ID $DEVICE_ID${NC}"
        else
            echo -e "${YELLOW}⚠ No microphone detected on device ID $DEVICE_ID${NC}"
            echo -e "  Audio capture may fail. Check your microphone connection"
        fi
    fi
fi

if [ "$BATTLEFIELD_MODE" = true ]; then
    CMD="$CMD --battlefield"
    echo -e "${BLUE}Enhancement: Battlefield Mode Enabled${NC}"
    echo -e "  ${GREEN}✓ Using specialized noise reduction${NC}"
    echo -e "  ${GREEN}✓ Adaptive VAD sensitivity${NC}"
    echo -e "  ${GREEN}✓ Signal enhancement active${NC}"
fi

if [ "$DISPLAY_ENABLED" = true ]; then
    CMD="$CMD --display"
    echo -e "${BLUE}Display: Enabled${NC}"
else
    echo -e "${BLUE}Display: Console Only${NC}"
fi

if [ "$HIGH_QUALITY" = true ]; then
    CMD="$CMD --high-quality"
    echo -e "${BLUE}Quality: High-Quality Processing${NC}"
fi

if [ "$OPTIMIZE_MEMORY" = true ]; then
    CMD="$CMD --optimize-memory"
    echo -e "${BLUE}Resources: Memory-Optimized Mode${NC}"
fi

CMD="$CMD --model $MODEL_SIZE"
echo -e "${BLUE}Model: $MODEL_SIZE${NC}"

# Check for Jetson platform
if [ -f "/etc/nv_tegra_release" ] || [ -d "/opt/nvidia/jetson-inference" ]; then
    echo -e "${BLUE}Hardware: Jetson Platform Detected${NC}"
    echo -e "  ${GREEN}✓ Applying Jetson-specific optimizations${NC}"
    CMD="$CMD --jetson-optimized"
fi

# Report system specs
echo -e "${BLUE}System Information:${NC}"
CPU_INFO=$(grep "model name" /proc/cpuinfo | head -1 | cut -d ":" -f 2 | sed 's/^[ \t]*//')
MEM_INFO=$(grep "MemTotal" /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')
echo -e "  CPU: $CPU_INFO"
echo -e "  Memory: $MEM_INFO"

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "  GPU: $GPU_INFO"
    echo -e "  ${GREEN}✓ GPU acceleration available${NC}"
else
    echo -e "  GPU: Not detected"
    echo -e "  ${YELLOW}⚠ Using CPU-only mode${NC}"
fi

# Display command and execute
echo -e "\n${BOLD}Configuration Summary:${NC}"
echo -e "  Model:       ${BOLD}$MODEL_SIZE${NC}"
echo -e "  Input:       ${BOLD}$([ "$USE_MIC" = true ] && echo "Microphone (ID: $DEVICE_ID)" || echo "File")${NC}"
echo -e "  Enhancement: ${BOLD}$([ "$BATTLEFIELD_MODE" = true ] && echo "Battlefield Mode" || echo "Standard")${NC}"
echo -e "  Quality:     ${BOLD}$([ "$HIGH_QUALITY" = true ] && echo "High" || echo "Standard")${NC}"
echo -e "  Output:      ${BOLD}$([ "$DISPLAY_ENABLED" = true ] && echo "Display + Console" || echo "Console Only")${NC}"
echo -e "  Transcript:  ${BOLD}$TRANSCRIPT_FILE${NC}"
echo -e "  Log file:    ${BOLD}$LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo "================================================================="

# Execute the command
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Audio-to-STT pipeline completed successfully${NC}"
    echo -e "  Transcripts saved to: ${BOLD}$TRANSCRIPT_FILE${NC}"
    
    # Check if transcript file has content
    if [ -f "$TRANSCRIPT_FILE" ] && [ -s "$TRANSCRIPT_FILE" ]; then
        echo -e "\n${BLUE}Transcript Summary:${NC}"
        echo -e "${YELLOW}----------------------------------------${NC}"
        tail -n 5 "$TRANSCRIPT_FILE"
        echo -e "${YELLOW}----------------------------------------${NC}"
        WORD_COUNT=$(wc -w < "$TRANSCRIPT_FILE")
        LINE_COUNT=$(wc -l < "$TRANSCRIPT_FILE")
        echo -e "  ${GREEN}✓ ${WORD_COUNT} words in ${LINE_COUNT} utterances transcribed${NC}"
    else
        echo -e "  ${YELLOW}⚠ No transcriptions generated${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ Audio-to-STT pipeline exited with error code: $EXIT_CODE${NC}"
    echo -e "  Check log file for details: ${BOLD}$LOG_FILE${NC}"
fi

# Display stats if available
if [ -f "stats_$LOG_DATE.json" ]; then
    echo -e "\n${BLUE}Performance Metrics:${NC}"
    python -c "import json; data=json.load(open('stats_$LOG_DATE.json')); print(f\"  Model loading time: {data.get('model_load_time', 0):.2f}s\"); print(f\"  Processing time: {data.get('processing_time', 0):.2f}s\"); print(f\"  Real-time factor: {data.get('rtf', 0):.2f}x\")"
fi

echo -e "\n${BLUE}For detailed options run: ${BOLD}$0 --help${NC}"

exit $EXIT_CODE