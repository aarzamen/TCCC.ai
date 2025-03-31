#!/bin/bash
#
# Optimized Audio-to-STT Pipeline Launcher
#
# This script launches the optimized audio capture to transcription pipeline
# with model caching for improved performance.
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate the virtual environment
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Activated virtual environment"
else
    echo "Warning: Virtual environment not found at $SCRIPT_DIR/venv"
fi

# Change to the script directory
cd "$SCRIPT_DIR" || { echo "Error: Failed to change to script directory"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Set date for log file
LOG_DATE=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/audio_stt_$LOG_DATE.log"
TRANSCRIPT_FILE="transcript_$LOG_DATE.txt"

# Check if we should use the microphone or a file
USE_MIC=true
INPUT_FILE=""
BATTLEFIELD_MODE=false
MODEL_SIZE="tiny.en"
DISPLAY_ENABLED=false

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command based on options
CMD="python audio_to_stt_optimized.py --output-file $TRANSCRIPT_FILE"

if [ "$USE_MIC" = false ]; then
    CMD="$CMD --file"
    if [ -n "$INPUT_FILE" ]; then
        CMD="$CMD --input-file $INPUT_FILE"
    fi
fi

if [ "$BATTLEFIELD_MODE" = true ]; then
    CMD="$CMD --battlefield"
fi

if [ "$DISPLAY_ENABLED" = true ]; then
    CMD="$CMD --display"
fi

CMD="$CMD --model $MODEL_SIZE"

# Display command and execute
echo "Running: $CMD"
echo "Transcripts will be saved to: $TRANSCRIPT_FILE"
echo "Log file: $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop"
echo "===================="

# Execute the command
$CMD 2>&1 | tee "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Audio-to-STT pipeline completed successfully"
    echo "Transcripts saved to: $TRANSCRIPT_FILE"
else
    echo ""
    echo "Audio-to-STT pipeline exited with error code: $EXIT_CODE"
    echo "Check log file for details: $LOG_FILE"
fi

exit $EXIT_CODE