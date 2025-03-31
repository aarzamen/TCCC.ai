#\!/bin/bash
# Enhanced Audio Pipeline Launcher
# Provides easy access to all audio enhancement options

# Define color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default settings
ENHANCEMENT_MODE="auto"
RECORDING_DURATION=15
OUTPUT_DIR="$(pwd)/enhanced_audio_output"
SHOW_STATUS=true

# Display banner
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}TCCC Enhanced Audio Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode|-m)
      ENHANCEMENT_MODE="$2"
      shift 2
      ;;
    --duration|-d)
      RECORDING_DURATION="$2"
      shift 2
      ;;
    --output|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-status)
      SHOW_STATUS=false
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --mode, -m MODE       Set enhancement mode: auto, fullsubnet, battlefield, both, none"
      echo "  --duration, -d SECS   Set recording duration in seconds (default: 15)"
      echo "  --output, -o DIR      Set output directory for recordings"
      echo "  --no-status           Disable status monitoring"
      echo "  --help, -h            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Validate enhancement mode
case "$ENHANCEMENT_MODE" in
  auto|fullsubnet|battlefield|both|none)
    # Valid mode, continue
    ;;
  *)
    echo -e "${RED}Error: Invalid enhancement mode '$ENHANCEMENT_MODE'${NC}"
    echo "Valid modes: auto, fullsubnet, battlefield, both, none"
    exit 1
    ;;
esac

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo -e " - Enhancement mode: ${GREEN}$ENHANCEMENT_MODE${NC}"
echo -e " - Recording duration: ${GREEN}$RECORDING_DURATION seconds${NC}"
echo -e " - Output directory: ${GREEN}$OUTPUT_DIR${NC}"

# Define timestamps for filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PREFIX="$OUTPUT_DIR/tccc_audio_$TIMESTAMP"

# Copy configuration files
echo -e "\n${YELLOW}Setting up environment...${NC}"
if [ "$ENHANCEMENT_MODE" = "fullsubnet" ] || [ "$ENHANCEMENT_MODE" = "both" ]; then
  # Check if FullSubNet is available
  if [ -d "fullsubnet_integration/fullsubnet" ]; then
    echo -e " - ${GREEN}FullSubNet enhancer is available${NC}"
  else
    echo -e " - ${YELLOW}FullSubNet repository not found, using offline mode${NC}"
  fi
fi

# Run audio capture with selected enhancement
echo -e "\n${YELLOW}Starting audio capture with '$ENHANCEMENT_MODE' enhancement...${NC}"
echo -e "${GREEN}Recording for $RECORDING_DURATION seconds${NC}"
echo -e "${YELLOW}Speak clearly into the microphone${NC}"
echo -e "${BLUE}============================================${NC}"

# Set environment variables for the script
export TCCC_AUDIO_OUTPUT_PREFIX="$OUTPUT_PREFIX"
export TCCC_AUDIO_RECORDING_DURATION="$RECORDING_DURATION"

# Launch status monitor in background if enabled
STATUS_PID=""
if [ "$SHOW_STATUS" = true ]; then
  # Start status monitor in background
  (
    while true; do
      echo -e "\n${YELLOW}Audio Enhancement Status:${NC}"
      # Check for status files updated by the enhancers
      if [ -f "$OUTPUT_PREFIX.stats" ]; then
        cat "$OUTPUT_PREFIX.stats"
      fi
      sleep 1
    done
  ) &
  STATUS_PID=$\!
  
  # Set trap to kill status monitor on exit
  trap "kill $STATUS_PID 2>/dev/null" EXIT
fi

# Run the microphone to text script
python3 microphone_to_text.py --enhancement "$ENHANCEMENT_MODE"

# Process completed
echo -e "\n${GREEN}Audio capture and enhancement complete\!${NC}"
echo -e "${YELLOW}Results:${NC}"
echo -e " - Transcription: ${GREEN}$OUTPUT_PREFIX.txt${NC}"
echo -e " - Enhanced audio: ${GREEN}$OUTPUT_PREFIX.wav${NC}"
echo -e " - Original audio: ${GREEN}$OUTPUT_PREFIX.original.wav${NC}"

# Copy the latest files to the output directory
cp improved_transcription.txt "$OUTPUT_PREFIX.txt" 2>/dev/null
cp improved_audio.wav "$OUTPUT_PREFIX.wav" 2>/dev/null
cp highquality_audio.wav "$OUTPUT_PREFIX.original.wav" 2>/dev/null

echo -e "\n${BLUE}============================================${NC}"
echo -e "${GREEN}Process completed successfully${NC}"
