#\!/bin/bash
# TCCC Audio Module - Master Script
# Provides a unified interface to all audio module functionality

# Define color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Display banner
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}${BOLD}TCCC Audio Module${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${YELLOW}Complete Speech-to-Text Pipeline with Enhanced Audio${NC}"
echo ""

# Check if command provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Available commands:${NC}"
    echo -e "  ${GREEN}run${NC}       - Run the audio pipeline"
    echo -e "  ${GREEN}configure${NC} - Configure the audio pipeline"
    echo -e "  ${GREEN}monitor${NC}   - Run the status monitor"
    echo -e "  ${GREEN}test${NC}      - Run tests on the audio pipeline"
    echo -e "  ${GREEN}demo${NC}      - Run a demonstration"
    echo -e "  ${GREEN}help${NC}      - Show help message"
    echo ""
    echo -e "Example: ${BOLD}$0 run --mode fullsubnet${NC}"
    exit 0
fi

# Parse command
COMMAND=$1
shift

case "$COMMAND" in
    run)
        # Run audio pipeline
        echo -e "${YELLOW}Running audio pipeline...${NC}"
        ${SCRIPT_DIR}/run_enhanced_audio.sh "$@"
        ;;
    configure)
        # Configure audio pipeline
        echo -e "${YELLOW}Configuring audio pipeline...${NC}"
        python3 ${SCRIPT_DIR}/configure_audio_enhancement.py "$@"
        ;;
    monitor)
        # Run status monitor
        echo -e "${YELLOW}Running status monitor...${NC}"
        python3 ${SCRIPT_DIR}/audio_status_monitor.py "$@"
        ;;
    test)
        # Run tests
        echo -e "${YELLOW}Running audio tests...${NC}"
        if [ -d "${SCRIPT_DIR}/fullsubnet_integration" ]; then
            ${SCRIPT_DIR}/fullsubnet_integration/run_comparison_test.sh "$@"
        else
            echo -e "${RED}FullSubNet integration not found.${NC}"
            echo -e "Running basic audio test instead."
            python3 ${SCRIPT_DIR}/microphone_to_text.py --enhancement none
        fi
        ;;
    demo)
        # Run demonstration
        echo -e "${YELLOW}Running audio demonstration...${NC}"
        
        # Check if mode provided
        MODE="auto"
        if [[ "$1" == "--mode" || "$1" == "-m" ]] && [ -n "$2" ]; then
            MODE="$2"
            shift 2
        fi
        
        # First run configuration
        echo -e "${BLUE}Step 1: Configuring for optimal performance...${NC}"
        python3 ${SCRIPT_DIR}/configure_audio_enhancement.py --mode "$MODE" --output "${SCRIPT_DIR}/config/audio_enhancement_demo.yaml"
        
        # Then run status monitor in background
        echo -e "${BLUE}Step 2: Starting status monitor...${NC}"
        xterm -T "TCCC Audio Status" -geometry 100x30+50+50 -e "python3 ${SCRIPT_DIR}/audio_status_monitor.py --simulate" &
        MONITOR_PID=$\!
        
        # Wait for monitor to start
        sleep 2
        
        # Run audio pipeline
        echo -e "${BLUE}Step 3: Running audio pipeline...${NC}"
        ${SCRIPT_DIR}/run_enhanced_audio.sh --mode "$MODE" --duration 20
        
        # Clean up
        kill $MONITOR_PID 2>/dev/null
        ;;
    help|--help|-h)
        # Show help message
        echo -e "${YELLOW}TCCC Audio Module Help${NC}"
        echo -e "${BLUE}============================================${NC}"
        echo -e "This module provides a complete speech-to-text pipeline with enhanced audio."
        echo -e "It includes multiple audio enhancement options and status monitoring."
        echo ""
        echo -e "${YELLOW}Available commands:${NC}"
        echo -e "  ${GREEN}run${NC} [options]       - Run the audio pipeline"
        echo -e "    Options:"
        echo -e "      --mode, -m MODE       Set enhancement mode: auto, fullsubnet, battlefield, both, none"
        echo -e "      --duration, -d SECS   Set recording duration in seconds (default: 15)"
        echo -e "      --output, -o DIR      Set output directory for recordings"
        echo -e ""
        echo -e "  ${GREEN}configure${NC} [options] - Configure the audio pipeline"
        echo -e "    Options:"
        echo -e "      --output, -o FILE     Set output file path"
        echo -e "      --mode, -m MODE       Force specific enhancement mode"
        echo -e "      --print, -p           Print configuration to stdout"
        echo -e ""
        echo -e "  ${GREEN}monitor${NC} [options]   - Run the status monitor"
        echo -e "    Options:"
        echo -e "      --update-interval N   Set update interval in seconds"
        echo -e "      --simulate            Simulate status updates for testing"
        echo -e ""
        echo -e "  ${GREEN}test${NC} [options]      - Run tests on the audio pipeline"
        echo -e "    Options:"
        echo -e "      FILE                  Path to audio file to test"
        echo -e ""
        echo -e "  ${GREEN}demo${NC} [options]      - Run a demonstration"
        echo -e "    Options:"
        echo -e "      --mode, -m MODE       Set enhancement mode for demo"
        echo -e ""
        echo -e "${BLUE}============================================${NC}"
        echo -e "${YELLOW}Examples:${NC}"
        echo -e "  $0 run --mode fullsubnet --duration 30"
        echo -e "  $0 configure --mode battlefield --print"
        echo -e "  $0 monitor --simulate"
        echo -e "  $0 demo --mode both"
        ;;
    *)
        # Unknown command
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo -e "Run '$0 help' for usage information"
        exit 1
        ;;
esac
