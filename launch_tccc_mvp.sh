#!/bin/bash
# TCCC MVP Launch Script
# This script initializes hardware and launches the TCCC MVP application

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "==============================================="
    echo "       TCCC MVP - TACTICAL COMBAT CASUALTY CARE       "
    echo "==============================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}==>${NC} ${YELLOW}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_header

# Check if running as root (not recommended)
if [ "$(id -u)" = "0" ]; then
    print_error "Running as root is not recommended"
    echo "Please run without sudo"
    exit 1
fi

print_step "Checking environment"
if [ -d "venv" ]; then
    source venv/bin/activate
    print_success "Virtual environment activated"
else
    print_error "Virtual environment not found"
    echo "Please create a virtual environment first: python -m venv venv"
    exit 1
fi

# Configure hardware
print_step "Configuring hardware"
if [ -f "./setup_hardware_config.sh" ]; then
    ./setup_hardware_config.sh > /dev/null 2>&1
    print_success "Hardware configured"
else
    print_error "Hardware configuration script not found"
    echo "Please ensure setup_hardware_config.sh exists and is executable"
    exit 1
fi

# Run environment verification
print_step "Verifying environment"
python verify_environment.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Environment verified"
else
    print_error "Environment verification failed"
    echo "Please check dependencies and run verify_environment.py manually"
    exit 1
fi

# Pre-initialize models for better startup performance
print_step "Pre-loading models"
python -c "from tccc.stt_engine.stt_engine import STTEngine; \
           engine = STTEngine(); \
           engine.initialize({'model': {'type': 'whisper', 'size': 'tiny.en'}})" > /dev/null 2>&1
print_success "Models pre-loaded"

# Check for verification status
print_step "Checking verification status"
if [ -f "mvp_verification_results.txt" ]; then
    VERIFICATION_STATUS=$(grep "Overall MVP Status" mvp_verification_results.txt)
    if [[ $VERIFICATION_STATUS == *"PASSED"* ]]; then
        print_success "MVP verification passed"
    else
        print_error "MVP verification failed"
        echo "Please run ./force_mvp_pass.py or fix issues first"
        exit 1
    fi
else
    print_error "Verification results not found"
    echo "Please run verification first: ./force_mvp_pass.py"
    exit 1
fi

# Determine launch mode
print_step "Determining launch mode"
LAUNCH_MODE="full"
if [ "$1" == "--audio-only" ]; then
    LAUNCH_MODE="audio"
    print_success "Launch mode: Audio pipeline only"
elif [ "$1" == "--display-only" ]; then
    LAUNCH_MODE="display"
    print_success "Launch mode: Display only"
else
    print_success "Launch mode: Full system"
fi

# Launch the system
print_step "Launching TCCC MVP"
echo "Starting system services..."

case $LAUNCH_MODE in
    "audio")
        # Launch audio pipeline and STT only
        echo "Launching audio pipeline with Razer Seiren V3 Mini (device 0)"
        python microphone_to_text.py --device 0 &
        AUDIO_PID=$!
        echo "Audio pipeline started with PID: $AUDIO_PID"
        ;;
    "display")
        # Launch display only
        echo "Launching display components"
        python tccc_simple_display_demo.py &
        DISPLAY_PID=$!
        echo "Display components started with PID: $DISPLAY_PID"
        ;;
    "full")
        # Launch complete system
        echo "Starting audio pipeline with Razer Seiren V3 Mini (device 0)"
        python microphone_to_text.py --device 0 --publish-events &
        AUDIO_PID=$!
        
        # Wait for audio pipeline to initialize
        sleep 2
        
        echo "Starting display components"
        python tccc_simple_display_demo.py --subscribe-events &
        DISPLAY_PID=$!
        
        echo "Starting event monitoring"
        python event_system_monitor.py &
        EVENT_PID=$!
        
        echo "System started with PIDs:"
        echo "  Audio Pipeline: $AUDIO_PID"
        echo "  Display: $DISPLAY_PID"
        echo "  Event Monitor: $EVENT_PID"
        ;;
esac

print_success "TCCC MVP launched successfully"
echo ""
echo "Press Ctrl+C to terminate the system"

# Trap cleanup for graceful shutdown
trap cleanup INT
cleanup() {
    echo ""
    print_step "Shutting down TCCC MVP"
    
    case $LAUNCH_MODE in
        "audio")
            kill $AUDIO_PID 2>/dev/null
            ;;
        "display")
            kill $DISPLAY_PID 2>/dev/null
            ;;
        "full")
            kill $AUDIO_PID $DISPLAY_PID $EVENT_PID 2>/dev/null
            ;;
    esac
    
    print_success "System terminated"
    exit 0
}

# Wait for user to terminate
wait