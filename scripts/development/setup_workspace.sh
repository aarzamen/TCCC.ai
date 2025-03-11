#!/bin/bash

# TCCC.ai Workspace Setup Script
# This script sets up a multi-terminal workspace for TCCC.ai development with Claude instances

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"
BROWSER="firefox"  # Change to your preferred browser

# Check if wmctrl is installed
if ! command -v wmctrl &> /dev/null; then
    echo "wmctrl is not installed. Installing now..."
    sudo apt-get update && sudo apt-get install -y wmctrl
fi

# Function to open a terminal with specific profile and commands
open_terminal() {
    local profile="$1"
    local title="$2"
    local commands="$3"
    local geometry="$4"
    
    gnome-terminal --window-with-profile="$profile" \
                   --title="$title" \
                   --geometry="$geometry" \
                   -- bash -c "cd $PROJECT_DIR && source $VENV_ACTIVATE && echo -e '\e]0;$title\a' && $commands; exec bash"
}

# Function to open Claude in a browser
open_claude() {
    local title="$1"
    local position="$2"
    
    $BROWSER "https://claude.ai/chat" &
    sleep 3
    
    # Get the window ID of the most recently opened browser window
    local window_id=$(wmctrl -l | grep "$BROWSER" | tail -1 | awk '{print $1}')
    
    # Set window name for easier identification
    wmctrl -i -r "$window_id" -T "$title"
    
    # Position the window based on the quadrant
    case "$position" in
        "top-left")
            wmctrl -i -r "$window_id" -e "0,0,0,$HALF_WIDTH,$TERMINAL_HEIGHT"
            ;;
        "top-right")
            wmctrl -i -r "$window_id" -e "0,$HALF_WIDTH,0,$HALF_WIDTH,$TERMINAL_HEIGHT"
            ;;
        "bottom-left")
            wmctrl -i -r "$window_id" -e "0,0,$TERMINAL_HEIGHT,$HALF_WIDTH,$TERMINAL_HEIGHT"
            ;;
        "bottom-right")
            wmctrl -i -r "$window_id" -e "0,$HALF_WIDTH,$TERMINAL_HEIGHT,$HALF_WIDTH,$TERMINAL_HEIGHT"
            ;;
    esac
}

# Function to position a window
position_window() {
    local title="$1"
    local x="$2"
    local y="$3"
    local width="$4"
    local height="$5"
    
    # Wait for the window to appear
    sleep 1
    wmctrl -r "$title" -e "0,$x,$y,$width,$height"
}

# Get screen dimensions
SCREEN_WIDTH=$(xdpyinfo | awk '/dimensions/{print $2}' | cut -d 'x' -f1)
SCREEN_HEIGHT=$(xdpyinfo | awk '/dimensions/{print $2}' | cut -d 'x' -f2)

# Calculate window dimensions (a bit smaller to account for window decorations)
HALF_WIDTH=$((SCREEN_WIDTH / 2 - 20))
TERMINAL_HEIGHT=$((SCREEN_HEIGHT / 2 - 180))  # Leave room for dashboard at bottom
DASHBOARD_HEIGHT=150
DASHBOARD_Y=$((SCREEN_HEIGHT - DASHBOARD_HEIGHT - 60))

# Create and position terminals
echo "Setting up TCCC.ai workspace with Claude instances..."

# Terminal 1: STT Engine Module (Top-left)
open_terminal "CC agent 1" "TCCC: STT Engine" "echo -e '\033[1;33m===== STT ENGINE MODULE =====\033[0m' && python -c \"from tccc.stt_engine.stt_engine import STTEngine; print('STT Engine ready for development')\"" "80x24+0+0"
position_window "TCCC: STT Engine" 0 0 $HALF_WIDTH $TERMINAL_HEIGHT

# Claude instance for STT Engine
echo "Opening Claude for STT Engine..."
open_claude "Claude - STT Engine" "top-left"

# Terminal 2: LLM Analysis Module (Top-right)
open_terminal "CC agent 2" "TCCC: LLM Analysis" "echo -e '\033[1;36m===== LLM ANALYSIS MODULE =====\033[0m' && python -c \"from tccc.llm_analysis.llm_analysis import LLMAnalysis; print('LLM Analysis ready for development')\"" "80x24+0+0"
position_window "TCCC: LLM Analysis" $HALF_WIDTH 0 $HALF_WIDTH $TERMINAL_HEIGHT

# Claude instance for LLM Analysis
echo "Opening Claude for LLM Analysis..."
open_claude "Claude - LLM Analysis" "top-right"

# Terminal 3: Document Library Module (Bottom-left)
open_terminal "CC agent 3" "TCCC: Document Library" "echo -e '\033[1;32m===== DOCUMENT LIBRARY MODULE =====\033[0m' && python -c \"from tccc.document_library.document_library import DocumentLibrary; print('Document Library ready for development')\"" "80x24+0+0"
position_window "TCCC: Document Library" 0 $TERMINAL_HEIGHT $HALF_WIDTH $TERMINAL_HEIGHT

# Claude instance for Document Library
echo "Opening Claude for Document Library..."
open_claude "Claude - Document Library" "bottom-left"

# Terminal 4: Integration Coordinator (Bottom-right)
open_terminal "CC agent 4" "TCCC: Integration" "echo -e '\033[1;35m===== INTEGRATION COORDINATOR =====\033[0m' && echo 'Integration environment ready' && ./run_all_verifications.sh" "80x24+0+0"
position_window "TCCC: Integration" $HALF_WIDTH $TERMINAL_HEIGHT $HALF_WIDTH $TERMINAL_HEIGHT

# Claude instance for Integration Coordinator
echo "Opening Claude for Integration Coordinator..."
open_claude "Claude - Integration Coordinator" "bottom-right"

# Reorder windows to bring terminals to the front
sleep 2
wmctrl -r "TCCC: STT Engine" -b add,above
wmctrl -r "TCCC: LLM Analysis" -b add,above
wmctrl -r "TCCC: Document Library" -b add,above
wmctrl -r "TCCC: Integration" -b add,above

# Launch dashboard in a terminal that spans the bottom
open_terminal "CC agent 1" "TCCC: Dashboard" "echo -e '\033[1;37m===== TCCC SYSTEM DASHBOARD =====\033[0m' && python dashboard.py" "160x10+0+0"
position_window "TCCC: Dashboard" 0 $DASHBOARD_Y $SCREEN_WIDTH $DASHBOARD_HEIGHT

# Show instructions for using the workspace
cat << EOF

===============================
TCCC.ai Workspace Setup Complete!
===============================

Workspace layout:
- Top-left: STT Engine Terminal + Claude instance
- Top-right: LLM Analysis Terminal + Claude instance
- Bottom-left: Document Library Terminal + Claude instance
- Bottom-right: Integration Coordinator Terminal + Claude instance
- Bottom: Dashboard spanning full width

Instructions:
1. In each Claude window, use the prompt: 
   "I am working on the TCCC.ai project, focusing on the [MODULE] module."
2. Use Alt+Tab to switch between terminals and Claude windows
3. When finished, close all windows or run: killall gnome-terminal firefox

EOF

echo "TCCC.ai workspace with Claude instances is ready!"