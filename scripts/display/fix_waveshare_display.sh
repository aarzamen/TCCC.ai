#!/bin/bash
# WaveShare Display Fix Launcher Script
# This script automatically fixes WaveShare display functionality for TCCC

# ANSI colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Echo with color
echo_color() {
  echo -e "${1}${2}${NC}"
}

# Title
echo_color "$BOLD$BLUE" "=== TCCC.ai WaveShare Display Fix ==="
echo ""

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
  echo_color "$YELLOW" "Virtual environment not active. Activating..."
  
  # Check if venv directory exists
  if [[ -d "venv" ]]; then
    source venv/bin/activate
    echo_color "$GREEN" "Virtual environment activated."
  else
    echo_color "$RED" "Error: Virtual environment directory 'venv' not found."
    echo "Please set up the virtual environment first."
    exit 1
  fi
fi

# Make the Python script executable
chmod +x fix_waveshare_display.py

# Run the Python script with appropriate arguments
echo_color "$YELLOW" "Running display fix script..."
echo ""

# Get the arguments
args=""
if [[ "$1" == "--env-only" ]]; then
  args="--env-only"
  echo_color "$YELLOW" "Environment setup only mode"
elif [[ "$1" == "--test-only" ]]; then
  args="--test-only"
  echo_color "$YELLOW" "Test only mode"
elif [[ "$1" == "--full-test" ]]; then
  args="--full-test"
  echo_color "$YELLOW" "Full test mode"
fi

# Run the script with any provided arguments
python fix_waveshare_display.py $args

# Check exit status
exit_status=$?
if [[ $exit_status -eq 0 ]]; then
  echo ""
  echo_color "$GREEN" "WaveShare display fix completed successfully!"
  echo ""
  echo "You can now run the full TCCC system with the WaveShare display:"
  echo_color "$BLUE" "python run_system.py --with-display"
  echo ""
  echo "Or test the display separately:"
  echo_color "$BLUE" "python test_waveshare_display.py"
else
  echo ""
  echo_color "$RED" "WaveShare display fix encountered errors (exit code: $exit_status)"
  echo "Please check the output above for details."
fi

exit $exit_status