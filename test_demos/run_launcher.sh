#!/bin/bash

# --- Determine script's own directory and project root ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"
LAUNCHER_SCRIPT="$SCRIPT_DIR/demo_launcher.py"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python3" # Changed from python to python3 for consistency

echo "--- Wrapper: Script Directory: $SCRIPT_DIR ---"
echo "--- Wrapper: Project Root: $PROJECT_ROOT ---"

# --- Check if venv exists ---
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Error: Virtual environment activation script not found at $VENV_ACTIVATE"
    echo "Please create the virtual environment in $PROJECT_ROOT/venv first."
    read -p "Press Enter to close terminal."
    exit 1
fi

# --- Activate Virtual Environment ---
echo "--- Wrapper: Activating Virtual Environment ($VENV_ACTIVATE) ---"
source "$VENV_ACTIVATE"

# --- Check if launcher script exists ---
if [ ! -f "$LAUNCHER_SCRIPT" ]; then
    echo "Error: Demo launcher script not found at $LAUNCHER_SCRIPT"
    read -p "Press Enter to close terminal."
    exit 1
fi

# --- Check Python executable in venv ---
if [ ! -x "$VENV_PYTHON" ]; then
    echo "Error: Python executable not found or not executable in venv at $VENV_PYTHON"
    read -p "Press Enter to close terminal."
    exit 1
fi

# --- Launch Demo Launcher ---
echo "--- Wrapper: Launching Demo Launcher ($LAUNCHER_SCRIPT) using $VENV_PYTHON ---"

"$VENV_PYTHON" "$LAUNCHER_SCRIPT"
EXIT_CODE=$?

echo ""
echo "--- Wrapper: Launcher finished with exit code: $EXIT_CODE ---"

# --- Deactivate (Optional but good practice) ---
deactivate

# --- Keep terminal open ---
read -p "Press Enter to close terminal."
exit $EXIT_CODE
