#\!/bin/bash
# Run FullSubNet demo using the microphone_to_text.py script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check if FullSubNet is installed
if [ \! -d "$SCRIPT_DIR/fullsubnet" ]; then
    echo "FullSubNet not installed. Running setup script..."
    cd "$SCRIPT_DIR"
    ./fullsubnet_setup.sh
    cd ..
fi

# Check if enhancement mode was provided
ENHANCEMENT_MODE="fullsubnet"
if [ $# -gt 0 ]; then
    ENHANCEMENT_MODE="$1"
fi

# Modify microphone_to_text.py to use the specified enhancement mode
sed -i "s/ENHANCEMENT_MODE = \"auto\"/ENHANCEMENT_MODE = \"$ENHANCEMENT_MODE\"/" microphone_to_text.py

echo "=== FullSubNet Speech Enhancement Demo ==="
echo "Running microphone_to_text.py with $ENHANCEMENT_MODE enhancement mode"

# Run the microphone to text script
python3 microphone_to_text.py

echo "Demo complete. Check output for enhanced transcription."

# Restore auto mode
sed -i "s/ENHANCEMENT_MODE = \"$ENHANCEMENT_MODE\"/ENHANCEMENT_MODE = \"auto\"/" microphone_to_text.py
