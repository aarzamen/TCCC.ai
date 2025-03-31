#!/bin/bash
# Shell script to run the whisper_to_prompt.py script with optimal settings

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Ensure required packages are installed
pip install sounddevice soundfile pyperclip requests python-dotenv

# Run the voice-to-prompt script in continuous mode with default device
# Using the special device ID 28 which is the default audio input
python whisper_to_prompt.py --continuous --device 28

# Deactivate virtual environment if activated
if [ -d "venv" ]; then
  deactivate
fi