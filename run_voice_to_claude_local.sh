#!/bin/bash
# Shell script to run the whisper_to_prompt_local.py script with optimal settings

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Ensure required packages are installed
pip install sounddevice soundfile pyperclip faster-whisper

# Download whisper model if needed
if [ ! -d "models/stt/tiny.en" ]; then
  echo "Downloading Whisper model..."
  python download_stt_model.py --model-size tiny.en
fi

# Run the local voice-to-prompt script in continuous mode with default device
# Using the special device ID 28 which is the default audio input
python whisper_to_prompt_local.py --continuous --device 28 --model tiny.en

# Deactivate virtual environment if activated
if [ -d "venv" ]; then
  deactivate
fi