#!/bin/bash
# TCCC Full End-to-End Demo - Console Version
# This script runs directly on the Jetson console and shows all components working together
# It captures speech through the Razer microphone, processes it through STT and LLM,
# and displays results on the WaveShare display

# Set the DISPLAY variable to target the Jetson's physical display
export DISPLAY=:0

# Use the actual models, not mocks
export USE_MOCK_STT=0
export USE_MOCK_LLM=0

# Clear the screen
clear

# ASCII art banner
echo -e "\e[1;36m"
echo "████████╗ ██████╗ ██████╗ ██████╗     ██████╗ ███████╗███╗   ███╗ ██████╗ "
echo "╚══██╔══╝██╔════╝██╔════╝██╔════╝    ██╔══██╗██╔════╝████╗ ████║██╔═══██╗"
echo "   ██║   ██║     ██║     ██║         ██║  ██║█████╗  ██╔████╔██║██║   ██║"
echo "   ██║   ██║     ██║     ██║         ██║  ██║██╔══╝  ██║╚██╔╝██║██║   ██║"
echo "   ██║   ╚██████╗╚██████╗╚██████╗    ██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝"
echo "   ╚═╝    ╚═════╝ ╚═════╝ ╚═════╝    ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ "
echo -e "\e[0m"
echo -e "\e[1;32mTactical Combat Casualty Care - Full System Demonstration\e[0m"
echo -e "\e[1;32m===================================================\e[0m"
echo

# Check for connected display
echo -e "\e[1;33mVerifying display connection...\e[0m"
if xrandr >/dev/null 2>&1; then
  display_info=$(xrandr | grep " connected")
  echo -e "Display detected: $display_info"
else
  echo -e "\e[1;31mWarning: Unable to detect display information\e[0m"
  echo -e "Continuing with default display settings"
fi

# Check for audio device
echo -e "\e[1;33mVerifying Razer Seiren V3 Mini connection...\e[0m"
if arecord -l | grep -q "Razer"; then
  mic_info=$(arecord -l | grep "Razer")
  echo -e "\e[1;32mRazer microphone detected: $mic_info\e[0m"
  MIC_DEVICE=0
else
  echo -e "\e[1;31mWarning: Razer microphone not detected!\e[0m"
  echo -e "Using default audio input device instead"
  MIC_DEVICE=$(arecord -l | grep -m1 -o "card [0-9]*" | cut -d' ' -f2)
fi

# Check for STT model
echo -e "\e[1;33mVerifying STT model availability...\e[0m"
if [ -d "models/stt" ]; then
  model_info=$(ls -la models/stt | grep -m1 ".bin\|.pt")
  echo -e "\e[1;32mSTT model available: $model_info\e[0m"
else
  echo -e "\e[1;31mWarning: STT model directory not found!\e[0m"
  echo -e "Will use default model location or download as needed"
fi

# Check for Phi-2 model
echo -e "\e[1;33mVerifying Phi-2 LLM availability...\e[0m"
if [ -d "models/llm" ] && ls models/llm/phi* >/dev/null 2>&1; then
  model_info=$(ls -la models/llm | grep -m1 "phi")
  echo -e "\e[1;32mPhi-2 model available: $model_info\e[0m"
else
  echo -e "\e[1;31mWarning: Phi-2 model not found in expected location!\e[0m"
  echo -e "Will use fallback model or download as needed"
fi

# Prepare environment
echo -e "\e[1;33mPreparing system environment...\e[0m"
if [ -d "venv" ]; then
  echo -e "Activating virtual environment..."
  source venv/bin/activate
fi

# Verify Python and required packages
echo -e "Checking Python environment..."
python_version=$(python3 --version)
echo -e "Python: $python_version"

# Launch the system integration test with all components
echo -e "\e[1;33m\nInitializing full TCCC system demonstration...\e[0m"
echo -e "This will show the complete pipeline from:"
echo -e "1. \e[1;36mRazer microphone audio capture\e[0m"
echo -e "2. \e[1;36mRealtime STT with Whisper\e[0m"
echo -e "3. \e[1;36mLLM analysis with Phi-2\e[0m"
echo -e "4. \e[1;36mRealtime display on WaveShare screen\e[0m"
echo

# Give the user instructions about what to say
echo -e "\e[1;33m================ DEMONSTRATION SCRIPT ================\e[0m"
echo -e "When prompted, please read the following medical scenario:"
echo
echo -e "\e[1;32mI have a 25-year-old male casualty with a gunshot wound to the right chest. He is conscious but having difficulty breathing. Oxygen saturation is 88% and dropping. I suspect a tension pneumothorax. Vital signs show heart rate of 135, blood pressure 90/60. What is the immediate treatment priority?\e[0m"
echo
echo -e "\e[1;33m=====================================================\e[0m"

echo -e "\nStarting in 5 seconds..."
for i in {5..1}; do
  echo -ne "\rStarting in $i seconds..."
  sleep 1
done
echo -e "\n"

# Execute the full pipeline with all actual components
echo -e "\e[1;32m================== STARTING SYSTEM ==================\e[0m"
python test_system_integration.py --use_display --real_mic --use_phi2 --use_whisper --battlemode

# Script completion
echo -e "\e[1;32m\nTCCC demonstration complete!\e[0m"
echo -e "Results saved to:"
echo -e "- Transcription: transcript_$(date +%Y%m%d_%H%M%S).txt"
echo -e "- Audio: recorded_audio.wav"
echo -e "- LLM Analysis: llm_analysis_output.json"

# If we're in a virtual environment, deactivate it
if [ -n "$VIRTUAL_ENV" ]; then
  deactivate
fi