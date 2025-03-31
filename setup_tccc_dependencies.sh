#!/bin/bash
# Setup script for TCCC dependencies on Jetson Nano

echo -e "\033[1;36m============================================================\033[0m"
echo -e "\033[1;36m Installing TCCC dependencies for Jetson Nano\033[0m"
echo -e "\033[1;36m============================================================\033[0m"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Install basic dependencies
echo "Installing basic dependencies..."
pip install pyaudio pygame numpy

# Test PyAudio installation
echo "Testing PyAudio installation..."
python -c "import pyaudio; p = pyaudio.PyAudio(); print('PyAudio device count:', p.get_device_count()); p.terminate()"

# Test PyGame installation
echo "Testing PyGame installation..."
python -c "import pygame; pygame.init(); print('PyGame version:', pygame.version.ver)"

# Create a desktop shortcut for the simple mic test
DESKTOP_FILE="/home/ama/Desktop/TCCC_Mic_Test_Simple.desktop"
echo "Creating desktop shortcut..."
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=TCCC Mic Test (Simple)
Comment=Test Razer microphone and display
Exec=/home/ama/tccc-project/run_tccc_mic_test_simple.sh
Terminal=false
Type=Application
Categories=Utility;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"
chmod +x /home/ama/tccc-project/run_tccc_mic_test_simple.sh

echo -e "\033[1;32mBasic dependencies installed successfully!\033[0m"

# Install PyTorch (optional - will be slow)
read -p "Do you want to install PyTorch for the full test? This will take a long time (y/n): " install_torch
if [[ "$install_torch" == "y" || "$install_torch" == "Y" ]]; then
  echo "Installing PyTorch..."
  pip install torch torchvision torchaudio
  
  # Test PyTorch installation
  python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
fi

# Deactivate the virtual environment if we activated it
if [ -d "venv" ]; then
  deactivate
fi

echo -e "\033[1;36m============================================================\033[0m"
echo -e "\033[1;36m Installation complete.\033[0m"
echo -e "\033[1;36m You can now run the TCCC Mic Test shortcut on your desktop.\033[0m"
echo -e "\033[1;36m============================================================\033[0m"