#!/bin/bash
# Install PyTorch for Jetson Nano/Orin

echo -e "\033[1;36m============================================================\033[0m"
echo -e "\033[1;36m Installing PyTorch for Jetson Nano/Orin\033[0m"
echo -e "\033[1;36m============================================================\033[0m"

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Check which Jetson platform we're on (Nano or Orin)
if [ -f /etc/nv_tegra_release ]; then
  # Extract the version
  JETSON_L4T=$(head -1 /etc/nv_tegra_release | cut -f 2 -d ' ' | cut -f 1 -d ',')
  
  echo "Detected Jetson L4T version: $JETSON_L4T"
  
  # Install PyTorch
  echo "Installing PyTorch for Jetson..."
  
  # Let's install a version known to work on most Jetsons
  pip install --upgrade pip
  pip install numpy wheel setuptools --upgrade
  
  # Install torch directly (standalone, no need for JetPack version matching)
  pip install torch torchvision torchaudio

  echo "Testing PyTorch installation..."
  python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
  
  echo -e "\033[1;32mPyTorch installation complete!\033[0m"
else
  echo -e "\033[1;31mThis doesn't appear to be a Jetson device.\033[0m"
  exit 1
fi

# Deactivate the virtual environment if we activated it
if [ -d "venv" ]; then
  deactivate
fi

echo -e "\033[1;36m============================================================\033[0m"
echo -e "\033[1;36m PyTorch installation complete. You can now run the TCCC test.\033[0m"
echo -e "\033[1;36m============================================================\033[0m"