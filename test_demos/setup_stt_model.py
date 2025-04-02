#!/usr/bin/env python3
"""
Setup STT model folder structure for TCCC project
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    """Main function"""
    # Define paths
    project_root = Path(__file__).parent
    model_dir = project_root / "models" / "stt"
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Setting up STT model directory at: {model_dir}")
    
    # Add placeholder for README
    readme_path = model_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write("""# TCCC STT Models

This directory contains the speech-to-text models used by the TCCC project.

## Models

Currently using:
- Faster Whisper tiny.en
""")
        print(f"Created README at: {readme_path}")
    
    # Create empty model file structure if they don't exist
    subdirs = ["tiny.en", "small.en", "base.en"]
    for subdir in subdirs:
        subdir_path = model_dir / subdir
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Created model directory: {subdir_path}")
    
    print("\nSTT model directory setup complete!")
    print("\nTo download and install models, run:")
    print("  python download_stt_model.py --model-size tiny.en")
    print("  python download_stt_model.py --model-size base.en")
    print("  python download_stt_model.py --model-size small.en")

if __name__ == "__main__":
    main()