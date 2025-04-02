#!/usr/bin/env python3
"""
Download Silero VAD model for faster-whisper
"""

import os
import sys
import requests
from pathlib import Path

def main():
    """Main function"""
    # Define paths
    assets_dir = Path(__file__).parent / "venv" / "lib" / "python3.10" / "site-packages" / "faster_whisper" / "assets"
    
    # Create assets directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)
    
    # Define model paths
    encoder_path = assets_dir / "silero_encoder_v5.onnx"
    decoder_path = assets_dir / "silero_decoder_v5.onnx"
    
    # Define model URLs
    encoder_url = "https://github.com/guillaumekln/faster-whisper/raw/master/faster_whisper/assets/silero_encoder_v5.onnx"
    decoder_url = "https://github.com/guillaumekln/faster-whisper/raw/master/faster_whisper/assets/silero_decoder_v5.onnx"
    
    # Download encoder
    print(f"Downloading Silero VAD encoder to {encoder_path}...")
    encoder_response = requests.get(encoder_url, stream=True)
    encoder_response.raise_for_status()
    
    # Save encoder
    with open(encoder_path, "wb") as f:
        for chunk in encoder_response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Encoder downloaded to: {encoder_path}")
    
    # Download decoder
    print(f"Downloading Silero VAD decoder to {decoder_path}...")
    decoder_response = requests.get(decoder_url, stream=True)
    decoder_response.raise_for_status()
    
    # Save decoder
    with open(decoder_path, "wb") as f:
        for chunk in decoder_response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Decoder downloaded to: {decoder_path}")
    print("Download complete!")

if __name__ == "__main__":
    main()