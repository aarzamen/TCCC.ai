#!/usr/bin/env python3
"""
Download Whisper model for STT Engine
"""

import os
import argparse
from faster_whisper import download_model

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download Whisper model for STT")
    parser.add_argument("--model-size", default="tiny", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2"],
                        help="Model size to download")
    parser.add_argument("--output-dir", default="models/stt",
                        help="Directory to save the model")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading {args.model_size} model to {args.output_dir}...")
    model_path = download_model(args.model_size, output_dir=args.output_dir)
    
    print(f"Model downloaded to: {model_path}")
    print("Model download complete!")

if __name__ == "__main__":
    main()