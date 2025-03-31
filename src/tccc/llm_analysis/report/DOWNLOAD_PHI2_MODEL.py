#!/usr/bin/env python3
"""
PHI-2 Model Downloader for TCCC.ai

This script downloads the complete PHI-2 model with full weights for the TCCC.ai system.
It includes authentication, resume capabilities, and verification of downloaded files.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PHI2Downloader")

def ensure_dependencies():
    """Ensure required dependencies are installed."""
    try:
        import torch
        import huggingface_hub
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing required dependencies...")
        
        import subprocess
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "transformers", "huggingface_hub", "tqdm",
                "--quiet"
            ])
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to install dependencies")
            return False

def setup_huggingface_auth(token=None):
    """Set up Hugging Face authentication."""
    import huggingface_hub
    
    if token is None:
        # Try to get token from environment variable
        token = os.environ.get("HUGGINGFACE_TOKEN")
        
    if token:
        huggingface_hub.login(token=token, add_to_git_credential=False)
        logger.info("Logged in to Hugging Face using token")
        return True
    else:
        logger.warning("No Hugging Face token provided, attempting anonymous download")
        return False

def verify_model_files(model_dir):
    """Verify if model files are complete and valid."""
    model_dir = Path(model_dir)
    
    # Check for essential files
    essential_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for file in essential_files:
        if not (model_dir / file).exists():
            logger.error(f"Missing essential file: {file}")
            return False
    
    # Check for model weights
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        logger.error("No model weight files found")
        return False
    
    # Verify weight files have reasonable size
    for weight_file in safetensors_files:
        file_size = weight_file.stat().st_size
        if file_size < 1_000_000:  # Less than 1MB is too small for a real weight file
            logger.error(f"Model weight file {weight_file.name} is too small ({file_size} bytes)")
            return False
            
    logger.info("All model files present and valid")
    return True

def download_phi2_model(output_dir, force=False, use_auth=True, resume=True):
    """Download the PHI-2 model with complete weights."""
    import huggingface_hub
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if not force and verify_model_files(output_path):
        logger.info(f"Model already exists in {output_dir} and appears valid")
        return True
    
    logger.info(f"Downloading PHI-2 model to {output_dir}")
    
    try:
        # Use snapshot_download for complete model with resume capability
        snapshot_download(
            repo_id="microsoft/phi-2",
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            resume_download=resume
        )
        
        # Verify downloaded model
        if verify_model_files(output_path):
            logger.info("Model downloaded and verified successfully")
            return True
        else:
            logger.error("Model downloaded but verification failed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def load_and_verify_model(model_dir):
    """Load and verify the downloaded model works correctly."""
    logger.info(f"Verifying model can be loaded from {model_dir}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Run a simple inference test
        prompt = "What are the key steps in treating a tension pneumothorax?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info("Model loaded and inference test successful")
        logger.info(f"Sample response: {response[:100]}...")
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download PHI-2 model for TCCC.ai")
    parser.add_argument("--output-dir", default="models/phi-2-instruct",
                       help="Output directory for model")
    parser.add_argument("--token", help="Hugging Face token for authentication")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if model exists")
    parser.add_argument("--skip-verify", action="store_true",
                       help="Skip model verification after download")
    
    args = parser.parse_args()
    
    # Ensure dependencies are installed
    if not ensure_dependencies():
        logger.error("Required dependencies are missing. Exiting.")
        sys.exit(1)
    
    # Setup authentication
    auth_success = setup_huggingface_auth(args.token)
    if not auth_success:
        logger.warning("Proceeding with anonymous download (may fail for gated models)")
    
    # Download model
    if download_phi2_model(args.output_dir, force=args.force, use_auth=auth_success):
        logger.info(f"PHI-2 model successfully downloaded to {args.output_dir}")
        
        # Verify model loading
        if not args.skip_verify:
            if load_and_verify_model(args.output_dir):
                logger.info("PHI-2 model verified and ready for use")
            else:
                logger.error("Model verification failed")
                sys.exit(1)
    else:
        logger.error("Failed to download PHI-2 model")
        sys.exit(1)
    
    # Create completion marker file
    with open(os.path.join(args.output_dir, "download_complete.json"), "w") as f:
        json.dump({
            "status": "complete",
            "timestamp": time.time(),
            "verified": not args.skip_verify
        }, f)
    
    logger.info("Download complete!")
    
if __name__ == "__main__":
    main()