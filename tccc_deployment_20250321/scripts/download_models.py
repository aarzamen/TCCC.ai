#!/usr/bin/env python3
"""
Model Downloader for TCCC.ai Jetson MVP
---------------------------------------
This script downloads the minimal set of models required for the
TCCC.ai MVP deployment on Jetson hardware.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelDownloader")

def ensure_deps():
    """Ensure required dependencies are installed."""
    try:
        import torch
        import huggingface_hub
        import tqdm
        from sentence_transformers import SentenceTransformer
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
                "torch", "tqdm", "huggingface_hub", 
                "transformers", "sentence-transformers",
                "faster-whisper", "--quiet"
            ])
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to install dependencies")
            return False

def download_whisper_model(model_size="tiny.en", output_dir="models/stt"):
    """Download Whisper STT model."""
    logger.info(f"Downloading Whisper {model_size} model...")
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if list(output_path.glob(f"{model_size}*")):
            logger.info(f"Whisper {model_size} model already exists in {output_dir}")
            return True
        
        # Use faster-whisper for downloading
        from faster_whisper import WhisperModel
        
        # This will download the model automatically
        _ = WhisperModel(model_size, download_root=output_dir)
        
        logger.info(f"Whisper {model_size} model downloaded to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {e}")
        return False

def download_llm_model(model_name="microsoft/phi-2", output_dir="models/llm/phi-2"):
    """Download LLM model."""
    logger.info(f"Downloading LLM model {model_name}...")
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if (output_path / "config.json").exists():
            logger.info(f"LLM model {model_name} already exists in {output_dir}")
            return True
        
        # Use Hugging Face hub to download
        from huggingface_hub import snapshot_download
        
        # Download only essential files (skip safetensors, etc.)
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            ignore_patterns=["*.safetensors", "*.msgpack", "*.h5", "*.ot"],
            local_dir_use_symlinks=False,
        )
        
        # Load tokenizer and model to verify
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        try:
            # Just load the tokenizer to verify download
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            logger.info(f"LLM model {model_name} downloaded and verified in {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download LLM model: {e}")
        return False

def download_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                          output_dir="models/embeddings/all-MiniLM-L6-v2"):
    """Download embedding model for RAG system."""
    logger.info(f"Downloading embedding model {model_name}...")
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        if (output_path / "config.json").exists():
            logger.info(f"Embedding model {model_name} already exists in {output_dir}")
            return True
        
        # Use sentence-transformers to download
        from sentence_transformers import SentenceTransformer
        
        # This will download the model
        model = SentenceTransformer(model_name)
        
        # Save the model to the specified directory
        model.save(output_dir)
        
        logger.info(f"Embedding model {model_name} downloaded to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to download embedding model: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download models for TCCC.ai Jetson MVP")
    parser.add_argument("--whisper-model", default="tiny.en", 
                        help="Whisper model size (tiny.en, base.en, small.en)")
    parser.add_argument("--llm-model", default="microsoft/phi-2",
                        help="LLM model name from Hugging Face")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model name from Hugging Face")
    parser.add_argument("--output-dir", default="models",
                        help="Base output directory for models")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip downloading Whisper model")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip downloading LLM model")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip downloading embedding model")
    
    args = parser.parse_args()
    
    # Ensure dependencies are installed
    if not ensure_deps():
        logger.error("Required dependencies are missing. Exiting.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track success/failure
    success = True
    
    # Download models
    if not args.skip_whisper:
        whisper_dir = os.path.join(args.output_dir, "stt")
        if not download_whisper_model(args.whisper_model, whisper_dir):
            success = False
    
    if not args.skip_llm:
        llm_model_name = args.llm_model.split("/")[-1]
        llm_dir = os.path.join(args.output_dir, "llm", llm_model_name)
        if not download_llm_model(args.llm_model, llm_dir):
            success = False
    
    if not args.skip_embedding:
        embedding_model_name = args.embedding_model.split("/")[-1]
        embedding_dir = os.path.join(args.output_dir, "embeddings", embedding_model_name)
        if not download_embedding_model(args.embedding_model, embedding_dir):
            success = False
    
    if success:
        logger.info("All models downloaded successfully!")
    else:
        logger.warning("Some models failed to download. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()