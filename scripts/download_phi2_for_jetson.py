#!/usr/bin/env python3
"""
Phi-2 Model Downloader for Jetson Orin Nano

This script downloads a quantized version of Microsoft's Phi-2 model
optimized for Jetson hardware.
"""

import os
import sys
import argparse
import hashlib
import json
import time
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Phi-2 model file information
MODEL_INFO = {
    "q4_k_m": {
        "name": "Phi-2 Q4_K_M (4-bit Quantized)",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "filename": "phi-2.Q4_K_M.gguf",
        "size": 1600000000,  # Approximate size in bytes
        "sha256": "0a7307c28f2fdabada0a95a58d46a8d8fdabda1bac281185b4976560328ac130",
        "description": "4-bit quantized model balanced for speed and accuracy"
    },
    "q5_k_m": {
        "name": "Phi-2 Q5_K_M (5-bit Quantized)",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_K_M.gguf",
        "filename": "phi-2.Q5_K_M.gguf",
        "size": 1900000000,  # Approximate size in bytes
        "sha256": "0f11ca26c8c01778a7dcf706fb78afbfcf871be6c47346aef9aab5b7f152a8bc",
        "description": "5-bit quantized model with better quality but larger size"
    },
    "q8_0": {
        "name": "Phi-2 Q8_0 (8-bit Quantized)",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q8_0.gguf",
        "filename": "phi-2.Q8_0.gguf",
        "size": 2800000000,  # Approximate size in bytes
        "sha256": "220bf4ffd0a6c978d366e2d4c3d6b6c3d69ffa2cca1ef17653fb76d9161c25f2",
        "description": "8-bit quantized model with high quality but larger size"
    },
    "tokenizer": {
        "url": "https://huggingface.co/Zierchi/phi-2-tokenizer-config/resolve/main/tokenizer_config.json",
        "filename": "tokenizer_config.json",
        "size": 10000,  # Approximate size in bytes
        "description": "Tokenizer configuration for Phi-2"
    }
}

# Setup colored output
COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m"
}

def print_colored(message, color="BLUE", bold=False):
    """Print colored text to terminal."""
    prefix = COLORS["BOLD"] if bold else ""
    print(f"{prefix}{COLORS[color]}{message}{COLORS['ENDC']}")

def check_file_exists(filepath, verify_hash=True, expected_hash=None):
    """Check if a file exists and verify its hash if requested."""
    file_path = Path(filepath)
    if not file_path.exists():
        return False
    
    if verify_hash and expected_hash:
        print_colored(f"Verifying hash for {file_path.name}...", "BLUE")
        file_hash = calculate_file_hash(file_path)
        if file_hash.lower() != expected_hash.lower():
            print_colored(f"Hash mismatch for {file_path.name}!", "RED", bold=True)
            print(f"Expected: {expected_hash}")
            print(f"Got: {file_hash}")
            return False
        print_colored(f"Hash verified for {file_path.name}", "GREEN")
    
    return True

def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_file(url, filename, expected_size=None, expected_hash=None):
    """Download a file with progress bar and verify its integrity."""
    try:
        # Setup destination path
        filepath = Path(filename)
        
        # Check if file already exists and is valid
        if check_file_exists(filepath, expected_hash is not None, expected_hash):
            print_colored(f"File {filepath.name} already exists and is valid. Skipping download.", "GREEN")
            return True
            
        # Create parent directories if they don't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if we need to resume download
        headers = {}
        mode = 'wb'
        if filepath.exists():
            existing_size = filepath.stat().st_size
            if expected_size and existing_size >= expected_size:
                print_colored(f"File {filepath.name} already exists with correct size.", "GREEN")
                # Still verify the hash
                if expected_hash and calculate_file_hash(filepath) != expected_hash:
                    print_colored(f"Hash verification failed for {filepath.name}, will re-download.", "YELLOW")
                    mode = 'wb'  # Rewrite the file
                else:
                    return True
            elif existing_size > 0:
                print_colored(f"Resuming download for {filepath.name} from {existing_size} bytes", "BLUE")
                headers['Range'] = f'bytes={existing_size}-'
                mode = 'ab'  # Append to the file
        
        # Start the download
        print_colored(f"Downloading {filepath.name} from {url}", "BLUE")
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        # Handle redirect
        if response.status_code == 302:
            url = response.headers['Location']
            print_colored(f"Redirected to {url}", "YELLOW")
            response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        if mode == 'ab':
            total_size += existing_size
        
        # Create progress bar for download
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=filepath.name,
            initial=existing_size if mode == 'ab' else 0
        )
        
        with open(filepath, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # Verify file size
        if expected_size:
            actual_size = filepath.stat().st_size
            if actual_size < expected_size * 0.9:  # Allow for some tolerance in expected size
                print_colored(f"Downloaded file size ({actual_size}) is less than expected ({expected_size})", "RED", bold=True)
                return False
        
        # Verify hash if provided
        if expected_hash:
            print_colored("Verifying file hash...", "BLUE")
            file_hash = calculate_file_hash(filepath)
            if file_hash.lower() != expected_hash.lower():
                print_colored("Hash verification failed!", "RED", bold=True)
                print(f"Expected: {expected_hash}")
                print(f"Got: {file_hash}")
                return False
            print_colored("Hash verification successful!", "GREEN")
        
        print_colored(f"Successfully downloaded {filepath.name}", "GREEN", bold=True)
        return True
        
    except Exception as e:
        print_colored(f"Error downloading {filename}: {str(e)}", "RED", bold=True)
        return False

def ensure_dependencies():
    """Check and install required dependencies."""
    try:
        import torch
        import transformers
        import llama_cpp
        print_colored("Required Python packages are already installed", "GREEN")
        return True
    except ImportError as e:
        print_colored(f"Missing dependency: {e}", "YELLOW")
        print_colored("Installing required dependencies...", "BLUE")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "transformers", "tqdm", "requests",
                "llama-cpp-python", "colorama", "numpy",
                "--upgrade"
            ])
            print_colored("Dependencies installed successfully", "GREEN")
            return True
        except subprocess.CalledProcessError:
            print_colored("Failed to install dependencies", "RED", bold=True)
            print("Please manually install required packages using:")
            print("pip install torch transformers tqdm requests llama-cpp-python colorama numpy")
            return False

def verify_model(model_path, tokenizer_path=None):
    """Verify that the downloaded model can be loaded and used for inference."""
    try:
        print_colored("\nVerifying model can be loaded...", "BLUE")
        
        # Try to load model with llama-cpp-python
        from llama_cpp import Llama
        
        # Load the model
        print_colored("Loading model with llama-cpp-python...", "BLUE")
        model = Llama(
            model_path=str(model_path),
            n_ctx=2048,  # Context window
            n_gpu_layers=-1  # Auto-detect GPU layers
        )
        
        # Test with a simple prompt
        test_prompt = "Patient presents with shortness of breath and chest pain radiating to the left arm. Vital signs: BP 160/95, HR 110, RR 22, SpO2 92%. History of hypertension. What is the most likely diagnosis?"
        
        print_colored("Running inference test...", "BLUE")
        print_colored("Test prompt: " + test_prompt, "YELLOW")
        
        # Generate response
        output = model.create_completion(
            test_prompt, 
            max_tokens=100,
            temperature=0.7,
            stop=["Patient:", "\n\n"]
        )
        
        print_colored("\nModel response:", "GREEN", bold=True)
        print(output['choices'][0]['text'])
        
        # Record stats
        stats = {
            "success": True,
            "model_path": str(model_path),
            "prompt_tokens": len(model.tokenize(test_prompt.encode())),
            "completion_tokens": len(model.tokenize(output['choices'][0]['text'].encode())),
            "timestamp": time.time()
        }
        
        # Save verification results
        results_path = Path("phi2_verification_results.json")
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print_colored("\nModel verification successful! Results saved to phi2_verification_results.json", "GREEN", bold=True)
        return True
        
    except Exception as e:
        print_colored(f"Model verification failed: {str(e)}", "RED", bold=True)
        print("Check that your model file is complete and not corrupted.")
        return False

def check_jetson_hardware():
    """Check if running on Jetson hardware and return info about the device."""
    try:
        # Check for Jetson-specific file
        soc_id_file = Path("/proc/device-tree/nvidia,soc-id")
        if not soc_id_file.exists():
            print_colored("Not running on Jetson hardware", "YELLOW")
            return None
            
        # Get model info
        jetson_info = {}
        
        # Try to get Jetson model
        model_file = Path("/proc/device-tree/model")
        if model_file.exists():
            with open(model_file, 'r') as f:
                jetson_info["model"] = f.read().strip()
        
        # Check for GPU
        try:
            import subprocess
            nvidia_smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).decode('utf-8').strip()
            jetson_info["gpu"] = nvidia_smi
        except:
            jetson_info["gpu"] = "Unknown"
            
        # Get CPU info
        try:
            with open("/proc/cpuinfo", 'r') as f:
                cpu_info = f.read()
                import re
                cpu_model = re.search(r"model name\s+:\s+(.*)", cpu_info)
                if cpu_model:
                    jetson_info["cpu"] = cpu_model.group(1)
                cores = len(re.findall(r"processor\s+:\s+\d+", cpu_info))
                jetson_info["cpu_cores"] = cores
        except:
            jetson_info["cpu"] = "Unknown"
            
        # Get RAM info
        try:
            with open("/proc/meminfo", 'r') as f:
                mem_info = f.read()
                import re
                mem_total = re.search(r"MemTotal:\s+(\d+)", mem_info)
                if mem_total:
                    jetson_info["ram"] = f"{int(mem_total.group(1)) // 1024} MB"
        except:
            jetson_info["ram"] = "Unknown"
                
        print_colored("Detected Jetson hardware:", "GREEN")
        for key, value in jetson_info.items():
            print(f"  {key}: {value}")
            
        return jetson_info
    except Exception as e:
        print_colored(f"Error detecting Jetson hardware: {str(e)}", "RED")
        return None

def recommend_quantization(jetson_info):
    """Recommend the best quantization level based on hardware."""
    if not jetson_info:
        # Default recommendation if not on Jetson
        return "q4_k_m"
        
    # Check if we can parse the GPU memory
    gpu_mem = 0
    try:
        if "gpu" in jetson_info:
            import re
            mem_match = re.search(r"(\d+)\s*MiB", jetson_info["gpu"])
            if mem_match:
                gpu_mem = int(mem_match.group(1))
    except:
        pass
        
    # Make recommendations based on available GPU memory
    if gpu_mem > 7500:  # More than 7.5GB
        return "q8_0"  # Recommend 8-bit quantization
    elif gpu_mem > 4000:  # More than 4GB
        return "q5_k_m"  # Recommend 5-bit quantization
    else:
        return "q4_k_m"  # Recommend 4-bit quantization for lower memory

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download Phi-2 model for Jetson Orin Nano")
    parser.add_argument("--output-dir", default="models/phi2_gguf",
                       help="Output directory for model")
    parser.add_argument("--quantization", default=None, choices=["q4_k_m", "q5_k_m", "q8_0"],
                       help="Quantization level for the model")
    parser.add_argument("--verify", action="store_true",
                       help="Verify model after downloading")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if model exists")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for key in COLORS:
            COLORS[key] = ""
    
    # Print header
    print_colored("\n" + "=" * 80, "HEADER")
    print_colored("PHI-2 MODEL DOWNLOADER FOR JETSON ORIN NANO".center(80), "HEADER", bold=True)
    print_colored("=" * 80 + "\n", "HEADER")
    
    # Ensure dependencies are installed
    if not ensure_dependencies():
        print_colored("Required dependencies are missing. Please install them and try again.", "RED")
        sys.exit(1)
    
    # Check Jetson hardware
    jetson_info = check_jetson_hardware()
    
    # Determine quantization level
    quantization = args.quantization
    if not quantization:
        quantization = recommend_quantization(jetson_info)
        print_colored(f"Auto-selected quantization level: {quantization}", "BLUE")
        print_colored(f"  {MODEL_INFO[quantization]['name']}", "BLUE")
        print_colored(f"  {MODEL_INFO[quantization]['description']}", "BLUE")
        print_colored("\nYou can override this with --quantization q4_k_m|q5_k_m|q8_0", "BLUE")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model file
    model_info = MODEL_INFO[quantization]
    model_path = output_dir / model_info["filename"]
    
    if args.force:
        print_colored("Forcing re-download of model files", "YELLOW")
        if model_path.exists():
            model_path.unlink()
    
    # Display model information
    print_colored(f"\nDownloading {model_info['name']}", "GREEN", bold=True)
    print_colored(f"Size: ~{model_info['size'] / 1024 / 1024:.1f} MB", "BLUE")
    print_colored(f"Description: {model_info['description']}", "BLUE")
    
    # Download the model
    model_success = download_file(
        url=model_info["url"],
        filename=model_path,
        expected_size=model_info["size"],
        expected_hash=model_info["sha256"]
    )
    
    # Download tokenizer
    tokenizer_info = MODEL_INFO["tokenizer"]
    tokenizer_path = output_dir / tokenizer_info["filename"]
    
    tokenizer_success = download_file(
        url=tokenizer_info["url"],
        filename=tokenizer_path,
        expected_size=tokenizer_info["size"]
    )
    
    if not model_success:
        print_colored("Failed to download the model. Please try again or use a different quantization level.", "RED", bold=True)
        sys.exit(1)
    
    # Create a configuration file
    config = {
        "model_type": "phi-2",
        "version": "1.0",
        "quantization": quantization,
        "file": model_info["filename"],
        "format": "gguf",
        "provider": "TheBloke",
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": model_info["description"]
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print_colored("\nModel download complete!", "GREEN", bold=True)
    print_colored(f"Model saved to: {model_path}", "GREEN")
    print_colored(f"Config saved to: {config_path}", "GREEN")
    
    # Create a simple README
    readme = f"""# Phi-2 Model for Jetson Orin Nano

## Model Information
- Name: {model_info['name']}
- Quantization: {quantization}
- Format: GGUF
- Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB
- SHA256: {model_info['sha256']}
- Downloaded: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Description
{model_info['description']}

## Usage
To use this model with llama-cpp-python:

```python
from llama_cpp import Llama

model = Llama(
    model_path="{model_path}",
    n_ctx=2048,  # Context window
    n_gpu_layers=-1  # Auto-detect GPU layers
)

output = model.create_completion(
    "Your prompt here",
    max_tokens=100,
    temperature=0.7
)
print(output['choices'][0]['text'])
```

## Verification
Run the verification script to test the model:

```
python verify_phi2.py --model-path {model_path}
```
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    # Verify the model if requested
    if args.verify:
        print_colored("\nVerifying model...", "BLUE", bold=True)
        verify_model(model_path, tokenizer_path)
    
    print_colored("\nDONE! You can now use this model in your applications.", "GREEN", bold=True)
    print_colored(f"Model path: {model_path}", "GREEN")
    print_colored("\nTo verify the model with a test inference:", "YELLOW")
    print_colored(f"python verify_phi2.py --model-path {model_path}", "YELLOW")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())