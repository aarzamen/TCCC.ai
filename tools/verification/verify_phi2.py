#!/usr/bin/env python3
"""
Phi-2 Model Verification for Jetson Orin Nano

This script verifies that a downloaded Phi-2 model can be loaded and run
on the Jetson hardware. It performs various tests to ensure the model
works correctly with the llama-cpp-python library.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import hashlib
import psutil
import subprocess
import numpy as np
import threading
import traceback

# Set up colored output
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

def resource_monitor(stop_event):
    """Monitor system resources while the model is running."""
    print_colored("\nStarting resource monitoring...", "BLUE")
    
    # Initialize data collection
    timestamps = []
    cpu_usages = []
    mem_usages = []
    gpu_usages = []
    
    # Initialize GPU monitoring if available
    has_gpu = False
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        has_gpu = result.returncode == 0
    except:
        pass
    
    # Print header
    print_colored("\nRESOURCE MONITORING:", "YELLOW", bold=True)
    print_colored("------------------", "YELLOW")
    header = f"{'Time':10} | {'CPU %':6} | {'MEM %':6} | {'MEM GB':7}"
    if has_gpu:
        header += f" | {'GPU %':6} | {'GPU MEM':7}"
    print_colored(header, "YELLOW")
    print_colored("------------------", "YELLOW")
    
    # Start monitoring
    start_time = time.time()
    try:
        while not stop_event.is_set():
            # Get current time
            current_time = time.time() - start_time
            timestamps.append(current_time)
            
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_usages.append(cpu_percent)
            
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            mem_gb = mem.used / (1024 ** 3)
            mem_usages.append(mem_percent)
            
            # Try to get GPU usage
            gpu_percent = 0
            gpu_mem = 0
            if has_gpu:
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    if result.returncode == 0:
                        output = result.stdout.strip().split(',')
                        if len(output) >= 2:
                            gpu_percent = float(output[0])
                            gpu_mem = float(output[1]) / 1024  # Convert MB to GB
                except Exception as e:
                    pass
            
            gpu_usages.append(gpu_percent)
            
            # Format output
            status = f"{current_time:8.2f}s | {cpu_percent:5.1f}% | {mem_percent:5.1f}% | {mem_gb:6.2f}GB"
            if has_gpu:
                status += f" | {gpu_percent:5.1f}% | {gpu_mem:6.2f}GB"
            print_colored(status, "BLUE")
            
            # Sleep briefly
            time.sleep(1)
    except Exception as e:
        print_colored(f"Error in resource monitor: {e}", "RED")
    
    # Print summary
    if timestamps:
        print_colored("\nRESOURCE USAGE SUMMARY:", "YELLOW", bold=True)
        print_colored("------------------", "YELLOW")
        print_colored(f"Duration: {timestamps[-1]:.2f} seconds", "BLUE")
        print_colored(f"CPU: Avg {np.mean(cpu_usages):.1f}%, Max {np.max(cpu_usages):.1f}%", "BLUE")
        print_colored(f"Memory: Avg {np.mean(mem_usages):.1f}%, Max {np.max(mem_usages):.1f}%", "BLUE")
        if has_gpu:
            print_colored(f"GPU: Avg {np.mean(gpu_usages):.1f}%, Max {np.max(gpu_usages):.1f}%", "BLUE")
        print_colored("------------------", "YELLOW")

def verify_model(model_path, medical_test=True, verbose=False, run_resource_monitor=True):
    """Verify that a model can be loaded and run on the current hardware."""
    results = {
        "success": False,
        "model_path": str(model_path),
        "timestamp": time.time(),
        "tests": {}
    }
    
    # 1. Check if the model file exists
    print_colored("\nStep 1: Checking if model file exists...", "BLUE", bold=True)
    if not Path(model_path).exists():
        print_colored(f"Model file not found: {model_path}", "RED", bold=True)
        results["tests"]["file_exists"] = False
        return results
    
    print_colored(f"Model file found: {model_path}", "GREEN")
    results["tests"]["file_exists"] = True
    
    # Get file size
    file_size = Path(model_path).stat().st_size
    print_colored(f"Model size: {file_size / 1024 / 1024:.2f} MB", "GREEN")
    results["model_size_mb"] = file_size / 1024 / 1024
    
    # 2. Verify file integrity
    print_colored("\nStep 2: Verifying file integrity...", "BLUE", bold=True)
    try:
        # Calculate file hash
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        
        file_hash = sha256.hexdigest()
        print_colored(f"SHA-256: {file_hash}", "GREEN")
        results["tests"]["file_hash"] = True
        results["file_hash"] = file_hash
    except Exception as e:
        print_colored(f"Error verifying file integrity: {e}", "RED", bold=True)
        results["tests"]["file_hash"] = False
        
    # 3. Load the model
    print_colored("\nStep 3: Loading model...", "BLUE", bold=True)
    
    try:
        # Start the resource monitor if requested
        stop_monitor = threading.Event()
        monitor_thread = None
        if run_resource_monitor:
            monitor_thread = threading.Thread(target=resource_monitor, args=(stop_monitor,))
            monitor_thread.daemon = True
            monitor_thread.start()
        
        # Set the start time
        start_time = time.time()
        
        # Import llama-cpp-python
        from llama_cpp import Llama
        
        try:
            # Load model
            print_colored("Loading model with llama-cpp-python...", "BLUE")
            model = Llama(
                model_path=str(model_path),
                n_ctx=2048,  # Context window
                n_gpu_layers=-1,  # Auto-detect GPU layers
                verbose=verbose  # Show detailed log if requested
            )
            
            # Calculate and record load time
            load_time = time.time() - start_time
            print_colored(f"Model loaded successfully in {load_time:.2f} seconds", "GREEN", bold=True)
            results["tests"]["model_load"] = True
            results["load_time"] = load_time
            
            # 4. Run a basic inference test
            print_colored("\nStep 4: Running basic inference test...", "BLUE", bold=True)
            basic_prompt = "Complete this sentence: The capital of France is"
            
            print_colored("Test prompt: " + basic_prompt, "YELLOW")
            
            # Time the inference
            inference_start = time.time()
            
            output = model.create_completion(
                basic_prompt, 
                max_tokens=20,
                temperature=0.7,
                stop=[".", "\n"]
            )
            
            inference_time = time.time() - inference_start
            
            print_colored("\nBasic inference result:", "GREEN", bold=True)
            print_colored(f"{basic_prompt} {output['choices'][0]['text']}", "GREEN")
            print_colored(f"Inference time: {inference_time:.2f} seconds", "BLUE")
            
            results["tests"]["basic_inference"] = True
            results["basic_inference_time"] = inference_time
            results["basic_inference_result"] = output['choices'][0]['text']
            
            # 5. Run a medical domain test if requested
            if medical_test:
                print_colored("\nStep 5: Running medical domain test...", "BLUE", bold=True)
                medical_prompt = "Patient presents with chest pain, shortness of breath, and diaphoresis. Vital signs show BP 90/60, HR 120, RR 24, SpO2 89%. What is the most likely diagnosis and initial treatment?"
                
                print_colored("Medical test prompt: " + medical_prompt, "YELLOW")
                
                # Time the inference
                medical_start = time.time()
                
                medical_output = model.create_completion(
                    medical_prompt, 
                    max_tokens=100,
                    temperature=0.7
                )
                
                medical_time = time.time() - medical_start
                
                print_colored("\nMedical inference result:", "GREEN", bold=True)
                print_colored(medical_output['choices'][0]['text'], "GREEN")
                print_colored(f"Medical inference time: {medical_time:.2f} seconds", "BLUE")
                
                results["tests"]["medical_inference"] = True
                results["medical_inference_time"] = medical_time
                results["medical_inference_result"] = medical_output['choices'][0]['text']
            
            # Stop resource monitoring
            if run_resource_monitor:
                stop_monitor.set()
                if monitor_thread:
                    monitor_thread.join(timeout=2)
            
            # Overall success
            results["success"] = True
            results["total_time"] = time.time() - start_time
            
            print_colored("\nModel verification completed successfully!", "GREEN", bold=True)
            print_colored(f"Total verification time: {results['total_time']:.2f} seconds", "GREEN")
            
            # Save results to file
            results_path = Path("phi2_verification_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print_colored(f"\nVerification results saved to {results_path}", "GREEN")
            
            return results
            
        except Exception as e:
            # Stop resource monitoring
            if run_resource_monitor:
                stop_monitor.set()
                if monitor_thread:
                    monitor_thread.join(timeout=2)
                
            print_colored(f"Error during model testing: {e}", "RED", bold=True)
            print_colored(traceback.format_exc(), "RED")
            results["error"] = str(e)
            results["tests"]["model_load"] = False
            
            # Save results to file
            results_path = Path("phi2_verification_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
    
    except ImportError as e:
        print_colored(f"Error importing required libraries: {e}", "RED", bold=True)
        print_colored("Please make sure llama-cpp-python is installed:", "RED")
        print_colored("pip install llama-cpp-python", "YELLOW")
        results["error"] = f"ImportError: {str(e)}"
        return results

def ensure_dependencies():
    """Check and install required dependencies."""
    required_packages = ["llama-cpp-python", "numpy", "psutil"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_colored(f"Missing dependencies: {', '.join(missing_packages)}", "YELLOW")
        print_colored("Installing required dependencies...", "BLUE")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
            subprocess.check_call(cmd)
            print_colored("Dependencies installed successfully", "GREEN")
            return True
        except Exception as e:
            print_colored(f"Failed to install dependencies: {e}", "RED", bold=True)
            print("Please manually install the required packages:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify Phi-2 model for Jetson")
    parser.add_argument("--model-path", required=True,
                       help="Path to the model file")
    parser.add_argument("--no-medical", action="store_true",
                       help="Skip medical domain test")
    parser.add_argument("--verbose", action="store_true",
                       help="Show verbose output")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Disable resource monitoring")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for key in COLORS:
            COLORS[key] = ""
    
    # Print header
    print_colored("\n" + "=" * 80, "HEADER")
    print_colored("PHI-2 MODEL VERIFICATION FOR JETSON".center(80), "HEADER", bold=True)
    print_colored("=" * 80 + "\n", "HEADER")
    
    # Ensure dependencies are installed
    if not ensure_dependencies():
        print_colored("Required dependencies are missing. Please install them and try again.", "RED")
        sys.exit(1)
    
    # Verify the model
    verify_model(
        model_path=args.model_path, 
        medical_test=not args.no_medical,
        verbose=args.verbose,
        run_resource_monitor=not args.no_monitor
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())