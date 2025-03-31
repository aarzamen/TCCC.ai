#!/usr/bin/env python3
"""
Phi-2 Model Integration for TCCC.ai

This script integrates the downloaded Phi-2 GGUF model with the existing TCCC.ai
codebase, replacing the mock implementation with the real model. It modifies the
necessary configuration files and creates an adapter class to interface between
llama-cpp-python and the existing code.
"""

import os
import sys
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
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

def check_model_exists(model_path):
    """Check if the model file exists and is valid."""
    path = Path(model_path)
    if not path.exists():
        print_colored(f"Model file not found: {model_path}", "RED", bold=True)
        return False
    
    # Check file size (should be at least 100MB)
    if path.stat().st_size < 100 * 1024 * 1024:
        print_colored(f"Model file seems too small: {path.stat().st_size / 1024 / 1024:.2f} MB", "YELLOW")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True

def create_llama_adapter(project_dir):
    """Create an adapter for llama-cpp-python to work with the codebase."""
    adapter_path = Path(project_dir) / "src" / "tccc" / "llm_analysis" / "llama_cpp_adapter.py"
    
    adapter_code = """#!/usr/bin/env python3
\"\"\"
LLAMA-CPP Adapter for TCCC.ai

This module provides an adapter between the llama-cpp-python library and
the TCCC.ai codebase, allowing the use of Phi-2 and other GGUF models.
\"\"\"

import os
import sys
import time
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import llama-cpp
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    
# Local imports
from tccc.utils.logging import get_logger

logger = get_logger(__name__)

class LlamaCppModel:
    \"\"\"
    Adapter for Phi-2 and other GGUF models using llama-cpp-python.
    
    This class provides a compatible interface with the PhiModel class
    but uses llama-cpp-python for inference.
    \"\"\"
    
    def __init__(self, config: Dict[str, Any]):
        \"\"\"Initialize the model adapter.
        
        Args:
            config: Configuration dictionary with model settings
        \"\"\"
        self.config = config
        self.model_path = Path(config.get("model_path", "models/phi2_gguf/phi-2.Q4_K_M.gguf"))
        self.use_gpu = config.get("use_gpu", True)
        self.quantization = config.get("quantization", "4-bit")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        
        # Check if llama-cpp is available
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python is not installed or not working")
            raise ImportError("llama-cpp-python is required but not installed")
        
        # Metrics for tracking
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency": 0.0
        }
        
        # Thread lock for shared model 
        self.lock = threading.Lock()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        \"\"\"Load the GGUF model using llama-cpp-python.\"\"\"
        try:
            logger.info(f"Loading model from {self.model_path} using llama-cpp-python")
            
            # Try to auto-detect optimal parameters based on the model name
            model_filename = self.model_path.name.lower()
            
            # Configure parameters based on quantization level in file name
            n_gpu_layers = -1  # Default: Auto-detect GPU layers
            
            if "q4" in model_filename:
                logger.info("Detected 4-bit quantization in filename")
                self.quantization = "4-bit"
            elif "q5" in model_filename:
                logger.info("Detected 5-bit quantization in filename")
                self.quantization = "5-bit" 
            elif "q8" in model_filename:
                logger.info("Detected 8-bit quantization in filename")
                self.quantization = "8-bit"
                
            # Adjust settings based on GPU availability
            if not self.use_gpu:
                logger.info("GPU usage disabled, using CPU only")
                n_gpu_layers = 0
            
            # Load the model with llama-cpp-python
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,  # Context window size
                n_gpu_layers=n_gpu_layers,
                verbose=False  # Set to True for debug info
            )
            
            # Set model info
            self.model_info = {
                "type": "llama-cpp-python",
                "model_path": str(self.model_path),
                "n_ctx": 2048,
                "n_gpu_layers": n_gpu_layers,
                "quantization": self.quantization
            }
            
            logger.info(f"Successfully loaded model with llama-cpp-python")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _prepare_prompt_for_phi(self, prompt: str) -> str:
        \"\"\"Format the prompt for Phi-2 model.
        
        Args:
            prompt: Raw input prompt
            
        Returns:
            Formatted prompt for the model
        \"\"\"
        # Phi-2 works well with this instruct format
        formatted_prompt = f\"\"\"<|system|>
You are an AI medical assistant for military medics, specializing in Tactical Combat Casualty Care.
Analyze the following transcript and extract the requested information accurately.
<|user|>
{prompt}
<|assistant|>\"\"\"
        
        return formatted_prompt
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None) -> Dict[str, Any]:
        \"\"\"Generate text based on the prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling value
            
        Returns:
            Dictionary with generated text
        \"\"\"
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        # Set generation parameters
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        try:
            # Format prompt
            formatted_prompt = self._prepare_prompt_for_phi(prompt)
            
            # Generate using llama-cpp (with thread safety)
            with self.lock:
                # Tokenize for counting tokens
                prompt_tokens = len(self.model.tokenize(formatted_prompt.encode()))
                
                # Run inference
                output = self.model.create_completion(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=["<|user|>", "<|system|>"]  # Stop at special tokens
                )
            
            # Extract generated text
            generated_text = output["choices"][0]["text"]
            
            # Count completion tokens
            completion_tokens = len(self.model.tokenize(generated_text.encode()))
            total_tokens = prompt_tokens + completion_tokens
            
            # Update metrics
            self.metrics["total_tokens"] += total_tokens
            elapsed = time.time() - start_time
            self.metrics["avg_latency"] = (
                (self.metrics["avg_latency"] * (self.metrics["total_requests"] - 1) + elapsed) / 
                self.metrics["total_requests"]
            )
            
            # Format response to match expected structure
            return {
                "id": str(uuid.uuid4()),
                "choices": [{"text": generated_text}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "model": "phi-2-gguf",
                "latency": elapsed
            }
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        \"\"\"Get model usage metrics.
        
        Returns:
            Dictionary with usage metrics
        \"\"\"
        return {
            "total_requests": self.metrics["total_requests"],
            "total_tokens": int(self.metrics["total_tokens"]),
            "avg_latency": round(self.metrics["avg_latency"], 3),
            "model": "phi-2-gguf",
            "quantization": self.quantization,
            "use_gpu": self.use_gpu
        }
"""
    
    # Write the adapter file
    with open(adapter_path, 'w') as f:
        f.write(adapter_code)
    
    print_colored(f"Created LLAMA-CPP adapter at: {adapter_path}", "GREEN")
    return adapter_path

def update_phi_model_init(project_dir, adapter_path):
    """Update the phi_model.py file to use the LLAMA-CPP adapter."""
    phi_model_path = Path(project_dir) / "src" / "tccc" / "llm_analysis" / "phi_model.py"
    
    if not phi_model_path.exists():
        print_colored(f"phi_model.py not found at {phi_model_path}", "RED", bold=True)
        return False
    
    try:
        # Read the original file
        with open(phi_model_path, 'r') as f:
            original_content = f.read()
        
        # Create a backup
        backup_path = phi_model_path.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.write(original_content)
        
        print_colored(f"Created backup at: {backup_path}", "BLUE")
        
        # Find the get_phi_model function
        if "def get_phi_model(" not in original_content:
            print_colored("Could not find get_phi_model function in phi_model.py", "RED", bold=True)
            return False
        
        # Find the appropriate insertion point for the new import
        import_section_end = original_content.find("# Local imports")
        if import_section_end > 0:
            # Find the end of the import section
            import_section_end = original_content.find("\n", import_section_end + 15)
            
            # Insert the new import
            modified_content = (
                original_content[:import_section_end] + 
                "\n# Import LLAMA-CPP adapter\ntry:\n    from tccc.llm_analysis.llama_cpp_adapter import LlamaCppModel\n    LLAMA_CPP_AVAILABLE = True\nexcept ImportError:\n    LLAMA_CPP_AVAILABLE = False\n" +
                original_content[import_section_end:]
            )
        else:
            # If we can't find the import section, just add it at the top
            modified_content = "# Import LLAMA-CPP adapter\ntry:\n    from tccc.llm_analysis.llama_cpp_adapter import LlamaCppModel\n    LLAMA_CPP_AVAILABLE = True\nexcept ImportError:\n    LLAMA_CPP_AVAILABLE = False\n\n" + original_content
        
        # Update the get_phi_model function to try the LLAMA-CPP adapter
        get_phi_model_start = modified_content.find("def get_phi_model(")
        if get_phi_model_start > 0:
            # Find the first line of the function body
            function_body_start = modified_content.find(":", get_phi_model_start) + 1
            
            # Find the relevant code section
            start_of_code = modified_content.find("    # Check if mock mode is explicitly enabled", function_body_start)
            if start_of_code > 0:
                # Insert code to try the LLAMA-CPP adapter
                updated_function = (
                    modified_content[:start_of_code] +
                    """    # Try the real LLAMA-CPP adapter if available
    try:
        if LLAMA_CPP_AVAILABLE:
            # Check if config contains a model_path with a .gguf extension
            if "model_path" in config and str(config["model_path"]).lower().endswith(".gguf"):
                logger.info("GGUF model detected, using LLAMA-CPP adapter")
                return LlamaCppModel(config)
    except Exception as e:
        logger.warning(f"Failed to use LLAMA-CPP adapter: {str(e)}")
    
""" +
                    modified_content[start_of_code:]
                )
                
                modified_content = updated_function
        
        # Write the modified file
        with open(phi_model_path, 'w') as f:
            f.write(modified_content)
        
        print_colored(f"Updated {phi_model_path} to use LLAMA-CPP adapter", "GREEN")
        return True
        
    except Exception as e:
        print_colored(f"Error updating phi_model.py: {e}", "RED", bold=True)
        print_colored(traceback.format_exc(), "RED")
        return False

def update_llm_config(project_dir, model_path):
    """Update the LLM config to use the real model."""
    config_path = Path(project_dir) / "config" / "llm_analysis.yaml"
    
    if not config_path.exists():
        print_colored(f"Config file not found at {config_path}", "RED", bold=True)
        return False
    
    try:
        # Read the original config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Create a backup
        backup_path = config_path.with_suffix('.yaml.bak')
        with open(backup_path, 'w') as f:
            f.write(config_content)
        
        print_colored(f"Created backup at: {backup_path}", "BLUE")
        
        # Find the model path setting
        path_line_idx = config_content.find("    path:")
        if path_line_idx > 0:
            # Find the end of this line
            line_end = config_content.find("\n", path_line_idx)
            
            # Update the path
            modified_config = (
                config_content[:path_line_idx] + 
                f"    path: \"{model_path}\"" +
                config_content[line_end:]
            )
            
            # Update force_real setting if it exists
            force_real_idx = modified_config.find("    force_real:")
            if force_real_idx > 0:
                line_end = modified_config.find("\n", force_real_idx)
                modified_config = (
                    modified_config[:force_real_idx] + 
                    "    force_real: true" +
                    modified_config[line_end:]
                )
            
            # Write the modified config
            with open(config_path, 'w') as f:
                f.write(modified_config)
            
            print_colored(f"Updated {config_path} to use real model at {model_path}", "GREEN")
            return True
        else:
            print_colored("Could not find path setting in config file", "RED", bold=True)
            return False
        
    except Exception as e:
        print_colored(f"Error updating LLM config: {e}", "RED", bold=True)
        print_colored(traceback.format_exc(), "RED")
        return False

def update_environment_variables():
    """Update environment variables to force real model usage."""
    # Create a helper script to set the environment variable
    script_path = Path.cwd() / "use_real_phi2.sh"
    
    script_content = """#!/bin/bash
# Enable real Phi-2 model usage for TCCC.ai
export TCCC_USE_MOCK_LLM=0

# Display a message
echo "Environment configured to use real Phi-2 model"
echo "Run your TCCC.ai commands in this shell session"

# Run the provided command or open a shell
if [ $# -gt 0 ]; then
    echo "Running command: $@"
    "$@"
else
    # Spawn a new shell with the environment variable set
    $SHELL
fi
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print_colored(f"Created environment setup script: {script_path}", "GREEN")
    print_colored("Run this script with: source use_real_phi2.sh", "YELLOW")
    return script_path

def create_test_script(project_dir, model_path):
    """Create a simple test script for the integrated model."""
    test_script_path = Path(project_dir) / "test_real_phi2.py"
    
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Test script for real Phi-2 model integration with TCCC.ai.

This script verifies that the real Phi-2 model is being used instead of the mock model.
\"\"\"

import os
import sys
import time
import json
from pathlib import Path

# Set the environment variable to ensure real model is used
os.environ["TCCC_USE_MOCK_LLM"] = "0"

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Import the required modules
from tccc.llm_analysis import LLMAnalysis
from tccc.utils.config_manager import ConfigManager

def main():
    \"\"\"Test the real Phi-2 model integration.\"\"\"
    print("\\n" + "=" * 80)
    print("TESTING REAL PHI-2 MODEL INTEGRATION".center(80))
    print("=" * 80 + "\\n")
    
    # Initialize LLM analysis
    print("Initializing LLM analysis module...")
    config_manager = ConfigManager()
    llm_config = config_manager.load_config("llm_analysis")
    
    # Force real model implementation
    if "model" in llm_config and "primary" in llm_config["model"]:
        llm_config["model"]["primary"]["force_real"] = True
        
    # Verify the model path
    model_path = llm_config.get("model", {}).get("primary", {}).get("path", "")
    print(f"Model path from config: {{model_path}}")
    
    # Initialize the LLM analysis module
    llm_analysis = LLMAnalysis()
    init_success = llm_analysis.initialize(llm_config)
    
    if not init_success:
        print("Failed to initialize LLM analysis module")
        return 1
    
    # Get model status
    status = llm_analysis.get_status()
    print("\\nModel status:")
    print(json.dumps(status["llm_engine"], indent=2))
    
    # Process a test medical text
    print("\\nProcessing test medical text...")
    test_text = "Patient presents with shortness of breath and chest pain radiating to the left arm. Vital signs: BP 160/95, HR 110, RR 22, SpO2 92%. History of hypertension."
    print(f"Test text: {{test_text}}")
    
    # Process the text
    start_time = time.time()
    transcription = {{'text': test_text}}
    results = llm_analysis.process_transcription(transcription)
    elapsed = time.time() - start_time
    
    # Generate a report
    report = llm_analysis.generate_report("tccc", results)
    
    # Display results
    print(f"\\nProcessing completed in {{elapsed:.2f}} seconds")
    print(f"Extracted {{len(results)}} entities")
    
    print("\\nEntities:")
    for i, entity in enumerate(results[:5]):  # Display first 5 entities
        print(f"  {{i+1}}. {{entity.get('type', '')}}: {{entity.get('value', entity.get('name', 'Unknown'))}}")
    
    if len(results) > 5:
        print(f"  ... and {{len(results) - 5}} more")
    
    print("\\nReport:")
    print(f"{{report['content'][:200]}}...")
    
    # Save results to a file
    output_path = Path("phi2_integration_test_results.json")
    with open(output_path, 'w') as f:
        json.dump({{"entities": results, "report": report}}, f, indent=2)
    
    print(f"\\nResults saved to {{output_path}}")
    
    # Check if mock mode was used
    is_mock = False
    if "model" in status["llm_engine"] and "primary" in status["llm_engine"]["model"]:
        model_name = status["llm_engine"]["model"]["primary"].get("name", "").lower()
        is_mock = "mock" in model_name
    
    if is_mock:
        print("\\n⚠️ WARNING: Mock model was used instead of real Phi-2 model!")
        print("Check that the model file exists and is correctly configured.")
        return 1
    else:
        print("\\n✅ SUCCESS: Real Phi-2 model was used for inference!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
"""
    
    with open(test_script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(test_script_path, 0o755)
    
    print_colored(f"Created test script: {test_script_path}", "GREEN")
    return test_script_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Integrate Phi-2 GGUF model with TCCC.ai")
    parser.add_argument("--model-path", required=True,
                       help="Path to the downloaded model file")
    parser.add_argument("--project-dir", default=".",
                       help="Path to the TCCC.ai project directory")
    parser.add_argument("--force", action="store_true",
                       help="Force integration even if model does not exist")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for key in COLORS:
            COLORS[key] = ""
    
    # Print header
    print_colored("\n" + "=" * 80, "HEADER")
    print_colored("PHI-2 MODEL INTEGRATION FOR TCCC.AI".center(80), "HEADER", bold=True)
    print_colored("=" * 80 + "\n", "HEADER")
    
    # Resolve paths
    model_path = os.path.abspath(args.model_path)
    project_dir = os.path.abspath(args.project_dir)
    
    # Check if the model exists
    if not args.force:
        if not check_model_exists(model_path):
            print_colored("Model integration aborted", "RED", bold=True)
            return 1
    
    print_colored(f"Integrating model: {model_path}", "GREEN")
    print_colored(f"Project directory: {project_dir}", "GREEN")
    
    # 1. Create the LLAMA-CPP adapter
    print_colored("\nStep 1: Creating LLAMA-CPP adapter...", "BLUE", bold=True)
    adapter_path = create_llama_adapter(project_dir)
    
    # 2. Update the phi_model.py file
    print_colored("\nStep 2: Updating phi_model.py...", "BLUE", bold=True)
    phi_model_updated = update_phi_model_init(project_dir, adapter_path)
    
    # 3. Update the LLM config
    print_colored("\nStep 3: Updating LLM config...", "BLUE", bold=True)
    config_updated = update_llm_config(project_dir, model_path)
    
    # 4. Update environment variables
    print_colored("\nStep 4: Creating environment setup script...", "BLUE", bold=True)
    env_script = update_environment_variables()
    
    # 5. Create a test script
    print_colored("\nStep 5: Creating test script...", "BLUE", bold=True)
    test_script = create_test_script(project_dir, model_path)
    
    # 6. Summary
    print_colored("\nIntegration Complete!", "GREEN", bold=True)
    print_colored("Summary of changes:", "GREEN")
    print_colored(f"- Created LLAMA-CPP adapter: {adapter_path}", "BLUE")
    print_colored(f"- Updated phi_model.py: {'Success' if phi_model_updated else 'Failed'}", "GREEN" if phi_model_updated else "RED")
    print_colored(f"- Updated LLM config: {'Success' if config_updated else 'Failed'}", "GREEN" if config_updated else "RED")
    print_colored(f"- Created environment script: {env_script}", "BLUE")
    print_colored(f"- Created test script: {test_script}", "BLUE")
    
    # 7. Next steps
    print_colored("\nNext Steps:", "YELLOW", bold=True)
    print_colored("1. Set the environment variable: source use_real_phi2.sh", "YELLOW")
    print_colored("2. Run the test script: ./test_real_phi2.py", "YELLOW")
    print_colored("3. Try the full system: python run_mic_pipeline.py", "YELLOW")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())