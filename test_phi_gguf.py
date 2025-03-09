#!/usr/bin/env python3
"""
Test script for the Phi-2 GGUF model implementation.

This script tests the PhiGGUFModel implementation to ensure it works properly
with the GGUF format. It sends a simple medical query and displays the response.
"""

import os
import sys
import logging
import time
import json
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PHI2_GGUF_TEST")

def main():
    """Main function to test the Phi-2 GGUF model."""
    parser = argparse.ArgumentParser(description="Test the Phi-2 GGUF model")
    parser.add_argument("--model-path", default="models/phi-2-gguf/phi-2.Q4_K_M.gguf",
                       help="Path to the GGUF model file")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU acceleration if available")
    parser.add_argument("--mock", action="store_true",
                       help="Force using mock implementation for testing")
    
    args = parser.parse_args()
    
    # Force mock mode if requested
    if args.mock:
        os.environ["TCCC_USE_MOCK_LLM"] = "1"
        logger.info("Forcing mock mode for testing")
    
    try:
        # Import modules only after logging is configured
        from tccc.llm_analysis import PhiGGUFModel, get_phi_gguf_model
        
        # Check if the model exists
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.info("To download the model, run: python -c \"from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='TheBloke/phi-2-GGUF', filename='phi-2.Q4_K_M.gguf'))\"")
            sys.exit(1)
        
        # Create model configuration
        model_config = {
            "gguf_model_path": str(model_path),
            "use_gpu": args.use_gpu,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024
        }
        
        # Print model configuration
        logger.info(f"Model configuration: {json.dumps(model_config, indent=2)}")
        
        # Initialize the model
        logger.info(f"Initializing Phi-2 GGUF model from {model_path}")
        start_time = time.time()
        model = get_phi_gguf_model(model_config)
        load_time = time.time() - start_time
        logger.info(f"Model initialization took {load_time:.2f} seconds")
        
        # Print model info
        try:
            model_info = str(type(model))
            logger.info(f"Loaded model type: {model_info}")
        except:
            pass
        
        # Test prompts
        test_prompts = [
            "What are the key steps in treating a tension pneumothorax in a battlefield setting?",
            "List the essential medications needed in a military medic's aid bag for tactical combat casualty care."
        ]
        
        # Process each prompt
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n--- Testing Prompt {i+1} ---")
            logger.info(f"Prompt: {prompt}")
            
            # Generate response
            start_time = time.time()
            response = model.generate(prompt)
            generation_time = time.time() - start_time
            
            # Extract text
            text = response["choices"][0]["text"]
            
            # Print results
            logger.info(f"Generation time: {generation_time:.2f} seconds")
            logger.info(f"Response:\n{text}\n")
            
            # Print token usage if available
            if "usage" in response:
                usage = response["usage"]
                logger.info(f"Token usage: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
        
        # Print model metrics
        metrics = model.get_metrics()
        logger.info(f"\nModel metrics: {json.dumps(metrics, indent=2)}")
        
        logger.info("Test completed successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.info("Make sure that llama-cpp-python is installed: pip install llama-cpp-python")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()