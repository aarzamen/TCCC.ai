#!/usr/bin/env python3
"""
Test script for Phi-2 model initialization.
This verifies that the tokenizer files are correctly installed and the model can be initialized.
"""

import os
import sys
from pathlib import Path

# Add project path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the llm_analysis module
from tccc.utils import ConfigManager
from tccc.llm_analysis.llm_analysis import LLMAnalysis

def test_phi2_initialization():
    """Test the Phi-2 model initialization."""
    print("Testing Phi-2 model initialization...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Load the configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("llm_analysis")
    
    print(f"Loaded configuration: {config.keys()}")
    print(f"Model path: {config['model']['primary']['path']}")
    
    # Check if the tokenizer files exist
    model_path = Path(config['model']['primary']['path'])
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json"
    ]
    
    print("\nChecking for tokenizer files:")
    for file in tokenizer_files:
        file_path = model_path / file
        exists = file_path.exists()
        print(f"  {file}: {'✅ Found' if exists else '❌ Missing'}")
    
    # Initialize the LLM Analysis module
    print("\nInitializing LLM Analysis module...")
    try:
        llm = LLMAnalysis()
        llm.initialize(config)
        print("✅ LLM Analysis initialized successfully!")
        
        # Check the model info
        primary_model_loaded = llm.llm_engine.model_info["primary"]["loaded"]
        fallback_model_loaded = llm.llm_engine.model_info["fallback"]["loaded"]
        
        print(f"\nPrimary model loaded: {'✅ Yes' if primary_model_loaded else '❌ No'}")
        print(f"Fallback model loaded: {'✅ Yes' if fallback_model_loaded else '❌ No'}")
        
        # Try accessing the LLM engine directly
        print("\nTesting LLM engine...")
        
        # Just verify we can access the engine
        if hasattr(llm, 'llm_engine'):
            print("LLM engine available")
            
            # Get engine status
            status = llm.llm_engine.get_status()
            print(f"Engine status: {status.keys()}")
            
            # Print model info
            print(f"Primary model: {status['models']['primary']['name']}")
            print(f"Fallback model: {status['models']['fallback']['name']}")
            
            return True
        else:
            print("LLM engine not found")
            return False
    except Exception as e:
        print(f"❌ LLM Analysis initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phi2_initialization()
    sys.exit(0 if success else 1)