#!/usr/bin/env python3
"""
Test script for model cache manager.
This script validates the basic functionality of the model cache manager
without requiring external model files.
"""

import os
import sys
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelCacheTest")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Create a mock model class for testing
class MockModel:
    def __init__(self, config):
        self.config = config
        self.name = config.get("name", "default")
        self.size = config.get("size", "tiny")
        print(f"Created MockModel: {self.name} (size: {self.size})")
        
    def __del__(self):
        print(f"Destroyed MockModel: {self.name}")
        
    def info(self):
        return f"MockModel: {self.name} (size: {self.size})"

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print(" Model Cache Manager Test ".center(60, "="))
    print("=" * 60 + "\n")
    
    # Check if we can import the necessary modules
    try:
        from tccc.stt_engine.model_cache_manager import get_model_cache_manager
        print("✓ Successfully imported model_cache_manager")
    except ImportError as e:
        print(f"✗ Failed to import model_cache_manager: {e}")
        return 1
    
    # Create cache manager
    try:
        cache_manager = get_model_cache_manager()
        print("✓ Created cache manager instance")
    except Exception as e:
        print(f"✗ Failed to create cache manager: {e}")
        return 1
    
    # Register mock model factory
    def create_mock_model(config):
        # Simulate loading delay
        time.sleep(0.5)
        return MockModel(config)
    
    cache_manager.register_model_factory("mock", create_mock_model)
    print("✓ Registered mock model factory")
    
    # Print initial cache status
    status = cache_manager.get_status()
    print(f"\nInitial cache status:")
    print(f"  Cache size: {status.get('cache_size', 0)}")
    print(f"  Max cache size: {status.get('max_cache_size', 0)}")
    
    # Test 1: Get first model
    print("\nTest 1: Get first model")
    model1_config = {"name": "model1", "size": "tiny"}
    start_time = time.time()
    model1, release_func1 = cache_manager.get_model("mock", model1_config)
    load_time1 = time.time() - start_time
    
    if model1 is not None:
        print(f"✓ Got model1: {model1.info()}")
        print(f"  Load time: {load_time1:.2f}s")
    else:
        print("✗ Failed to get model1")
        return 1
    
    # Test 2: Get second model (different config)
    print("\nTest 2: Get second model (different config)")
    model2_config = {"name": "model2", "size": "small"}
    start_time = time.time()
    model2, release_func2 = cache_manager.get_model("mock", model2_config)
    load_time2 = time.time() - start_time
    
    if model2 is not None:
        print(f"✓ Got model2: {model2.info()}")
        print(f"  Load time: {load_time2:.2f}s")
    else:
        print("✗ Failed to get model2")
        return 1
    
    # Test 3: Get model1 again (should be cached)
    print("\nTest 3: Get model1 again (should be cached)")
    start_time = time.time()
    model1_again, release_func1_again = cache_manager.get_model("mock", model1_config)
    load_time1_again = time.time() - start_time
    
    if model1_again is not None:
        print(f"✓ Got model1 again: {model1_again.info()}")
        print(f"  Load time: {load_time1_again:.2f}s")
        print(f"  Cache hit: {load_time1_again < 0.1}")
    else:
        print("✗ Failed to get model1 again")
        return 1
    
    # Print cache status after loading models
    status = cache_manager.get_status()
    print(f"\nCache status after loading models:")
    print(f"  Cache size: {status.get('cache_size', 0)}")
    cached_models = status.get('cached_models', [])
    for i, model in enumerate(cached_models):
        print(f"  Model {i+1}: {model.get('key', 'unknown')} (refs: {model.get('references', 0)})")
    
    # Test 4: Release models
    print("\nTest 4: Release models")
    release_func1()
    print("✓ Released model1")
    
    release_func2()
    print("✓ Released model2")
    
    release_func1_again()
    print("✓ Released model1_again")
    
    # Print cache status after releasing
    status = cache_manager.get_status()
    print(f"\nCache status after releasing models:")
    print(f"  Cache size: {status.get('cache_size', 0)}")
    cached_models = status.get('cached_models', [])
    for i, model in enumerate(cached_models):
        print(f"  Model {i+1}: {model.get('key', 'unknown')} (refs: {model.get('references', 0)})")
    
    # Test 5: Force cleanup
    print("\nTest 5: Force cleanup")
    cache_manager._cleanup_unused_models()
    
    # Check status after cleanup
    status = cache_manager.get_status()
    print(f"Cache status after cleanup:")
    print(f"  Cache size: {status.get('cache_size', 0)}")
    
    # Test 6: Shutdown
    print("\nTest 6: Shutdown")
    cache_manager.shutdown()
    print("✓ Cache manager shutdown complete")
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())