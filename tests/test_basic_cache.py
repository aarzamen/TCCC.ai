#!/usr/bin/env python3
"""
Basic test of model caching with minimal dependencies.
"""

import os
import sys
import time
from typing import Dict, Any, Tuple, Callable

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class DummyModel:
    """Simple dummy model for testing."""
    
    def __init__(self, name: str, size: str):
        self.name = name
        self.size = size
        print(f"Created DummyModel: {name} (size: {size})")
    
    def __str__(self):
        return f"DummyModel({self.name}, {self.size})"

def create_dummy_model(config: Dict[str, Any]) -> DummyModel:
    """Factory function for creating dummy models."""
    name = config.get("name", "default")
    size = config.get("size", "tiny")
    
    # Simulate loading delay
    print(f"Loading model {name}...")
    time.sleep(1.0)
    
    return DummyModel(name, size)

def main():
    """Main function."""
    print("\n=== Basic Model Caching Test ===\n")
    
    # Create a simple model cache
    cache = {}
    
    # Function to generate cache key
    def get_cache_key(config: Dict[str, Any]) -> str:
        return f"{config.get('name', 'default')}_{config.get('size', 'tiny')}"
    
    # Function to get model from cache or create new
    def get_model(config: Dict[str, Any]) -> Tuple[DummyModel, Callable]:
        key = get_cache_key(config)
        
        if key in cache:
            print(f"Cache hit for {key}")
            model = cache[key]["model"]
            cache[key]["refs"] += 1
        else:
            print(f"Cache miss for {key}")
            model = create_dummy_model(config)
            cache[key] = {"model": model, "refs": 1}
        
        # Create release function
        def release_func():
            if key in cache:
                cache[key]["refs"] -= 1
                print(f"Released {key}, refs: {cache[key]['refs']}")
        
        return model, release_func
    
    # Test with first model
    print("\n--- Test 1: First model ---")
    config1 = {"name": "model1", "size": "tiny"}
    start = time.time()
    model1, release1 = get_model(config1)
    print(f"Got model: {model1}")
    print(f"Time: {time.time() - start:.2f}s")
    
    # Test with second model (different config)
    print("\n--- Test 2: Second model ---")
    config2 = {"name": "model2", "size": "small"}
    start = time.time()
    model2, release2 = get_model(config2)
    print(f"Got model: {model2}")
    print(f"Time: {time.time() - start:.2f}s")
    
    # Test with first model again (should be cached)
    print("\n--- Test 3: First model again ---")
    start = time.time()
    model1_again, release1_again = get_model(config1)
    print(f"Got model: {model1_again}")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Same object as first: {model1 is model1_again}")
    
    # Print cache status
    print("\n--- Cache Status ---")
    for key, entry in cache.items():
        print(f"Key: {key}, Refs: {entry['refs']}, Model: {entry['model']}")
    
    # Release models
    print("\n--- Release Models ---")
    release1()
    release2()
    release1_again()
    
    # Print cache status after release
    print("\n--- Cache Status After Release ---")
    for key, entry in cache.items():
        print(f"Key: {key}, Refs: {entry['refs']}, Model: {entry['model']}")
    
    # Clean up cache
    print("\n--- Cleanup Cache ---")
    for key in list(cache.keys()):
        if cache[key]["refs"] <= 0:
            print(f"Removing {key} from cache")
            del cache[key]
    
    # Print final cache status
    print("\n--- Final Cache Status ---")
    for key, entry in cache.items():
        print(f"Key: {key}, Refs: {entry['refs']}, Model: {entry['model']}")
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())