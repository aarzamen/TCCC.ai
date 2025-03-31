"""
Model cache manager for TCCC.ai system.

This module provides a caching system for machine learning models
to reduce load times and memory usage across different components.
"""

import os
import time
import threading
import logging
import atexit
from typing import Dict, Any, Optional, Callable, List, Tuple
from collections import defaultdict
import weakref

# Import Jetson optimizations if available
try:
    from tccc.utils.jetson_integration import initialize_jetson_optimizations
    from tccc.utils.tensor_optimization import TensorOptimizer
    JETSON_INTEGRATION_AVAILABLE = True
except ImportError:
    JETSON_INTEGRATION_AVAILABLE = False

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from tccc.utils.logging import get_logger
logger = get_logger(__name__)

class ModelCacheEntry:
    """
    Represents a cached model entry with reference counting.
    """
    
    def __init__(self, model, model_config: Dict[str, Any], preload_time: float):
        """
        Initialize a model cache entry.
        
        Args:
            model: The model object
            model_config: Configuration used to create the model
            preload_time: Time taken to load the model
        """
        self.model = model
        self.model_config = model_config
        self.last_access_time = time.time()
        self.creation_time = time.time()
        self.preload_time = preload_time
        self.access_count = 0
        self.reference_count = 0
        self.is_preloaded = False
        
        # Create a lock for thread-safety
        self.lock = threading.RLock()
        
    def increment_ref(self):
        """Increment the reference count."""
        with self.lock:
            self.reference_count += 1
            self.access_count += 1
            self.last_access_time = time.time()
        return self.reference_count
    
    def decrement_ref(self):
        """Decrement the reference count."""
        with self.lock:
            if self.reference_count > 0:
                self.reference_count -= 1
            self.last_access_time = time.time()
        return self.reference_count
    
    def get_ref_count(self):
        """Get the current reference count."""
        with self.lock:
            return self.reference_count
    
    def can_be_unloaded(self):
        """Check if the model can be unloaded."""
        with self.lock:
            # Preloaded models are kept until explicitly unloaded
            if self.is_preloaded:
                return False
            # Otherwise, models with no references can be unloaded
            return self.reference_count == 0
    
    def release_resources(self):
        """
        Release model resources to free memory.
        This depends on the model type.
        """
        try:
            with self.lock:
                # Handle different model types
                if hasattr(self.model, 'to'):
                    # PyTorch model: move to CPU first to free GPU memory
                    if TORCH_AVAILABLE:
                        self.model.to('cpu')
                
                # Clear CUDA cache if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Set reference to None
                self.model = None
                logger.info(f"Released model resources for {self.model_config.get('type', 'unknown')}")
                return True
        except Exception as e:
            logger.error(f"Error releasing model resources: {e}")
            return False


class ModelCacheManager:
    """
    Singleton manager for model caching across the system.
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCacheManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the model cache manager."""
        with self._lock:
            if self._initialized:
                return
                
            # Initialize cache
            self.model_cache = {}
            self.model_factories = {}
            self.preload_queue = []
            self.cleanup_interval = 300  # seconds
            self.max_cache_size = 3  # maximum number of models to keep in cache
            self.memory_threshold = 0.8  # percentage of system memory to trigger cleanup
            
            # Optimizations for Jetson
            self.is_jetson = False
            self.available_memory = float('inf')
            self.max_models_jetson = 1  # More conservative for Jetson
            
            # Configure for Jetson if detected
            if JETSON_INTEGRATION_AVAILABLE:
                try:
                    jetson_integration = initialize_jetson_optimizations()
                    self.is_jetson = jetson_integration.is_jetson
                    
                    if self.is_jetson:
                        logger.info("Model cache configured for Jetson environment")
                        # More conservative settings for Jetson
                        self.cleanup_interval = 120  # seconds
                        self.max_cache_size = self.max_models_jetson
                        self.memory_threshold = 0.7  # Lower threshold for Jetson
                except Exception as e:
                    logger.warning(f"Failed to initialize Jetson optimizations: {e}")
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
            # Register cleanup on exit
            atexit.register(self.shutdown)
            
            self._initialized = True
            logger.info("Model cache manager initialized")
    
    def register_model_factory(self, model_type: str, factory_func: Callable[..., Any]):
        """
        Register a factory function for creating models.
        
        Args:
            model_type: Type identifier for the model
            factory_func: Function that creates the model
        """
        with self._lock:
            self.model_factories[model_type] = factory_func
            logger.info(f"Registered model factory for {model_type}")
    
    def get_model(self, model_type: str, model_config: Dict[str, Any]) -> Tuple[Any, Callable]:
        """
        Get a model from the cache or create a new one.
        
        Args:
            model_type: Type of model to get
            model_config: Configuration for the model
            
        Returns:
            Tuple of (model, release_func)
        """
        with self._lock:
            # Create a cache key from model type and config
            cache_key = self._create_cache_key(model_type, model_config)
            
            # Check if model is in cache
            if cache_key in self.model_cache:
                cache_entry = self.model_cache[cache_key]
                cache_entry.increment_ref()
                logger.info(f"Using cached model {model_type} (refs: {cache_entry.get_ref_count()})")
                
                # Create a release function using weakref to avoid circular references
                release_func = lambda key=cache_key: self._release_model(key)
                
                return cache_entry.model, release_func
            
            # Not in cache, check if we have a factory
            if model_type in self.model_factories:
                # Measure loading time
                start_time = time.time()
                
                # Create the model
                model = self.model_factories[model_type](model_config)
                
                # Calculate loading time
                loading_time = time.time() - start_time
                
                # Add to cache
                cache_entry = ModelCacheEntry(model, model_config, loading_time)
                cache_entry.increment_ref()
                self.model_cache[cache_key] = cache_entry
                
                logger.info(f"Created and cached new model {model_type} in {loading_time:.2f}s")
                
                # Create a release function
                release_func = lambda key=cache_key: self._release_model(key)
                
                return model, release_func
            
            # No factory found
            logger.error(f"No model factory registered for type {model_type}")
            return None, lambda: None
    
    def preload_model(self, model_type: str, model_config: Dict[str, Any]) -> bool:
        """
        Preload a model to avoid first-use delay.
        
        Args:
            model_type: Type of model to preload
            model_config: Configuration for the model
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                # Check if already in cache
                cache_key = self._create_cache_key(model_type, model_config)
                if cache_key in self.model_cache:
                    # Already cached, mark as preloaded
                    self.model_cache[cache_key].is_preloaded = True
                    logger.info(f"Model {model_type} already in cache, marked as preloaded")
                    return True
                
                # Add to preload queue
                self.preload_queue.append((model_type, model_config, cache_key))
                
                # Start preloading in a background thread
                thread = threading.Thread(
                    target=self._preload_model_thread,
                    args=(model_type, model_config, cache_key),
                    daemon=True
                )
                thread.start()
                
                logger.info(f"Started preloading model {model_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to preload model {model_type}: {e}")
            return False
    
    def _preload_model_thread(self, model_type: str, model_config: Dict[str, Any], cache_key: str):
        """
        Thread function for preloading a model.
        
        Args:
            model_type: Type of model to preload
            model_config: Configuration for the model
            cache_key: Cache key for the model
        """
        try:
            # Check if there's enough memory for preloading
            if not self._check_memory_available():
                logger.warning(f"Not enough memory for preloading {model_type}, skipping")
                return
            
            # Check if we have a factory
            if model_type in self.model_factories:
                # Measure loading time
                start_time = time.time()
                
                # Create the model
                model = self.model_factories[model_type](model_config)
                
                # Calculate loading time
                loading_time = time.time() - start_time
                
                # Add to cache with preloaded flag
                with self._lock:
                    cache_entry = ModelCacheEntry(model, model_config, loading_time)
                    cache_entry.is_preloaded = True
                    self.model_cache[cache_key] = cache_entry
                
                logger.info(f"Preloaded model {model_type} in {loading_time:.2f}s")
            else:
                logger.error(f"No model factory registered for type {model_type}")
                
        except Exception as e:
            logger.error(f"Error in preload thread for {model_type}: {e}")
    
    def _release_model(self, cache_key: str) -> bool:
        """
        Release a model back to the cache.
        
        Args:
            cache_key: Cache key for the model
            
        Returns:
            Success status
        """
        with self._lock:
            if cache_key in self.model_cache:
                cache_entry = self.model_cache[cache_key]
                ref_count = cache_entry.decrement_ref()
                logger.debug(f"Released model {cache_key} (refs: {ref_count})")
                return True
            
            logger.warning(f"Tried to release unknown model {cache_key}")
            return False
    
    def _cleanup_loop(self):
        """
        Thread function for periodic cleanup of unused models.
        """
        while True:
            try:
                # Sleep first to allow initialization
                time.sleep(self.cleanup_interval)
                
                # Check memory usage and perform cleanup if needed
                self._check_memory_and_cleanup()
                
            except Exception as e:
                logger.error(f"Error in model cache cleanup loop: {e}")
                time.sleep(60)  # Sleep a bit longer if there was an error
    
    def _check_memory_and_cleanup(self) -> bool:
        """
        Check memory usage and clean up unused models if needed.
        
        Returns:
            True if cleanup was performed
        """
        with self._lock:
            # Check current memory usage
            memory_usage = self._get_memory_usage()
            cache_size = len(self.model_cache)
            
            # Decide if cleanup is needed
            cleanup_needed = (
                memory_usage > self.memory_threshold or
                cache_size > self.max_cache_size
            )
            
            if cleanup_needed:
                logger.info(f"Cleaning up model cache (memory: {memory_usage:.1%}, models: {cache_size})")
                return self._cleanup_unused_models()
            
            return False
    
    def _cleanup_unused_models(self) -> bool:
        """
        Clean up unused models from the cache.
        
        Returns:
            True if any models were cleaned up
        """
        with self._lock:
            # Find models with no references
            unused_models = [
                key for key, entry in self.model_cache.items()
                if entry.can_be_unloaded()
            ]
            
            # If running on Jetson, only keep max_models_jetson
            if self.is_jetson and len(self.model_cache) > self.max_models_jetson:
                # Find oldest accessed models
                old_models = sorted(
                    [
                        (key, entry) for key, entry in self.model_cache.items()
                        if not entry.is_preloaded and entry.get_ref_count() == 0
                    ],
                    key=lambda x: x[1].last_access_time
                )
                
                # Add old models to unused_models, keeping only max_models_jetson
                extra_models = len(self.model_cache) - self.max_models_jetson
                for i in range(min(extra_models, len(old_models))):
                    if old_models[i][0] not in unused_models:
                        unused_models.append(old_models[i][0])
            
            # Remove unused models
            for key in unused_models:
                entry = self.model_cache[key]
                if entry.release_resources():
                    del self.model_cache[key]
                    logger.info(f"Removed unused model {key} from cache")
            
            return len(unused_models) > 0
    
    def _get_memory_usage(self) -> float:
        """
        Get the current memory usage as a percentage.
        
        Returns:
            Memory usage as a percentage (0.0-1.0)
        """
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Get GPU memory usage
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                return allocated / total
            else:
                # Get system memory usage
                import psutil
                return psutil.virtual_memory().percent / 100.0
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def _check_memory_available(self) -> bool:
        """
        Check if there's enough memory available for loading a new model.
        
        Returns:
            True if enough memory is available
        """
        try:
            memory_usage = self._get_memory_usage()
            enough_memory = memory_usage < self.memory_threshold
            
            if not enough_memory:
                # Try to free memory
                if self._cleanup_unused_models():
                    # Check again after cleanup
                    memory_usage = self._get_memory_usage()
                    enough_memory = memory_usage < self.memory_threshold
            
            return enough_memory
        except Exception as e:
            logger.warning(f"Error checking memory availability: {e}")
            return True  # Assume it's fine
    
    def _create_cache_key(self, model_type: str, model_config: Dict[str, Any]) -> str:
        """
        Create a cache key from model type and config.
        
        Args:
            model_type: Type of model
            model_config: Configuration for the model
            
        Returns:
            Cache key string
        """
        # Extract relevant config parameters for the key
        relevant_params = {}
        
        if model_type == "faster-whisper":
            # For faster-whisper, the key parameters are model size, compute_type and language
            relevant_params = {
                "size": model_config.get("size", "tiny"),
                "compute_type": model_config.get("compute_type", "int8"),
                "language": model_config.get("language", "en")
            }
        elif model_type == "whisper":
            # For standard whisper, similar parameters
            relevant_params = {
                "size": model_config.get("size", "tiny"),
                "language": model_config.get("language", "en")
            }
        else:
            # For other model types, use the full config
            relevant_params = model_config
        
        # Create a string representation
        params_str = "_".join(f"{k}={v}" for k, v in sorted(relevant_params.items()))
        
        return f"{model_type}_{params_str}"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the model cache.
        
        Returns:
            Status dictionary
        """
        with self._lock:
            cached_models = []
            
            for key, entry in self.model_cache.items():
                cached_models.append({
                    "key": key,
                    "type": entry.model_config.get("type", "unknown"),
                    "references": entry.get_ref_count(),
                    "preloaded": entry.is_preloaded,
                    "age": time.time() - entry.creation_time,
                    "access_count": entry.access_count,
                    "preload_time": entry.preload_time
                })
            
            # Get memory usage
            memory_usage = self._get_memory_usage()
            
            # Detect if we're on Jetson
            is_jetson = self.is_jetson
            if not is_jetson and TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    device_name = torch.cuda.get_device_name(0).lower()
                    is_jetson = any(name in device_name for name in ["tegra", "orin", "xavier", "jetson"])
                except Exception:
                    pass
            
            return {
                "cached_models": cached_models,
                "cache_size": len(self.model_cache),
                "max_cache_size": self.max_cache_size,
                "memory_usage": memory_usage,
                "memory_threshold": self.memory_threshold,
                "cleanup_interval": self.cleanup_interval,
                "is_jetson": is_jetson,
                "preload_queue_size": len(self.preload_queue)
            }
    
    def shutdown(self):
        """
        Shutdown the model cache manager, releasing all resources.
        """
        with self._lock:
            logger.info("Shutting down model cache manager")
            
            # Release all models
            for key, entry in list(self.model_cache.items()):
                entry.release_resources()
            
            # Clear cache
            self.model_cache.clear()
            self.preload_queue.clear()
            
            logger.info("Model cache manager shutdown complete")


# Create singleton instance
_model_cache_manager = None

def get_model_cache_manager() -> ModelCacheManager:
    """
    Get the singleton model cache manager instance.
    
    Returns:
        ModelCacheManager instance
    """
    global _model_cache_manager
    if _model_cache_manager is None:
        _model_cache_manager = ModelCacheManager()
    return _model_cache_manager