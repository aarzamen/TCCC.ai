#!/usr/bin/env python3
"""
Jetson Integration Module for TCCC.ai
-------------------------------------
This module provides integration points between the TCCC.ai modules
and Jetson Orin Nano optimizations. It handles:

1. Integration of Jetson-optimized model loading with STT Engine
2. Integration of Jetson-optimized model loading with LLM Analysis
3. Dynamic resource allocation based on system state
4. Monitoring and reporting for system health
"""

import os
import sys
import yaml
import logging
from typing import Dict, Optional, Any, Union

# Import the Jetson optimizer
from tccc.utils.jetson_optimizer import (
    JetsonOptimizer,
    optimize_whisper_for_jetson,
    optimize_llm_for_jetson,
    optimize_embeddings_for_jetson,
    configure_audio_for_jetson,
    get_jetson_resource_monitor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JetsonIntegration")

# Path to Jetson optimizer config
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "config",
    "jetson_optimizer.yaml"
)


class JetsonIntegration:
    """
    Main class for integrating Jetson optimizations with TCCC modules
    """
    
    def __init__(self, config_path: Optional[str] = None, auto_setup: bool = False):
        """
        Initialize the Jetson integration with optional config.
        
        Args:
            config_path: Path to configuration file (default: config/jetson_optimizer.yaml)
            auto_setup: Automatically set up environment and start monitoring when True
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        
        # Initialize the Jetson optimizer
        self.optimizer = JetsonOptimizer(self.config_path, auto_setup=auto_setup)
        
        # Check if we're running on a Jetson platform
        self.is_jetson = self.optimizer.is_jetson
        if self.is_jetson:
            logger.info("Jetson platform detected, optimizations enabled")
        else:
            logger.warning("Not running on Jetson platform, optimizations disabled")
        
        # Initialize resource monitoring
        self.resource_monitor = None
    
    def _load_config(self) -> Dict:
        """Load the Jetson optimizer configuration"""
        # Default configuration
        default_config = {
            "power_mode": "balanced",
            "cuda_enabled": True,
            "tensorrt_enabled": True,
            "audio": {
                "sample_rate": 48000,
                "bit_depth": 24,
                "channels": 1,
                "buffer_size": 4096
            },
            "models": {
                "whisper": {
                    "compute_type": "float16",
                    "device": "cuda",
                    "model_size": "tiny.en",
                    "cpu_threads": 2,
                    "beam_size": 3
                },
                "llm": {
                    "quantization": "int8",
                    "device": "cuda",
                    "max_tokens": 512
                },
                "embeddings": {
                    "device": "cuda",
                    "half_precision": True
                }
            }
        }
        
        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        logger.info(f"Loaded configuration from {self.config_path}")
                        return user_config
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
        
        # If we get here, use default config
        logger.info("Using default Jetson optimization configuration")
        return default_config
    
    def setup_audio_devices(self) -> Dict:
        """
        Configure audio devices for the TCCC system
        
        Returns:
            Dict with audio device information
        """
        # Configure audio devices
        audio_devices = configure_audio_for_jetson()
        
        # Set up environment variables
        if audio_devices["razer_mini"]["detected"]:
            os.environ["TCCC_AUDIO_CARD"] = audio_devices["razer_mini"]["card_id"]
            os.environ["TCCC_AUDIO_DEVICE"] = audio_devices["razer_mini"]["device_id"]
            os.environ["TCCC_AUDIO_RATE"] = str(self.config.get("audio", {}).get("sample_rate", 48000))
            os.environ["TCCC_AUDIO_FORMAT"] = "S24_LE" if self.config.get("audio", {}).get("bit_depth", 24) == 24 else "S16_LE"
            os.environ["TCCC_AUDIO_CHANNELS"] = str(self.config.get("audio", {}).get("channels", 1))
            
            logger.info(f"Audio device configured: Razer Seiren V3 Mini (card {audio_devices['razer_mini']['card_id']})")
        else:
            logger.warning("Razer Seiren V3 Mini not detected")
        
        return audio_devices
    
    def optimize_faster_whisper(self, model_size: Optional[str] = None) -> Dict:
        """
        Get optimized settings for faster-whisper on Jetson
        
        Args:
            model_size: Optional model size override
            
        Returns:
            Dict of optimized parameters for faster-whisper
        """
        # Get the default model size from config
        if not model_size:
            model_size = self.config.get("models", {}).get("whisper", {}).get("model_size", "tiny.en")
        
        # Get optimized settings
        whisper_opts = optimize_whisper_for_jetson(model_size=model_size)
        
        if self.is_jetson:
            logger.info(f"Optimized whisper settings: {whisper_opts}")
        
        return whisper_opts
    
    def optimize_llm(self) -> Dict:
        """
        Get optimized settings for LLM on Jetson
        
        Returns:
            Dict of optimized parameters for LLM
        """
        llm_opts = optimize_llm_for_jetson()
        
        if self.is_jetson:
            logger.info(f"Optimized LLM settings: {llm_opts}")
        
        return llm_opts
    
    def optimize_embeddings(self) -> Dict:
        """
        Get optimized settings for embedding models on Jetson
        
        Returns:
            Dict of optimized parameters for embedding models
        """
        embedding_opts = optimize_embeddings_for_jetson()
        
        if self.is_jetson:
            logger.info(f"Optimized embedding settings: {embedding_opts}")
        
        return embedding_opts
    
    def start_resource_monitoring(self, interval: float = 5.0):
        """
        Start monitoring system resources
        
        Args:
            interval: Time in seconds between measurements
        """
        if self.resource_monitor:
            logger.warning("Resource monitoring already active")
            return
        
        self.resource_monitor = get_jetson_resource_monitor(interval)
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    def stop_resource_monitoring(self):
        """Stop resource monitoring"""
        if self.resource_monitor:
            self.resource_monitor.stop_resource_monitoring()
            self.resource_monitor = None
            logger.info("Stopped resource monitoring")
    
    def get_resource_stats(self) -> Dict:
        """
        Get current resource statistics
        
        Returns:
            Dict with current resource usage statistics
        """
        if not self.resource_monitor:
            logger.warning("Resource monitoring not active")
            return {}
        
        return self.resource_monitor.get_resource_stats()
    
    def apply_profile(self, profile_name: str) -> bool:
        """
        Apply a predefined performance profile
        
        Args:
            profile_name: Name of the profile to apply
            
        Returns:
            True if profile was applied successfully, False otherwise
        """
        if not self.is_jetson:
            logger.warning("Not running on Jetson, profile not applied")
            return False
        
        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            logger.error(f"Profile '{profile_name}' not found in configuration")
            return False
        
        profile = profiles[profile_name]
        logger.info(f"Applying profile: {profile_name} - {profile.get('description', '')}")
        
        # Apply power mode if specified
        if "power_mode" in profile:
            self.optimizer._set_power_mode(profile["power_mode"])
        
        # Apply other profile settings
        # These will be picked up by the modules on next load
        
        return True
    
    def suggest_optimal_settings(self) -> Dict:
        """
        Analyze system and suggest optimal settings
        
        Returns:
            Dict of suggested settings for different modules
        """
        return self.optimizer.suggest_optimal_settings()


# Module initialization function
def initialize_jetson_optimizations(auto_setup: bool = False) -> Optional[JetsonIntegration]:
    """
    Initialize Jetson optimizations for TCCC.ai
    
    Args:
        auto_setup: Automatically set up audio devices and start monitoring
    
    Returns:
        JetsonIntegration instance if successful, None otherwise
    """
    try:
        integration = JetsonIntegration(auto_setup=auto_setup)
        
        # Only set up devices and monitoring if auto_setup is True
        if integration.is_jetson and auto_setup:
            integration.setup_audio_devices()
            integration.start_resource_monitoring()
            
        return integration
    except Exception as e:
        logger.error(f"Failed to initialize Jetson optimizations: {e}")
        return None


# Helper functions
def get_whisper_params() -> Dict:
    """
    Get optimized Whisper parameters for current environment
    
    Returns:
        Dict of optimized parameters for faster-whisper
    """
    integration = JetsonIntegration()
    return integration.optimize_faster_whisper()


def get_llm_params() -> Dict:
    """
    Get optimized LLM parameters for current environment
    
    Returns:
        Dict of optimized parameters for LLM
    """
    integration = JetsonIntegration()
    return integration.optimize_llm()


if __name__ == "__main__":
    # When run directly, perform diagnostics and print suggested settings
    integration = JetsonIntegration()
    suggestions = integration.suggest_optimal_settings()
    
    print("\n" + "="*50)
    print("TCCC.ai Jetson Integration Diagnostic")
    print("="*50 + "\n")
    
    print(f"Running on Jetson: {suggestions['hardware_detected']['is_jetson']}")
    print(f"CUDA available: {suggestions['hardware_detected']['has_cuda']}")
    print(f"Available memory: {suggestions['hardware_detected']['available_memory_gb']} GB")
    
    print("\nDetected audio devices:")
    audio_devices = suggestions['hardware_detected']['audio_devices']
    if audio_devices['razer_mini']['detected']:
        print(f"✅ Razer Seiren V3 Mini detected (card {audio_devices['razer_mini']['card_id']})")
    else:
        print("❌ Razer Seiren V3 Mini not detected")
    
    print("\nRecommended settings:")
    print(f"STT Engine: {suggestions['whisper_settings']['suggested_model']} model with {suggestions['whisper_settings']['compute_type']} precision")
    print(f"LLM: {suggestions['llm_settings']['quantization']} quantization with max {suggestions['llm_settings']['max_tokens']} tokens")
    print(f"Power mode: {suggestions['general']['power_mode']}")
    
    print("\nAvailable profiles:")
    for profile_name, profile in integration.config.get("profiles", {}).items():
        print(f"- {profile_name}: {profile.get('description', '')}")
    
    print("\n" + "="*50)