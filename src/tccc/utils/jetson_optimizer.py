#!/usr/bin/env python3
"""
Jetson Orin Nano Optimization Module for TCCC.ai
------------------------------------------------
This module provides hardware-specific optimizations for running TCCC.ai on Jetson Orin Nano hardware.
It handles:
1. Model optimization for CUDA and TensorRT
2. Audio device configuration for Razer Seiren V3 Mini
3. Resource monitoring and allocation
4. Power management profiles
"""

import os
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JetsonOptimizer")

# Hardware constants for Jetson Orin Nano
JETSON_SPECS = {
    "max_memory_gb": 8,
    "compute_capability": "8.7",  # Ampere architecture
    "cpu_cores": 6,
    "gpu_tflops": 0.5,  # Rough estimate for AI workloads
    "tdp_watts": 7.0,   # Thermal Design Power
}

# Audio device identifiers
RAZER_MINI_CARD_NAME = "Razer Seiren V3 Mini"
LOGITECH_HEADSET_CARD_NAME = "Logitech USB Headset"

class JetsonOptimizer:
    """Main class for Jetson hardware optimizations"""
    
    def __init__(self, config_path: Optional[str] = None, auto_setup: bool = False):
        """
        Initialize the Jetson optimizer with optional config.
        
        Args:
            config_path: Path to configuration file (optional)
            auto_setup: Automatically set up the environment when True
        """
        self.config = self._load_config(config_path)
        self.is_jetson = self._detect_jetson()
        
        # Initialize monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.resource_stats = {
            "cpu_usage": [],
            "gpu_usage": [],
            "memory_usage": [],
            "temperature": [],
            "power_usage": []
        }
        
        if self.is_jetson:
            logger.info("Jetson Orin Nano platform detected")
            if auto_setup:
                self._setup_jetson_environment()
        else:
            logger.warning("Not running on Jetson platform. Some optimizations will be disabled.")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "power_mode": "balanced",  # Options: "max_performance", "balanced", "power_saver"
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
                    "cpu_threads": 2
                },
                "llm": {
                    "quantization": "int8",
                    "device": "cuda"
                },
                "embeddings": {
                    "device": "cuda"
                }
            },
            "memory_limits": {
                "whisper": 1.0,  # GB
                "llm": 4.0,      # GB
                "embeddings": 0.5 # GB
            }
        }
        
        if not config_path:
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge user config with defaults
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _detect_jetson(self) -> bool:
        """Detect if running on Jetson hardware"""
        try:
            # Check for Jetson-specific CPU info
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                if 'tegra' in cpu_info.lower() or 'nvidia' in cpu_info.lower():
                    return True
            
            # Try nvidia-smi with Jetson-specific args
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            return "Orin" in result.stdout
        except:
            return False
    
    def _setup_jetson_environment(self) -> None:
        """Set up Jetson-specific environment variables and settings"""
        # Set environment variables for better performance
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # TensorRT optimizations
        if self.config["tensorrt_enabled"]:
            os.environ["TF_TRT_ALLOW_GPU_FALLBACK"] = "1"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
        
        # Set thread priority for audio processing
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Apply power mode
        self._set_power_mode(self.config["power_mode"])
    
    def _set_power_mode(self, mode: str) -> None:
        """Set Jetson power mode"""
        if not self.is_jetson:
            return
        
        logger.info(f"Setting Jetson power mode to: {mode}")
        
        try:
            # Check if we have sudo access
            have_sudo = False
            try:
                result = subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False)
                have_sudo = result.returncode == 0
            except:
                have_sudo = False
            
            if not have_sudo:
                logger.warning("No sudo access. Power mode changes require elevated permissions.")
                logger.info("Skipping power mode change due to permissions.")
                return
                
            if mode == "max_performance":
                # Maximum performance mode
                subprocess.run(["sudo", "nvpmodel", "-m", "0"], check=False)
                subprocess.run(["sudo", "jetson_clocks"], check=False)
            elif mode == "balanced":
                # Balanced mode
                subprocess.run(["sudo", "nvpmodel", "-m", "1"], check=False)
            elif mode == "power_saver":
                # Power saver mode
                subprocess.run(["sudo", "nvpmodel", "-m", "2"], check=False)
        except Exception as e:
            logger.error(f"Failed to set power mode: {e}")
            logger.info("If running in emulation or non-Jetson hardware, this is expected.")
    
    def configure_audio_devices(self) -> Dict:
        """Configure audio devices for optimal performance"""
        audio_config = {
            "razer_mini": {
                "detected": False,
                "card_id": None,
                "device_id": None
            },
            "logitech_headset": {
                "detected": False,
                "card_id": None,
                "device_id": None
            }
        }
        
        try:
            # Run arecord -l to get audio devices
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=False)
            lines = result.stdout.split('\n')
            
            # Parse the output to find our devices
            for line in lines:
                if not line.startswith('card '):
                    continue
                    
                try:
                    parts = line.split(':')
                    if len(parts) < 2:
                        continue
                        
                    card_info = parts[0].strip()
                    card_name = parts[1].strip()
                    
                    # Extract card and device IDs more safely
                    card_id_match = card_info.split('card ')
                    if len(card_id_match) < 2:
                        continue
                    
                    card_id_parts = card_id_match[1].split(',')
                    if not card_id_parts:
                        continue
                        
                    card_id = card_id_parts[0].split()[0]
                    
                    # Try to extract device ID
                    device_id = "0"  # Default device ID
                    if "device" in card_info:
                        device_parts = card_info.split('device ')
                        if len(device_parts) >= 2:
                            device_id_parts = device_parts[1].split(',')
                            if device_id_parts:
                                device_id = device_id_parts[0].split()[0].rstrip(':')
                    
                    # Check for our target devices
                    if RAZER_MINI_CARD_NAME in card_name:
                        audio_config["razer_mini"] = {
                            "detected": True,
                            "card_id": card_id,
                            "device_id": device_id
                        }
                    
                    if LOGITECH_HEADSET_CARD_NAME in card_name:
                        audio_config["logitech_headset"] = {
                            "detected": True,
                            "card_id": card_id,
                            "device_id": device_id
                        }
                except Exception as parsing_error:
                    logger.warning(f"Error parsing audio device line '{line}': {parsing_error}")
                    continue
            
            # Configure detected devices
            if audio_config["razer_mini"]["detected"]:
                logger.info(f"Detected Razer Seiren V3 Mini: card {audio_config['razer_mini']['card_id']}")
                # Create optimal ALSA config for the microphone if wanted
                if self.config.get("auto_configure_audio", False):
                    self._configure_alsa_for_device(
                        audio_config["razer_mini"]["card_id"],
                        audio_config["razer_mini"]["device_id"],
                        self.config.get("audio", {}).get("sample_rate", 48000),
                        self.config.get("audio", {}).get("bit_depth", 24),
                        self.config.get("audio", {}).get("channels", 1)
                    )
            
            if audio_config["logitech_headset"]["detected"]:
                logger.info(f"Detected Logitech USB Headset: card {audio_config['logitech_headset']['card_id']}")
            
            # If no target devices were found, check for any audio devices
            if not any(device["detected"] for device in audio_config.values()):
                logger.info("No target audio devices found, checking for any available audio devices")
                
                # Look for any audio input device
                for line in lines:
                    if line.startswith('card '):
                        logger.info(f"Found generic audio device: {line}")
                        break
        
        except Exception as e:
            logger.error(f"Failed to configure audio devices: {e}")
            logger.debug(f"Audio device detection exception details:", exc_info=True)
        
        return audio_config
    
    def _configure_alsa_for_device(self, card_id, device_id, sample_rate, bit_depth, channels):
        """Configure ALSA settings for a specific audio device"""
        try:
            # Set default audio device
            home_dir = os.path.expanduser("~")
            asound_path = os.path.join(home_dir, ".asoundrc")
            
            asound_config = f"""
pcm.!default {{
    type hw
    card {card_id}
    device {device_id}
}}

ctl.!default {{
    type hw
    card {card_id}
}}

pcm.tccc_mic {{
    type hw
    card {card_id}
    device {device_id}
    rate {sample_rate}
    format {'S24_LE' if bit_depth == 24 else 'S16_LE'}
    channels {channels}
}}
"""
            # Only write if we have permission
            try:
                with open(asound_path, 'w') as f:
                    f.write(asound_config)
                logger.info(f"Created ALSA configuration at {asound_path}")
            except PermissionError:
                logger.warning(f"Could not write to {asound_path} due to permissions")
                # Create temp file instead
                temp_path = "tccc_asound.conf"
                with open(temp_path, 'w') as f:
                    f.write(asound_config)
                logger.info(f"Created temporary ALSA configuration at {temp_path}")
        
        except Exception as e:
            logger.error(f"Failed to configure ALSA: {e}")
    
    def optimize_model_loading(self, model_type: str) -> Dict:
        """
        Optimize the loading of models based on Jetson capability
        
        Args:
            model_type: One of "whisper", "llm", "embeddings"
            
        Returns:
            Dict of parameter overrides to use when loading the model
        """
        if not self.is_jetson:
            return {}  # No optimization on non-Jetson platforms
        
        # Get model-specific optimizations
        model_config = self.config["models"].get(model_type, {})
        
        # Default optimizations
        optimizations = {
            "device": model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        }
        
        # Model-specific optimizations
        if model_type == "whisper":
            optimizations.update({
                "compute_type": model_config.get("compute_type", "float16"),
                "num_workers": model_config.get("cpu_threads", 2),
                "beam_size": 3,  # Reduced beam size for faster inference
                "vad_filter": True,  # Use VAD to skip silence
                "vad_parameters": {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 100
                }
            })
        
        elif model_type == "llm":
            optimizations.update({
                "load_in_8bit": model_config.get("quantization") == "int8",
                "load_in_4bit": model_config.get("quantization") == "int4",
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "use_cache": True,
                "torch_dtype": torch.float16
            })
        
        elif model_type == "embeddings":
            optimizations.update({
                "half_precision": True
            })
        
        return optimizations
    
    def start_resource_monitoring(self, interval: float = 5.0) -> None:
        """
        Start monitoring system resources
        
        Args:
            interval: Time in seconds between measurements
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Resource monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._resource_monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    def stop_resource_monitoring(self) -> None:
        """Stop resource monitoring thread"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
    
    def _resource_monitoring_loop(self, interval: float) -> None:
        """Resource monitoring background thread"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.resource_stats["cpu_usage"].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.resource_stats["memory_usage"].append(memory_percent)
                
                # GPU usage (Jetson-specific)
                if self.is_jetson:
                    try:
                        # This is a simplified approach - full implementation would parse tegrastats output
                        gpu_usage = self._get_gpu_usage()
                        self.resource_stats["gpu_usage"].append(gpu_usage)
                    except:
                        self.resource_stats["gpu_usage"].append(0)
                
                # Cap history length
                max_history = 100
                for key in self.resource_stats:
                    if len(self.resource_stats[key]) > max_history:
                        self.resource_stats[key] = self.resource_stats[key][-max_history:]
                
                # Log if resources are critically high
                if cpu_percent > 90 or memory_percent > 90:
                    logger.warning(f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%")
            
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
            
            time.sleep(interval)
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage on Jetson"""
        if not self.is_jetson:
            return 0.0
        
        try:
            # This could use tegrastats or other Jetson-specific tools
            # Simplified implementation - real version would parse tegrastats output
            return 0.0
        except:
            return 0.0
    
    def get_resource_stats(self) -> Dict:
        """
        Get current resource statistics
        
        Returns:
            Dict with current resource usage statistics
        """
        return {
            "cpu_percent": self.resource_stats["cpu_usage"][-1] if self.resource_stats["cpu_usage"] else 0,
            "memory_percent": self.resource_stats["memory_usage"][-1] if self.resource_stats["memory_usage"] else 0,
            "gpu_percent": self.resource_stats["gpu_usage"][-1] if self.resource_stats["gpu_usage"] else 0,
            "averages": {
                "cpu": np.mean(self.resource_stats["cpu_usage"]) if self.resource_stats["cpu_usage"] else 0,
                "memory": np.mean(self.resource_stats["memory_usage"]) if self.resource_stats["memory_usage"] else 0,
                "gpu": np.mean(self.resource_stats["gpu_usage"]) if self.resource_stats["gpu_usage"] else 0,
            }
        }
    
    def suggest_optimal_settings(self) -> Dict:
        """
        Analyze system and suggest optimal settings
        
        Returns:
            Dict of suggested settings for different modules
        """
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        has_cuda = torch.cuda.is_available()
        
        suggestions = {
            "hardware_detected": {
                "is_jetson": self.is_jetson,
                "has_cuda": has_cuda,
                "available_memory_gb": round(available_memory, 2),
                "audio_devices": self.configure_audio_devices()
            },
            "whisper_settings": {},
            "llm_settings": {},
            "general": {}
        }
        
        # Whisper suggestions
        compute_type = "float16" if has_cuda else "int8"
        if available_memory < 2.0:
            compute_type = "int8"
        
        suggestions["whisper_settings"] = {
            "compute_type": compute_type,
            "device": "cuda" if has_cuda else "cpu",
            "beam_size": 1 if available_memory < 1.0 else 3,
            "suggested_model": "tiny.en" if available_memory < 2.0 else "base.en"
        }
        
        # LLM suggestions
        quantization = "int8"
        if available_memory < 4.0:
            quantization = "int4"
        
        suggestions["llm_settings"] = {
            "quantization": quantization,
            "device": "cuda" if has_cuda else "cpu",
            "max_tokens": 512 if available_memory < 3.0 else 1024
        }
        
        # General suggestions
        suggestions["general"] = {
            "batch_size": 1 if available_memory < 2.0 else 2,
            "power_mode": "balanced" if has_cuda else "power_saver",
            "parallel_processes": min(4, psutil.cpu_count(logical=False) or 2)
        }
        
        return suggestions


# Module helpers

def optimize_whisper_for_jetson(model_size="tiny.en", compute_type="float16") -> Dict:
    """Get optimal Whisper model settings for Jetson"""
    optimizer = JetsonOptimizer()
    base_options = optimizer.optimize_model_loading("whisper")
    
    # Override with specific parameters if provided
    if model_size:
        base_options["model_size"] = model_size
    if compute_type:
        base_options["compute_type"] = compute_type
        
    return base_options


def optimize_llm_for_jetson() -> Dict:
    """Get optimal LLM settings for Jetson"""
    optimizer = JetsonOptimizer()
    return optimizer.optimize_model_loading("llm")


def optimize_embeddings_for_jetson() -> Dict:
    """Get optimal embedding model settings for Jetson"""
    optimizer = JetsonOptimizer()
    return optimizer.optimize_model_loading("embeddings")


def configure_audio_for_jetson() -> Dict:
    """Configure audio devices for Jetson"""
    optimizer = JetsonOptimizer()
    return optimizer.configure_audio_devices()


def get_jetson_resource_monitor(interval=5.0):
    """Get a resource monitor for Jetson"""
    monitor = JetsonOptimizer()
    monitor.start_resource_monitoring(interval)
    return monitor


if __name__ == "__main__":
    # When run directly, perform a diagnostic and suggest optimal settings
    optimizer = JetsonOptimizer()
    suggestions = optimizer.suggest_optimal_settings()
    
    print("\n" + "="*50)
    print("TCCC.ai Jetson Optimization Diagnostic")
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
        
    if audio_devices['logitech_headset']['detected']:
        print(f"✅ Logitech USB Headset detected (card {audio_devices['logitech_headset']['card_id']})")
    else:
        print("❌ Logitech USB Headset not detected")
    
    print("\nRecommended settings:")
    print(f"STT Engine: {suggestions['whisper_settings']['suggested_model']} model with {suggestions['whisper_settings']['compute_type']} precision")
    print(f"LLM: {suggestions['llm_settings']['quantization']} quantization with max {suggestions['llm_settings']['max_tokens']} tokens")
    print(f"Power mode: {suggestions['general']['power_mode']}")
    print(f"Batch size: {suggestions['general']['batch_size']}")
    
    print("\n" + "="*50)