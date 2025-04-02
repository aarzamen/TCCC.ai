#\!/usr/bin/env python3
"""
TCCC Audio Enhancement Configuration Selector
Generates an optimized configuration for the audio enhancement pipeline.
"""

import os
import sys
import json
import argparse
import yaml
import time
import platform
import subprocess
import shutil
from pathlib import Path

def detect_system_capabilities():
    """
    Detect system capabilities for optimal configuration.
    
    Returns:
        Dictionary of system capabilities
    """
    capabilities = {
        'has_gpu': False,
        'gpu_name': 'none',
        'gpu_memory_mb': 0,
        'cpu_cores': os.cpu_count() or 1,
        'cpu_model': 'unknown',
        'total_memory_mb': 0,
        'is_jetson': False,
        'cuda_available': False,
        'cuda_version': 'none',
        'torch_available': False,
        'has_microphone': False,
        'audio_devices': [],
        'fullsubnet_available': False,
        'battlefield_available': False,
    }
    
    # Get basic system info
    system = platform.system()
    capabilities['system'] = system
    
    # Get memory info
    try:
        if system == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        capabilities['total_memory_mb'] = int(line.split()[1]) // 1024
                        break
        elif system == 'Darwin':  # macOS
            output = subprocess.check_output(['sysctl', 'hw.memsize']).decode().strip()
            capabilities['total_memory_mb'] = int(output.split()[1]) // (1024 * 1024)
        elif system == 'Windows':
            output = subprocess.check_output(['wmic', 'computersystem', 'get', 'totalphysicalmemory']).decode().strip()
            capabilities['total_memory_mb'] = int(output.split('\n')[1]) // (1024 * 1024)
    except Exception:
        pass
    
    # Check if running on Jetson
    try:
        if os.path.exists('/etc/nv_tegra_release') or os.path.exists('/etc/tegra-release'):
            capabilities['is_jetson'] = True
            
            # Get Jetson model
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    capabilities['cpu_model'] = f.read().strip()
            except:
                pass
    except Exception:
        pass
    
    # Check for NVIDIA GPU
    try:
        # Try nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        if result.returncode == 0:
            output = result.stdout.decode().strip()
            if output:
                lines = output.split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        capabilities['has_gpu'] = True
                        capabilities['gpu_name'] = parts[0].strip()
                        memory_str = parts[1].strip()
                        try:
                            # Parse memory value and unit
                            memory_parts = memory_str.split()
                            if len(memory_parts) == 2:
                                value, unit = memory_parts
                                # Convert to MB
                                if unit.lower() == 'mib':
                                    capabilities['gpu_memory_mb'] = int(float(value))
                                elif unit.lower() == 'gib':
                                    capabilities['gpu_memory_mb'] = int(float(value) * 1024)
                        except Exception:
                            pass
                        break
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check CUDA availability with PyTorch
    try:
        import torch
        capabilities['torch_available'] = True
        capabilities['cuda_available'] = torch.cuda.is_available()
        if capabilities['cuda_available']:
            capabilities['cuda_version'] = torch.version.cuda or 'unknown'
            if not capabilities['has_gpu']:
                capabilities['has_gpu'] = True
                try:
                    capabilities['gpu_name'] = torch.cuda.get_device_name(0)
                    capabilities['gpu_memory_mb'] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                except Exception:
                    pass
    except ImportError:
        pass
    
    # Check audio devices
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        capabilities['has_microphone'] = False
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info.get('maxInputChannels', 0) > 0:
                capabilities['has_microphone'] = True
                capabilities['audio_devices'].append({
                    'index': i,
                    'name': dev_info.get('name', 'Unknown'),
                    'channels': dev_info.get('maxInputChannels', 0),
                    'sample_rate': dev_info.get('defaultSampleRate', 0)
                })
        p.terminate()
    except ImportError:
        pass
    
    # Check for FullSubNet
    fullsubnet_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'fullsubnet_integration',
        'fullsubnet'
    )
    capabilities['fullsubnet_available'] = os.path.exists(fullsubnet_path)
    
    # Check for Battlefield Enhancer
    battlefield_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'battlefield_audio_enhancer.py'
    )
    capabilities['battlefield_available'] = os.path.exists(battlefield_path)
    
    return capabilities

def generate_optimal_config(capabilities, output_file=None, mode=None):
    """
    Generate optimal configuration based on system capabilities.
    
    Args:
        capabilities: Dictionary of system capabilities
        output_file: Path to output file
        mode: Force specific enhancement mode
        
    Returns:
        Configuration dictionary
    """
    # Base configuration
    config = {
        'audio': {
            'sample_rate': 44100,
            'channels': 1,
            'chunk_size': 1024,
            'format': 'int16',
        },
        'enhancement': {
            'mode': 'auto',
            'target_level_db': -18,
            'noise_reduction_strength': 0.7,
        },
        'battlefield': {
            'enabled': capabilities['battlefield_available'],
            'outdoor_mode': True,
            'transient_protection': True,
            'distance_compensation': True,
            'voice_isolation': {
                'enabled': True,
                'strength': 0.8,
                'focus_width': 200,
                'voice_boost_db': 6
            }
        },
        'fullsubnet': {
            'enabled': capabilities['fullsubnet_available'] and capabilities['cuda_available'],
            'use_gpu': capabilities['cuda_available'],
            'model_path': os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'fullsubnet_integration',
                'models',
                'fullsubnet_best_model_58epochs.pth'
            ),
            'sample_rate': 16000,
            'batch_size': 1,
            'chunk_size': 1024,
            'n_fft': 512,
            'hop_length': 256,
            'win_length': 512,
            'normalized_input': True,
            'normalized_output': True,
            'gpu_acceleration': capabilities['cuda_available'],
            'fallback_to_cpu': True
        },
        'stt': {
            'engine': 'faster-whisper',
            'model_size': 'tiny',  # Adjust based on system
            'compute_type': 'int8',
            'vad_filter': False,
            'language': 'en',
        },
        'system': {
            'output_directory': os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'enhanced_audio_output'
            ),
            'recording_duration': 15,
            'show_status': True,
        }
    }
    
    # Adjust based on GPU capabilities
    if capabilities['has_gpu']:
        if capabilities['gpu_memory_mb'] >= 4000:
            config['stt']['model_size'] = 'small'
            config['stt']['compute_type'] = 'float16'
            config['fullsubnet']['batch_size'] = 2
        elif capabilities['gpu_memory_mb'] >= 2000:
            config['stt']['model_size'] = 'tiny'
            config['stt']['compute_type'] = 'int8'
        
        # Larger chunk size for more powerful GPUs
        if capabilities['gpu_memory_mb'] >= 6000:
            config['audio']['chunk_size'] = 2048
            config['fullsubnet']['chunk_size'] = 2048
    
    # Adjust based on CPU capabilities
    if capabilities['cpu_cores'] <= 2:
        # Low-end CPU, use minimal settings
        config['audio']['sample_rate'] = 16000
        config['audio']['chunk_size'] = 512
        config['stt']['model_size'] = 'tiny'
    
    # Adjust based on Jetson
    if capabilities['is_jetson']:
        # Jetson-specific optimizations
        config['audio']['sample_rate'] = 16000  # Lower sample rate to reduce CPU load
        config['fullsubnet']['gpu_acceleration'] = True
    
    # Force specific mode if requested
    if mode and mode in ['auto', 'fullsubnet', 'battlefield', 'both', 'none']:
        config['enhancement']['mode'] = mode
    elif not config['fullsubnet']['enabled'] and config['battlefield']['enabled']:
        # If FullSubNet not available but Battlefield is, use Battlefield
        config['enhancement']['mode'] = 'battlefield'
    elif config['fullsubnet']['enabled'] and not config['battlefield']['enabled']:
        # If FullSubNet available but Battlefield is not, use FullSubNet
        config['enhancement']['mode'] = 'fullsubnet'
    elif config['fullsubnet']['enabled'] and config['battlefield']['enabled']:
        # If both available, use the one that makes most sense for the hardware
        if capabilities['cuda_available'] and capabilities['gpu_memory_mb'] >= 2000:
            config['enhancement']['mode'] = 'fullsubnet'  # Prefer GPU-accelerated method
        else:
            config['enhancement']['mode'] = 'battlefield'  # Prefer CPU-friendly method
    else:
        # If neither available, use none
        config['enhancement']['mode'] = 'none'
    
    # Save configuration if output file specified
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save configuration
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    return config

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Audio Enhancement Configuration Selector")
    parser.add_argument("--output", "-o", type=str, default="config/audio_enhancement.yaml",
                      help="Output file path")
    parser.add_argument("--mode", "-m", type=str, choices=['auto', 'fullsubnet', 'battlefield', 'both', 'none'],
                      help="Force specific enhancement mode")
    parser.add_argument("--print", "-p", action="store_true",
                      help="Print configuration to stdout")
    args = parser.parse_args()
    
    # Detect system capabilities
    print("Detecting system capabilities...")
    capabilities = detect_system_capabilities()
    
    # Generate optimal configuration
    print("Generating optimal configuration...")
    config = generate_optimal_config(capabilities, args.output, args.mode)
    
    # Print configuration
    if args.print:
        print("\nGenerated Configuration:")
        print(yaml.dump(config, default_flow_style=False))
    
    # Print summary
    print("\nConfiguration Summary:")
    print(f"- Enhancement mode: {config['enhancement']['mode']}")
    print(f"- Sample rate: {config['audio']['sample_rate']} Hz")
    print(f"- STT engine: {config['stt']['engine']} ({config['stt']['model_size']})")
    
    if config['enhancement']['mode'] in ['fullsubnet', 'both', 'auto'] and capabilities['fullsubnet_available']:
        print("\nFullSubNet Configuration:")
        print(f"- GPU enabled: {config['fullsubnet']['use_gpu']}")
        print(f"- GPU name: {capabilities['gpu_name']}")
        print(f"- GPU memory: {capabilities['gpu_memory_mb']} MB")
    
    if config['enhancement']['mode'] in ['battlefield', 'both', 'auto'] and capabilities['battlefield_available']:
        print("\nBattlefield Enhancer Configuration:")
        print(f"- Outdoor mode: {config['battlefield']['outdoor_mode']}")
        print(f"- Distance compensation: {config['battlefield']['distance_compensation']}")
    
    print(f"\nConfiguration saved to: {args.output}")
    print("To use this configuration:")
    print(f"  ./run_enhanced_audio.sh --config {args.output}")

if __name__ == "__main__":
    main()
