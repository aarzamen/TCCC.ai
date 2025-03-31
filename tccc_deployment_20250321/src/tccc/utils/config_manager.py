"""
Configuration Manager utility for TCCC.ai system.

This module provides a backward-compatible ConfigManager class that works
with both the old and new configuration systems.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, IO

from tccc.utils.config import Config, load_config


class ConfigManager:
    """
    Configuration Manager class (compatibility wrapper)
    
    This class provides backward compatibility with code expecting the 
    ConfigManager interface while using the newer Config implementation.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files.
                If None, defaults to the 'config' directory in the project root.
        """
        self.config = Config(config_dir)
        self.configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_name: Name of the configuration file (without extension).
                
        Returns:
            The loaded configuration as a dictionary.
        """
        filename = f"{config_name}.yaml"
        config_data = self.config.load(filename)
        self.configs[config_name] = config_data
        return config_data
    
    def get_config(self, config_name: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get a configuration by name.
        
        Args:
            config_name: Name of the configuration.
            default: Default value to return if configuration not found.
                
        Returns:
            The configuration dictionary or default if not found.
        """
        if config_name not in self.configs:
            try:
                return self.load_config(config_name)
            except Exception:
                return default if default is not None else {}
        return self.configs[config_name]
    
    @staticmethod
    def load_yaml(file_path: Union[str, IO]) -> Dict[str, Any]:
        """
        Load YAML configuration from a file path or file object.
        
        Args:
            file_path: Path to the YAML file or a file-like object
                
        Returns:
            The loaded configuration as a dictionary
        """
        try:
            if isinstance(file_path, str):
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = yaml.safe_load(file_path)
                
            if not isinstance(config, dict):
                raise ValueError(f"Invalid configuration format. Expected a dictionary.")
                
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")