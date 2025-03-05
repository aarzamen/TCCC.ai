"""
Configuration management for TCCC.ai system.

This module provides utilities for loading and validating configuration from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, IO


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class Config:
    """
    Configuration management class.
    
    Handles loading configuration from YAML files and providing access to
    configuration values.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files.
                If None, defaults to the 'config' directory in the project root.
        """
        if config_dir is None:
            # Assume config directory is at project root
            self.config_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    'config'
                )
            )
        else:
            self.config_dir = os.path.abspath(config_dir)
            
        if not os.path.isdir(self.config_dir):
            raise ConfigError(f"Configuration directory not found: {self.config_dir}")
            
        self.config = {}
    
    def load(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            filename: Name of the YAML file in the config directory.
                
        Returns:
            The loaded configuration as a dictionary.
            
        Raises:
            ConfigError: If the file cannot be loaded or parsed.
        """
        filepath = os.path.join(self.config_dir, filename)
        
        if not os.path.isfile(filepath):
            raise ConfigError(f"Configuration file not found: {filepath}")
            
        try:
            with open(filepath, 'r') as file:
                config = yaml.safe_load(file)
                
            if not isinstance(config, dict):
                raise ConfigError(f"Invalid configuration format in {filepath}. Expected a dictionary.")
                
            self.config.update(config)
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML in {filepath}: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration from {filepath}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key.
            default: Default value to return if the key is not found.
                
        Returns:
            The configuration value, or the default if not found.
        """
        return self.config.get(key, default)
    
    def require(self, key: str) -> Any:
        """
        Get a required configuration value.
        
        Args:
            key: The configuration key.
                
        Returns:
            The configuration value.
            
        Raises:
            ConfigError: If the key is not found in the configuration.
        """
        if key not in self.config:
            raise ConfigError(f"Required configuration key not found: {key}")
            
        return self.config[key]


def load_config(config_files: list, config_dir: Optional[str] = None) -> Config:
    """
    Load configuration from multiple files.
    
    Args:
        config_files: List of configuration files to load.
        config_dir: Directory containing configuration files.
            
    Returns:
        A Config instance with all configuration loaded.
    """
    config = Config(config_dir)
    
    for filename in config_files:
        config.load(filename)
        
    return config


@classmethod
def load_yaml(cls, file_path: Union[str, IO]) -> Dict[str, Any]:
    """
    Load YAML configuration from a file path or file object.
    
    Args:
        file_path: Path to the YAML file or a file-like object
            
    Returns:
        The loaded configuration as a dictionary
        
    Raises:
        ConfigError: If the file cannot be loaded or parsed
    """
    try:
        if isinstance(file_path, str):
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = yaml.safe_load(file_path)
            
        if not isinstance(config, dict):
            raise ConfigError(f"Invalid configuration format. Expected a dictionary.")
            
        return config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML: {str(e)}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration: {str(e)}")