"""
Plugin manager module for TCCC.ai processing core.

This module provides plugin management functionality for the Processing Core.
"""

import os
import importlib.util
import inspect
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable, Set, Tuple
import threading

from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class ProcessingPlugin(ABC):
    """
    Base class for processing plugins that can be registered with the Processing Core.
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Configuration for the plugin.
            
        Returns:
            True if initialization was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using the plugin.
        
        Args:
            data: The data to process.
            
        Returns:
            The processed data.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the plugin.
        
        Returns:
            A dictionary with plugin metadata.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the plugin.
        
        Returns:
            The plugin name.
        """
        pass


class PluginManager:
    """
    Manages plugins for the Processing Core.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the plugin manager.
        
        Args:
            config: Configuration for the plugin manager.
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.plugin_dirs = config.get("directories", [])
        self.default_plugins = config.get("default_plugins", [])
        self.isolation = config.get("isolation", "thread")
        
        # Add the built-in plugins directory
        builtin_dir = os.path.join(os.path.dirname(__file__), "plugins")
        if os.path.exists(builtin_dir) and builtin_dir not in self.plugin_dirs:
            self.plugin_dirs.append(builtin_dir)
        
        # Maps plugin name to plugin class
        self.available_plugins: Dict[str, Type[ProcessingPlugin]] = {}
        
        # Maps plugin name to plugin instance
        self.active_plugins: Dict[str, ProcessingPlugin] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Load available plugins
        self._discover_plugins()
        
        # Initialize default plugins
        if self.enabled and self.default_plugins:
            for plugin_name in self.default_plugins:
                self.register_plugin(plugin_name)
    
    def _discover_plugins(self):
        """
        Discover available plugins in the plugin directories.
        """
        if not self.enabled:
            logger.info("Plugin system is disabled")
            return
        
        with self.lock:
            discovered = 0
            
            for plugin_dir in self.plugin_dirs:
                if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
                    logger.warning(f"Plugin directory not found: {plugin_dir}")
                    continue
                
                logger.debug(f"Searching for plugins in {plugin_dir}")
                
                # Walk through all Python files in the directory
                for root, _, files in os.walk(plugin_dir):
                    for file in files:
                        if not file.endswith(".py") or file.startswith("_"):
                            continue
                        
                        plugin_path = os.path.join(root, file)
                        try:
                            # Load the module
                            module_name = os.path.splitext(file)[0]
                            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
                            if spec is None or spec.loader is None:
                                continue
                            
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Find plugin classes
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and 
                                    issubclass(obj, ProcessingPlugin) and 
                                    obj != ProcessingPlugin):
                                    # Create plugin instance
                                    plugin_name = getattr(obj, "name", name.lower())
                                    self.available_plugins[plugin_name] = obj
                                    discovered += 1
                                    logger.debug(f"Discovered plugin: {plugin_name}")
                        
                        except Exception as e:
                            logger.error(f"Error loading plugin from {plugin_path}: {str(e)}")
            
            logger.info(f"Discovered {discovered} plugins in {len(self.plugin_dirs)} directories")
    
    def register_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """
        Register and initialize a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to register.
            config: Configuration for the plugin.
            
        Returns:
            True if the plugin was registered successfully, False otherwise.
        """
        if not self.enabled:
            logger.warning("Cannot register plugin: Plugin system is disabled")
            return False
        
        with self.lock:
            # Check if already registered
            if plugin_name in self.active_plugins:
                logger.info(f"Plugin {plugin_name} is already registered")
                return True
            
            # Check if plugin exists
            if plugin_name not in self.available_plugins:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            # Create plugin instance
            try:
                plugin_class = self.available_plugins[plugin_name]
                plugin = plugin_class()
                
                # Initialize plugin
                if not plugin.initialize(config):
                    logger.error(f"Failed to initialize plugin {plugin_name}")
                    return False
                
                # Register plugin
                self.active_plugins[plugin_name] = plugin
                logger.info(f"Registered plugin: {plugin_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error registering plugin {plugin_name}: {str(e)}")
                return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to unregister.
            
        Returns:
            True if the plugin was unregistered successfully, False otherwise.
        """
        if not self.enabled:
            return False
        
        with self.lock:
            if plugin_name not in self.active_plugins:
                logger.warning(f"Plugin {plugin_name} is not registered")
                return False
            
            # Remove plugin
            del self.active_plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
    
    def get_plugin(self, plugin_name: str) -> Optional[ProcessingPlugin]:
        """
        Get a registered plugin by name.
        
        Args:
            plugin_name: Name of the plugin.
            
        Returns:
            The plugin instance, or None if not found.
        """
        if not self.enabled:
            return None
        
        with self.lock:
            return self.active_plugins.get(plugin_name)
    
    def get_active_plugins(self) -> List[str]:
        """
        Get the names of all active plugins.
        
        Returns:
            A list of active plugin names.
        """
        if not self.enabled:
            return []
        
        with self.lock:
            return list(self.active_plugins.keys())
    
    def get_available_plugins(self) -> List[str]:
        """
        Get the names of all available plugins.
        
        Returns:
            A list of available plugin names.
        """
        if not self.enabled:
            return []
        
        with self.lock:
            return list(self.available_plugins.keys())
    
    def process_data(self, data: Dict[str, Any], plugin_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process data through plugins.
        
        Args:
            data: The data to process.
            plugin_names: Names of plugins to use. If None, all active plugins are used.
            
        Returns:
            The processed data.
        """
        if not self.enabled or not self.active_plugins:
            return data
        
        result = data.copy()
        
        with self.lock:
            plugins_to_use = []
            
            if plugin_names:
                # Use specified plugins
                for name in plugin_names:
                    if name in self.active_plugins:
                        plugins_to_use.append(self.active_plugins[name])
                    else:
                        logger.warning(f"Plugin {name} is not active, skipping")
            else:
                # Use all active plugins
                plugins_to_use = list(self.active_plugins.values())
            
            # Process data through each plugin
            for plugin in plugins_to_use:
                try:
                    if self.isolation == "thread":
                        # Process in current thread
                        result = plugin.process(result)
                    elif self.isolation == "process":
                        # TODO: Implement process isolation if needed
                        # This would require using multiprocessing to run plugins in separate processes
                        logger.warning("Process isolation not implemented, using thread isolation")
                        result = plugin.process(result)
                    else:
                        # No isolation
                        result = plugin.process(result)
                        
                except Exception as e:
                    logger.error(f"Error in plugin {plugin.name}: {str(e)}")
        
        return result
    
    def reload_plugins(self):
        """
        Reload all plugins.
        """
        if not self.enabled:
            return
        
        with self.lock:
            # Remember active plugins
            active_names = list(self.active_plugins.keys())
            
            # Clear plugins
            self.active_plugins.clear()
            self.available_plugins.clear()
            
            # Discover plugins again
            self._discover_plugins()
            
            # Re-register active plugins
            for name in active_names:
                if name in self.available_plugins:
                    self.register_plugin(name)
                else:
                    logger.warning(f"Plugin {name} is no longer available")
            
            logger.info(f"Reloaded plugins: {len(self.active_plugins)} active, {len(self.available_plugins)} available")
    
    def get_plugin_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all active plugins.
        
        Returns:
            A dictionary mapping plugin names to metadata.
        """
        if not self.enabled:
            return {}
        
        with self.lock:
            metadata = {}
            for name, plugin in self.active_plugins.items():
                try:
                    metadata[name] = plugin.get_metadata()
                except Exception as e:
                    logger.error(f"Error getting metadata for plugin {name}: {str(e)}")
                    metadata[name] = {"error": str(e)}
            
            return metadata