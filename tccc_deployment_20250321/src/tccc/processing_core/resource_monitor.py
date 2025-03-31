"""
Resource monitoring module for TCCC.ai processing core.

This module provides resource monitoring functionality for the Jetson Orin Nano platform.
"""

import os
import time
import threading
from typing import Dict, List, Optional, Any, Callable
import psutil
try:
    # Jetson-specific modules
    import jtop
    import nvgpu
    JETSON_AVAILABLE = True
except ImportError:
    JETSON_AVAILABLE = False

from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class ResourceUsage:
    """
    Represents resource usage information.
    """
    
    def __init__(self,
                 cpu_usage: float,
                 memory_usage: float,
                 gpu_usage: Optional[float] = None,
                 gpu_memory_usage: Optional[float] = None,
                 temperature: Optional[Dict[str, float]] = None,
                 power_usage: Optional[Dict[str, float]] = None,
                 timestamp: Optional[float] = None):
        """
        Initialize resource usage data.
        
        Args:
            cpu_usage: CPU usage as a percentage.
            memory_usage: Memory usage as a percentage.
            gpu_usage: GPU usage as a percentage (if available).
            gpu_memory_usage: GPU memory usage as a percentage (if available).
            temperature: Temperature readings in Celsius.
            power_usage: Power usage in watts.
            timestamp: Timestamp when the data was collected.
        """
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.gpu_usage = gpu_usage
        self.gpu_memory_usage = gpu_memory_usage
        self.temperature = temperature or {}
        self.power_usage = power_usage or {}
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the resource usage to a dictionary.
        
        Returns:
            A dictionary representation of the resource usage.
        """
        result = {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "timestamp": self.timestamp
        }
        
        if self.gpu_usage is not None:
            result["gpu_usage"] = self.gpu_usage
        
        if self.gpu_memory_usage is not None:
            result["gpu_memory_usage"] = self.gpu_memory_usage
        
        if self.temperature:
            result["temperature"] = self.temperature
        
        if self.power_usage:
            result["power_usage"] = self.power_usage
        
        return result
    
    def __repr__(self) -> str:
        result = f"ResourceUsage(cpu={self.cpu_usage:.1f}%, memory={self.memory_usage:.1f}%"
        
        if self.gpu_usage is not None:
            result += f", gpu={self.gpu_usage:.1f}%"
        
        if self.gpu_memory_usage is not None:
            result += f", gpu_memory={self.gpu_memory_usage:.1f}%"
        
        if self.temperature:
            temp_str = ", ".join([f"{k}={v:.1f}°C" for k, v in self.temperature.items()])
            result += f", temp={{{temp_str}}}"
        
        result += ")"
        return result


class JetsonResourceMonitor:
    """
    Resource monitor for Jetson platforms using jtop.
    """
    
    def __init__(self):
        """
        Initialize the Jetson resource monitor.
        """
        if not JETSON_AVAILABLE:
            raise ImportError("jtop and nvgpu modules required for Jetson monitoring")
        
        self.jtop_instance = None
        self.is_running = False
        logger.info("Initialized Jetson resource monitor")
    
    def start(self):
        """
        Start the Jetson monitoring.
        """
        if self.is_running:
            return
        
        try:
            self.jtop_instance = jtop.jtop()
            self.jtop_instance.start()
            self.is_running = True
            logger.info("Started Jetson resource monitoring")
        except Exception as e:
            logger.error(f"Failed to start Jetson monitoring: {str(e)}")
            self.jtop_instance = None
    
    def stop(self):
        """
        Stop the Jetson monitoring.
        """
        if not self.is_running or self.jtop_instance is None:
            return
        
        try:
            self.jtop_instance.close()
            self.is_running = False
            logger.info("Stopped Jetson resource monitoring")
        except Exception as e:
            logger.error(f"Error stopping Jetson monitoring: {str(e)}")
        finally:
            self.jtop_instance = None
    
    def get_usage(self) -> ResourceUsage:
        """
        Get current resource usage information.
        
        Returns:
            A ResourceUsage object with the current resource data.
        """
        if not self.is_running or self.jtop_instance is None:
            self.start()
            # Wait for data to be available
            time.sleep(0.5)
        
        try:
            if not self.jtop_instance.ok():
                logger.warning("jtop is not running properly, restarting")
                self.stop()
                self.start()
                time.sleep(0.5)
            
            # CPU usage
            cpu_usage = 0.0
            if hasattr(self.jtop_instance, 'cpu') and self.jtop_instance.cpu:
                cpu_usage = self.jtop_instance.cpu['usage']
            else:
                # Fallback to psutil
                cpu_usage = psutil.cpu_percent()
            
            # Memory usage
            memory_usage = 0.0
            if hasattr(self.jtop_instance, 'memory') and self.jtop_instance.memory:
                memory_usage = (self.jtop_instance.memory['used'] / 
                               self.jtop_instance.memory['total']) * 100.0
            else:
                # Fallback to psutil
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
            
            # GPU usage
            gpu_usage = None
            gpu_memory_usage = None
            if hasattr(self.jtop_instance, 'gpu') and self.jtop_instance.gpu:
                gpu_usage = self.jtop_instance.gpu['usage']
            
            # Temperature
            temperature = {}
            if hasattr(self.jtop_instance, 'temperature') and self.jtop_instance.temperature:
                for name, temp in self.jtop_instance.temperature.items():
                    if isinstance(temp, (int, float)):
                        temperature[name] = temp
            
            # Power usage
            power_usage = {}
            if hasattr(self.jtop_instance, 'power') and self.jtop_instance.power:
                for name, power in self.jtop_instance.power.items():
                    if isinstance(power, (int, float)):
                        power_usage[name] = power
            
            return ResourceUsage(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                temperature=temperature,
                power_usage=power_usage
            )
            
        except Exception as e:
            logger.error(f"Error getting Jetson resource usage: {str(e)}")
            # Fallback to basic monitoring
            return self._get_basic_usage()
    
    def _get_basic_usage(self) -> ResourceUsage:
        """
        Get basic resource usage when jtop fails.
        
        Returns:
            A ResourceUsage object with basic resource data.
        """
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        return ResourceUsage(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent
        )


class GenericResourceMonitor:
    """
    Generic resource monitor for non-Jetson platforms.
    """
    
    def __init__(self):
        """
        Initialize the generic resource monitor.
        """
        self.nvidia_available = False
        
        try:
            import torch
            self.nvidia_available = torch.cuda.is_available()
        except ImportError:
            pass
        
        logger.info(f"Initialized generic resource monitor (NVIDIA GPU: {self.nvidia_available})")
    
    def start(self):
        """
        Start monitoring (no-op for generic monitor).
        """
        pass
    
    def stop(self):
        """
        Stop monitoring (no-op for generic monitor).
        """
        pass
    
    def get_usage(self) -> ResourceUsage:
        """
        Get current resource usage information.
        
        Returns:
            A ResourceUsage object with the current resource data.
        """
        # CPU usage
        cpu_usage = psutil.cpu_percent()
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU usage (if available)
        gpu_usage = None
        gpu_memory_usage = None
        
        if self.nvidia_available:
            try:
                import torch
                import pynvml
                
                # Initialize NVML
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                if device_count > 0:
                    # Get first device
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # Get GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                    
                    # Get memory utilization
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_usage = (mem_info.used / mem_info.total) * 100.0
                
                # Shutdown NVML
                pynvml.nvmlShutdown()
                
            except (ImportError, Exception) as e:
                logger.debug(f"Error getting NVIDIA GPU stats: {str(e)}")
        
        # Temperature (if available)
        temperature = {}
        try:
            temp_sensors = psutil.sensors_temperatures()
            for name, entries in temp_sensors.items():
                for entry in entries:
                    key = f"{name}_{entry.label}" if entry.label else name
                    temperature[key] = entry.current
        except (AttributeError, Exception):
            # sensors_temperatures not available on all platforms
            pass
        
        return ResourceUsage(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            temperature=temperature if temperature else None
        )


class ResourceMonitor:
    """
    Resource monitor for tracking system resource usage.
    
    Automatically detects if running on Jetson platform and uses the appropriate
    monitoring implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the resource monitor.
        
        Args:
            config: Configuration for the resource monitor.
        """
        self.config = config
        self.enable_monitoring = config.get("enable_monitoring", True)
        self.monitoring_interval_sec = config.get("monitoring_interval_sec", 5)
        self.cpu_threshold = config.get("cpu_threshold", 80)
        self.gpu_threshold = config.get("gpu_threshold", 80)
        self.memory_threshold = config.get("memory_threshold", 80)
        
        # Detect Jetson platform
        self.is_jetson = self._detect_jetson()
        
        # Initialize appropriate monitor
        if self.is_jetson and JETSON_AVAILABLE:
            self.monitor = JetsonResourceMonitor()
            logger.info("Using Jetson-specific resource monitor")
        else:
            self.monitor = GenericResourceMonitor()
            logger.info("Using generic resource monitor")
        
        # Variables for monitoring thread
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.callbacks = []
        self.history = []
        self.max_history = 100
        
        # Current resource usage
        self.current_usage = None
    
    def _detect_jetson(self) -> bool:
        """
        Detect if running on a Jetson platform.
        
        Returns:
            True if running on Jetson, False otherwise.
        """
        # Check for Jetson-specific files
        jetson_files = [
            "/proc/device-tree/model",
            "/etc/nv_tegra_release"
        ]
        
        for path in jetson_files:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        content = f.read().lower()
                        if "jetson" in content or "tegra" in content:
                            return True
                except:
                    pass
        
        return False
    
    def start(self):
        """
        Start the resource monitoring.
        """
        if not self.enable_monitoring:
            logger.info("Resource monitoring is disabled")
            return
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Resource monitoring is already running")
            return
        
        self.stop_event.clear()
        self.monitor.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started resource monitoring (interval: {self.monitoring_interval_sec}s)")
    
    def stop(self):
        """
        Stop the resource monitoring.
        """
        if not self.enable_monitoring or not self.monitoring_thread:
            return
        
        self.stop_event.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        self.monitor.stop()
        logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self):
        """
        Background thread for periodic resource monitoring.
        """
        while not self.stop_event.is_set():
            try:
                usage = self.monitor.get_usage()
                self.current_usage = usage
                
                # Add to history
                self.history.append(usage)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                
                # Check thresholds and notify callbacks
                self._check_thresholds(usage)
                
                # Trigger periodic callbacks
                for callback, args, kwargs in self.callbacks:
                    try:
                        callback(usage, *args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in resource monitoring callback: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {str(e)}")
            
            # Wait for next interval or until stopped
            self.stop_event.wait(self.monitoring_interval_sec)
    
    def _check_thresholds(self, usage: ResourceUsage):
        """
        Check if resource usage exceeds configured thresholds.
        
        Args:
            usage: Current resource usage.
        """
        warnings = []
        
        # CPU threshold
        if usage.cpu_usage > self.cpu_threshold:
            warnings.append(f"CPU usage ({usage.cpu_usage:.1f}%) exceeds threshold ({self.cpu_threshold}%)")
        
        # Memory threshold
        if usage.memory_usage > self.memory_threshold:
            warnings.append(f"Memory usage ({usage.memory_usage:.1f}%) exceeds threshold ({self.memory_threshold}%)")
        
        # GPU threshold (if available)
        if usage.gpu_usage is not None and usage.gpu_usage > self.gpu_threshold:
            warnings.append(f"GPU usage ({usage.gpu_usage:.1f}%) exceeds threshold ({self.gpu_threshold}%)")
        
        # Log warnings
        if warnings:
            logger.warning(f"Resource threshold exceeded: {', '.join(warnings)}")
    
    def get_current_usage(self) -> Optional[ResourceUsage]:
        """
        Get the most recent resource usage data.
        
        Returns:
            The most recent ResourceUsage object, or None if not available.
        """
        if self.current_usage is None and self.enable_monitoring:
            # Get an initial reading
            self.current_usage = self.monitor.get_usage()
        
        return self.current_usage
    
    def get_average_usage(self, window_size: int = 5) -> Optional[ResourceUsage]:
        """
        Get the average resource usage over a window of recent measurements.
        
        Args:
            window_size: Number of most recent measurements to average.
            
        Returns:
            A ResourceUsage object with averaged values, or None if no data available.
        """
        if not self.history:
            return self.get_current_usage()
        
        # Use at most window_size recent measurements
        window = self.history[-min(window_size, len(self.history)):]
        
        # Calculate averages
        avg_cpu = sum(u.cpu_usage for u in window) / len(window)
        avg_memory = sum(u.memory_usage for u in window) / len(window)
        
        # GPU may not be available on all platforms
        gpu_values = [u.gpu_usage for u in window if u.gpu_usage is not None]
        avg_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else None
        
        gpu_mem_values = [u.gpu_memory_usage for u in window if u.gpu_memory_usage is not None]
        avg_gpu_memory = sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else None
        
        # Temperature averages
        avg_temp = {}
        for u in window:
            if u.temperature:
                for key, value in u.temperature.items():
                    if key not in avg_temp:
                        avg_temp[key] = []
                    avg_temp[key].append(value)
        
        avg_temp = {k: sum(v) / len(v) for k, v in avg_temp.items()}
        
        # Power usage averages
        avg_power = {}
        for u in window:
            if u.power_usage:
                for key, value in u.power_usage.items():
                    if key not in avg_power:
                        avg_power[key] = []
                    avg_power[key].append(value)
        
        avg_power = {k: sum(v) / len(v) for k, v in avg_power.items()}
        
        return ResourceUsage(
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu,
            gpu_memory_usage=avg_gpu_memory,
            temperature=avg_temp if avg_temp else None,
            power_usage=avg_power if avg_power else None
        )
    
    def register_callback(self, callback: Callable[[ResourceUsage, Any], None], *args, **kwargs):
        """
        Register a callback to be called on each resource update.
        
        Args:
            callback: Function to call with resource usage data.
            *args: Additional positional arguments to pass to the callback.
            **kwargs: Additional keyword arguments to pass to the callback.
        """
        self.callbacks.append((callback, args, kwargs))
        logger.debug(f"Registered resource monitoring callback: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable):
        """
        Unregister a previously registered callback.
        
        Args:
            callback: The callback function to unregister.
        """
        self.callbacks = [(cb, args, kwargs) for cb, args, kwargs in self.callbacks if cb != callback]
        logger.debug(f"Unregistered resource monitoring callback: {callback.__name__}")
    
    def clear_callbacks(self):
        """
        Remove all registered callbacks.
        """
        self.callbacks = []
        logger.debug("Cleared all resource monitoring callbacks")
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """
        Get a dictionary of current resource metrics.
        
        Returns:
            Dictionary with resource metrics.
        """
        usage = self.get_current_usage()
        if not usage:
            return {}
        
        metrics = usage.to_dict()
        
        # Add average values
        avg_usage = self.get_average_usage()
        if avg_usage:
            metrics["avg_cpu_usage"] = avg_usage.cpu_usage
            metrics["avg_memory_usage"] = avg_usage.memory_usage
            if avg_usage.gpu_usage is not None:
                metrics["avg_gpu_usage"] = avg_usage.gpu_usage
        
        return metrics