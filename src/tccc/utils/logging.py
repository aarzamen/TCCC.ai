"""
Logging setup for TCCC.ai system.

This module provides structured logging with context and log rotation for all TCCC.ai components.
"""

import logging
import json
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class ContextLogger:
    """
    Logger class that provides structured logging with context and rotation.
    
    Supports adding persistent context that will be included with all log messages,
    as well as adding per-message context. Includes both size-based and time-based
    log rotation for better log management.
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO, 
                 log_to_file: bool = True, log_dir: str = None,
                 rotation_size_mb: int = 10, max_log_files: int = 10,
                 time_rotation: str = 'midnight'):
        """
        Initialize the logger with rotation support.
        
        Args:
            name: The name of the logger.
            log_level: The minimum log level to log.
            log_to_file: Whether to log to a file.
            log_dir: Directory to store log files, if logging to files.
            rotation_size_mb: Maximum size of log files in MB before rotation.
            max_log_files: Maximum number of log files to keep.
            time_rotation: When to rotate logs based on time ('midnight', 'h' for hourly, etc.)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.context = {}
        
        # Remove existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handlers with rotation if requested
        if log_to_file:
            if log_dir is None:
                log_dir = os.path.join(os.getcwd(), 'logs')
            
            os.makedirs(log_dir, exist_ok=True)
            
            # Base log file name
            base_log_file = os.path.join(log_dir, f"{name}.log")
            
            # Create formatter for file handlers
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Size-based rotation handler
            size_handler = RotatingFileHandler(
                filename=base_log_file,
                maxBytes=rotation_size_mb * 1024 * 1024,  # Convert MB to bytes
                backupCount=max_log_files
            )
            size_handler.setLevel(log_level)
            size_handler.setFormatter(file_formatter)
            self.logger.addHandler(size_handler)
            
            # Time-based rotation handler for daily logs
            # Store logs in a separate directory to avoid conflicts
            time_log_dir = os.path.join(log_dir, "daily")
            os.makedirs(time_log_dir, exist_ok=True)
            
            time_log_file = os.path.join(time_log_dir, f"{name}.log")
            time_handler = TimedRotatingFileHandler(
                filename=time_log_file,
                when=time_rotation,  # Rotate at midnight by default
                interval=1,  # One day
                backupCount=max_log_files
            )
            time_handler.setLevel(log_level)
            time_handler.setFormatter(file_formatter)
            self.logger.addHandler(time_handler)
    
    def add_context(self, **kwargs):
        """
        Add persistent context that will be included with all future log messages.
        
        Args:
            **kwargs: Key-value pairs to add to the context.
        """
        self.context.update(kwargs)
    
    def _format_message(self, msg: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format the log message with context.
        
        Args:
            msg: The log message.
            context: Additional context for this specific log message.
            
        Returns:
            The formatted log message with context.
        """
        log_context = self.context.copy()
        if context:
            log_context.update(context)
        
        if log_context:
            context_str = json.dumps(log_context)
            return f"{msg} - Context: {context_str}"
        
        return msg
    
    def debug(self, msg: str, context: Optional[Dict[str, Any]] = None):
        """Log a debug message with context."""
        self.logger.debug(self._format_message(msg, context))
    
    def info(self, msg: str, context: Optional[Dict[str, Any]] = None):
        """Log an info message with context."""
        self.logger.info(self._format_message(msg, context))
    
    def warning(self, msg: str, context: Optional[Dict[str, Any]] = None):
        """Log a warning message with context."""
        self.logger.warning(self._format_message(msg, context))
    
    def error(self, msg: str, context: Optional[Dict[str, Any]] = None):
        """Log an error message with context."""
        self.logger.error(self._format_message(msg, context))
    
    def critical(self, msg: str, context: Optional[Dict[str, Any]] = None):
        """Log a critical message with context."""
        self.logger.critical(self._format_message(msg, context))


def get_logger(name: str, log_level: int = logging.INFO, 
               log_to_file: bool = True, log_dir: str = None,
               rotation_size_mb: int = 10, max_log_files: int = 10,
               time_rotation: str = 'midnight') -> ContextLogger:
    """
    Get a logger instance with the given name and rotation settings.
    
    Args:
        name: The name of the logger.
        log_level: The minimum log level to log.
        log_to_file: Whether to log to a file.
        log_dir: Directory to store log files, if logging to files.
        rotation_size_mb: Maximum size of log files in MB before rotation.
        max_log_files: Maximum number of log files to keep.
        time_rotation: When to rotate logs based on time.
        
    Returns:
        A ContextLogger instance.
    """
    return ContextLogger(
        name=name, 
        log_level=log_level, 
        log_to_file=log_to_file, 
        log_dir=log_dir,
        rotation_size_mb=rotation_size_mb,
        max_log_files=max_log_files,
        time_rotation=time_rotation
    )