"""
Logging setup for TCCC.ai system.

This module provides structured logging with context for all TCCC.ai components.
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional


class ContextLogger:
    """
    Logger class that provides structured logging with context.
    
    Supports adding persistent context that will be included with all log messages,
    as well as adding per-message context.
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO, 
                 log_to_file: bool = True, log_dir: str = None):
        """
        Initialize the logger.
        
        Args:
            name: The name of the logger.
            log_level: The minimum log level to log.
            log_to_file: Whether to log to a file.
            log_dir: Directory to store log files, if logging to files.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.context = {}
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if requested
        if log_to_file:
            if log_dir is None:
                log_dir = os.path.join(os.getcwd(), 'logs')
            
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(
                log_dir, 
                f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
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
               log_to_file: bool = True, log_dir: str = None) -> ContextLogger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: The name of the logger.
        log_level: The minimum log level to log.
        log_to_file: Whether to log to a file.
        log_dir: Directory to store log files, if logging to files.
        
    Returns:
        A ContextLogger instance.
    """
    return ContextLogger(name, log_level, log_to_file, log_dir)