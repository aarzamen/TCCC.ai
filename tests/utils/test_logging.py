"""
Unit tests for the logging system.
"""

import os
import pytest
import tempfile
import time
import logging
import json
from pathlib import Path

from tccc.utils.logging import get_logger, ContextLogger

@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestLogging:
    """Tests for the logging system."""
    
    def test_context_logger_initialization(self, temp_log_dir):
        """Test initialization of ContextLogger."""
        logger = get_logger(
            name="test_logger",
            log_level=logging.DEBUG,
            log_to_file=True,
            log_dir=temp_log_dir
        )
        
        # Verify logger was created
        assert isinstance(logger, ContextLogger)
        
        # Verify log files were created
        log_files = os.listdir(temp_log_dir)
        assert any(file.startswith("test_logger") for file in log_files)
        
        # Verify daily log directory was created
        daily_dir = os.path.join(temp_log_dir, "daily")
        assert os.path.isdir(daily_dir)
        
        # Verify daily log file was created
        daily_files = os.listdir(daily_dir)
        assert any(file.startswith("test_logger") for file in daily_files)
    
    def test_log_rotation_setup(self, temp_log_dir):
        """Test that log rotation handlers are properly set up."""
        logger = get_logger(
            name="rotation_test",
            log_to_file=True,
            log_dir=temp_log_dir,
            rotation_size_mb=1,  # Small size for testing
            max_log_files=5
        )
        
        # Get the Python logger
        py_logger = logger.logger
        
        # Verify handlers are present
        handlers = py_logger.handlers
        
        # Should have 3 handlers: console, size rotation, time rotation
        assert len(handlers) == 3
        
        # Verify at least one rotating handler is present
        rotating_handlers = [h for h in handlers if hasattr(h, 'backupCount')]
        assert len(rotating_handlers) == 2
        
        # Verify rotation settings
        for handler in rotating_handlers:
            assert handler.backupCount == 5  # max_log_files
    
    def test_context_addition(self):
        """Test adding context to logger."""
        logger = get_logger("context_test", log_to_file=False)
        
        # Add context
        logger.add_context(user="test_user", session_id="12345")
        
        # Verify context was added
        assert logger.context["user"] == "test_user"
        assert logger.context["session_id"] == "12345"
    
    def test_log_with_context(self, caplog):
        """Test logging with context."""
        logger = get_logger("context_log_test", log_to_file=False)
        
        # Add persistent context
        logger.add_context(user="test_user")
        
        # Log with additional context
        with caplog.at_level(logging.INFO):
            logger.info("Test message", {"request_id": "abcd1234"})
        
        # Verify log message contains both contexts
        assert "Test message" in caplog.text
        assert "test_user" in caplog.text
        assert "abcd1234" in caplog.text
    
    def test_log_levels(self, caplog):
        """Test different log levels."""
        logger = get_logger("level_test", log_level=logging.WARNING, log_to_file=False)
        
        # Debug and info should not be logged at WARNING level
        with caplog.at_level(logging.WARNING):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        # Verify only warning and above were logged
        assert "Debug message" not in caplog.text
        assert "Info message" not in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text
        assert "Critical message" in caplog.text
    
    def test_file_logging(self, temp_log_dir):
        """Test that logs are written to file."""
        logger_name = "file_test"
        logger = get_logger(
            name=logger_name,
            log_to_file=True,
            log_dir=temp_log_dir
        )
        
        # Log a message
        logger.info("Test file logging")
        
        # Get the log file
        log_file = os.path.join(temp_log_dir, f"{logger_name}.log")
        
        # Verify file exists and contains the message
        assert os.path.exists(log_file)
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test file logging" in content
    
    def test_size_based_rotation(self, temp_log_dir):
        """Test size-based log rotation."""
        logger = get_logger(
            name="size_rotation_test",
            log_to_file=True,
            log_dir=temp_log_dir,
            rotation_size_mb=0.001  # Very small size (approximately 1KB)
        )
        
        # Write enough logs to trigger rotation
        large_msg = "X" * 500  # 500 bytes per message
        for i in range(10):  # 5KB total
            logger.info(f"{large_msg} {i}")
        
        # Check for rotated files
        files = [f for f in os.listdir(temp_log_dir) if f.startswith("size_rotation_test")]
        
        # Should have at least 2 files (current + rotated)
        assert len(files) >= 2
    
    def test_memory_only_logger(self):
        """Test logger with no file output."""
        logger = get_logger("memory_only", log_to_file=False)
        
        # Log a message
        logger.info("Memory only test")
        
        # Verify the logger has only one handler (console)
        assert len(logger.logger.handlers) == 1
        assert isinstance(logger.logger.handlers[0], logging.StreamHandler)
    
    def test_handler_uniqueness(self):
        """Test that creating multiple loggers with the same name doesn't duplicate handlers."""
        # Get logger with same name twice
        logger1 = get_logger("unique_test", log_to_file=False)
        logger2 = get_logger("unique_test", log_to_file=False)
        
        # Should have same number of handlers
        assert len(logger1.logger.handlers) == len(logger2.logger.handlers)
        
        # The logger objects should be different but reference the same Python logger
        assert logger1 is not logger2
        assert logger1.logger is logger2.logger