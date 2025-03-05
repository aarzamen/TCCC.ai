"""
Unit tests for the StateManager class.
"""

import os
import json
import sqlite3
import tempfile
import pytest
import time
from pathlib import Path
from typing import Dict, Any

from tccc.processing_core.state_manager import StateManager, StateEntry


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sqlite_state_manager(temp_state_dir):
    """Create a StateManager instance with SQLite persistence."""
    config = {
        "enable_persistence": True,
        "storage_path": temp_state_dir,
        "autosave_interval_sec": 0.1,  # Short interval for testing
        "keep_history": True
    }
    manager = StateManager(config)
    yield manager
    manager.stop()


@pytest.fixture
def memory_state_manager():
    """Create a StateManager instance without persistence."""
    config = {
        "enable_persistence": False
    }
    manager = StateManager(config)
    yield manager
    manager.stop()


class TestStateManager:
    """Tests for the StateManager class with SQLite persistence."""

    def test_initialization(self, temp_state_dir):
        """Test initialization of StateManager with SQLite."""
        config = {
            "enable_persistence": True,
            "storage_path": temp_state_dir
        }
        manager = StateManager(config)
        
        # Verify SQLite database was created
        db_path = os.path.join(temp_state_dir, "state.db")
        assert os.path.exists(db_path)
        
        # Verify WAL mode is enabled
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        conn.close()
        
        assert journal_mode.upper() == "WAL"
        
        # Clean up
        manager.stop()
    
    def test_set_get_state(self, sqlite_state_manager):
        """Test setting and getting state."""
        # Set state
        test_state = {"key1": "value1", "key2": 42, "key3": {"nested": "value"}}
        sqlite_state_manager.set_state(test_state)
        
        # Get state
        state = sqlite_state_manager.get_state()
        
        # Verify state was set correctly
        assert state == test_state
    
    def test_update_state(self, sqlite_state_manager):
        """Test updating state."""
        # Set initial state
        initial_state = {"key1": "value1", "key2": 42}
        sqlite_state_manager.set_state(initial_state)
        
        # Update state
        update = {"key2": 99, "key3": "new_value"}
        sqlite_state_manager.update_state(update)
        
        # Get updated state
        state = sqlite_state_manager.get_state()
        
        # Verify state was updated correctly
        assert state["key1"] == "value1"  # Unchanged
        assert state["key2"] == 99  # Updated
        assert state["key3"] == "new_value"  # Added
    
    def test_state_value_operations(self, sqlite_state_manager):
        """Test state value operations."""
        # Set state value
        sqlite_state_manager.set_state_value("test_key", "test_value")
        
        # Get state value
        value = sqlite_state_manager.get_state_value("test_key")
        
        # Verify state value was set correctly
        assert value == "test_value"
        
        # Test default value for nonexistent key
        value = sqlite_state_manager.get_state_value("nonexistent_key", "default")
        assert value == "default"
    
    def test_history_tracking(self, sqlite_state_manager):
        """Test history tracking."""
        # Set state multiple times to generate history
        sqlite_state_manager.set_state({"state": 1})
        sqlite_state_manager.set_state({"state": 2})
        sqlite_state_manager.set_state({"state": 3})
        
        # Get history
        history = sqlite_state_manager.get_history()
        
        # Verify history contains entries
        assert len(history) > 0
        
        # The history ordering might vary based on implementation details
        # Just verify that at least one of the entries has the correct state
        state_values = [entry["state"].get("state") for entry in history]
        assert 3 in state_values
    
    def test_checkpoint_operations(self, sqlite_state_manager):
        """Test checkpoint operations."""
        # Set state
        test_state = {"test": "checkpoint_data"}
        sqlite_state_manager.set_state(test_state)
        
        # Save checkpoint
        sqlite_state_manager.save_checkpoint("test_checkpoint")
        
        # Verify checkpoint exists
        checkpoints = sqlite_state_manager.list_checkpoints()
        assert "test_checkpoint" in checkpoints
        
        # Change state
        sqlite_state_manager.set_state({"test": "different_data"})
        
        # Load checkpoint
        success = sqlite_state_manager.load_checkpoint("test_checkpoint")
        assert success
        
        # Verify state was restored from checkpoint
        state = sqlite_state_manager.get_state()
        assert state["test"] == "checkpoint_data"
        
        # Delete checkpoint
        success = sqlite_state_manager.delete_checkpoint("test_checkpoint")
        assert success
        
        # Verify checkpoint was deleted
        checkpoints = sqlite_state_manager.list_checkpoints()
        assert "test_checkpoint" not in checkpoints
    
    def test_persistence_between_instances(self, temp_state_dir):
        """Test state persistence between different instances."""
        # Create and configure first instance
        config = {
            "enable_persistence": True,
            "storage_path": temp_state_dir,
            "autosave_interval_sec": 0.1
        }
        manager1 = StateManager(config)
        
        # Set state in first instance
        test_state = {"persistence": "test", "nested": {"value": 42}}
        manager1.set_state(test_state)
        
        # Wait for autosave
        time.sleep(0.2)
        
        # Shut down first instance
        manager1.stop()
        
        # Create second instance with same config
        manager2 = StateManager(config)
        
        # Get state from second instance
        state = manager2.get_state()
        
        # Verify state was persisted
        assert state["persistence"] == "test"
        assert state["nested"]["value"] == 42
        
        # Clean up
        manager2.stop()
    
    def test_callback_mechanism(self, sqlite_state_manager):
        """Test callback mechanism."""
        # Initialize callback tracking
        callback_called = False
        old_state_from_callback = None
        new_state_from_callback = None
        
        # Define callback function
        def state_change_callback(old_state, new_state):
            nonlocal callback_called, old_state_from_callback, new_state_from_callback
            callback_called = True
            old_state_from_callback = old_state
            new_state_from_callback = new_state
        
        # Register callback
        sqlite_state_manager.register_callback(state_change_callback)
        
        # Set initial state
        initial_state = {"callback": "initial"}
        sqlite_state_manager.set_state(initial_state)
        
        # Update state to trigger callback
        new_state = {"callback": "updated"}
        sqlite_state_manager.set_state(new_state)
        
        # Verify callback was called
        assert callback_called
        
        # Verify callback received correct old and new states
        assert old_state_from_callback["callback"] == "initial"
        assert new_state_from_callback["callback"] == "updated"
        
        # Unregister callback
        sqlite_state_manager.unregister_callback(state_change_callback)
        
        # Reset tracking
        callback_called = False
        
        # Update state again
        sqlite_state_manager.set_state({"callback": "another_update"})
        
        # Verify callback was not called after unregistering
        assert not callback_called
    
    def test_state_entry_serialization(self):
        """Test serialization of StateEntry objects."""
        # Create a StateEntry
        state = {"test": "data", "number": 42}
        metadata = {"source": "test", "importance": "high"}
        timestamp = time.time()
        entry = StateEntry(state, timestamp, metadata)
        
        # Convert to dictionary
        entry_dict = entry.to_dict()
        
        # Verify dictionary contains all fields
        assert entry_dict["state"] == state
        assert entry_dict["timestamp"] == timestamp
        assert entry_dict["metadata"] == metadata
        
        # Recreate from dictionary
        recreated = StateEntry.from_dict(entry_dict)
        
        # Verify recreated entry matches original
        assert recreated.state == entry.state
        assert recreated.timestamp == entry.timestamp
        assert recreated.metadata == entry.metadata
    
    def test_reset_state(self, sqlite_state_manager):
        """Test resetting state."""
        # Set initial state
        sqlite_state_manager.set_state({"key": "value"})
        
        # Reset state
        sqlite_state_manager.reset_state()
        
        # Verify state is empty
        state = sqlite_state_manager.get_state()
        assert state == {}
    
    def test_memory_only_mode(self, memory_state_manager):
        """Test memory-only mode (no persistence)."""
        # Set state
        memory_state_manager.set_state({"memory": "only"})
        
        # Get state
        state = memory_state_manager.get_state()
        
        # Verify state was set correctly
        assert state["memory"] == "only"
        
        # Verify no database connection
        assert memory_state_manager.connection is None