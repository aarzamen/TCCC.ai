"""
State management module for TCCC.ai processing core.

This module provides state management functionality for the Processing Core.
Uses SQLite with WAL mode for persistence.
"""

import os
import json
import sqlite3
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

from tccc.utils.logging import get_logger

# Define SystemState enum here to avoid circular imports
class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    READY = "ready"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    REPORTING = "reporting"
    ERROR = "error"
    SHUTDOWN = "shutdown"

logger = get_logger(__name__)


class StateEntry:
    """
    Represents a state entry with timestamp and metadata.
    """
    
    def __init__(self, state: Dict[str, Any], timestamp: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a state entry.
        
        Args:
            state: The state data.
            timestamp: The timestamp for the state entry.
            metadata: Additional metadata about the state entry.
        """
        self.state = state.copy()
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state entry to a dictionary.
        
        Returns:
            A dictionary representation of the state entry.
        """
        return {
            "state": self.state,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateEntry':
        """
        Create a state entry from a dictionary.
        
        Args:
            data: Dictionary containing state entry data.
            
        Returns:
            A StateEntry instance.
        """
        return cls(
            state=data.get("state", {}),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata")
        )


class StateManager:
    """
    Manages state for the Processing Core.
    Uses SQLite with WAL mode for efficient and resilient persistence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state manager.
        
        Args:
            config: Configuration for the state manager.
        """
        self.config = config
        self.enable_persistence = config.get("enable_persistence", True)
        self.storage_path = config.get("storage_path", "data/processing_core/state")
        self.autosave_interval_sec = config.get("autosave_interval_sec", 60)
        self.keep_history = config.get("keep_history", True)
        self.max_history_entries = config.get("max_history_entries", 10)
        self.db_path = os.path.join(self.storage_path, "state.db")
        self.connection = None
        
        # Create storage directory if needed
        if self.enable_persistence and not os.path.exists(self.storage_path):
            try:
                os.makedirs(self.storage_path, exist_ok=True)
                logger.info(f"Created state storage directory: {self.storage_path}")
            except Exception as e:
                logger.error(f"Failed to create state storage directory: {str(e)}")
                self.enable_persistence = False
        
        # State storage
        self.current_state: Dict[str, Any] = {}
        self.history: List[StateEntry] = []
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # State change callbacks
        self.callbacks: List[Callable[[Dict[str, Any], Dict[str, Any]], None]] = []
        
        # Autosave thread
        self.autosave_thread = None
        self.stop_autosave = threading.Event()
        
        # Initialize database
        if self.enable_persistence:
            self._init_database()
        
        # Load initial state
        self._load_state()
        
        # Start autosave thread if enabled
        if self.enable_persistence and self.autosave_interval_sec > 0:
            self._start_autosave()
    
    def _init_database(self):
        """
        Initialize the SQLite database with WAL mode.
        """
        try:
            # Create a connection to the database
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Enable WAL mode for better concurrency and resilience
            self.connection.execute("PRAGMA journal_mode=WAL;")
            
            # Create tables if they don't exist
            cursor = self.connection.cursor()
            
            # Current state table - stores key-value pairs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS current_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    modified_at REAL
                )
            ''')
            
            # State history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT,
                    timestamp REAL,
                    metadata TEXT
                )
            ''')
            
            # Checkpoints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    name TEXT PRIMARY KEY,
                    state TEXT,
                    timestamp REAL,
                    metadata TEXT
                )
            ''')
            
            self.connection.commit()
            logger.info("Initialized SQLite database with WAL mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            self.enable_persistence = False
    
    def _load_state(self):
        """
        Load state from SQLite database.
        """
        if not self.enable_persistence or self.connection is None:
            return
        
        with self.lock:
            try:
                # Load current state
                cursor = self.connection.cursor()
                cursor.execute("SELECT key, value FROM current_state")
                rows = cursor.fetchall()
                
                for key, value in rows:
                    try:
                        self.current_state[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # Store as-is if not valid JSON
                        self.current_state[key] = value
                
                logger.info(f"Loaded {len(rows)} state keys from database")
                
                # Load state history if enabled
                if self.keep_history:
                    cursor.execute(
                        "SELECT state, timestamp, metadata FROM state_history ORDER BY timestamp DESC LIMIT ?",
                        (self.max_history_entries,)
                    )
                    history_rows = cursor.fetchall()
                    
                    for state_json, timestamp, metadata_json in history_rows:
                        try:
                            state = json.loads(state_json)
                            metadata = json.loads(metadata_json) if metadata_json else None
                            self.history.append(StateEntry(state, timestamp, metadata))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing history entry: {e}")
                    
                    # Reverse to get chronological order
                    self.history.reverse()
                    logger.info(f"Loaded {len(self.history)} history entries from database")
                
            except Exception as e:
                logger.error(f"Failed to load state from database: {str(e)}")
    
    def _save_state(self):
        """
        Save state to SQLite database.
        """
        if not self.enable_persistence or self.connection is None:
            return
        
        with self.lock:
            try:
                cursor = self.connection.cursor()
                current_time = time.time()
                
                # Begin transaction
                self.connection.execute("BEGIN TRANSACTION")
                
                # Save current state
                for key, value in self.current_state.items():
                    value_json = json.dumps(value)
                    cursor.execute(
                        "INSERT OR REPLACE INTO current_state (key, value, modified_at) VALUES (?, ?, ?)",
                        (key, value_json, current_time)
                    )
                
                # Save latest history entry if available and history is enabled
                if self.keep_history and self.history:
                    latest = self.history[-1]
                    state_json = json.dumps(latest.state)
                    metadata_json = json.dumps(latest.metadata) if latest.metadata else None
                    
                    cursor.execute(
                        "INSERT INTO state_history (state, timestamp, metadata) VALUES (?, ?, ?)",
                        (state_json, latest.timestamp, metadata_json)
                    )
                    
                    # Cleanup old history entries if we have too many
                    cursor.execute(
                        "DELETE FROM state_history WHERE id NOT IN (SELECT id FROM state_history ORDER BY timestamp DESC LIMIT ?)",
                        (self.max_history_entries,)
                    )
                
                # Commit transaction
                self.connection.commit()
                logger.debug(f"Saved state to database ({len(self.current_state)} keys)")
                
            except Exception as e:
                logger.error(f"Failed to save state to database: {str(e)}")
                # Try to rollback transaction
                try:
                    self.connection.rollback()
                except:
                    pass
    
    def _start_autosave(self):
        """
        Start the autosave thread.
        """
        if self.autosave_thread is not None and self.autosave_thread.is_alive():
            return
        
        self.stop_autosave.clear()
        self.autosave_thread = threading.Thread(
            target=self._autosave_loop,
            daemon=True
        )
        self.autosave_thread.start()
        logger.info(f"Started state autosave (interval: {self.autosave_interval_sec}s)")
    
    def _autosave_loop(self):
        """
        Background thread for periodic state saving.
        """
        while not self.stop_autosave.is_set():
            try:
                # Save state
                self._save_state()
            except Exception as e:
                logger.error(f"Error in autosave loop: {str(e)}")
            
            # Wait for next interval or until stopped
            self.stop_autosave.wait(self.autosave_interval_sec)
    
    def stop(self):
        """
        Stop the state manager, save state, and close database connection.
        """
        if self.autosave_thread is not None and self.autosave_thread.is_alive():
            self.stop_autosave.set()
            self.autosave_thread.join(timeout=2.0)
        
        # Save state one last time
        self._save_state()
        
        # Close database connection
        if self.connection is not None:
            try:
                self.connection.close()
                self.connection = None
                logger.info("Closed database connection")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")
        
        logger.info("Stopped state manager")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.
        
        Returns:
            The current state dictionary.
        """
        with self.lock:
            return self.current_state.copy()
    
    def set_state(self, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the current state.
        
        Args:
            state: The new state to set.
            metadata: Additional metadata about the state change.
        """
        with self.lock:
            old_state = self.current_state.copy()
            
            # Create a copy of the new state
            new_state = state.copy()
            
            # Update current state
            self.current_state = new_state
            
            # Add to history if enabled
            if self.keep_history:
                self.history.append(StateEntry(new_state, metadata=metadata))
                
                # Trim history if needed
                if len(self.history) > self.max_history_entries:
                    self.history = self.history[-self.max_history_entries:]
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {str(e)}")
            
            # Save state if persistence is enabled
            if self.enable_persistence and self.autosave_interval_sec <= 0:
                # If autosave is disabled, save immediately
                self._save_state()
    
    def update_state(self, update: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the current state with new values.
        
        Args:
            update: Dictionary with state updates.
            metadata: Additional metadata about the state change.
        """
        with self.lock:
            old_state = self.current_state.copy()
            
            # Apply update to current state
            self.current_state.update(update)
            
            # Add to history if enabled
            if self.keep_history:
                self.history.append(StateEntry(self.current_state.copy(), metadata=metadata))
                
                # Trim history if needed
                if len(self.history) > self.max_history_entries:
                    self.history = self.history[-self.max_history_entries:]
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(old_state, self.current_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {str(e)}")
            
            # Save state if persistence is enabled
            if self.enable_persistence and self.autosave_interval_sec <= 0:
                # If autosave is disabled, save immediately
                self._save_state()
    
    def get_state_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific value from the current state.
        
        Args:
            key: The state key to get.
            default: Default value to return if the key is not found.
            
        Returns:
            The value for the specified key, or the default value if not found.
        """
        with self.lock:
            return self.current_state.get(key, default)
    
    def set_state_value(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set a specific value in the current state.
        
        Args:
            key: The state key to set.
            value: The value to set.
            metadata: Additional metadata about the state change.
        """
        with self.lock:
            old_state = self.current_state.copy()
            
            # Update state
            self.current_state[key] = value
            
            # Add to history if enabled
            if self.keep_history:
                self.history.append(StateEntry(
                    self.current_state.copy(),
                    metadata=metadata or {"updated_key": key}
                ))
                
                # Trim history if needed
                if len(self.history) > self.max_history_entries:
                    self.history = self.history[-self.max_history_entries:]
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(old_state, self.current_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {str(e)}")
            
            # Save state if persistence is enabled
            if self.enable_persistence and self.autosave_interval_sec <= 0:
                # If autosave is disabled, save immediately
                self._save_state()
    
    def register_callback(self, callback: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> None:
        """
        Register a callback for state changes.
        
        Args:
            callback: Function to call when the state changes. The function should accept
                two arguments: the old state and the new state.
        """
        with self.lock:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable) -> None:
        """
        Unregister a previously registered callback.
        
        Args:
            callback: The callback function to unregister.
        """
        with self.lock:
            self.callbacks = [cb for cb in self.callbacks if cb != callback]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the state history.
        
        Returns:
            A list of state history entries.
        """
        with self.lock:
            return [entry.to_dict() for entry in self.history]
    
    def clear_history(self) -> None:
        """
        Clear the state history.
        """
        with self.lock:
            self.history.clear()
            
            # Save state if persistence is enabled
            if self.enable_persistence:
                self._save_state()
    
    def reset_state(self) -> None:
        """
        Reset the state to an empty dictionary.
        """
        with self.lock:
            old_state = self.current_state.copy()
            
            # Reset state
            self.current_state = {}
            
            # Add to history if enabled
            if self.keep_history:
                self.history.append(StateEntry(
                    self.current_state.copy(),
                    metadata={"action": "reset"}
                ))
                
                # Trim history if needed
                if len(self.history) > self.max_history_entries:
                    self.history = self.history[-self.max_history_entries:]
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(old_state, self.current_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {str(e)}")
            
            # Save state if persistence is enabled
            if self.enable_persistence:
                self._save_state()
    
    def save_checkpoint(self, name: str) -> None:
        """
        Save the current state as a named checkpoint.
        
        Args:
            name: The name of the checkpoint.
        """
        if not self.enable_persistence or self.connection is None:
            logger.warning("Cannot save checkpoint: Persistence is disabled")
            return
        
        with self.lock:
            try:
                cursor = self.connection.cursor()
                checkpoint_data = {
                    "state": self.current_state,
                    "timestamp": time.time(),
                    "metadata": {
                        "name": name,
                        "date": datetime.now().isoformat()
                    }
                }
                
                state_json = json.dumps(checkpoint_data["state"])
                timestamp = checkpoint_data["timestamp"]
                metadata_json = json.dumps(checkpoint_data["metadata"])
                
                cursor.execute(
                    "INSERT OR REPLACE INTO checkpoints (name, state, timestamp, metadata) VALUES (?, ?, ?, ?)",
                    (name, state_json, timestamp, metadata_json)
                )
                
                self.connection.commit()
                logger.info(f"Saved state checkpoint: {name}")
            except Exception as e:
                logger.error(f"Failed to save state checkpoint: {str(e)}")
                try:
                    self.connection.rollback()
                except:
                    pass
    
    def load_checkpoint(self, name: str) -> bool:
        """
        Load a named checkpoint as the current state.
        
        Args:
            name: The name of the checkpoint.
            
        Returns:
            True if the checkpoint was loaded successfully, False otherwise.
        """
        if not self.enable_persistence or self.connection is None:
            logger.warning("Cannot load checkpoint: Persistence is disabled")
            return False
        
        with self.lock:
            try:
                cursor = self.connection.cursor()
                cursor.execute(
                    "SELECT state, timestamp, metadata FROM checkpoints WHERE name = ?",
                    (name,)
                )
                row = cursor.fetchone()
                
                if not row:
                    logger.error(f"Checkpoint not found: {name}")
                    return False
                
                state_json, timestamp, metadata_json = row
                old_state = self.current_state.copy()
                
                # Update current state
                self.current_state = json.loads(state_json)
                
                # Add to history if enabled
                if self.keep_history:
                    metadata = {"action": "load_checkpoint", "name": name}
                    if metadata_json:
                        try:
                            metadata.update(json.loads(metadata_json))
                        except:
                            pass
                    
                    self.history.append(StateEntry(
                        self.current_state.copy(),
                        timestamp=time.time(),
                        metadata=metadata
                    ))
                    
                    # Trim history if needed
                    if len(self.history) > self.max_history_entries:
                        self.history = self.history[-self.max_history_entries:]
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(old_state, self.current_state)
                    except Exception as e:
                        logger.error(f"Error in state change callback: {str(e)}")
                
                # Save current state to database
                self._save_state()
                
                logger.info(f"Loaded state checkpoint: {name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load state checkpoint: {str(e)}")
                return False
    
    def list_checkpoints(self) -> List[str]:
        """
        List available state checkpoints.
        
        Returns:
            A list of checkpoint names.
        """
        if not self.enable_persistence or self.connection is None:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM checkpoints ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []
            
    def delete_checkpoint(self, name: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            name: The name of the checkpoint to delete.
            
        Returns:
            True if the checkpoint was deleted, False otherwise.
        """
        if not self.enable_persistence or self.connection is None:
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE name = ?", (name,))
            self.connection.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Deleted checkpoint: {name}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {str(e)}")
            return False