"""
State management module for TCCC.ai processing core.

This module provides state management functionality for the Processing Core.
"""

import os
import json
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from tccc.utils.logging import get_logger

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
        
        # Load initial state
        self._load_state()
        
        # Start autosave thread if enabled
        if self.enable_persistence and self.autosave_interval_sec > 0:
            self._start_autosave()
    
    def _load_state(self):
        """
        Load state from persistent storage.
        """
        if not self.enable_persistence:
            return
        
        state_file = os.path.join(self.storage_path, "current_state.json")
        history_file = os.path.join(self.storage_path, "state_history.json")
        
        with self.lock:
            # Load current state
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r') as f:
                        self.current_state = json.load(f)
                    logger.info(f"Loaded state from {state_file}")
                except Exception as e:
                    logger.error(f"Failed to load state from {state_file}: {str(e)}")
            
            # Load state history
            if self.keep_history and os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                        self.history = [StateEntry.from_dict(entry) for entry in history_data]
                    logger.info(f"Loaded state history from {history_file} ({len(self.history)} entries)")
                except Exception as e:
                    logger.error(f"Failed to load state history from {history_file}: {str(e)}")
    
    def _save_state(self):
        """
        Save state to persistent storage.
        """
        if not self.enable_persistence:
            return
        
        state_file = os.path.join(self.storage_path, "current_state.json")
        history_file = os.path.join(self.storage_path, "state_history.json")
        
        with self.lock:
            # Save current state
            try:
                with open(state_file, 'w') as f:
                    json.dump(self.current_state, f, indent=2)
                logger.debug(f"Saved state to {state_file}")
            except Exception as e:
                logger.error(f"Failed to save state to {state_file}: {str(e)}")
            
            # Save state history
            if self.keep_history and self.history:
                try:
                    history_data = [entry.to_dict() for entry in self.history]
                    with open(history_file, 'w') as f:
                        json.dump(history_data, f, indent=2)
                    logger.debug(f"Saved state history to {history_file} ({len(self.history)} entries)")
                except Exception as e:
                    logger.error(f"Failed to save state history to {history_file}: {str(e)}")
    
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
        Stop the state manager and save state.
        """
        if self.autosave_thread is not None and self.autosave_thread.is_alive():
            self.stop_autosave.set()
            self.autosave_thread.join(timeout=2.0)
        
        # Save state one last time
        self._save_state()
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
        if not self.enable_persistence:
            logger.warning("Cannot save checkpoint: Persistence is disabled")
            return
        
        checkpoint_file = os.path.join(self.storage_path, f"checkpoint_{name}.json")
        
        with self.lock:
            try:
                checkpoint_data = {
                    "state": self.current_state,
                    "timestamp": time.time(),
                    "metadata": {
                        "name": name,
                        "date": datetime.now().isoformat()
                    }
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                logger.info(f"Saved state checkpoint: {name}")
            except Exception as e:
                logger.error(f"Failed to save state checkpoint: {str(e)}")
    
    def load_checkpoint(self, name: str) -> bool:
        """
        Load a named checkpoint as the current state.
        
        Args:
            name: The name of the checkpoint.
            
        Returns:
            True if the checkpoint was loaded successfully, False otherwise.
        """
        if not self.enable_persistence:
            logger.warning("Cannot load checkpoint: Persistence is disabled")
            return False
        
        checkpoint_file = os.path.join(self.storage_path, f"checkpoint_{name}.json")
        
        if not os.path.exists(checkpoint_file):
            logger.error(f"Checkpoint not found: {name}")
            return False
        
        with self.lock:
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                old_state = self.current_state.copy()
                
                # Update current state
                self.current_state = checkpoint_data.get("state", {})
                
                # Add to history if enabled
                if self.keep_history:
                    self.history.append(StateEntry(
                        self.current_state.copy(),
                        metadata={"action": "load_checkpoint", "name": name}
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
        if not self.enable_persistence:
            return []
        
        try:
            checkpoints = []
            
            for filename in os.listdir(self.storage_path):
                if filename.startswith("checkpoint_") and filename.endswith(".json"):
                    checkpoint_name = filename[11:-5]  # Remove "checkpoint_" prefix and ".json" suffix
                    checkpoints.append(checkpoint_name)
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []