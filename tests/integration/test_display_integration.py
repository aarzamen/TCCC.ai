#!/usr/bin/env python3
"""
Integration test for the display interface with system components
"""

import os
import sys
import time
import unittest
import multiprocessing
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock environment - this allows testing without actual display
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

try:
    from tccc.display.display_interface import DisplayInterface
    from tccc.system.display_integration import DisplayIntegration
except ImportError as e:
    print(f"Error importing display modules: {e}")
    print("Make sure the TCCC package is installed and the environment is activated")
    sys.exit(1)

class TestDisplayIntegration(unittest.TestCase):
    """Test cases for display interface integration"""
    
    def setUp(self):
        """Set up test environment"""
        # Run display in a separate process to avoid issues with pygame display
        self.queue = multiprocessing.Queue()
        self.process = None
    
    def tearDown(self):
        """Clean up after tests"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
    
    def test_display_initialization(self):
        """Test that the display can initialize"""
        def run_display():
            try:
                display = DisplayInterface(width=800, height=480, fullscreen=False)
                result = display.initialize()
                self.queue.put(("init_result", result))
                if result:
                    self.queue.put(("message", "Display initialized successfully"))
                    display.stop()
            except Exception as e:
                self.queue.put(("error", str(e)))
        
        # Run in separate process
        self.process = multiprocessing.Process(target=run_display)
        self.process.start()
        
        # Wait for results
        try:
            results = {}
            timeout = time.time() + 10  # 10 second timeout
            
            while time.time() < timeout and "init_result" not in results:
                try:
                    key, value = self.queue.get(timeout=0.5)
                    results[key] = value
                except multiprocessing.queues.Empty:
                    pass
            
            # Verify results
            self.assertIn("init_result", results, "Display initialization result not received")
            self.assertTrue(results.get("init_result", False), "Display failed to initialize")
            
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
    
    def test_display_integration_creation(self):
        """Test that DisplayIntegration can be created"""
        def run_integration():
            try:
                integration = DisplayIntegration(
                    display_width=800,
                    display_height=480,
                    fullscreen=False
                )
                self.queue.put(("created", True))
                self.queue.put(("message", "Display integration created successfully"))
            except Exception as e:
                self.queue.put(("error", str(e)))
        
        # Run in separate process
        self.process = multiprocessing.Process(target=run_integration)
        self.process.start()
        
        # Wait for results
        try:
            results = {}
            timeout = time.time() + 10  # 10 second timeout
            
            while time.time() < timeout and "created" not in results:
                try:
                    key, value = self.queue.get(timeout=0.5)
                    results[key] = value
                except multiprocessing.queues.Empty:
                    pass
            
            # Verify results
            self.assertIn("created", results, "Display integration creation result not received")
            self.assertTrue(results.get("created", False), "Failed to create DisplayIntegration")
            
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
    
    def test_event_handling(self):
        """Test that events can be processed"""
        def run_event_test():
            try:
                # Create integration with dummy display
                integration = DisplayIntegration(
                    display_width=800, 
                    display_height=480,
                    fullscreen=False
                )
                
                # Mock the actual display to avoid UI
                integration.display.initialize = lambda: True
                integration.display.start = lambda: None
                integration.display.update_transcription = lambda text: self.queue.put(("transcription", text))
                integration.display.add_significant_event = lambda event: self.queue.put(("event", event))
                integration.display.update_card_data = lambda data: self.queue.put(("card_data", data))
                integration.display.set_display_mode = lambda mode: self.queue.put(("mode", mode))
                
                # Start integration
                integration.start()
                self.queue.put(("started", True))
                
                # Send events
                integration.on_transcription("Test transcription")
                integration.on_significant_event("Test event")
                integration.on_card_data_update({"name": "Test Patient"})
                integration.on_care_complete()
                
                # Let events process
                time.sleep(1)
                
                # Stop integration
                integration.stop()
                self.queue.put(("stopped", True))
                
            except Exception as e:
                self.queue.put(("error", str(e)))
        
        # Run in separate process
        self.process = multiprocessing.Process(target=run_event_test)
        self.process.start()
        
        # Wait for results
        try:
            results = {}
            events_received = []
            timeout = time.time() + 10  # 10 second timeout
            
            while time.time() < timeout and len(events_received) < 5:  # expect 5 events
                try:
                    key, value = self.queue.get(timeout=0.5)
                    if key in ["transcription", "event", "card_data", "mode"]:
                        events_received.append(key)
                    results[key] = value
                except multiprocessing.queues.Empty:
                    pass
            
            # Verify results
            self.assertIn("started", results, "Display integration failed to start")
            self.assertIn("transcription", results, "Transcription event not processed")
            self.assertIn("event", results, "Significant event not processed")
            self.assertIn("card_data", results, "Card data not processed")
            self.assertIn("mode", results, "Display mode not set")
            self.assertEqual(results.get("mode"), "card", "Display mode not set to card on care_complete")
            
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)


if __name__ == "__main__":
    unittest.main()