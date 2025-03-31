#!/usr/bin/env python3
"""
TCCC.ai Display Integration
--------------------------
Connects the display interface with the STT engine, LLM analysis,
and other components of the TCCC.ai system.
"""

import os
import sys
import time
import logging
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayIntegration")

# Try to import TCCC modules
try:
    from tccc.display.display_interface import DisplayInterface
    from tccc.stt_engine.stt_engine import STTEngine
    from tccc.llm_analysis.llm_analysis import LLMAnalysis
    from tccc.form_generator.form_generator import FormGenerator
except ImportError as e:
    logger.error(f"Failed to import TCCC modules: {e}")
    logger.info("This module should be run from within the TCCC project environment")
    sys.exit(1)

class DisplayIntegration:
    """
    Integrates the display interface with other TCCC.ai components.
    Acts as a listener that watches for events from various components
    and updates the display accordingly.
    """
    
    def __init__(self, 
                config: Optional[Dict] = None, 
                display_width: int = 800, 
                display_height: int = 480,
                fullscreen: bool = False):
        """
        Initialize the display integration.
        
        Args:
            config: Configuration dictionary
            display_width: Width of the display in pixels
            display_height: Height of the display in pixels
            fullscreen: Whether to run in fullscreen mode
        """
        self.config = config or {}
        
        # Create the display interface
        logger.info(f"Creating display with size {display_width}x{display_height}")
        self.display = DisplayInterface(
            width=display_width, 
            height=display_height,
            fullscreen=fullscreen
        )
        
        # Queue for events from other components
        self.event_queue = queue.Queue()
        
        # State
        self.care_complete = False
        self.active = False
        self.processing_thread = None
        
    def start(self):
        """Start the display integration and UI"""
        if self.active:
            logger.warning("Display integration already active")
            return
            
        # Start the display
        self.display.start()
        
        # Start the event processing thread
        self.active = True
        self.processing_thread = threading.Thread(target=self._event_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Display integration started")
        
    def stop(self):
        """Stop the display integration and UI"""
        self.active = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
        # Stop the display
        self.display.stop()
        logger.info("Display integration stopped")
        
    def _event_processing_loop(self):
        """Process events from the queue and update the display"""
        while self.active:
            try:
                # Get an event from the queue with a 0.1s timeout
                try:
                    event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                # Process the event based on type
                event_type = event.get('type', '')
                
                if event_type == 'transcription':
                    # Add transcription text to the display
                    text = event.get('text', '')
                    if text:
                        self.display.update_transcription(text)
                        
                elif event_type == 'significant_event':
                    # Add a significant event to the display
                    self.display.add_significant_event({
                        'time': event.get('time', datetime.now().strftime("%H:%M:%S")),
                        'description': event.get('description', '')
                    })
                    
                elif event_type == 'card_data':
                    # Update the TCCC card data
                    self.display.update_card_data(event.get('data', {}))
                    
                elif event_type == 'care_complete':
                    # Mark care as complete and switch to card view
                    self.care_complete = True
                    self.display.set_display_mode('card')
                    
                elif event_type == 'display_mode':
                    # Set the display mode
                    self.display.set_display_mode(event.get('mode', 'live'))
                    
                # Mark the event as processed
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing display event: {e}")
                
    def on_transcription(self, text: str):
        """
        Handle new transcription from STT engine
        
        Args:
            text: Transcribed text
        """
        if not text.strip():
            return
            
        self.event_queue.put({
            'type': 'transcription',
            'text': text
        })
        
    def on_significant_event(self, description: str, time_str: Optional[str] = None):
        """
        Handle a significant event from LLM analysis
        
        Args:
            description: Description of the event
            time_str: Optional timestamp (defaults to current time)
        """
        self.event_queue.put({
            'type': 'significant_event',
            'description': description,
            'time': time_str or datetime.now().strftime("%H:%M:%S")
        })
        
    def on_card_data_update(self, card_data: Dict[str, Any]):
        """
        Handle TCCC card data updates from form generator
        
        Args:
            card_data: Dictionary of card data fields
        """
        self.event_queue.put({
            'type': 'card_data',
            'data': card_data
        })
        
    def on_care_complete(self):
        """Handle care complete signal"""
        self.event_queue.put({
            'type': 'care_complete'
        })
        
    def set_display_mode(self, mode: str):
        """
        Set the display mode
        
        Args:
            mode: Either 'live' or 'card'
        """
        self.event_queue.put({
            'type': 'display_mode',
            'mode': mode
        })
        
    @staticmethod
    def connect_to_system(system, display_integration):
        """
        Connect the display integration to system components
        
        Args:
            system: The TCCC.ai system instance
            display_integration: The DisplayIntegration instance
        """
        # Connect to STT engine
        if hasattr(system, 'stt_engine'):
            logger.info("Connecting display to STT engine")
            original_transcribe_cb = system.stt_engine.on_transcription_complete
            
            def transcription_callback(text):
                # Call the original callback
                if original_transcribe_cb:
                    original_transcribe_cb(text)
                # Update the display
                display_integration.on_transcription(text)
                
            system.stt_engine.on_transcription_complete = transcription_callback
            
        # Connect to LLM analysis
        if hasattr(system, 'llm_analysis'):
            logger.info("Connecting display to LLM analysis")
            original_event_cb = system.llm_analysis.on_significant_event
            
            def event_callback(event):
                # Call the original callback
                if original_event_cb:
                    original_event_cb(event)
                # Update the display
                display_integration.on_significant_event(event)
                
            system.llm_analysis.on_significant_event = event_callback
            
        # Connect to form generator if available
        if hasattr(system, 'form_generator'):
            logger.info("Connecting display to form generator")
            original_form_cb = system.form_generator.on_form_update
            
            def form_callback(form_data):
                # Call the original callback
                if original_form_cb:
                    original_form_cb(form_data)
                # Update the display
                display_integration.on_card_data_update(form_data)
                
            system.form_generator.on_form_update = form_callback
        
        logger.info("Display integration connected to system components")


# Example usage
if __name__ == "__main__":
    # Simple demonstration without actual system components
    
    display_integration = DisplayIntegration(
        display_width=800,
        display_height=480,
        fullscreen=False
    )
    
    display_integration.start()
    
    try:
        # Simulate STT transcriptions
        display_integration.on_transcription("Starting patient assessment")
        time.sleep(1)
        
        display_integration.on_transcription("I have a patient with a gunshot wound to the left thigh")
        time.sleep(1)
        
        # Simulate significant events from LLM
        display_integration.on_significant_event("Patient care started")
        time.sleep(0.5)
        display_integration.on_significant_event("GSW identified in left thigh")
        time.sleep(1)
        
        display_integration.on_transcription("Checking for arterial bleeding... yes, there's bright red blood")
        time.sleep(1)
        display_integration.on_significant_event("Arterial bleeding identified")
        
        # Simulate more transcription and events
        display_integration.on_transcription("I'm applying a tourniquet now")
        time.sleep(1)
        display_integration.on_significant_event("Tourniquet applied to left thigh")
        
        display_integration.on_transcription("Checking vital signs")
        time.sleep(1)
        display_integration.on_transcription("BP is 100/60, pulse is 110, respiratory rate is 22")
        
        # Simulate card data updates
        display_integration.on_card_data_update({
            "name": "John Doe",
            "rank": "SGT",
            "unit": "1st Battalion, 3rd Marines",
            "mechanism_of_injury": "GSW",
            "injuries": "GSW left thigh, arterial bleeding",
            "treatment_given": "Tourniquet applied to left thigh at 14:32",
            "vital_signs": "HR 110, BP 100/60, RR 22"
        })
        
        time.sleep(1)
        
        display_integration.on_transcription("Administering 10mg morphine IV")
        time.sleep(1)
        display_integration.on_significant_event("Medication given: Morphine 10mg IV")
        
        # Update card with medication
        display_integration.on_card_data_update({
            "medications": "Morphine 10mg IV at 14:35"
        })
        
        time.sleep(1)
        display_integration.on_transcription("Patient is stable, care complete")
        time.sleep(1)
        display_integration.on_significant_event("Patient stabilized, treatment complete")
        
        # Final card update
        display_integration.on_card_data_update({
            "evacuation_priority": "Urgent"
        })
        
        # Signal care is complete
        time.sleep(2)
        display_integration.on_care_complete()
        
        # Wait for user to exit
        print("Press Ctrl+C to exit demonstration")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Demonstration interrupted")
    finally:
        display_integration.stop()