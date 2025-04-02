#!/usr/bin/env python3
"""
Event System Monitor for TCCC.

This script monitors events flowing through the system and displays them
in a readable format for debugging and verification.
"""

import os
import sys
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EventMonitor")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Dictionary to store event statistics
event_stats = {
    "total_events": 0,
    "events_by_type": {},
    "events_by_source": {},
    "start_time": datetime.now(),
}

# Lock for thread-safe access to stats
stats_lock = threading.Lock()

def subscribe_to_events():
    """Subscribe to all events in the system."""
    try:
        from tccc.utils.event_bus import get_event_bus
        from tccc.utils.event_schema import EventType
        
        event_bus = get_event_bus()
        if not event_bus:
            logger.error("Failed to get event bus")
            return False
        
        # Get all event types
        event_types = []
        if hasattr(EventType, '__members__'):
            # If EventType is an enum
            event_types = list(EventType.__members__.values())
        else:
            # If EventType is a class with constants
            event_types = [getattr(EventType, attr) for attr in dir(EventType) 
                         if not attr.startswith('_') and not callable(getattr(EventType, attr))]
        
        # Also add string versions of event types for compatibility
        event_types_str = ["audio_segment", "transcription", "processed_text", 
                          "error", "system_status", "command", "vital_signs"]
        
        # Subscribe to all event types
        for event_type in event_types:
            event_bus.subscribe("event_monitor", event_type, handle_event)
        
        # Subscribe to string event types for safety
        for event_type in event_types_str:
            try:
                event_bus.subscribe("event_monitor", event_type, handle_event)
            except:
                pass
        
        logger.info(f"Subscribed to {len(event_types)} event types")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import event bus: {e}")
        return False
    except Exception as e:
        logger.error(f"Error subscribing to events: {e}")
        return False

def handle_event(event: Any) -> None:
    """
    Handle and log events from the system.
    
    Args:
        event: The event object
    """
    try:
        # Get event type and source
        event_type = getattr(event, 'event_type', 'unknown')
        event_source = getattr(event, 'source', 'unknown')
        
        # Update stats
        with stats_lock:
            event_stats["total_events"] += 1
            
            # Update by type
            if event_type not in event_stats["events_by_type"]:
                event_stats["events_by_type"][event_type] = 0
            event_stats["events_by_type"][event_type] += 1
            
            # Update by source
            if event_source not in event_stats["events_by_source"]:
                event_stats["events_by_source"][event_source] = 0
            event_stats["events_by_source"][event_source] += 1
        
        # Handle specific event types
        if hasattr(event, 'event_type'):
            event_type_str = str(event.event_type)
            
            if 'transcription' in event_type_str.lower():
                # Handle transcription event
                text = getattr(event, 'text', 'No text available')
                is_partial = getattr(event, 'is_partial', False)
                confidence = getattr(event, 'confidence', 0.0)
                
                # Only log if not a partial result or significant content
                if not is_partial or len(text) > 20:
                    logger.info(f"📝 Transcription ({confidence:.2f}): {text}")
                
            elif 'audio' in event_type_str.lower():
                # Just log audio events without details
                duration_sec = 0
                if hasattr(event, 'audio_data') and hasattr(event.audio_data, 'shape'):
                    # Assuming 16kHz sample rate
                    duration_sec = len(event.audio_data) / 16000
                
                is_speech = getattr(event, 'is_speech', False)
                logger.debug(f"🔊 Audio: {duration_sec:.2f}s {'(speech)' if is_speech else '(non-speech)'}")
                
            elif 'error' in event_type_str.lower():
                # Handle error event
                error_msg = getattr(event, 'message', 'Unknown error')
                component = getattr(event, 'component', 'unknown')
                severity = getattr(event, 'severity', 'ERROR')
                
                logger.error(f"❌ Error [{component}] {severity}: {error_msg}")
                
            elif 'vital' in event_type_str.lower():
                # Handle vital signs event
                vitals = getattr(event, 'vitals', {})
                if vitals:
                    vital_str = ", ".join([f"{k}: {v}" for k, v in vitals.items()])
                    logger.info(f"💓 Vitals: {vital_str}")
                
            elif 'command' in event_type_str.lower():
                # Handle command event
                command = getattr(event, 'command', 'unknown')
                params = getattr(event, 'parameters', {})
                
                logger.info(f"🎮 Command: {command} - Params: {params}")
                
            elif 'status' in event_type_str.lower():
                # Handle status event - just log a marker
                logger.debug(f"ℹ️ Status update from {event_source}")
                
            else:
                # Generic event logging
                logger.debug(f"Event: {event_type} from {event_source}")
                
    except Exception as e:
        logger.error(f"Error handling event: {e}")

def print_stats():
    """Print event statistics periodically."""
    with stats_lock:
        runtime = datetime.now() - event_stats["start_time"]
        runtime_sec = runtime.total_seconds()
        
        total = event_stats["total_events"]
        rate = total / runtime_sec if runtime_sec > 0 else 0
        
        print("\n--- Event System Statistics ---")
        print(f"Runtime: {runtime.seconds} seconds")
        print(f"Total events: {total} ({rate:.1f} events/sec)")
        
        print("\nEvents by type:")
        for event_type, count in event_stats["events_by_type"].items():
            print(f"  {event_type}: {count}")
        
        print("\nEvents by source:")
        for source, count in event_stats["events_by_source"].items():
            print(f"  {source}: {count}")
        
        print("\nPress Ctrl+C to exit")

def stats_loop():
    """Background thread to periodically print stats."""
    while True:
        time.sleep(10)  # Print stats every 10 seconds
        print_stats()

def main():
    """Main entry point."""
    print("TCCC Event System Monitor")
    print("-------------------------")
    print("Monitoring events flowing through the system")
    
    # Subscribe to events
    if not subscribe_to_events():
        print("Failed to subscribe to events - check event bus")
        return 1
    
    # Start stats thread
    stats_thread = threading.Thread(target=stats_loop, daemon=True)
    stats_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down event monitor")
        print_stats()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())