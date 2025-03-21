#!/usr/bin/env python3
"""
Simplified verification script for event schema.
This script verifies the core functionality of the event system.
"""

import os
import sys
import time
import threading
import logging
import queue
from typing import Dict, List, Any

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import TCCC modules
from src.tccc.utils.event_schema import BaseEvent, EventType, create_event
from src.tccc.utils.event_bus import get_event_bus

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EventChecker:
    """Simple class to check event reception."""
    
    def __init__(self):
        """Initialize the event checker."""
        self.received_events = []
        self.event_received = threading.Event()
        
    def handle_event(self, event: BaseEvent):
        """Handle received events."""
        self.received_events.append(event)
        logger.info(f"Received event: {event.type} from {event.source}")
        self.event_received.set()
        
    def wait_for_event(self, timeout: float = 2.0) -> bool:
        """
        Wait for any event to be received.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if event was received, False if timeout
        """
        return self.event_received.wait(timeout)

def test_event_bus_core_functionality():
    """Test the core event bus functionality."""
    logger.info("\n=== Testing core event bus functionality ===")
    
    # Create event bus
    event_bus = get_event_bus()
    
    # Create event checker
    checker = EventChecker()
    
    # Subscribe to events
    event_bus.subscribe(
        subscriber="test_subscriber",
        event_types=["test_event"],
        callback=checker.handle_event
    )
    
    # Create and publish a test event
    test_event = create_event(
        event_type="test_event",
        source="verification_script",
        data={"message": "Test event from verification script"}
    )
    
    logger.info("Publishing test event...")
    event_bus.publish(test_event)
    
    # Wait for the event to be received
    if checker.wait_for_event(timeout=2.0):
        logger.info("✓ Test event successfully received")
        return True
    else:
        logger.error("✗ Test event was not received")
        return False

def test_event_bus_wildcard_subscription():
    """Test wildcard subscription functionality."""
    logger.info("\n=== Testing wildcard subscription ===")
    
    # Create event bus
    event_bus = get_event_bus()
    
    # Create event checker
    checker = EventChecker()
    
    # Subscribe to all events with wildcard
    event_bus.subscribe(
        subscriber="wildcard_subscriber",
        event_types=["*"],
        callback=checker.handle_event
    )
    
    # Create and publish events of different types
    event_types = ["type1", "type2", "type3"]
    
    for event_type in event_types:
        event = create_event(
            event_type=event_type,
            source="verification_script",
            data={"message": f"Event of type {event_type}"}
        )
        
        logger.info(f"Publishing event of type {event_type}...")
        event_bus.publish(event)
        time.sleep(0.1)  # Small delay between events
    
    # Wait for events to be received
    time.sleep(1.0)
    
    # Check if all events were received
    received_types = [event.type for event in checker.received_events]
    
    for event_type in event_types:
        if event_type not in received_types:
            logger.error(f"✗ Event of type {event_type} was not received")
            return False
    
    logger.info(f"✓ All events successfully received via wildcard subscription")
    return True

def test_event_bus_filtering():
    """Test event filtering functionality."""
    logger.info("\n=== Testing event filtering ===")
    
    # Create event bus
    event_bus = get_event_bus()
    
    # Test events counter
    even_counter = 0
    
    # Create a filter function
    def only_even_ids(event: BaseEvent) -> bool:
        event_id = event.data.get("id", 0)
        return event_id % 2 == 0
    
    # Event handler that counts events
    def handle_filtered_event(event: BaseEvent):
        nonlocal even_counter
        even_counter += 1
        logger.info(f"Received filtered event with ID: {event.data.get('id')}")
    
    # Subscribe with filter
    event_bus.subscribe(
        subscriber="filter_subscriber",
        event_types=["filtered_event"],
        callback=handle_filtered_event,
        filter_fn=only_even_ids
    )
    
    # Publish 10 events with different IDs
    for i in range(10):
        event = create_event(
            event_type="filtered_event",
            source="verification_script",
            data={"id": i, "message": f"Event with ID {i}"}
        )
        
        event_bus.publish(event)
    
    # Wait for events to be processed
    time.sleep(1.0)
    
    # Check if only events with even IDs were received
    expected_count = 5  # Events with IDs 0, 2, 4, 6, 8
    
    if even_counter == expected_count:
        logger.info(f"✓ Event filtering working correctly (received {even_counter}/{expected_count} events)")
        return True
    else:
        logger.error(f"✗ Event filtering failed (received {even_counter} events, expected {expected_count})")
        return False

def test_event_bus_async_delivery():
    """Test asynchronous delivery of events."""
    logger.info("\n=== Testing asynchronous event delivery ===")
    
    # Create event bus with async delivery
    event_bus = get_event_bus(async_delivery=True)
    
    # Create a queue for received events
    event_queue = queue.Queue()
    
    # Event handler that adds to queue
    def handle_async_event(event: BaseEvent):
        logger.info(f"Received async event: {event.data.get('message')}")
        event_queue.put(event)
    
    # Subscribe to async events
    event_bus.subscribe(
        subscriber="async_subscriber",
        event_types=["async_event"],
        callback=handle_async_event
    )
    
    # Publish a batch of events rapidly
    event_count = 5
    
    for i in range(event_count):
        event = create_event(
            event_type="async_event",
            source="verification_script",
            data={"message": f"Async event {i}"}
        )
        
        event_bus.publish(event)
    
    # Wait for events to be delivered
    time.sleep(1.0)
    
    # Check if all events were received
    received_count = event_queue.qsize()
    
    if received_count == event_count:
        logger.info(f"✓ Async delivery working correctly (received {received_count}/{event_count} events)")
        return True
    else:
        logger.error(f"✗ Async delivery failed (received {received_count} events, expected {event_count})")
        return False

def run_all_tests():
    """Run all event schema tests."""
    logger.info("Starting event schema verification...")
    
    tests = [
        ("Core Functionality", test_event_bus_core_functionality),
        ("Wildcard Subscription", test_event_bus_wildcard_subscription),
        ("Event Filtering", test_event_bus_filtering),
        ("Async Delivery", test_event_bus_async_delivery)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Error running test {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n=== Event Schema Verification Results ===")
    all_passed = True
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\nEvent system verification complete: All tests PASSED")
        return 0
    else:
        logger.error("\nEvent system verification complete: Some tests FAILED")
        return 1

if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        sys.exit(1)