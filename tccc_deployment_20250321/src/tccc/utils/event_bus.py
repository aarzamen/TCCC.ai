"""
TCCC Event Bus Implementation

This module provides the event bus for inter-component communication
in the TCCC system using the standardized event schema.
"""

import time
import threading
import logging
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Set

from tccc.utils.logging import get_logger
from tccc.utils.event_schema import BaseEvent, EventType, ErrorSeverity, create_event

logger = get_logger(__name__)


class EventSubscription:
    """Manages a subscription to specific event types."""
    
    def __init__(
        self, 
        subscriber: str,
        event_types: List[Union[EventType, str]],
        callback: Callable[[BaseEvent], None],
        filter_fn: Optional[Callable[[BaseEvent], bool]] = None
    ):
        """
        Initialize event subscription.
        
        Args:
            subscriber: Subscriber identifier
            event_types: List of event types to subscribe to
            callback: Callback function to handle events
            filter_fn: Optional filter function to determine if an event should be delivered
        """
        self.subscriber = subscriber
        self.event_types = [et.value if isinstance(et, EventType) else et for et in event_types]
        self.callback = callback
        self.filter_fn = filter_fn


class EventBus:
    """
    Event bus for TCCC inter-component communication.
    
    Provides a pub-sub mechanism for components to communicate through
    standardized events. Supports filtering, async delivery, and more.
    """
    
    def __init__(self, async_delivery: bool = True, max_queue_size: int = 1000):
        """
        Initialize event bus.
        
        Args:
            async_delivery: Whether to deliver events asynchronously
            max_queue_size: Maximum size of the event queue for async delivery
        """
        self.subscriptions: Dict[str, List[EventSubscription]] = {}
        self.subscribers: Set[str] = set()
        
        # Async delivery settings
        self.async_delivery = async_delivery
        self.event_queue = queue.Queue(maxsize=max_queue_size)
        self.delivery_thread = None
        self.running = False
        
        # Thread lock for subscription modifications
        self.lock = threading.RLock()
        
        # Event stats
        self.stats = {
            'events_published': 0,
            'events_delivered': 0,
            'events_dropped': 0,
            'start_time': time.time()
        }
        
        # Start delivery thread if async mode is enabled
        if self.async_delivery:
            self._start_delivery_thread()
    
    def _start_delivery_thread(self):
        """Start the asynchronous event delivery thread."""
        if self.delivery_thread is not None and self.delivery_thread.is_alive():
            return
            
        self.running = True
        self.delivery_thread = threading.Thread(
            target=self._delivery_loop,
            name="EventBus-Delivery",
            daemon=True
        )
        self.delivery_thread.start()
        logger.info("Event bus delivery thread started")
    
    def _delivery_loop(self):
        """Main loop for asynchronous event delivery."""
        while self.running:
            try:
                # Get event from queue with timeout to allow for graceful shutdown
                try:
                    event, subscriptions = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Deliver event to all matching subscriptions
                self._deliver_event(event, subscriptions)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in event delivery thread: {e}")
                time.sleep(0.1)  # Prevent tight loop in case of repeated errors
    
    def stop(self):
        """Stop the event bus and clean up resources."""
        self.running = False
        
        if self.delivery_thread:
            self.delivery_thread.join(timeout=1.0)
            if self.delivery_thread.is_alive():
                logger.warning("Event delivery thread did not terminate cleanly")
        
        logger.info("Event bus stopped")
    
    def subscribe(
        self, 
        subscriber: str, 
        event_types: List[Union[EventType, str]], 
        callback: Callable[[BaseEvent], None],
        filter_fn: Optional[Callable[[BaseEvent], bool]] = None
    ) -> bool:
        """
        Subscribe to specific event types.
        
        Args:
            subscriber: Subscriber identifier
            event_types: List of event types to subscribe to
            callback: Callback function to handle events
            filter_fn: Optional filter function to determine if an event should be delivered
            
        Returns:
            Success status
        """
        if not event_types:
            logger.warning(f"Subscriber '{subscriber}' attempted to subscribe without event types")
            return False
            
        try:
            with self.lock:
                # Create subscription object
                subscription = EventSubscription(
                    subscriber=subscriber,
                    event_types=event_types,
                    callback=callback,
                    filter_fn=filter_fn
                )
                
                # Add subscriber to set of all subscribers
                self.subscribers.add(subscriber)
                
                # Append subscription for each event type
                for event_type in event_types:
                    event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
                    
                    if event_type_str not in self.subscriptions:
                        self.subscriptions[event_type_str] = []
                    
                    self.subscriptions[event_type_str].append(subscription)
                
                logger.info(f"Subscriber '{subscriber}' subscribed to event types: {event_types}")
                return True
                
        except Exception as e:
            logger.error(f"Error subscribing '{subscriber}' to events: {e}")
            return False
    
    def unsubscribe(self, subscriber: str, event_types: Optional[List[Union[EventType, str]]] = None) -> bool:
        """
        Unsubscribe from specific or all event types.
        
        Args:
            subscriber: Subscriber identifier
            event_types: List of event types to unsubscribe from (None for all)
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                # If no event types specified, unsubscribe from all
                if event_types is None:
                    # Remove from all event types
                    for event_type in list(self.subscriptions.keys()):
                        self.subscriptions[event_type] = [
                            sub for sub in self.subscriptions[event_type]
                            if sub.subscriber != subscriber
                        ]
                    
                    # Remove empty event type lists
                    for event_type in list(self.subscriptions.keys()):
                        if not self.subscriptions[event_type]:
                            del self.subscriptions[event_type]
                    
                    # Remove from set of subscribers
                    if subscriber in self.subscribers:
                        self.subscribers.remove(subscriber)
                    
                    logger.info(f"Subscriber '{subscriber}' unsubscribed from all events")
                    return True
                
                # Convert event types to strings
                event_type_strs = [et.value if isinstance(et, EventType) else et for et in event_types]
                
                # Remove from specified event types
                for event_type in event_type_strs:
                    if event_type in self.subscriptions:
                        self.subscriptions[event_type] = [
                            sub for sub in self.subscriptions[event_type]
                            if sub.subscriber != subscriber
                        ]
                        
                        # Remove empty event type list
                        if not self.subscriptions[event_type]:
                            del self.subscriptions[event_type]
                
                # Check if subscriber still has any subscriptions
                has_subscriptions = False
                for subs in self.subscriptions.values():
                    if any(sub.subscriber == subscriber for sub in subs):
                        has_subscriptions = True
                        break
                
                # If no more subscriptions, remove from set of subscribers
                if not has_subscriptions and subscriber in self.subscribers:
                    self.subscribers.remove(subscriber)
                
                logger.info(f"Subscriber '{subscriber}' unsubscribed from event types: {event_types}")
                return True
                
        except Exception as e:
            logger.error(f"Error unsubscribing '{subscriber}' from events: {e}")
            return False
    
    def publish(self, event: BaseEvent) -> bool:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
            
        Returns:
            Success status
        """
        if not event:
            logger.warning("Attempted to publish None event")
            return False
            
        try:
            # Get event type string
            event_type = event.type
            
            # Update stats
            self.stats['events_published'] += 1
            
            # Get matching subscriptions
            matching_subscriptions = []
            
            # Check if we have subscribers for this event type
            if event_type in self.subscriptions:
                matching_subscriptions.extend(self.subscriptions[event_type])
            
            # Check for wildcard subscribers
            if "*" in self.subscriptions:
                matching_subscriptions.extend(self.subscriptions["*"])
            
            # If no matching subscriptions, just return
            if not matching_subscriptions:
                return True
            
            # Deliver based on mode
            if self.async_delivery:
                try:
                    # Put event in queue for asynchronous delivery
                    self.event_queue.put((event, matching_subscriptions), block=False)
                except queue.Full:
                    logger.warning("Event queue full, dropping event")
                    self.stats['events_dropped'] += 1
                    return False
            else:
                # Deliver synchronously
                self._deliver_event(event, matching_subscriptions)
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    def _deliver_event(self, event: BaseEvent, subscriptions: List[EventSubscription]):
        """
        Deliver an event to matching subscriptions.
        
        Args:
            event: Event to deliver
            subscriptions: List of matching subscriptions
        """
        for subscription in subscriptions:
            try:
                # Apply filter if provided
                if subscription.filter_fn and not subscription.filter_fn(event):
                    continue
                
                # Deliver event
                subscription.callback(event)
                self.stats['events_delivered'] += 1
                
            except Exception as e:
                logger.error(f"Error delivering event to subscriber '{subscription.subscriber}': {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.
        
        Returns:
            Statistics dictionary
        """
        # Calculate uptime
        uptime = time.time() - self.stats['start_time']
        
        stats = {
            'events_published': self.stats['events_published'],
            'events_delivered': self.stats['events_delivered'],
            'events_dropped': self.stats['events_dropped'],
            'subscriber_count': len(self.subscribers),
            'subscription_count': sum(len(subs) for subs in self.subscriptions.values()),
            'event_types': list(self.subscriptions.keys()),
            'uptime_seconds': uptime,
            'events_per_second': self.stats['events_published'] / uptime if uptime > 0 else 0,
            'async_mode': self.async_delivery,
            'queue_size': self.event_queue.qsize() if self.async_delivery else 0
        }
        
        return stats


# Module-level event bus instance for singleton access
_default_event_bus = None


def get_event_bus(async_delivery: bool = True, max_queue_size: int = 1000) -> EventBus:
    """
    Get the default event bus instance (singleton pattern).
    
    Args:
        async_delivery: Whether to deliver events asynchronously
        max_queue_size: Maximum size of the event queue for async delivery
        
    Returns:
        EventBus instance
    """
    global _default_event_bus
    
    if _default_event_bus is None:
        _default_event_bus = EventBus(async_delivery=async_delivery, max_queue_size=max_queue_size)
        logger.info(f"Created default event bus (async={async_delivery})")
    
    return _default_event_bus