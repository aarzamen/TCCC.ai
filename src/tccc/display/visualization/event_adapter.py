#!/usr/bin/env python3
"""
TCCC.ai Display Event Adapter
-----------------------------
Provides integration between the event bus system and visualization components.
Subscribes to relevant event types and transforms them into visualization-ready formats.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from tccc.utils.event_bus import get_event_bus, EventSubscription
from tccc.utils.event_schema import BaseEvent, EventType, LLMAnalysisEvent, TranscriptionEvent
from tccc.display.visualization.timeline import TimelineVisualization, TimelineEvent
from tccc.display.visualization.vital_signs import VitalSignsVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayEventAdapter")


class DisplayEventAdapter:
    """
    Adapter to connect event bus events to display components.
    Subscribes to relevant event types and updates visualization components.
    """
    
    def __init__(
        self,
        timeline: Optional[TimelineVisualization] = None,
        vitals: Optional[VitalSignsVisualization] = None,
        component_name: str = "display_adapter"
    ):
        """
        Initialize the display event adapter.
        
        Args:
            timeline: Timeline visualization component
            vitals: Vital signs visualization component
            component_name: Component name for event source identification
        """
        self.timeline = timeline
        self.vitals = vitals
        self.component_name = component_name
        self.event_bus = get_event_bus()
        self.subscriptions = []
        self.session_id = None
        self.last_event_time = time.time()
        
        # Register with event bus if components are provided
        if timeline or vitals:
            self.register()
            
    def set_timeline(self, timeline: TimelineVisualization):
        """Set the timeline visualization component."""
        self.timeline = timeline
        if not self.subscriptions:
            self.register()
            
    def set_vitals(self, vitals: VitalSignsVisualization):
        """Set the vital signs visualization component."""
        self.vitals = vitals
        if not self.subscriptions:
            self.register()
            
    def register(self):
        """Register with the event bus for relevant event types."""
        # Unregister first if already registered
        self.unregister()
        
        # Subscribe to relevant event types
        subscribed_events = [
            EventType.TRANSCRIPTION,
            EventType.LLM_ANALYSIS,
            EventType.PROCESSED_TEXT,
        ]
        
        # Register for each event type
        self.event_bus.subscribe(
            subscriber=self.component_name,
            event_types=subscribed_events,
            callback=self.handle_event
        )
        
        logger.info(f"Registered with event bus for event types: {subscribed_events}")
        
    def unregister(self):
        """Unregister from the event bus."""
        if not self.event_bus:
            return
            
        self.event_bus.unsubscribe(
            subscriber=self.component_name
        )
        self.subscriptions = []
        
    def handle_event(self, event: BaseEvent):
        """
        Handle incoming events from the event bus.
        
        Args:
            event: The event to handle
        """
        # Store session ID from first event
        if not self.session_id:
            self.session_id = event.session_id
            
        # Update last event time
        self.last_event_time = time.time()
        
        # Handle specific event types
        if event.type == EventType.TRANSCRIPTION.value:
            self._handle_transcription_event(event)
        elif event.type == EventType.LLM_ANALYSIS.value:
            self._handle_llm_analysis_event(event)
        elif event.type == EventType.PROCESSED_TEXT.value:
            self._handle_processed_text_event(event)
            
    def _handle_transcription_event(self, event: BaseEvent):
        """
        Handle transcription events from STT engine.
        
        Args:
            event: Transcription event
        """
        if not self.timeline:
            return
            
        # Extract text from transcription
        text = event.data.get("text", "")
        if not text:
            return
            
        # Add basic event to timeline for transcript
        # Use lowercased text for automatic categorization
        self.timeline.add_event_from_text(text, event_type="info")
        
        # Check for vital signs in the text and update if available
        if self.vitals:
            self._extract_vitals_from_text(text)
            
    def _handle_llm_analysis_event(self, event: BaseEvent):
        """
        Handle LLM analysis events for timeline visualization.
        
        Args:
            event: LLM analysis event
        """
        if not self.timeline:
            return
            
        # Extract key fields
        summary = event.data.get("summary", "")
        medical_terms = event.data.get("medical_terms", [])
        actions = event.data.get("actions", [])
        
        # Create an event for the summary
        if summary:
            self.timeline.add_event(TimelineEvent(
                timestamp=datetime.now(),
                event_type="assessment",
                title="Medical Assessment",
                description=summary,
                source="llm_analysis"
            ))
            
        # Create events for critical medical terms
        for term in medical_terms:
            term_text = term.get("term", "")
            category = term.get("category", "")
            severity = term.get("severity", "")
            
            if severity == "critical":
                self.timeline.add_event(TimelineEvent(
                    timestamp=datetime.now(),
                    event_type="critical",
                    title=f"Critical: {term_text}",
                    description=f"Critical {category}: {term_text}",
                    source="llm_analysis"
                ))
                
        # Create events for recommended actions
        for action in actions:
            action_type = action.get("type", "")
            priority = action.get("priority", "")
            description = action.get("description", "")
            
            event_type = "warning"
            if action_type == "treatment":
                event_type = "treatment"
            elif action_type == "medication":
                event_type = "medication"
            elif action_type == "transport":
                event_type = "transport"
                
            title = f"{action_type.title()}: {priority.title()}"
            
            self.timeline.add_event(TimelineEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                title=title,
                description=description,
                source="llm_analysis"
            ))
            
    def _handle_processed_text_event(self, event: BaseEvent):
        """
        Handle processed text events for visualization.
        
        Args:
            event: Processed text event
        """
        if not self.timeline:
            return
            
        # Extract entities and intent
        entities = event.data.get("entities", [])
        intent = event.data.get("intent", {})
        
        # Process medical entities
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_type = entity.get("type", "")
            
            # Only add significant medical entities to timeline
            if entity_type == "MEDICAL_CONDITION":
                self.timeline.add_event(TimelineEvent(
                    timestamp=datetime.now(),
                    event_type="warning",
                    title=f"Medical Condition: {entity_text}",
                    description=f"Detected medical condition: {entity_text}",
                    source="processing_core"
                ))
            elif entity_type == "MEDICATION":
                self.timeline.add_event(TimelineEvent(
                    timestamp=datetime.now(),
                    event_type="medication",
                    title=f"Medication: {entity_text}",
                    description=f"Medication mentioned: {entity_text}",
                    source="processing_core"
                ))
                
        # Process intent if it's a clear medical action
        if intent:
            intent_name = intent.get("name", "")
            confidence = intent.get("confidence", 0.0)
            slots = intent.get("slots", {})
            
            # Only add high-confidence, actionable intents
            if confidence > 0.7 and intent_name in ["report_medical_condition", "request_treatment"]:
                # Create a more descriptive title based on intent
                if intent_name == "report_medical_condition":
                    title = "Medical Condition Reported"
                    event_type = "assessment"
                else:
                    title = "Treatment Requested"
                    event_type = "treatment"
                    
                # Create description from slots
                description = ", ".join([f"{k}: {v}" for k, v in slots.items()])
                
                self.timeline.add_event(TimelineEvent(
                    timestamp=datetime.now(),
                    event_type=event_type,
                    title=title,
                    description=description,
                    source="processing_core"
                ))
                
    def _extract_vitals_from_text(self, text: str):
        """
        Extract vital signs from transcribed text and update vital signs visualization.
        
        Args:
            text: Transcribed text to extract vitals from
        """
        if not self.vitals:
            return
            
        # Simple regex-free pattern matching for vital signs
        # In a production system, this would use more sophisticated NLP
        text = text.lower()
        
        # Heart rate extraction
        hr_indicators = ["heart rate", "pulse", "hr", "bpm"]
        for indicator in hr_indicators:
            if indicator in text:
                # Find numeric values near the indicator
                words = text.split()
                for i, word in enumerate(words):
                    if word == indicator or word.startswith(indicator):
                        # Look for numbers before and after
                        for j in range(max(0, i-3), min(len(words), i+4)):
                            if words[j].isdigit():
                                try:
                                    hr_value = int(words[j])
                                    if 30 <= hr_value <= 200:  # Reasonable HR range
                                        self.vitals.update_vital("heart_rate", hr_value)
                                        break
                                except ValueError:
                                    pass
                                    
        # Similar pattern for other vitals
        # Blood pressure
        if "blood pressure" in text or "bp" in text.split():
            for i in range(len(text)):
                if i+7 < len(text) and text[i:i+7].isdigit() and '/' in text[i+7:i+10]:
                    try:
                        systolic = int(text[i:i+7].strip())
                        j = text.find('/', i)
                        if j > 0 and j < i+10:
                            k = j+1
                            while k < len(text) and (text[k].isdigit() or text[k].isspace()):
                                k += 1
                            diastolic = int(text[j+1:k].strip())
                            self.vitals.update_vital("blood_pressure", (systolic, diastolic))
                    except ValueError:
                        pass
                        
        # Temperature extraction
        temp_indicators = ["temperature", "temp", "degrees"]
        for indicator in temp_indicators:
            if indicator in text:
                words = text.split()
                for i, word in enumerate(words):
                    if word == indicator or word.startswith(indicator):
                        for j in range(max(0, i-3), min(len(words), i+4)):
                            try:
                                temp_value = float(words[j].replace("°", ""))
                                if 35 <= temp_value <= 43:  # Celsius range
                                    self.vitals.update_vital("temperature", temp_value)
                                    break
                                elif 95 <= temp_value <= 108:  # Fahrenheit range
                                    # Convert to Celsius
                                    celsius = (temp_value - 32) * 5/9
                                    self.vitals.update_vital("temperature", celsius)
                                    break
                            except ValueError:
                                pass
                                
        # Oxygen saturation extraction
        if "oxygen" in text or "o2" in text or "spo2" in text or "saturation" in text:
            words = text.split()
            for i, word in enumerate(words):
                if word in ["oxygen", "o2", "spo2", "saturation"]:
                    for j in range(max(0, i-3), min(len(words), i+4)):
                        try:
                            if "%" in words[j]:
                                spo2_value = int(words[j].replace("%", ""))
                                if 50 <= spo2_value <= 100:
                                    self.vitals.update_vital("spo2", spo2_value)
                                    break
                            elif words[j].isdigit():
                                spo2_value = int(words[j])
                                if 50 <= spo2_value <= 100:
                                    self.vitals.update_vital("spo2", spo2_value)
                                    break
                        except ValueError:
                            pass
                            
        # Respiratory rate extraction
        if "respiratory rate" in text or "respiration" in text or "breathing rate" in text or "rr" in text.split():
            words = text.split()
            for i, word in enumerate(words):
                if word in ["respiratory", "respiration", "breathing"] or (word == "rr" and i > 0):
                    for j in range(max(0, i-2), min(len(words), i+5)):
                        try:
                            if words[j].isdigit():
                                rr_value = int(words[j])
                                if 5 <= rr_value <= 60:  # Reasonable RR range
                                    self.vitals.update_vital("respiratory_rate", rr_value)
                                    break
                        except ValueError:
                            pass


# Example usage
if __name__ == "__main__":
    import pygame
    from tccc.display.visualization.timeline import TimelineVisualization
    from tccc.display.visualization.vital_signs import VitalSignsVisualization
    
    # Initialize pygame
    pygame.init()
    
    # Create a test window
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Display Event Adapter Test")
    
    # Create visualization components
    timeline_rect = pygame.Rect(50, 50, 700, 250)
    timeline = TimelineVisualization(screen, timeline_rect)
    
    vitals_rect = pygame.Rect(50, 320, 700, 250)
    vitals = VitalSignsVisualization(screen, vitals_rect)
    
    # Create the adapter
    adapter = DisplayEventAdapter(timeline, vitals)
    
    # Create a test LLM analysis event
    llm_event = {
        "type": EventType.LLM_ANALYSIS.value,
        "source": "llm_analysis",
        "timestamp": time.time(),
        "session_id": "test_session",
        "data": {
            "summary": "Patient presents with signs of tension pneumothorax requiring immediate intervention.",
            "topics": ["tension pneumothorax", "chest trauma", "respiratory distress"],
            "medical_terms": [
                {
                    "term": "tension pneumothorax",
                    "category": "medical_condition",
                    "severity": "critical",
                    "confidence": 0.95
                }
            ],
            "actions": [
                {
                    "type": "treatment",
                    "priority": "immediate",
                    "description": "Perform needle decompression followed by chest tube placement"
                }
            ]
        }
    }
    
    # Create a test transcription event
    transcription_event = {
        "type": EventType.TRANSCRIPTION.value,
        "source": "stt_engine",
        "timestamp": time.time(),
        "session_id": "test_session",
        "data": {
            "text": "Patient has severe respiratory distress with heart rate 130, blood pressure 90/60, and oxygen saturation 85%",
            "segments": [],
            "language": "en",
            "confidence": 0.92,
            "is_partial": False
        }
    }
    
    # Create the events from dictionaries
    llm_event_obj = BaseEvent.from_dict(llm_event)
    transcription_event_obj = BaseEvent.from_dict(transcription_event)
    
    # Simulate receiving events
    adapter.handle_event(transcription_event_obj)
    adapter.handle_event(llm_event_obj)
    
    # Create a clock for consistent frame rate
    clock = pygame.time.Clock()
    
    # Main loop
    running = True
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw visualizations
        timeline.draw()
        vitals.draw()
        
        # Update display
        pygame.display.flip()
        
        # Cap frame rate
        clock.tick(30)
    
    # Clean up
    pygame.quit()