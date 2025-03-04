"""
Unit tests for the Processing Core module.
"""

import os
import pytest
import asyncio
from typing import Dict, Any

from tccc.processing_core.processing_core import ProcessingCore, TranscriptionSegment
from tccc.processing_core.entity_extractor import Entity
from tccc.processing_core.intent_classifier import Intent
from tccc.processing_core.sentiment_analyzer import SentimentAnalysis


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Fixture for mock configuration."""
    return {
        "general": {
            "enabled": True,
            "debug": True,
            "processing_mode": "async",
            "max_concurrent_tasks": 2,
            "processing_timeout_ms": 1000
        },
        "entity_extraction": {
            "model_type": "spacy",
            "spacy_model": "en_core_web_sm",
            "confidence_threshold": 0.5,
            "custom_entity_types": [
                {"name": "TEST_ENTITY", "pattern": "test\\d+"}
            ]
        },
        "intent_classification": {
            "model_type": "rule_based",
            "confidence_threshold": 0.5,
            "categories": [
                "inquiry",
                "request",
                "gratitude"
            ],
            "fallback_intent": "unknown"
        },
        "sentiment_analysis": {
            "model_type": "rule_based",
            "fine_grained": False,
            "detect_emotions": False
        },
        "resource_management": {
            "enable_monitoring": False
        },
        "plugins": {
            "enabled": False
        },
        "state_management": {
            "enable_persistence": False
        }
    }


@pytest.fixture
def processing_core(mock_config) -> ProcessingCore:
    """Fixture for a processing core instance."""
    core = ProcessingCore()
    # Run the async initialization in a separate event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(core.initialize(mock_config))
    loop.close()
    
    # Return the initialized core
    yield core
    
    # Clean up
    core.shutdown()


class TestProcessingCore:
    """Tests for the ProcessingCore class."""
    
    @pytest.mark.asyncio
    async def test_initialize(self, mock_config):
        """Test initialization of ProcessingCore."""
        core = ProcessingCore()
        result = await core.initialize(mock_config)
        
        assert result is True
        assert core.initialized is True
        
        core.shutdown()
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, processing_core):
        """Test entity extraction."""
        text = "Apple is looking at buying a company for $1 billion in the UK."
        entities = await processing_core.extractEntities(text)
        
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert all(isinstance(entity, Entity) for entity in entities)
    
    @pytest.mark.asyncio
    async def test_identify_intents(self, processing_core):
        """Test intent identification."""
        text = "Can you please help me with my account?"
        intents = await processing_core.identifyIntents(text)
        
        assert isinstance(intents, list)
        assert len(intents) > 0
        assert all(isinstance(intent, Intent) for intent in intents)
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, processing_core):
        """Test sentiment analysis."""
        text = "I'm really happy with the service you provided!"
        sentiment = await processing_core.analyzeSentiment(text)
        
        assert isinstance(sentiment, SentimentAnalysis)
        assert sentiment.sentiment in ["positive", "neutral", "negative", "very_positive", "very_negative"]
        assert 0.0 <= sentiment.score <= 1.0
    
    @pytest.mark.asyncio
    async def test_process_transcription(self, processing_core):
        """Test transcription processing."""
        segment = TranscriptionSegment(
            text="I need help with my account please.",
            speaker="customer",
            start_time=0.0,
            end_time=3.5,
            confidence=0.95,
            is_final=True,
            metadata={
                "conversation_id": "test-conv-1234",
                "session_id": "test-session-5678"
            }
        )
        
        processed = await processing_core.processTranscription(segment)
        
        assert processed.text == segment.text
        assert processed.speaker == segment.speaker
        assert processed.entities is not None
        assert processed.intents is not None
        assert processed.sentiment is not None
        assert "conversation_id" in processed.metadata
    
    def test_get_processing_metrics(self, processing_core):
        """Test getting processing metrics."""
        metrics = processing_core.getProcessingMetrics()
        
        assert isinstance(metrics, dict)
        assert "segments_processed" in metrics
        assert "entities_extracted" in metrics
        assert "intents_identified" in metrics
        assert "sentiment_analyzed" in metrics
        assert "average_processing_time" in metrics
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, processing_core):
        """Test handling of empty input."""
        # Test empty text for entity extraction
        entities = await processing_core.extractEntities("")
        assert isinstance(entities, list)
        assert len(entities) == 0
        
        # Test empty text for intent classification
        intents = await processing_core.identifyIntents("")
        assert isinstance(intents, list)
        assert len(intents) == 1  # Should return unknown intent
        assert intents[0].intent_type == "unknown"
        
        # Test empty text for sentiment analysis
        sentiment = await processing_core.analyzeSentiment("")
        assert isinstance(sentiment, SentimentAnalysis)
        assert sentiment.sentiment == "neutral"
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_config):
        """Test shutdown procedure."""
        core = ProcessingCore()
        await core.initialize(mock_config)
        
        assert core.initialized is True
        
        core.shutdown()
        assert core.initialized is False