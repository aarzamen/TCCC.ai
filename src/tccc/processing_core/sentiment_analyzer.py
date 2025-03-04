"""
Sentiment analysis module for TCCC.ai processing core.

This module provides sentiment analysis functionality for the Processing Core.
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalysis:
    """
    Represents sentiment analysis results for text.
    """
    
    def __init__(self, 
                 sentiment: str, 
                 score: float, 
                 compound_score: Optional[float] = None,
                 emotions: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment analysis results.
        
        Args:
            sentiment: The sentiment classification (e.g., "positive", "negative", "neutral").
            score: The confidence score for the sentiment.
            compound_score: A normalized compound score that represents overall sentiment.
            emotions: Dictionary mapping emotion labels to scores.
            metadata: Additional metadata about the sentiment analysis.
        """
        self.sentiment = sentiment
        self.score = score
        self.compound_score = compound_score
        self.emotions = emotions or {}
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the sentiment analysis to a dictionary.
        
        Returns:
            A dictionary representation of the sentiment analysis.
        """
        result = {
            "sentiment": self.sentiment,
            "score": self.score,
            "metadata": self.metadata
        }
        
        if self.compound_score is not None:
            result["compound_score"] = self.compound_score
        
        if self.emotions:
            result["emotions"] = self.emotions
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentAnalysis':
        """
        Create a sentiment analysis from a dictionary.
        
        Args:
            data: Dictionary containing sentiment analysis data.
            
        Returns:
            A SentimentAnalysis instance.
        """
        return cls(
            sentiment=data["sentiment"],
            score=data["score"],
            compound_score=data.get("compound_score"),
            emotions=data.get("emotions"),
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        result = f"SentimentAnalysis(sentiment='{self.sentiment}', score={self.score:.2f}"
        if self.compound_score is not None:
            result += f", compound={self.compound_score:.2f}"
        if self.emotions:
            emotion_str = ", ".join([f"{k}={v:.2f}" for k, v in self.emotions.items()])
            result += f", emotions={{{emotion_str}}}"
        result += ")"
        return result


class RuleBasedSentimentAnalyzer:
    """
    Simple rule-based sentiment analyzer using lexicon approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the rule-based sentiment analyzer.
        
        Args:
            config: Configuration for the analyzer.
        """
        self.config = config
        self.fine_grained = config.get("fine_grained", False)
        self.detect_emotions = config.get("detect_emotions", False)
        
        # Load lexicons
        self.positive_words = self._load_lexicon("positive_words.txt", config)
        self.negative_words = self._load_lexicon("negative_words.txt", config)
        self.emotion_lexicon = self._load_emotion_lexicon(config) if self.detect_emotions else {}
        
        # Default emotion detection is disabled
        self.emotions_to_detect = set(config.get("emotions", []))
        
        logger.info("Initialized rule-based sentiment analyzer")
    
    def _load_lexicon(self, filename: str, config: Dict[str, Any]) -> set:
        """
        Load a lexicon of words from a file.
        
        Args:
            filename: The name of the lexicon file.
            config: Configuration dictionary.
            
        Returns:
            A set of words from the lexicon.
        """
        lexicon_path = None
        model_path = config.get("model_path")
        
        # Try to find the lexicon file
        if model_path:
            lexicon_path = os.path.join(model_path, filename)
        
        words = set()
        if lexicon_path and os.path.exists(lexicon_path):
            try:
                with open(lexicon_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            words.add(line.lower())
                logger.info(f"Loaded {len(words)} words from {lexicon_path}")
            except Exception as e:
                logger.error(f"Failed to load lexicon from {lexicon_path}: {str(e)}")
                words = self._get_default_lexicon(filename)
        else:
            words = self._get_default_lexicon(filename)
        
        return words
    
    def _get_default_lexicon(self, filename: str) -> set:
        """
        Get a default lexicon when the lexicon file is not available.
        
        Args:
            filename: The name of the lexicon file.
            
        Returns:
            A default set of words.
        """
        if filename == "positive_words.txt":
            return {
                "good", "great", "excellent", "amazing", "wonderful", "fantastic", "terrific",
                "outstanding", "perfect", "awesome", "brilliant", "happy", "pleased", "satisfied",
                "enjoy", "like", "love", "best", "better", "success", "successful", "thank",
                "thanks", "appreciate", "recommended", "positive", "helpful", "glad", "impressed"
            }
        elif filename == "negative_words.txt":
            return {
                "bad", "terrible", "awful", "horrible", "poor", "worst", "disappointing",
                "disappointed", "fail", "failed", "failure", "problem", "issue", "error",
                "wrong", "difficult", "hard", "complicated", "confusing", "unhappy", "sad",
                "angry", "upset", "annoyed", "frustrated", "hate", "dislike", "negative",
                "useless", "waste", "expensive", "overpriced", "slow", "difficult"
            }
        else:
            return set()
    
    def _load_emotion_lexicon(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Load the emotion lexicon.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            A dictionary mapping emotions to lists of associated words.
        """
        emotion_lexicon = {}
        model_path = config.get("model_path")
        emotions = config.get("emotions", [])
        
        if not emotions:
            return {}
        
        # Try to load from file
        if model_path:
            emotion_path = os.path.join(model_path, "emotion_lexicon.json")
            if os.path.exists(emotion_path):
                try:
                    import json
                    with open(emotion_path, 'r') as f:
                        emotion_lexicon = json.load(f)
                    logger.info(f"Loaded emotion lexicon from {emotion_path}")
                    return emotion_lexicon
                except Exception as e:
                    logger.error(f"Failed to load emotion lexicon: {str(e)}")
        
        # Use default lexicon
        default_emotions = {
            "anger": ["angry", "mad", "furious", "outraged", "frustrated", "irritated", "annoyed"],
            "joy": ["happy", "delighted", "pleased", "glad", "joyful", "thrilled", "excited"],
            "sadness": ["sad", "unhappy", "depressed", "gloomy", "miserable", "disappointed"],
            "fear": ["afraid", "scared", "frightened", "terrified", "worried", "anxious"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
            "trust": ["trust", "believe", "confidence", "reliable", "dependable"],
            "anticipation": ["expect", "anticipate", "look forward", "await", "hopeful"]
        }
        
        # Only include emotions that are in the config
        emotion_lexicon = {k: v for k, v in default_emotions.items() if k in emotions}
        logger.info(f"Using default emotion lexicon for {len(emotion_lexicon)} emotions")
        
        return emotion_lexicon
    
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A SentimentAnalysis object with the results.
        """
        if not text.strip():
            return SentimentAnalysis("neutral", 0.5)
        
        # Preprocess text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        # Calculate sentiment score
        if total_words > 0:
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            
            # Calculate compound score
            compound_score = (positive_score - negative_score) / (positive_score + negative_score + 0.001)
            compound_score = max(-1.0, min(1.0, compound_score))  # Ensure in range [-1, 1]
            
            # Determine sentiment label
            if self.fine_grained:
                if compound_score > 0.5:
                    sentiment = "very_positive"
                    score = 0.8 + (compound_score - 0.5) * 0.4  # Map [0.5, 1.0] to [0.8, 1.0]
                elif compound_score > 0.1:
                    sentiment = "positive"
                    score = 0.6 + (compound_score - 0.1) * 0.5  # Map [0.1, 0.5] to [0.6, 0.8]
                elif compound_score > -0.1:
                    sentiment = "neutral"
                    score = 0.5 + compound_score * 0.5  # Map [-0.1, 0.1] to [0.45, 0.55]
                elif compound_score > -0.5:
                    sentiment = "negative"
                    score = 0.4 - (compound_score + 0.1) * 0.5  # Map [-0.5, -0.1] to [0.6, 0.4]
                else:
                    sentiment = "very_negative"
                    score = 0.2 - (compound_score + 0.5) * 0.4  # Map [-1.0, -0.5] to [0.4, 0.0]
            else:
                if compound_score > 0.1:
                    sentiment = "positive"
                    score = 0.6 + compound_score * 0.4  # Map [0.1, 1.0] to [0.64, 1.0]
                elif compound_score > -0.1:
                    sentiment = "neutral"
                    score = 0.5 + compound_score * 0.5  # Map [-0.1, 0.1] to [0.45, 0.55]
                else:
                    sentiment = "negative"
                    score = 0.4 + compound_score * 0.4  # Map [-1.0, -0.1] to [0.0, 0.36]
        else:
            # No words found
            sentiment = "neutral"
            score = 0.5
            compound_score = 0.0
        
        # Detect emotions if enabled
        emotions = None
        if self.detect_emotions and self.emotion_lexicon:
            emotions = {}
            for emotion, emotion_words in self.emotion_lexicon.items():
                matches = sum(1 for word in words if word in emotion_words)
                if total_words > 0:
                    emotions[emotion] = matches / total_words
                else:
                    emotions[emotion] = 0.0
        
        # Return sentiment analysis
        return SentimentAnalysis(
            sentiment=sentiment,
            score=score,
            compound_score=compound_score,
            emotions=emotions,
            metadata={
                "positive_words": positive_count,
                "negative_words": negative_count,
                "total_words": total_words
            }
        )


class TransformersSentimentAnalyzer:
    """
    Sentiment analyzer using transformer models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer-based sentiment analyzer.
        
        Args:
            config: Configuration for the analyzer.
        """
        self.config = config
        self.fine_grained = config.get("fine_grained", False)
        self.detect_emotions = config.get("detect_emotions", False)
        self.emotions_to_detect = set(config.get("emotions", []))
        
        # Configure hardware acceleration
        device = -1  # CPU by default
        if config.get("enable_cuda", False) and torch.cuda.is_available():
            device = config.get("cuda_device", 0)
            logger.info(f"Using CUDA device {device} for sentiment analysis")
        
        # Use quantized model if specified
        self.use_quantized = config.get("use_quantized", False)
        
        # Choose appropriate model
        model_name = config.get("model_name", "distilbert-base-uncased-finetuned-sst-2-english")
        
        try:
            # Load sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
            logger.info(f"Loaded sentiment analysis model: {model_name}")
            
            # Load emotion detection pipeline if enabled
            if self.detect_emotions and self.emotions_to_detect:
                # In a real implementation, we would use a dedicated emotion detection model
                # For simplicity, we're reusing the sentiment model here
                self.emotion_pipeline = None
                logger.info("Emotion detection is enabled, but using rule-based fallback")
                self.emotion_lexicon = self._load_default_emotion_lexicon()
        
        except Exception as e:
            logger.error(f"Failed to load transformer sentiment model: {str(e)}")
            raise
    
    def _load_default_emotion_lexicon(self) -> Dict[str, List[str]]:
        """
        Load a default emotion lexicon for rule-based emotion detection.
        
        Returns:
            A dictionary mapping emotions to lists of words.
        """
        default_emotions = {
            "anger": ["angry", "mad", "furious", "outraged", "frustrated", "irritated", "annoyed"],
            "joy": ["happy", "delighted", "pleased", "glad", "joyful", "thrilled", "excited"],
            "sadness": ["sad", "unhappy", "depressed", "gloomy", "miserable", "disappointed"],
            "fear": ["afraid", "scared", "frightened", "terrified", "worried", "anxious"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
            "trust": ["trust", "believe", "confidence", "reliable", "dependable"],
            "anticipation": ["expect", "anticipate", "look forward", "await", "hopeful"]
        }
        
        # Only include emotions that are in the config
        return {k: v for k, v in default_emotions.items() if k in self.emotions_to_detect}
    
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A SentimentAnalysis object with the results.
        """
        if not text.strip():
            return SentimentAnalysis("neutral", 0.5)
        
        try:
            # Get sentiment from transformer model
            results = self.sentiment_pipeline(text, truncation=True, max_length=512)
            result = results[0]  # Most models return a list with a single result
            
            sentiment = result["label"].lower()
            score = result["score"]
            
            # Map to our standard sentiment categories
            if sentiment == "positive" or sentiment == "positive" or sentiment == "5 stars":
                if self.fine_grained and score > 0.9:
                    sentiment = "very_positive"
                else:
                    sentiment = "positive"
            elif sentiment == "negative" or sentiment == "negative" or sentiment == "1 star":
                if self.fine_grained and score > 0.9:
                    sentiment = "very_negative"
                else:
                    sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Calculate compound score if not provided directly
            compound_score = None
            if sentiment == "positive" or sentiment == "very_positive":
                compound_score = score * 2 - 1  # Map [0.5, 1.0] to [0, 1]
            elif sentiment == "negative" or sentiment == "very_negative":
                compound_score = 1 - score * 2  # Map [0.5, 1.0] to [0, -1]
            else:
                # For neutral, use a small range around 0
                compound_score = (score - 0.5) * 0.2  # Map [0, 1] to [-0.1, 0.1]
            
            # Detect emotions if enabled
            emotions = None
            if self.detect_emotions and self.emotions_to_detect:
                if self.emotion_pipeline:
                    # Use dedicated emotion pipeline
                    emotion_results = self.emotion_pipeline(text, truncation=True)
                    emotions = {r["label"].lower(): r["score"] for r in emotion_results 
                               if r["label"].lower() in self.emotions_to_detect}
                else:
                    # Fallback to rule-based approach
                    emotions = {}
                    words = re.findall(r'\b\w+\b', text.lower())
                    total_words = len(words)
                    
                    for emotion, emotion_words in self.emotion_lexicon.items():
                        matches = sum(1 for word in words if word in emotion_words)
                        if total_words > 0:
                            emotions[emotion] = matches / total_words
                        else:
                            emotions[emotion] = 0.0
            
            # Return sentiment analysis
            return SentimentAnalysis(
                sentiment=sentiment,
                score=score,
                compound_score=compound_score,
                emotions=emotions
            )
            
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {str(e)}")
            # Return neutral as fallback
            return SentimentAnalysis("neutral", 0.5)


class SentimentAnalyzer:
    """
    Sentiment analyzer that can use different backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration for the analyzer.
        """
        self.config = config
        self.model_type = config.get("model_type", "rule_based")
        
        # Initialize the appropriate backend
        if self.model_type == "rule_based":
            self.analyzer = RuleBasedSentimentAnalyzer(config)
        elif self.model_type == "transformers":
            self.analyzer = TransformersSentimentAnalyzer(config)
        else:
            raise ValueError(f"Unsupported sentiment analyzer model type: {self.model_type}")
        
        logger.info(f"SentimentAnalyzer initialized with {self.model_type} backend")
    
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A SentimentAnalysis object with the results.
        """
        sentiment = await self.analyzer.analyze_sentiment(text)
        logger.debug(f"Sentiment analysis result: {sentiment}")
        return sentiment