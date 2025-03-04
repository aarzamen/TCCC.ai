"""
Intent classification module for TCCC.ai processing core.

This module provides intent classification functionality for the Processing Core.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class Intent:
    """
    Represents a detected intent in text.
    """
    
    def __init__(self, intent_type: str, confidence: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an intent.
        
        Args:
            intent_type: The type/category of the intent.
            confidence: The confidence score for this intent.
            metadata: Additional metadata about the intent.
        """
        self.intent_type = intent_type
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the intent to a dictionary.
        
        Returns:
            A dictionary representation of the intent.
        """
        return {
            "type": self.intent_type,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """
        Create an intent from a dictionary.
        
        Args:
            data: Dictionary containing intent data.
            
        Returns:
            An Intent instance.
        """
        return cls(
            intent_type=data["type"],
            confidence=data["confidence"],
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        return f"Intent(type='{self.intent_type}', confidence={self.confidence:.2f})"


class RuleBasedIntentClassifier:
    """
    Simple rule-based intent classifier using keywords and patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the rule-based intent classifier.
        
        Args:
            config: Configuration for the classifier.
        """
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.fallback_intent = config.get("fallback_intent", "unknown")
        self.rules = {}
        
        # Load rules from file if provided
        model_path = config.get("model_path")
        if model_path and os.path.exists(os.path.join(model_path, "intent_rules.json")):
            self._load_rules(os.path.join(model_path, "intent_rules.json"))
        else:
            # Default rules based on categories in config
            self._create_default_rules(config.get("categories", []))
    
    def _load_rules(self, rules_file: str) -> None:
        """
        Load intent rules from a JSON file.
        
        Args:
            rules_file: Path to the rules file.
        """
        try:
            with open(rules_file, 'r') as f:
                rules_data = json.load(f)
            
            for intent_type, rule_data in rules_data.items():
                patterns = rule_data.get("patterns", [])
                keywords = rule_data.get("keywords", [])
                compiled_patterns = []
                
                for pattern in patterns:
                    try:
                        compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                    except re.error as e:
                        logger.error(f"Invalid regex pattern for {intent_type}: {e}")
                
                self.rules[intent_type] = {
                    "patterns": compiled_patterns,
                    "keywords": [k.lower() for k in keywords]
                }
            
            logger.info(f"Loaded {len(self.rules)} intent rules from {rules_file}")
        
        except Exception as e:
            logger.error(f"Failed to load intent rules: {str(e)}")
            self._create_default_rules(self.config.get("categories", []))
    
    def _create_default_rules(self, categories: List[str]) -> None:
        """
        Create default rules based on the provided categories.
        
        Args:
            categories: List of intent categories.
        """
        default_rules = {
            "inquiry": {
                "keywords": ["what", "how", "tell me", "show", "explain", "details", "information"],
                "patterns": [r"^(?:can|could) you tell me", r"^(?:what|how|when)(?:\s+is|\s+are)"]
            },
            "complaint": {
                "keywords": ["unhappy", "disappointed", "issue", "problem", "wrong", "error", "failed"],
                "patterns": [r"not (?:working|happy|satisfied)", r"(?:issue|problem) with"]
            },
            "request": {
                "keywords": ["please", "I want", "I need", "give me", "send", "provide"],
                "patterns": [r"^(?:can|could) you (?:please |)(?:help|send|provide)"]
            },
            "escalation": {
                "keywords": ["supervisor", "manager", "escalate", "speak with someone else"],
                "patterns": [r"(?:speak|talk) to (?:your|a) (?:supervisor|manager)"]
            },
            "gratitude": {
                "keywords": ["thanks", "thank you", "appreciate", "grateful", "helpful"],
                "patterns": [r"^thank you", r"thanks for your (?:help|assistance)"]
            },
            "cancellation": {
                "keywords": ["cancel", "stop", "terminate", "end subscription", "no longer want"],
                "patterns": [r"(?:want|would like) to cancel", r"stop (?:my|the) (?:service|subscription)"]
            },
            "confirmation": {
                "keywords": ["confirm", "yes", "correct", "right", "agreed", "I agree"],
                "patterns": [r"^yes,? (?:please|that's right)", r"^(?:that's|that is) correct"]
            }
        }
        
        # Only use rules for categories that were provided
        if categories:
            self.rules = {k: v for k, v in default_rules.items() if k in categories}
        else:
            self.rules = default_rules
        
        for rule_type, rule_data in self.rules.items():
            patterns = rule_data.get("patterns", [])
            compiled_patterns = []
            
            for pattern in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.error(f"Invalid regex pattern for {rule_type}: {e}")
            
            self.rules[rule_type]["patterns"] = compiled_patterns
        
        logger.info(f"Created {len(self.rules)} default intent rules")
    
    async def classify_intent(self, text: str) -> List[Intent]:
        """
        Classify the intent in the given text.
        
        Args:
            text: The text to classify.
            
        Returns:
            A list of detected intents, sorted by confidence.
        """
        if not text.strip():
            return [Intent(self.fallback_intent, 0.0)]
        
        text_lower = text.lower()
        matches = []
        
        # Check each intent type
        for intent_type, rule_data in self.rules.items():
            score = 0.0
            match_details = {}
            
            # Check for keyword matches
            keyword_matches = []
            for keyword in rule_data["keywords"]:
                if keyword in text_lower:
                    keyword_matches.append(keyword)
                    score += 0.1  # Each keyword adds to the score
            
            if keyword_matches:
                match_details["keyword_matches"] = keyword_matches
            
            # Check for pattern matches
            pattern_matches = []
            for pattern in rule_data["patterns"]:
                if pattern.search(text):
                    pattern_matches.append(pattern.pattern)
                    score += 0.3  # Patterns have higher weight than keywords
            
            if pattern_matches:
                match_details["pattern_matches"] = pattern_matches
            
            # Cap score at 1.0
            score = min(score, 1.0)
            
            # Add to matches if above threshold
            if score >= self.confidence_threshold:
                matches.append(Intent(
                    intent_type=intent_type,
                    confidence=score,
                    metadata=match_details if match_details else None
                ))
        
        # If no matches above threshold, return fallback
        if not matches:
            return [Intent(self.fallback_intent, 0.0)]
        
        # Sort by confidence, highest first
        return sorted(matches, key=lambda x: x.confidence, reverse=True)


class NeuralIntentClassifier:
    """
    Neural intent classifier using transformer models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the neural intent classifier.
        
        Args:
            config: Configuration for the classifier.
        """
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.fallback_intent = config.get("fallback_intent", "unknown")
        self.categories = config.get("categories", [])
        
        # Default model for intent classification
        model_path = config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            # Use a default model if no custom model is provided
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            # In a real implementation, we'd use a model fine-tuned for intent classification
        else:
            model_name = model_path
        
        try:
            # Configure hardware acceleration
            device = -1  # CPU by default
            if config.get("enable_cuda", False) and torch.cuda.is_available():
                device = config.get("cuda_device", 0)
                logger.info(f"Using CUDA device {device} for intent classification")
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Apply quantization if enabled
            if config.get("use_quantized", False) and device != -1:
                logger.info("Applying quantization to intent classifier model")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            # Move model to correct device
            if device != -1:
                self.model = self.model.to(f"cuda:{device}")
            
            # Load ID to label mapping
            id2label_path = os.path.join(model_path, "id2label.json") if model_path else None
            if id2label_path and os.path.exists(id2label_path):
                with open(id2label_path, 'r') as f:
                    self.id2label = json.load(f)
            else:
                # Fallback mapping (would be different in a real implementation)
                self.id2label = {0: "negative", 1: "positive"}
            
            logger.info(f"Loaded neural intent classifier model: {model_name}")
        
        except Exception as e:
            logger.error(f"Failed to load neural intent classifier: {str(e)}")
            raise
    
    async def classify_intent(self, text: str) -> List[Intent]:
        """
        Classify the intent in the given text.
        
        Args:
            text: The text to classify.
            
        Returns:
            A list of detected intents, sorted by confidence.
        """
        if not text.strip():
            return [Intent(self.fallback_intent, 0.0)]
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to same device as model
            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to intents
            intents = []
            for i, prob in enumerate(probs[0]):
                confidence = prob.item()
                intent_type = self.id2label.get(str(i), f"intent_{i}")
                
                if confidence >= self.confidence_threshold:
                    intents.append(Intent(
                        intent_type=intent_type,
                        confidence=confidence
                    ))
            
            # If no intents above threshold, return fallback
            if not intents:
                return [Intent(self.fallback_intent, 0.0)]
            
            # Sort by confidence, highest first
            return sorted(intents, key=lambda x: x.confidence, reverse=True)
        
        except Exception as e:
            logger.error(f"Error in neural intent classification: {str(e)}")
            return [Intent(self.fallback_intent, 0.0)]


class IntentClassifier:
    """
    Intent classifier that can use different backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the intent classifier.
        
        Args:
            config: Configuration for the classifier.
        """
        self.config = config
        self.model_type = config.get("model_type", "rule_based")
        
        # Initialize the appropriate backend
        if self.model_type == "rule_based":
            self.classifier = RuleBasedIntentClassifier(config)
        elif self.model_type == "neural":
            self.classifier = NeuralIntentClassifier(config)
        else:
            raise ValueError(f"Unsupported intent classifier model type: {self.model_type}")
        
        logger.info(f"IntentClassifier initialized with {self.model_type} backend")
    
    async def identify_intents(self, text: str) -> List[Intent]:
        """
        Identify intents in the given text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A list of identified intents, sorted by confidence.
        """
        intents = await self.classifier.classify_intent(text)
        logger.debug(f"Identified intents: {intents}")
        return intents