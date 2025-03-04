"""
Entity extraction module for TCCC.ai processing core.

This module provides entity extraction functionality for the Processing Core.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class Entity:
    """
    Represents an extracted entity from text.
    """
    
    def __init__(self, 
                 text: str, 
                 entity_type: str, 
                 start: int, 
                 end: int, 
                 confidence: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an entity.
        
        Args:
            text: The text of the entity.
            entity_type: The type/category of the entity.
            start: The start character position in the original text.
            end: The end character position in the original text.
            confidence: The confidence score for this entity.
            metadata: Additional metadata about the entity.
        """
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary.
        
        Returns:
            A dictionary representation of the entity.
        """
        return {
            "text": self.text,
            "type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """
        Create an entity from a dictionary.
        
        Args:
            data: Dictionary containing entity data.
            
        Returns:
            An Entity instance.
        """
        return cls(
            text=data["text"],
            entity_type=data["type"],
            start=data["start"],
            end=data["end"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
    
    def __repr__(self) -> str:
        return f"Entity(text='{self.text}', type='{self.entity_type}', confidence={self.confidence:.2f})"


class CustomPatternMatcher:
    """
    Custom pattern matcher for entity extraction using regular expressions.
    """
    
    def __init__(self, patterns: List[Dict[str, str]]):
        """
        Initialize the pattern matcher.
        
        Args:
            patterns: List of patterns with format {"name": "ENTITY_TYPE", "pattern": "regex_pattern"}
        """
        self.patterns = []
        for pattern_def in patterns:
            try:
                self.patterns.append((
                    pattern_def["name"],
                    re.compile(pattern_def["pattern"])
                ))
                logger.info(f"Loaded custom pattern: {pattern_def['name']}")
            except re.error as e:
                logger.error(f"Invalid regex pattern for {pattern_def['name']}: {e}")
    
    def find_matches(self, text: str) -> List[Entity]:
        """
        Find all pattern matches in the text.
        
        Args:
            text: The text to search for entities.
            
        Returns:
            A list of Entity objects for matched patterns.
        """
        entities = []
        
        for entity_type, pattern in self.patterns:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0
                ))
        
        return entities


class EntityExtractor:
    """
    Entity extraction component that can use different backends (spaCy, transformers).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the entity extractor.
        
        Args:
            config: Configuration for the entity extractor.
        """
        self.config = config
        self.model_type = config.get("model_type", "spacy")
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.custom_matcher = None
        
        # Initialize custom pattern matcher if specified
        if "custom_entity_types" in config:
            self.custom_matcher = CustomPatternMatcher(config["custom_entity_types"])
        
        # Initialize the appropriate backend
        if self.model_type == "spacy":
            self._init_spacy(config)
        elif self.model_type == "transformers":
            self._init_transformers(config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"EntityExtractor initialized with {self.model_type} backend")
    
    def _init_spacy(self, config: Dict[str, Any]) -> None:
        """
        Initialize spaCy backend.
        
        Args:
            config: Configuration dictionary.
        """
        model_name = config.get("spacy_model", "en_core_web_sm")
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"SpaCy model {model_name} not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
    
    def _init_transformers(self, config: Dict[str, Any]) -> None:
        """
        Initialize transformers backend.
        
        Args:
            config: Configuration dictionary.
        """
        model_name = config.get("transformers_model", "dslim/bert-base-NER")
        device = -1  # CPU by default
        
        # Configure hardware acceleration
        if config.get("use_quantized", False):
            logger.info("Using quantized model for memory efficiency")
        
        if config.get("enable_cuda", False) and torch.cuda.is_available():
            device = config.get("cuda_device", 0)
            logger.info(f"Using CUDA device {device} for entity extraction")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Apply quantization if enabled
            if config.get("use_quantized", False) and device != -1:
                logger.info("Applying quantization to transformers model")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                aggregation_strategy="simple"
            )
            logger.info(f"Loaded transformers model: {model_name}")
        
        except Exception as e:
            logger.error(f"Failed to load transformers model: {str(e)}")
            raise
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: The text to extract entities from.
            
        Returns:
            A list of extracted entities.
        """
        if not text.strip():
            return []
        
        # Extract entities using the appropriate backend
        if self.model_type == "spacy":
            entities = self._extract_with_spacy(text)
        else:  # transformers
            entities = self._extract_with_transformers(text)
        
        # Add custom pattern matches if available
        if self.custom_matcher:
            custom_entities = self.custom_matcher.find_matches(text)
            entities.extend(custom_entities)
        
        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]
        
        logger.debug(f"Extracted {len(entities)} entities from text")
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """
        Extract entities using spaCy.
        
        Args:
            text: The text to process.
            
        Returns:
            A list of extracted entities.
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                entity_type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence scores
                metadata={"description": spacy.explain(ent.label_)}
            ))
        
        return entities
    
    def _extract_with_transformers(self, text: str) -> List[Entity]:
        """
        Extract entities using transformers.
        
        Args:
            text: The text to process.
            
        Returns:
            A list of extracted entities.
        """
        results = self.ner_pipeline(text)
        entities = []
        
        for result in results:
            entities.append(Entity(
                text=result["word"],
                entity_type=result["entity_group"],
                start=result["start"],
                end=result["end"],
                confidence=result["score"]
            ))
        
        return entities