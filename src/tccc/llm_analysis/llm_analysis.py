"""
LLM Analysis module for TCCC.ai.

This module extracts medical information from transcriptions and generates
structured reports using optimized LLMs for Jetson hardware.
"""

import os
import json
import time
import logging
import threading
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import re
import uuid
import hashlib

# For LLM management
import numpy as np
import torch

# Utilities
from tccc.utils.config import Config
from tccc.utils.logging import get_logger
from tccc.document_library import DocumentLibrary

logger = get_logger(__name__)

class LLMEngine:
    """
    Manages LLM inference with support for multiple model backends
    optimized for Jetson hardware.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM engine with configuration.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.model_config = config["model"]
        self.hardware_config = config["hardware"]
        
        # Initialize model state
        self.primary_model = None
        self.fallback_model = None
        self.tokenizer = None
        
        # Model metadata
        self.model_info = {
            "primary": {
                "loaded": False,
                "provider": self.model_config["primary"]["provider"],
                "name": self.model_config["primary"]["name"]
            },
            "fallback": {
                "loaded": False,
                "provider": self.model_config["fallback"]["provider"],
                "name": self.model_config["fallback"]["name"]
            }
        }
        
        # Initialize hardware settings
        self._setup_hardware()
        
        # Load models
        self._load_models()
    
    def _setup_hardware(self):
        """Configure hardware acceleration for Jetson."""
        if not self.hardware_config["enable_acceleration"]:
            logger.info("Hardware acceleration disabled")
            return
        
        try:
            # Set CUDA device if available
            if torch.cuda.is_available():
                cuda_device = self.hardware_config["cuda_device"]
                torch.cuda.set_device(cuda_device)
                logger.info(f"Set CUDA device to: {cuda_device}")
                
                # Get device properties
                device_props = torch.cuda.get_device_properties(cuda_device)
                logger.info(f"Using GPU: {device_props.name} with {device_props.total_memory / 1024**2:.0f}MB memory")
                
                # Set memory limit
                memory_limit = self.hardware_config["memory_limit_mb"] * 1024 * 1024
                if memory_limit > 0:
                    # This is a recommended approach for Jetson devices to limit memory usage
                    torch.cuda.set_per_process_memory_fraction(
                        memory_limit / device_props.total_memory
                    )
                    logger.info(f"Set GPU memory limit to {memory_limit / 1024**2:.0f}MB")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self.hardware_config["cuda_device"] = -1
                
        except Exception as e:
            logger.error(f"Error setting up hardware acceleration: {str(e)}")
            logger.warning("Falling back to CPU-only mode")
            self.hardware_config["enable_acceleration"] = False
            self.hardware_config["cuda_device"] = -1
    
    def _load_models(self):
        """Load primary and fallback LLM models."""
        # First try to load primary model
        try:
            self._load_primary_model()
        except Exception as e:
            logger.error(f"Failed to load primary model: {str(e)}")
            logger.warning("Will use fallback model for all requests")
        
        # Load fallback model if configured
        try:
            self._load_fallback_model()
        except Exception as e:
            logger.error(f"Failed to load fallback model: {str(e)}")
            if not self.model_info["primary"]["loaded"]:
                logger.error("No models available - LLM functionality will be limited")
    
    def _load_primary_model(self):
        """Load the primary model based on provider."""
        provider = self.model_config["primary"]["provider"]
        model_name = self.model_config["primary"]["name"]
        
        logger.info(f"Loading primary model: {model_name} from {provider}")
        
        if provider == "local":
            self._load_local_model(self.model_config["primary"])
        elif provider == "openai":
            self._setup_openai_model(self.model_config["primary"])
        else:
            logger.error(f"Unsupported model provider: {provider}")
            raise ValueError(f"Unsupported model provider: {provider}")
        
        self.model_info["primary"]["loaded"] = True
        logger.info(f"Primary model loaded successfully")
    
    def _load_fallback_model(self):
        """Load the fallback model based on provider."""
        provider = self.model_config["fallback"]["provider"]
        model_name = self.model_config["fallback"]["name"]
        
        logger.info(f"Loading fallback model: {model_name} from {provider}")
        
        if provider == "local":
            self._load_local_model(self.model_config["fallback"], is_primary=False)
        elif provider == "openai":
            self._setup_openai_model(self.model_config["fallback"], is_primary=False)
        else:
            logger.error(f"Unsupported model provider: {provider}")
            raise ValueError(f"Unsupported model provider: {provider}")
        
        self.model_info["fallback"]["loaded"] = True
        logger.info(f"Fallback model loaded successfully")
    
    def _load_local_model(self, model_config: Dict[str, Any], is_primary: bool = True):
        """Load a local LLM model optimized for Jetson hardware.
        
        Args:
            model_config: Model configuration
            is_primary: Whether this is the primary model
        """
        model_path = model_config["path"]
        model_name = model_config["name"]
        
        try:
            # Check if path exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            # Determine model type based on files
            # In a real implementation, you would use different loaders based on model format
            
            # This is a simplified implementation using CTransformers for llama models
            # or Transformers for other models
            if "llama" in model_name.lower():
                self._load_llama_model(model_path, model_config, is_primary)
            elif "phi" in model_name.lower():
                self._load_phi_model(model_path, model_config, is_primary)
            else:
                # Default to transformers
                self._load_transformers_model(model_path, model_config, is_primary)
                
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    def _load_llama_model(self, model_path: str, model_config: Dict[str, Any], is_primary: bool = True):
        """Load a Llama model using CTransformers or another optimized library.
        
        In a real implementation, this would use a library like llama-cpp-python or CTransformers
        that supports quantized Llama models efficiently on Jetson hardware.
        
        For this implementation, we'll create a simplified placeholder.
        """
        # In a real implementation, we would do:
        # from llama_cpp import Llama
        # model = Llama(
        #     model_path=f"{model_path}/model.gguf",
        #     n_gpu_layers=-1,  # Auto-detect
        #     n_ctx=model_config.get("max_context_length", 2048),
        # )
        
        # For this implementation, we'll use a placeholder
        logger.info(f"Loaded Llama model from {model_path} (placeholder for actual implementation)")
        
        # Create a minimal interface that mimics what we would expect from a real model
        class LlamaModelPlaceholder:
            def __init__(self, model_path, config):
                self.model_path = model_path
                self.config = config
                
            def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
                return {"id": str(uuid.uuid4()), 
                        "choices": [{"text": "[LLM would generate text here]"}]}
                
        # Store model in appropriate attribute
        if is_primary:
            self.primary_model = LlamaModelPlaceholder(model_path, model_config)
        else:
            self.fallback_model = LlamaModelPlaceholder(model_path, model_config)
    
    def _load_phi_model(self, model_path: str, model_config: Dict[str, Any], is_primary: bool = True):
        """Load a Phi model using transformers with ONNX Runtime.
        
        In a real implementation, this would use optimum.onnxruntime for Phi models.
        """
        # In a real implementation, we would do:
        # from optimum.onnxruntime import ORTModelForCausalLM
        # from transformers import AutoTokenizer
        # 
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = ORTModelForCausalLM.from_pretrained(
        #     model_path,
        #     provider="CUDAExecutionProvider" if self.hardware_config["cuda_device"] >= 0 else "CPUExecutionProvider"
        # )
        
        # For this implementation, we'll use a placeholder
        logger.info(f"Loaded Phi model from {model_path} (placeholder for actual implementation)")
        
        # Create a minimal interface
        class PhiModelPlaceholder:
            def __init__(self, model_path, config):
                self.model_path = model_path
                self.config = config
                
            def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
                return {"id": str(uuid.uuid4()), 
                        "choices": [{"text": "[LLM would generate text here]"}]}
                
        # Store model in appropriate attribute
        if is_primary:
            self.primary_model = PhiModelPlaceholder(model_path, model_config)
            self.tokenizer = object()  # Placeholder for tokenizer
        else:
            self.fallback_model = PhiModelPlaceholder(model_path, model_config)
    
    def _load_transformers_model(self, model_path: str, model_config: Dict[str, Any], is_primary: bool = True):
        """Load a model using HuggingFace Transformers."""
        # In a real implementation, you would do something like:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # 
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     load_in_4bit=self.hardware_config["quantization"] == "4-bit",
        #     load_in_8bit=self.hardware_config["quantization"] == "8-bit",
        # )
        
        # For this implementation, we'll use a placeholder
        logger.info(f"Loaded Transformers model from {model_path} (placeholder for actual implementation)")
        
        # Create a minimal interface
        class TransformersModelPlaceholder:
            def __init__(self, model_path, config):
                self.model_path = model_path
                self.config = config
                
            def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
                return {"id": str(uuid.uuid4()), 
                        "choices": [{"text": "[LLM would generate text here]"}]}
                
        # Store model in appropriate attribute
        if is_primary:
            self.primary_model = TransformersModelPlaceholder(model_path, model_config)
            self.tokenizer = object()  # Placeholder for tokenizer
        else:
            self.fallback_model = TransformersModelPlaceholder(model_path, model_config)
    
    def _setup_openai_model(self, model_config: Dict[str, Any], is_primary: bool = True):
        """Set up OpenAI API client."""
        # In a real implementation, we would use:
        # import openai
        # openai.api_key = model_config.get("api_key")
        
        # For this implementation, use a placeholder
        logger.info(f"Set up OpenAI client for model: {model_config['name']}")
        
        # Create minimal interface
        class OpenAIClientPlaceholder:
            def __init__(self, config):
                self.config = config
                
            def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
                return {"id": str(uuid.uuid4()), 
                        "choices": [{"text": "[LLM would generate text here]"}]}
        
        # Store client in appropriate attribute
        if is_primary:
            self.primary_model = OpenAIClientPlaceholder(model_config)
        else:
            self.fallback_model = OpenAIClientPlaceholder(model_config)
    
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, 
                     temperature: Optional[float] = None, top_p: Optional[float] = None,
                     use_fallback: bool = False) -> Dict[str, Any]:
        """Generate text using the LLM.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling value
            use_fallback: Whether to use the fallback model
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        # Determine which model to use
        if use_fallback or not self.model_info["primary"]["loaded"]:
            if not self.model_info["fallback"]["loaded"]:
                raise RuntimeError("No models available for generation")
            
            model = self.fallback_model
            model_config = self.model_config["fallback"]
            model_type = "fallback"
        else:
            model = self.primary_model
            model_config = self.model_config["primary"]
            model_type = "primary"
        
        # Set generation parameters
        if max_tokens is None:
            max_tokens = model_config.get("max_tokens", 1024)
            
        if temperature is None:
            temperature = model_config.get("temperature", 0.7)
            
        if top_p is None:
            top_p = model_config.get("top_p", 0.95)
        
        # Perform generation
        try:
            response = model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # In a real implementation, we would extract text based on model type
            # Here, we use a simplified approach
            generated_text = response["choices"][0]["text"] if "choices" in response else ""
            
            # Build result
            result = {
                "text": generated_text,
                "model": {
                    "type": model_type,
                    "name": model_config["name"],
                    "provider": model_config["provider"]
                },
                "metrics": {
                    "latency": time.time() - start_time
                }
            }
            
            # Log if configured
            if self.config["monitoring"]["log_latency"]:
                logger.debug(f"Generation latency: {result['metrics']['latency']:.2f}s")
                
            return result
            
        except Exception as e:
            # If using primary and it fails, try fallback
            if not use_fallback and self.model_info["primary"]["loaded"] and self.model_info["fallback"]["loaded"]:
                logger.warning(f"Primary model failed, falling back: {str(e)}")
                return self.generate_text(prompt, max_tokens, temperature, top_p, use_fallback=True)
            else:
                logger.error(f"Text generation failed: {str(e)}")
                raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the LLM engine.
        
        Returns:
            Dictionary with model status information
        """
        return {
            "models": self.model_info,
            "hardware": {
                "acceleration": self.hardware_config["enable_acceleration"],
                "cuda_device": self.hardware_config["cuda_device"],
                "cuda_available": torch.cuda.is_available(),
                "quantization": self.hardware_config["quantization"]
            }
        }


class MedicalEntityExtractor:
    """
    Extracts medical entities and events from transcriptions using
    optimized prompts and structured output parsing.
    """
    
    def __init__(self, llm_engine: LLMEngine, config: Dict[str, Any]):
        """Initialize the medical entity extractor.
        
        Args:
            llm_engine: LLMEngine instance
            config: Configuration dictionary
        """
        self.llm_engine = llm_engine
        self.config = config
        
        # Load prompt templates
        self.prompt_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for entity extraction."""
        # In a real implementation, these would be loaded from files
        # For this implementation, we'll use hardcoded templates
        
        templates = {
            "entity_extraction": """Extract medical entities from the following medical conversation transcript. 
Focus on identifying medical events, procedures, symptoms, measurements, and medications.

For each entity, extract:
1. Type: [procedure, symptom, measurement, medication, diagnosis, treatment]
2. Value: The specific entity name/value
3. Time: When it occurred, if mentioned
4. Context: Any relevant context or details

Format your output as a JSON array of objects with these fields.

TRANSCRIPT:
{{transcription}}

OUTPUT:""",

            "temporal_extraction": """Given the medical conversation transcript below, identify all temporal references 
and timeline information. For each event mentioned, determine when it occurred.

Parse absolute times (e.g., "10:30am"), relative times (e.g., "5 minutes ago", "earlier today"), 
and sequence information (e.g., "before intubation", "after administering medication").

Format your output as a JSON array with these fields:
1. event_id: Unique identifier
2. event: The medical event/procedure mentioned
3. timestamp: Absolute time if available (ISO format)
4. relative_time: Relative time reference if available
5. sequence: Sequence information if available
6. confidence: Your confidence in the temporal extraction (low, medium, high)

TRANSCRIPT:
{{transcription}}

OUTPUT:""",

            "vital_signs": """Extract all vital sign measurements from the following medical conversation.

For each measurement, identify:
1. Type: [heart_rate, blood_pressure, temperature, respiratory_rate, oxygen_saturation, etc.]
2. Value: The numeric value
3. Unit: The measurement unit
4. Time: When it was measured, if mentioned
5. Trend: Whether it's improving, worsening, or stable (if mentioned)

Format your output as a JSON array of objects with these fields.

TRANSCRIPT:
{{transcription}}

OUTPUT:""",

            "medication": """Extract all medication information from the following medical conversation.

For each medication, identify:
1. Name: Medication name
2. Dosage: Amount given
3. Route: Administration route (IV, oral, etc.)
4. Time: When administered or to be administered
5. Frequency: How often to be administered
6. Purpose: Reason for administration (if mentioned)

Format your output as a JSON array of objects with these fields.

TRANSCRIPT:
{{transcription}}

OUTPUT:""",

            "procedures": """Extract all medical procedures mentioned in the following conversation.

For each procedure, identify:
1. Name: Procedure name
2. Status: [planned, in_progress, completed, abandoned]
3. Time: When performed or planned
4. Performer: Who performed it (if mentioned)
5. Outcome: Result or outcome (if mentioned)
6. Details: Any specific details about the procedure

Format your output as a JSON array of objects with these fields.

TRANSCRIPT:
{{transcription}}

OUTPUT:"""
        }
        
        return templates
    
    def _render_prompt(self, template_name: str, **kwargs) -> str:
        """Render a prompt template with the given variables.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Variables to use in rendering
            
        Returns:
            Rendered prompt string
        """
        if template_name not in self.prompt_templates:
            raise ValueError(f"Template not found: {template_name}")
            
        template = self.prompt_templates[template_name]
        
        # Simple template rendering with string formatting
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
            
        return template
    
    def _parse_llm_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse LLM output into structured entities.
        
        Args:
            output: LLM-generated output string
            
        Returns:
            List of parsed entities as dictionaries
        """
        # Extract JSON from the LLM output
        try:
            # Find JSON-like structure in the output
            json_match = re.search(r'(\[.*\]|\{.*\})', output, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning(f"Could not find JSON structure in output")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse output as JSON: {str(e)}")
            logger.debug(f"Output was: {output}")
            return []
    
    def extract_entities(self, transcription: str) -> List[Dict[str, Any]]:
        """Extract medical entities from transcription.
        
        Args:
            transcription: Conversation transcription text
            
        Returns:
            List of extracted entities
        """
        # Generate prompt for entity extraction
        prompt = self._render_prompt("entity_extraction", transcription=transcription)
        
        # Generate LLM response
        response = self.llm_engine.generate_text(prompt)
        
        # Parse entities from response
        entities = self._parse_llm_output(response["text"])
        
        return entities
    
    def extract_temporal_information(self, transcription: str) -> List[Dict[str, Any]]:
        """Extract temporal information from transcription.
        
        Args:
            transcription: Conversation transcription text
            
        Returns:
            List of events with temporal information
        """
        # Generate prompt for temporal extraction
        prompt = self._render_prompt("temporal_extraction", transcription=transcription)
        
        # Generate LLM response
        response = self.llm_engine.generate_text(prompt)
        
        # Parse temporal information
        temporal_events = self._parse_llm_output(response["text"])
        
        return temporal_events
    
    def extract_vital_signs(self, transcription: str) -> List[Dict[str, Any]]:
        """Extract vital sign measurements from transcription.
        
        Args:
            transcription: Conversation transcription text
            
        Returns:
            List of vital sign measurements
        """
        # Generate prompt for vital sign extraction
        prompt = self._render_prompt("vital_signs", transcription=transcription)
        
        # Generate LLM response
        response = self.llm_engine.generate_text(prompt)
        
        # Parse vital signs
        vitals = self._parse_llm_output(response["text"])
        
        return vitals
    
    def extract_medications(self, transcription: str) -> List[Dict[str, Any]]:
        """Extract medication information from transcription.
        
        Args:
            transcription: Conversation transcription text
            
        Returns:
            List of medication information
        """
        # Generate prompt for medication extraction
        prompt = self._render_prompt("medication", transcription=transcription)
        
        # Generate LLM response
        response = self.llm_engine.generate_text(prompt)
        
        # Parse medications
        medications = self._parse_llm_output(response["text"])
        
        return medications
    
    def extract_procedures(self, transcription: str) -> List[Dict[str, Any]]:
        """Extract procedure information from transcription.
        
        Args:
            transcription: Conversation transcription text
            
        Returns:
            List of procedure information
        """
        # Generate prompt for procedure extraction
        prompt = self._render_prompt("procedures", transcription=transcription)
        
        # Generate LLM response
        response = self.llm_engine.generate_text(prompt)
        
        # Parse procedures
        procedures = self._parse_llm_output(response["text"])
        
        return procedures
    
    def extract_all(self, transcription: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all medical information from transcription.
        
        Args:
            transcription: Conversation transcription text
            
        Returns:
            Dictionary with categorized medical information
        """
        # Extract different types of information
        entities = self.extract_entities(transcription)
        temporal = self.extract_temporal_information(transcription)
        vitals = self.extract_vital_signs(transcription)
        medications = self.extract_medications(transcription)
        procedures = self.extract_procedures(transcription)
        
        # Combine all information
        return {
            "entities": entities,
            "temporal": temporal,
            "vitals": vitals,
            "medications": medications,
            "procedures": procedures
        }


class TemporalEventSequencer:
    """
    Orders medical events chronologically based on temporal information.
    """
    
    def __init__(self):
        """Initialize the temporal event sequencer."""
        pass
    
    def _extract_timestamp(self, event: Dict[str, Any]) -> Tuple[Optional[datetime], float]:
        """Extract a timestamp or temporal reference from an event.
        
        Args:
            event: Event dictionary
            
        Returns:
            Tuple of (datetime or None, confidence)
        """
        # Try to find absolute timestamp
        timestamp = None
        confidence = 0.5  # Default medium confidence
        
        # Check for ISO format timestamp
        if "timestamp" in event and event["timestamp"]:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                confidence = 0.9  # High confidence for explicit timestamp
                return timestamp, confidence
            except (ValueError, TypeError):
                pass
                
        # Check for time field in various formats
        if "time" in event and event["time"]:
            time_str = event["time"]
            
            # Try common time formats
            formats = [
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p"
            ]
            
            for fmt in formats:
                try:
                    timestamp = datetime.strptime(time_str, fmt)
                    confidence = 0.8  # Good confidence for parsable time
                    return timestamp, confidence
                except ValueError:
                    continue
                    
        # Check for relative time references
        if "relative_time" in event and event["relative_time"]:
            # This would require sophisticated NLP to convert to absolute time
            # For now, we'll just extract the relative reference for sorting
            confidence = 0.6  # Medium confidence for relative time
                    
        # Check for sequence information
        if "sequence" in event and event["sequence"]:
            # Sequence info helps with ordering
            confidence = 0.7 if confidence < 0.7 else confidence
                    
        # If confidence specified in the event, use that
        if "confidence" in event:
            if event["confidence"] == "high":
                confidence = 0.9
            elif event["confidence"] == "medium":
                confidence = 0.6
            elif event["confidence"] == "low":
                confidence = 0.3
                
        return timestamp, confidence
    
    def _calculate_sequence_score(self, event: Dict[str, Any]) -> float:
        """Calculate a sequence score for sorting events without timestamps.
        
        Args:
            event: Event dictionary
            
        Returns:
            Sequence score (higher means later in sequence)
        """
        score = 0.0
        
        # Check sequence terms
        if "sequence" in event and event["sequence"]:
            seq = event["sequence"].lower()
            
            # Terms indicating earlier in sequence
            if any(term in seq for term in ["before", "prior to", "initially", "first", "start"]):
                score -= 10.0
                
            # Terms indicating later in sequence
            if any(term in seq for term in ["after", "following", "then", "next", "subsequently"]):
                score += 10.0
                
            # Terms indicating end of sequence
            if any(term in seq for term in ["finally", "lastly", "in the end", "eventually"]):
                score += 20.0
                
        # Check relative time terms
        if "relative_time" in event and event["relative_time"]:
            rel_time = event["relative_time"].lower()
            
            # Terms indicating earlier
            if any(term in rel_time for term in ["ago", "earlier", "previously", "past"]):
                score -= 5.0
                
            # Terms indicating later
            if any(term in rel_time for term in ["later", "after", "in the future"]):
                score += 5.0
                
            # Try to extract numeric values (e.g., "5 minutes ago")
            match = re.search(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+(ago|later)', rel_time)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                direction = match.group(3)
                
                # Convert to a rough score
                unit_multiplier = {
                    "second": 1,
                    "minute": 60,
                    "hour": 3600,
                    "day": 86400,
                    "week": 604800,
                    "month": 2592000,
                    "year": 31536000
                }
                
                time_score = value * unit_multiplier.get(unit, 1)
                if direction == "ago":
                    score -= time_score / 3600  # Scale to hours
                else:
                    score += time_score / 3600
                    
        return score
    
    def sequence_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sequence events chronologically based on temporal information.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of events with sequencing information added
        """
        if not events:
            return []
            
        # Extract timestamps and confidence for each event
        event_times = []
        for i, event in enumerate(events):
            timestamp, confidence = self._extract_timestamp(event)
            sequence_score = self._calculate_sequence_score(event)
            
            event_times.append({
                "index": i,
                "event": event,
                "timestamp": timestamp,
                "confidence": confidence,
                "sequence_score": sequence_score
            })
            
        # Sort events
        # First by timestamp if available
        # Then by sequence score for relative ordering
        # Then by extraction order as fallback
        def sort_key(item):
            if item["timestamp"] is not None:
                # Convert timestamp to sortable value
                return (0, item["timestamp"].timestamp(), item["sequence_score"], item["index"])
            else:
                # No timestamp, use sequence score
                return (1, 0, item["sequence_score"], item["index"])
                
        sorted_events = sorted(event_times, key=sort_key)
        
        # Add sequence information to events
        result = []
        for i, item in enumerate(sorted_events):
            event = item["event"].copy()
            
            # Add sequencing metadata
            event["sequence_metadata"] = {
                "position": i,
                "confidence": item["confidence"],
                "has_timestamp": item["timestamp"] is not None,
                "sequence_score": item["sequence_score"]
            }
            
            result.append(event)
            
        return result


class ReportGenerator:
    """
    Generates structured medical reports from extracted events.
    """
    
    def __init__(self, llm_engine: LLMEngine, config: Dict[str, Any]):
        """Initialize the report generator.
        
        Args:
            llm_engine: LLMEngine instance
            config: Configuration dictionary
        """
        self.llm_engine = llm_engine
        self.config = config
        
        # Load report templates
        self.report_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load report templates."""
        # In a real implementation, these would be loaded from files
        # For this implementation, we'll use hardcoded templates
        
        templates = {
            "medevac": """Generate a standardized MEDEVAC report based on the following medical events.
Format the report according to the 9-line MEDEVAC request format:

Line 1: Pickup location
Line 2: Radio frequency and call sign
Line 3: Number of patients by precedence
Line 4: Special equipment required
Line 5: Number of patients by type
Line 6: Security at pickup site
Line 7: Method of marking pickup site
Line 8: Patient nationality and status
Line 9: NBC contamination

MEDICAL EVENTS:
{{events}}

OUTPUT:""",

            "zmist": """Generate a ZMIST trauma report based on the following medical events.
Format the report in the standard ZMIST format:

Z - Mechanism of injury
M - Injuries sustained
I - Signs (vital signs)
S - Treatment given
T - Trends (changes in condition)

MEDICAL EVENTS:
{{events}}

OUTPUT:""",

            "soap": """Generate a SOAP medical note based on the following medical events.
Format the note in the standard SOAP format:

S - Subjective (patient's reported symptoms)
O - Objective (measurable observations, vital signs)
A - Assessment (diagnosis or clinical impression)
P - Plan (treatment plan, next steps)

MEDICAL EVENTS:
{{events}}

OUTPUT:""",

            "tccc": """Generate a Tactical Combat Casualty Care (TCCC) Card based on the following medical events.
Include the standard TCCC sections:

1. Casualty Information
2. Mechanism of Injury
3. Injuries
4. Signs and Symptoms
5. Treatments
6. Medications
7. Fluid Therapy
8. Notes

MEDICAL EVENTS:
{{events}}

OUTPUT:"""
        }
        
        return templates
    
    def _render_prompt(self, template_name: str, **kwargs) -> str:
        """Render a report template with the given variables.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Variables to use in rendering
            
        Returns:
            Rendered prompt string
        """
        if template_name not in self.report_templates:
            raise ValueError(f"Template not found: {template_name}")
            
        template = self.report_templates[template_name]
        
        # Handle events specially to format them properly
        if "events" in kwargs and isinstance(kwargs["events"], list):
            # Convert events to a string representation
            events_str = json.dumps(kwargs["events"], indent=2)
            kwargs["events"] = events_str
            
        # Simple template rendering with string formatting
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
            
        return template
    
    def generate_report(self, report_type: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a structured report from medical events.
        
        Args:
            report_type: Type of report to generate (medevac, zmist, soap, tccc)
            events: List of medical events
            
        Returns:
            Dictionary with report content and metadata
        """
        if not report_type in self.report_templates:
            raise ValueError(f"Unsupported report type: {report_type}")
            
        # Generate prompt
        prompt = self._render_prompt(report_type, events=events)
        
        # Generate report using LLM
        response = self.llm_engine.generate_text(prompt)
        
        # Format result
        result = {
            "report_type": report_type,
            "content": response["text"],
            "generated_at": datetime.now().isoformat(),
            "events_count": len(events),
            "model": response["model"]
        }
        
        return result


class ContextIntegrator:
    """
    Enhances analysis by integrating document library context.
    """
    
    def __init__(self, document_library: DocumentLibrary, config: Dict[str, Any]):
        """Initialize the context integrator.
        
        Args:
            document_library: DocumentLibrary instance
            config: Configuration dictionary
        """
        self.document_library = document_library
        self.config = config
    
    def get_relevant_context(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Get relevant context from document library.
        
        Args:
            query: Query string
            n_results: Number of results to return
            
        Returns:
            List of context snippets
        """
        try:
            # Query document library
            results = self.document_library.query(query, n_results=n_results)
            
            # Format results
            context = []
            for result in results.get("results", []):
                context.append({
                    "text": result["text"],
                    "score": result["score"],
                    "source": result["metadata"].get("file_name", "Unknown source"),
                    "document_id": result["document_id"]
                })
                
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return []
    
    def enhance_with_context(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance events with relevant medical context.
        
        Args:
            events: List of medical events
            
        Returns:
            Enhanced events with context
        """
        enhanced_events = []
        
        for event in events:
            # Create query from event
            if "type" in event and "value" in event:
                query = f"{event['type']} {event['value']}"
                
                # Get relevant context
                context = self.get_relevant_context(query, n_results=1)
                
                # Add context to event
                enhanced_event = event.copy()
                enhanced_event["context_reference"] = context[0] if context else None
                
                enhanced_events.append(enhanced_event)
            else:
                # No type/value, just copy the event
                enhanced_events.append(event.copy())
                
        return enhanced_events


class LLMAnalysis:
    """
    LLM Analysis module for extracting medical information from transcriptions
    and generating structured reports.
    """
    
    def __init__(self):
        """Initialize the LLM analysis module."""
        self.config = None
        self.llm_engine = None
        self.entity_extractor = None
        self.event_sequencer = None
        self.report_generator = None
        self.context_integrator = None
        self.document_library = None
        self.initialized = False
        
        # Cache for storing analysis results
        self.cache = {}
        self.cache_lock = threading.Lock()
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the LLM analysis module with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.config = config
            
            # Initialize LLM engine
            logger.info("Initializing LLM engine")
            self.llm_engine = LLMEngine(config)
            
            # Initialize document library if integration enabled
            if config["policy_qa"]["enabled"]:
                logger.info("Initializing document library integration")
                # Note: In a real implementation, this would use dependency injection
                # rather than creating the DocumentLibrary directly
                self.document_library = DocumentLibrary()
                doc_lib_config = ConfigManager().load_config("document_library")
                self.document_library.initialize(doc_lib_config)
                
                # Initialize context integrator
                self.context_integrator = ContextIntegrator(self.document_library, config)
            
            # Initialize entity extractor
            logger.info("Initializing medical entity extractor")
            self.entity_extractor = MedicalEntityExtractor(self.llm_engine, config)
            
            # Initialize event sequencer
            logger.info("Initializing temporal event sequencer")
            self.event_sequencer = TemporalEventSequencer()
            
            # Initialize report generator
            logger.info("Initializing report generator")
            self.report_generator = ReportGenerator(self.llm_engine, config)
            
            # Initialize cache if enabled
            if config["caching"]["enabled"]:
                logger.info("Initializing analysis cache")
                self.cache = {}
                
            self.initialized = True
            logger.info("LLM analysis module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"LLM analysis initialization failed: {str(e)}")
            logger.debug(traceback.format_exc())
            self.initialized = False
            return False
    
    def _cache_key(self, transcription: Dict[str, Any]) -> str:
        """Generate a cache key for a transcription.
        
        Args:
            transcription: Transcription dictionary
            
        Returns:
            Cache key string
        """
        # Generate a consistent hash from the transcription content
        if isinstance(transcription, dict):
            if "text" in transcription:
                content = transcription["text"]
            elif "transcript" in transcription:
                content = transcription["transcript"]
            else:
                # Serialize the whole dict if no text field found
                content = json.dumps(transcription, sort_keys=True)
        else:
            content = str(transcription)
            
        # Create MD5 hash
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    
    def _check_cache(self, transcription: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Check if analysis results are in cache.
        
        Args:
            transcription: Transcription dictionary
            
        Returns:
            Cached results or None if not found/expired
        """
        if not self.config["caching"]["enabled"]:
            return None
            
        cache_key = self._cache_key(transcription)
        
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if entry is expired
                cache_ttl = self.config["caching"]["ttl_seconds"]
                if time.time() - entry["timestamp"] < cache_ttl:
                    logger.debug(f"Cache hit for transcription analysis")
                    return entry["results"]
                else:
                    # Entry expired
                    del self.cache[cache_key]
                    
        return None
    
    def _update_cache(self, transcription: Dict[str, Any], results: List[Dict[str, Any]]):
        """Update cache with analysis results.
        
        Args:
            transcription: Transcription dictionary
            results: Analysis results
        """
        if not self.config["caching"]["enabled"]:
            return
            
        cache_key = self._cache_key(transcription)
        
        with self.cache_lock:
            # Check cache size limit
            max_size = self.config["caching"]["max_size_mb"] * 1024 * 1024
            
            # Very simple size estimation (not accurate for complex objects)
            current_size = sum(len(json.dumps(v)) for v in self.cache.values())
            
            # If cache is full, remove oldest entries
            if current_size > max_size:
                # Sort entries by timestamp
                sorted_entries = sorted(
                    [(k, v["timestamp"]) for k, v in self.cache.items()],
                    key=lambda x: x[1]
                )
                
                # Remove oldest entries until below 80% of max size
                for key, _ in sorted_entries:
                    del self.cache[key]
                    
                    # Check if we've cleared enough space
                    current_size = sum(len(json.dumps(v)) for v in self.cache.values())
                    if current_size < max_size * 0.8:
                        break
            
            # Add new entry
            self.cache[cache_key] = {
                "timestamp": time.time(),
                "results": results
            }
    
    def process_transcription(self, transcription: Dict[str, Any], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process transcription and extract medical events.
        
        Args:
            transcription: Dictionary with transcription data
                Must contain either 'text' or 'transcript' field
            context: Optional context dictionary
                
        Returns:
            List of extracted medical events
        """
        if not self.initialized:
            logger.error("LLM analysis module not initialized")
            return []
            
        # Validate transcription
        if not isinstance(transcription, dict):
            logger.error("Transcription must be a dictionary")
            return []
            
        # Extract text from transcription
        if "text" in transcription:
            text = transcription["text"]
        elif "transcript" in transcription:
            text = transcription["transcript"]
        else:
            logger.error("Transcription missing 'text' or 'transcript' field")
            return []
            
        # Check cache first
        cached_results = self._check_cache(transcription)
        if cached_results is not None:
            return cached_results
            
        try:
            # Extract entities from transcription
            logger.info("Extracting medical entities from transcription")
            all_entities = self.entity_extractor.extract_all(text)
            
            # Combine all entities into a single list for sequencing
            combined_entities = []
            for category, entities in all_entities.items():
                for entity in entities:
                    # Add category to each entity
                    entity["category"] = category
                    combined_entities.append(entity)
            
            # Sequence events chronologically
            logger.info("Sequencing medical events")
            sequenced_events = self.event_sequencer.sequence_events(combined_entities)
            
            # Enhance with document context if available
            if self.context_integrator and context and context.get("enhance_with_context", True):
                logger.info("Enhancing events with document context")
                enhanced_events = self.context_integrator.enhance_with_context(sequenced_events)
            else:
                enhanced_events = sequenced_events
                
            # Update cache
            self._update_cache(transcription, enhanced_events)
            
            return enhanced_events
            
        except Exception as e:
            logger.error(f"Error processing transcription: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def generate_report(self, report_type: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate structured report from medical events.
        
        Args:
            report_type: Type of report to generate (medevac, zmist, soap, tccc)
            events: List of medical events
                
        Returns:
            Dictionary with report content and metadata
        """
        if not self.initialized:
            logger.error("LLM analysis module not initialized")
            return {"error": "Module not initialized"}
            
        try:
            # Validate report type
            valid_types = ["medevac", "zmist", "soap", "tccc"]
            if report_type.lower() not in valid_types:
                logger.error(f"Invalid report type: {report_type}")
                return {"error": f"Invalid report type. Supported types: {', '.join(valid_types)}"}
                
            # Generate report
            logger.info(f"Generating {report_type} report")
            report = self.report_generator.generate_report(report_type.lower(), events)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Return current status of the LLM analysis module.
        
        Returns:
            Dictionary with module status
        """
        status = {
            "initialized": self.initialized,
            "cache_enabled": self.config["caching"]["enabled"] if self.config else False,
            "cache_entries": len(self.cache) if hasattr(self, "cache") else 0
        }
        
        # Add LLM engine status if available
        if self.llm_engine:
            status["llm_engine"] = self.llm_engine.get_status()
            
        # Add document library status if available
        if self.document_library:
            doc_lib_status = self.document_library.get_status()
            status["document_library"] = {
                "initialized": doc_lib_status.get("status") == "initialized",
                "documents": doc_lib_status.get("documents", {}).get("count", 0)
            }
            
        return status