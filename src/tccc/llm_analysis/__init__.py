"""
LLM Analysis module for TCCC.ai system.

This module provides advanced conversation analysis using language models
for medical transcription processing with reliable, reproducible behavior.
"""

import os
import uuid
import logging
from typing import Dict, Any

# Import main components
from .llm_analysis import LLMAnalysis

# Get logger
from tccc.utils.logging import get_logger
logger = get_logger(__name__)

# Import LLM implementations
try:
    from .mock_llm import DeterministicRulesEngine, get_phi_model, MockPhiModel
    __has_phi_model = True
except ImportError:
    __has_phi_model = False
    logger.warning("Phi model implementation not available")

# Import GGUF model implementation
try:
    from .phi_gguf_model import PhiGGUFModel, get_phi_gguf_model
    __has_gguf_model = True
except ImportError:
    __has_gguf_model = False
    logger.warning("Phi GGUF model implementation not available")

__all__ = ["LLMAnalysis", "get_phi_model", "DeterministicRulesEngine", "MockPhiModel"]

# Add GGUF model to exports if available
if __has_gguf_model:
    __all__.extend(["PhiGGUFModel", "get_phi_gguf_model"])