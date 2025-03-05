"""
LLM Analysis module for TCCC.ai system.

This module provides advanced conversation analysis and agent recommendations using LLMs
with Microsoft's Phi-2 model optimized for Jetson Orin Nano hardware.
"""

import os
import uuid
import logging
from typing import Dict, Any

# Import main components
from .llm_analysis import LLMAnalysis

# Import Phi models
try:
    from .phi_model import PhiModel, get_phi_model  # Use our new implementation
    __has_phi_model = True
except ImportError:
    __has_phi_model = False

# Import mock implementation
try:
    from .mock_llm import MockPhiModel
    __has_mock = True
except ImportError:
    __has_mock = False

# Get logger
from tccc.utils.logging import get_logger
logger = get_logger(__name__)

__all__ = ["LLMAnalysis", "get_phi_model"]