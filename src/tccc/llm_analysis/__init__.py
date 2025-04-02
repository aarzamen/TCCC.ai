"""
LLM Analysis module for TCCC.ai system.

This module provides advanced conversation analysis using language models
for medical transcription processing with reliable, reproducible behavior.
"""

import os
import uuid
import logging
from typing import Dict, Any
import sys
import traceback

# Import main components
print("DEBUG [LLMAnalysis __init__]: Importing LLMAnalysis class...")
try:
    from .llm_analysis import LLMAnalysis
    print("DEBUG [LLMAnalysis __init__]: LLMAnalysis class imported.")
except Exception as e:
    print(f"ERROR [LLMAnalysis __init__] importing LLMAnalysis class: {e}\n{traceback.format_exc()}")
    LLMAnalysis = None
    # Decide if this is critical
    sys.exit(1)

# Get logger
from tccc.utils.logging import get_logger
logger = get_logger(__name__)

# Import LLM implementations
print("DEBUG [LLMAnalysis __init__]: Importing mock_llm...")
try:
    from .mock_llm import DeterministicRulesEngine, get_phi_model, MockPhiModel
    __has_phi_model = True
    print("DEBUG [LLMAnalysis __init__]: mock_llm imported.")
except ImportError:
    __has_phi_model = False
    logger.warning("Phi model implementation not available")
except Exception as e:
    print(f"ERROR [LLMAnalysis __init__] importing mock_llm: {e}\n{traceback.format_exc()}")
    __has_phi_model = False

# Import GGUF model implementation
print("DEBUG [LLMAnalysis __init__]: Importing phi_gguf_model...")
try:
    from .phi_gguf_model import PhiGGUFModel, get_phi_gguf_model
    __has_gguf_model = True
    print("DEBUG [LLMAnalysis __init__]: phi_gguf_model imported.")
except ImportError:
    __has_gguf_model = False
    logger.warning("Phi GGUF model implementation not available")
except Exception as e:
    print(f"ERROR [LLMAnalysis __init__] importing phi_gguf_model: {e}\n{traceback.format_exc()}")
    __has_gguf_model = False

__all__ = ["LLMAnalysis"]
if __has_phi_model:
    __all__.extend(["get_phi_model", "DeterministicRulesEngine", "MockPhiModel"])
# Add GGUF model to exports if available
if __has_gguf_model:
    __all__.extend(["PhiGGUFModel", "get_phi_gguf_model"])