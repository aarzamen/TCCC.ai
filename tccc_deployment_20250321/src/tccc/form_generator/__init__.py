"""
Form Generator module for TCCC.ai system.

This module provides capabilities for generating military medical forms 
based on extracted data from audio transcriptions and LLM analysis.
"""

from tccc.form_generator.form_generator import FormGenerator
from tccc.form_generator.tccc_card import TCCCCasualtyCard

__all__ = ["FormGenerator", "TCCCCasualtyCard"]