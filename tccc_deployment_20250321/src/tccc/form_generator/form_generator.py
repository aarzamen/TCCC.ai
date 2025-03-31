"""
Base Form Generator module for TCCC.ai system.

This module provides the base functionality for generating various military medical forms.
"""

import os
import json
import logging
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Import utilities
from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class FormGenerator(ABC):
    """
    Abstract base class for form generators.
    
    This class defines the interface for all form generators in the system
    and provides common functionality for form generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the form generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initialized = False
        self.forms_dir = None
        self.template_dir = None
        
        # Set up output directory
        if "forms_dir" in config:
            self.forms_dir = Path(config["forms_dir"])
            os.makedirs(self.forms_dir, exist_ok=True)
        
        # Set up template directory
        if "template_dir" in config:
            self.template_dir = Path(config["template_dir"])
        
        self.initialized = True
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def generate_form(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate a form based on the provided data.
        
        Args:
            data: Data to use for form generation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with form information
        """
        pass
    
    def validate_data(self, data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validate that required fields are present in the data.
        
        Args:
            data: Data to validate
            required_fields: List of required field names
            
        Returns:
            Tuple of (is_valid, missing_fields)
        """
        missing_fields = []
        
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    def save_form(self, form_data: Dict[str, Any], file_path: Optional[str] = None) -> str:
        """Save form data to disk.
        
        Args:
            form_data: Form data to save
            file_path: Optional file path to save to
            
        Returns:
            Path to the saved file
        """
        if file_path is None:
            # Generate a file path if not provided
            form_type = form_data.get("form_type", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{form_type}_{timestamp}.json"
            
            if self.forms_dir:
                file_path = self.forms_dir / filename
            else:
                file_path = Path(tempfile.gettempdir()) / filename
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save form data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(form_data, f, indent=2)
            
        logger.info(f"Saved form data to {file_path}")
        return str(file_path)
    
    def merge_data(self, base_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new data with base data, keeping base values if keys already exist.
        
        Args:
            base_data: Base data dictionary
            new_data: New data to merge
            
        Returns:
            Merged data
        """
        result = base_data.copy()
        
        for key, value in new_data.items():
            if key not in result:
                result[key] = value
        
        return result
    
    def get_form_template(self, template_name: str) -> Dict[str, Any]:
        """Get a form template.
        
        Args:
            template_name: Name of the template to get
            
        Returns:
            Template data
        """
        if not self.template_dir:
            logger.warning("No template directory configured")
            return {}
        
        template_path = self.template_dir / f"{template_name}.json"
        
        if not os.path.exists(template_path):
            logger.warning(f"Template {template_name} not found at {template_path}")
            return {}
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the form generator.
        
        Returns:
            Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "forms_dir": str(self.forms_dir) if self.forms_dir else None,
            "template_dir": str(self.template_dir) if self.template_dir else None
        }