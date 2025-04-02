#!/usr/bin/env python3
"""
Enhance the document processor with support for code files and additional formats.
This script is automatically run when the RAG explorer is started.
"""

import os
import sys
import logging
from pathlib import Path

from tccc.document_library.document_processor import DocumentProcessor
from tccc.utils.logging import get_logger

logger = get_logger("extend_document_processor")

def extend_document_processor():
    """Add support for additional file formats to DocumentProcessor."""
    
    logger.info("Extending DocumentProcessor with support for additional formats")
    
    # Add method to process code files (Python, JavaScript, etc.)
    def _process_code_file(self, file_path):
        """Process a code file (Python, JavaScript, etc.)
        
        Args:
            file_path: Path to the code file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()[1:] or "unknown"
            
            # Extract basic metadata
            metadata = {
                "file_name": os.path.basename(file_path),
                "content_type": f"text/{ext}",
                "file_size": os.path.getsize(file_path),
                "code_language": ext
            }
            
            # Add minimal formatting to separate comments from code
            lines = text.split('\n')
            formatted_lines = []
            
            for line in lines:
                # Add spacing around section headers (common in many languages)
                if line.strip().startswith('#') and len(line.strip()) > 2:
                    formatted_lines.append('\n' + line)
                # Handle docstrings in Python
                elif '"""' in line or "'''" in line:
                    formatted_lines.append('\n' + line)
                else:
                    formatted_lines.append(line)
            
            text = '\n'.join(formatted_lines)
            
            # Clean text
            text = self._clean_text(text)
            
            return {
                "success": True,
                "text": text,
                "metadata": metadata,
                "format": f"code-{ext}"
            }
            
        except Exception as e:
            logger.error(f"Error processing code file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {},
                "format": "unknown"
            }
    
    # Add method to document processor
    DocumentProcessor._process_code_file = _process_code_file
    
    # Add code format handlers to format_handlers
    def patched_init(self, config):
        # Call original __init__
        self._original_init(config)
        
        # Add code formats to format handlers
        self.format_handlers.update({
            "code-py": self._process_code_file,
            "code-js": self._process_code_file,
            "code-java": self._process_code_file,
            "code-cpp": self._process_code_file,
            "code-c": self._process_code_file,
            "code-go": self._process_code_file,
            "code-rs": self._process_code_file,
            "code-sh": self._process_code_file,
            "code-yaml": self._process_code_file,
            "code-json": self._process_code_file,
            "code-xml": self._process_code_file,
            "code-sql": self._process_code_file,
        })
        
        # Add more extensions to supported_extensions
        code_extensions = {
            "py": "code-py",
            "js": "code-js",
            "java": "code-java",
            "cpp": "code-cpp",
            "c": "code-c",
            "h": "code-c",
            "go": "code-go", 
            "rs": "code-rs",
            "sh": "code-sh",
            "bash": "code-sh",
            "yaml": "code-yaml",
            "yml": "code-yaml",
            "json": "code-json",
            "xml": "code-xml",
            "sql": "code-sql",
        }
        
        self.supported_extensions.update(code_extensions)
        
        logger.info(f"Added support for code file formats: {', '.join(code_extensions.keys())}")
    
    # Save original __init__ method
    DocumentProcessor._original_init = DocumentProcessor.__init__
    
    # Replace with patched version
    DocumentProcessor.__init__ = patched_init
    
    # Patch detect_format method to handle code files better
    original_detect_format = DocumentProcessor._detect_format
    
    def patched_detect_format(self, file_path):
        """Patched version of _detect_format that better handles code files."""
        
        # Get original detected format
        original_format = original_detect_format(self, file_path)
        
        # If it's a text file, check if it's actually a code file
        if original_format == "text":
            ext = os.path.splitext(file_path)[1].lower()[1:]
            code_format = f"code-{ext}"
            
            # If we have a specific handler for this code type, use it
            if code_format in self.format_handlers:
                return code_format
        
        return original_format
    
    # Replace with patched version
    DocumentProcessor._detect_format = patched_detect_format
    
    logger.info("DocumentProcessor successfully extended with additional format support")

if __name__ == "__main__":
    extend_document_processor()
    print("✅ DocumentProcessor extended with support for additional file formats")