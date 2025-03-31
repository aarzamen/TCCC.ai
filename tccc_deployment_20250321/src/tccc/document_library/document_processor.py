"""
Document Processor module for the Document Library.

This module handles the extraction and processing of text from various document formats,
including PDF, DOCX, HTML, and plain text files.
"""

import os
import io
import re
import logging
import tempfile
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import for text extraction from different file formats
try:
    import pdfplumber  # For PDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx  # For DOCX
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    from bs4 import BeautifulSoup  # For HTML
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False

# Local imports
from tccc.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Document Processor for extracting text from various file formats.
    
    This class provides methods to:
    - Detect document format based on file extension or content
    - Extract text from PDF, DOCX, HTML, and plain text files
    - Clean and normalize extracted text
    - Handle document metadata extraction when available
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the document processor.
        
        Args:
            config: Configuration dictionary with document processing settings
        """
        self.config = config
        self.supported_extensions = self._get_supported_extensions()
        
        # Initialize format handlers
        self.format_handlers = {
            "text": self._process_text_file,
            "pdf": self._process_pdf_file,
            "docx": self._process_docx_file,
            "html": self._process_html_file
        }
        
        # Log supported formats
        supported_formats = []
        if PDF_SUPPORT:
            supported_formats.append("PDF")
        if DOCX_SUPPORT:
            supported_formats.append("DOCX")
        if HTML_SUPPORT:
            supported_formats.append("HTML")
        supported_formats.append("TXT")
        
        logger.info(f"Document Processor initialized with support for: {', '.join(supported_formats)}")
    
    def _get_supported_extensions(self) -> Dict[str, str]:
        """Get mapping of supported file extensions to format types.
        
        Returns:
            Dictionary mapping extensions to formats
        """
        extensions = {}
        
        # Get extension mappings from config
        document_types = self.config["indexing"].get("document_types", [])
        for doc_type_entry in document_types:
            for format_type, format_extensions in doc_type_entry.items():
                for ext in format_extensions:
                    # Remove leading dot if present
                    ext = ext.lower()
                    if ext.startswith("."):
                        ext = ext[1:]
                    extensions[ext] = format_type
        
        # Always include basic text formats
        if "txt" not in extensions:
            extensions["txt"] = "text"
        if "md" not in extensions:
            extensions["md"] = "text"
        
        return extensions
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if the file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the format is supported, False otherwise
        """
        # Get file extension
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()[1:]  # Remove leading dot
        
        # Check if extension is in supported list
        if extension in self.supported_extensions:
            format_type = self.supported_extensions[extension]
            
            # Check if required libraries are available
            if format_type == "pdf" and not PDF_SUPPORT:
                logger.warning(f"PDF support is not available. Install pdfplumber to enable PDF processing.")
                return False
            elif format_type == "docx" and not DOCX_SUPPORT:
                logger.warning(f"DOCX support is not available. Install python-docx to enable DOCX processing.")
                return False
            elif format_type == "html" and not HTML_SUPPORT:
                logger.warning(f"HTML support is not available. Install beautifulsoup4 to enable HTML processing.")
                return False
            
            return True
        
        return False
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document file and extract text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size_kb = os.path.getsize(file_path) / 1024
            max_size_kb = self.config["indexing"].get("max_document_size_kb", 5120)
            
            if file_size_kb > max_size_kb:
                logger.warning(f"File size ({file_size_kb:.2f} KB) exceeds maximum allowed size ({max_size_kb} KB): {file_path}")
                return {
                    "success": False,
                    "error": f"File size exceeds maximum allowed size",
                    "file_path": file_path,
                    "text": "",
                    "metadata": {}
                }
            
            # Detect format
            format_type = self._detect_format(file_path)
            
            if format_type in self.format_handlers:
                # Process document with appropriate handler
                result = self.format_handlers[format_type](file_path)
                result["file_path"] = file_path
                result["format"] = format_type
                return result
            else:
                logger.warning(f"Unsupported file format: {format_type} for {file_path}")
                return {
                    "success": False,
                    "error": f"Unsupported file format: {format_type}",
                    "file_path": file_path,
                    "text": "",
                    "metadata": {}
                }
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "text": "",
                "metadata": {}
            }
    
    def _detect_format(self, file_path: str) -> str:
        """Detect the format of a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Format type string
        """
        # Get file extension
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()[1:]  # Remove leading dot
        
        # Look up format in supported extensions
        if extension in self.supported_extensions:
            return self.supported_extensions[extension]
        
        # Try to detect by content
        mime_type = mimetypes.guess_type(file_path)[0]
        
        if mime_type:
            if mime_type.startswith("text/"):
                return "text"
            elif mime_type == "application/pdf":
                return "pdf"
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return "docx"
            elif mime_type in ["text/html", "application/xhtml+xml"]:
                return "html"
        
        # Default to text format
        return "text"
    
    def _process_text_file(self, file_path: str) -> Dict[str, Any]:
        """Process a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Clean text
            text = self._clean_text(text)
            
            # Extract basic metadata
            metadata = {
                "file_name": os.path.basename(file_path),
                "content_type": "text/plain",
                "file_size": os.path.getsize(file_path)
            }
            
            return {
                "success": True,
                "text": text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {}
            }
    
    def _process_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not PDF_SUPPORT:
            logger.error("PDF processing is not supported. Install pdfplumber to enable PDF processing.")
            return {
                "success": False,
                "error": "PDF processing is not supported",
                "text": "",
                "metadata": {}
            }
        
        try:
            # Extract text using pdfplumber
            text = ""
            metadata = {
                "file_name": os.path.basename(file_path),
                "content_type": "application/pdf",
                "file_size": os.path.getsize(file_path),
                "pages": 0,
                "title": "",
                "author": "",
                "creator": ""
            }
            
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                if hasattr(pdf, "metadata") and pdf.metadata:
                    for key, value in pdf.metadata.items():
                        if isinstance(value, str):
                            metadata[key.lower()] = value
                
                # Get page count
                metadata["pages"] = len(pdf.pages)
                
                # Extract text from each page
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
            
            # Clean text
            text = self._clean_text(text)
            
            return {
                "success": True,
                "text": text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {}
            }
    
    def _process_docx_file(self, file_path: str) -> Dict[str, Any]:
        """Process a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not DOCX_SUPPORT:
            logger.error("DOCX processing is not supported. Install python-docx to enable DOCX processing.")
            return {
                "success": False,
                "error": "DOCX processing is not supported",
                "text": "",
                "metadata": {}
            }
        
        try:
            # Extract text using python-docx
            doc = docx.Document(file_path)
            
            # Extract text from paragraphs
            text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Extract metadata
            metadata = {
                "file_name": os.path.basename(file_path),
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "file_size": os.path.getsize(file_path)
            }
            
            # Try to extract core properties
            if hasattr(doc, "core_properties"):
                props = doc.core_properties
                if hasattr(props, "title") and props.title:
                    metadata["title"] = props.title
                if hasattr(props, "author") and props.author:
                    metadata["author"] = props.author
                if hasattr(props, "subject") and props.subject:
                    metadata["subject"] = props.subject
                if hasattr(props, "keywords") and props.keywords:
                    metadata["keywords"] = props.keywords
            
            # Clean text
            text = self._clean_text(text)
            
            return {
                "success": True,
                "text": text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing DOCX file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {}
            }
    
    def _process_html_file(self, file_path: str) -> Dict[str, Any]:
        """Process an HTML file.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not HTML_SUPPORT:
            logger.error("HTML processing is not supported. Install beautifulsoup4 to enable HTML processing.")
            return {
                "success": False,
                "error": "HTML processing is not supported",
                "text": "",
                "metadata": {}
            }
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            metadata = {
                "file_name": os.path.basename(file_path),
                "content_type": "text/html",
                "file_size": os.path.getsize(file_path)
            }
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.string
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                if meta.get('name') and meta.get('content'):
                    metadata[meta['name'].lower()] = meta['content']
            
            # Extract text
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Clean text
            text = self._clean_text(text)
            
            return {
                "success": True,
                "text": text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing HTML file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {}
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text