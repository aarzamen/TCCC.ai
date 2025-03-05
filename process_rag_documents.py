#!/usr/bin/env python3
"""
Process downloaded military medicine documents for the TCCC.ai RAG database.

This script processes downloaded documents (PDFs, HTML, etc.) into plain text,
splits them into manageable chunks, and adds them to the Document Library
for use with the RAG system.
"""

import os
import sys
import time
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Attempt to import document processing libraries
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    
try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import bs4
    from bs4 import BeautifulSoup
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_Document_Processor")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import TCCC.ai modules
from tccc.utils import ConfigManager
from tccc.document_library import DocumentLibrary

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF document.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text as string
    """
    if not PDF_SUPPORT:
        logger.error("PDF support not available. Install pdfplumber package.")
        return ""
    
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX document.
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        Extracted text as string
    """
    if not DOCX_SUPPORT:
        logger.error("DOCX support not available. Install python-docx package.")
        return ""
    
    try:
        text = ""
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX {file_path}: {str(e)}")
        return ""

def extract_text_from_html(file_path: str) -> str:
    """
    Extract text from HTML document.
    
    Args:
        file_path: Path to HTML file
        
    Returns:
        Extracted text as string
    """
    if not HTML_SUPPORT:
        logger.error("HTML support not available. Install beautifulsoup4 package.")
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing spaces
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Join lines, removing empty ones
        text = '\\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from HTML {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from plain text document.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Extracted text as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to extract text from TXT {file_path}: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Failed to extract text from TXT {file_path}: {str(e)}")
        return ""

def extract_text(file_path: str) -> str:
    """
    Extract text from document based on file extension.
    
    Args:
        file_path: Path to document file
        
    Returns:
        Extracted text as string
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == '.docx':
        return extract_text_from_docx(file_path)
    elif file_ext in ['.html', '.htm']:
        return extract_text_from_html(file_path)
    elif file_ext in ['.txt', '.md']:
        return extract_text_from_txt(file_path)
    else:
        logger.warning(f"Unsupported file format: {file_ext}")
        return ""

def process_document(file_path: str, doc_library: DocumentLibrary, metadata: Dict[str, Any]) -> bool:
    """
    Process document and add to document library.
    
    Args:
        file_path: Path to document file
        doc_library: Document Library instance
        metadata: Document metadata
        
    Returns:
        Success status
    """
    try:
        logger.info(f"Processing document: {os.path.basename(file_path)}")
        
        # Extract text from document
        text = extract_text(file_path)
        
        if not text:
            logger.error(f"No text extracted from {file_path}")
            return False
        
        # Save text to temporary file for document library
        temp_dir = os.path.join(os.path.dirname(file_path), "processed")
        os.makedirs(temp_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        text_file = os.path.join(temp_dir, f"{base_name}.txt")
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Add document to library
        document = {
            "file_path": text_file,
            "metadata": metadata
        }
        
        doc_id = doc_library.add_document(document)
        
        if doc_id:
            logger.info(f"Document added to library with ID: {doc_id}")
            return True
        else:
            logger.error(f"Failed to add document to library")
            return False
        
    except Exception as e:
        logger.error(f"Failed to process document {file_path}: {str(e)}")
        return False

def process_documents(input_dir: str, doc_library: DocumentLibrary) -> Tuple[int, int]:
    """
    Process all documents in a directory.
    
    Args:
        input_dir: Input directory containing documents
        doc_library: Document Library instance
        
    Returns:
        Tuple of (success count, failure count)
    """
    success_count = 0
    failure_count = 0
    
    # Walk through input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Skip processed files
            if "processed" in root:
                continue
                
            file_path = os.path.join(root, file)
            
            # Skip non-document files
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in ['.pdf', '.docx', '.html', '.htm', '.txt', '.md']:
                continue
            
            # Create metadata
            category = os.path.basename(root)
            metadata = {
                "title": os.path.splitext(file)[0],
                "source_file": file,
                "category": category,
                "source": "TCCC.ai RAG Documents",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Process document
            success = process_document(file_path, doc_library, metadata)
            
            if success:
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count

def main() -> int:
    """
    Main function to process documents and add to document library.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Process documents for TCCC.ai RAG database")
    parser.add_argument("--input", "-i", type=str, default="data/rag_documents",
                        help="Input directory containing downloaded documents")
    parser.add_argument("--config", "-c", type=str, default="config/document_library.yaml",
                        help="Document library configuration file")
    args = parser.parse_args()
    
    # Check if input directory exists
    input_dir = args.input
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("document_library")
    
    # Initialize document library
    doc_library = DocumentLibrary()
    success = doc_library.initialize(config)
    
    if not success:
        logger.error("Failed to initialize Document Library")
        return 1
    
    # Get initial status
    initial_status = doc_library.get_status()
    initial_doc_count = initial_status.get("documents", {}).get("count", 0)
    logger.info(f"Initial document count: {initial_doc_count}")
    
    # Process documents
    logger.info(f"Processing documents in {input_dir}")
    success_count, failure_count = process_documents(input_dir, doc_library)
    
    # Get final status
    final_status = doc_library.get_status()
    final_doc_count = final_status.get("documents", {}).get("count", 0)
    
    # Print summary
    logger.info(f"Documents processed: {success_count} successful, {failure_count} failed")
    logger.info(f"Document library now contains {final_doc_count} documents (+{final_doc_count - initial_doc_count})")
    
    if failure_count > 0:
        logger.warning("Some documents could not be processed. Check the logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())