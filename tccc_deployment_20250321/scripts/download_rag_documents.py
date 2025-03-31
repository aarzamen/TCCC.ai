#!/usr/bin/env python3
"""
Download military medicine documents for TCCC.ai RAG database.

This script downloads key military medicine documents for the TCCC.ai system's
Document Library, focusing on topics like 9-line MEDEVAC, MIST reporting,
TCCC guidelines, and the Valkyrie Whole Blood Transfusion Program.
"""

import os
import sys
import time
import logging
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_Document_Downloader")

# Document definitions - key military medicine documents
DOCUMENTS = [
    {
        "name": "TCCC Guidelines for Medical Personnel",
        "url": "https://irp.fas.org/doddir/army/tccc-1.pdf",
        "description": "Foundational document for tactical combat casualty care, covering essential procedures like MIST and 9-line reporting",
        "priority": 1,
        "category": "core_guidelines"
    },
    {
        "name": "Joint Publication 4-02: Health Service Support",
        "url": "https://irp.fas.org/doddir/dod/jp4_02.pdf",
        "description": "Details health service support doctrine, including 9-line MEDEVAC requests, critical for combat evacuation",
        "priority": 2,
        "category": "core_guidelines"
    },
    {
        "name": "DD Form 1380: TCCC Card",
        "url": "https://www.esd.whs.mil/Portals/54/Documents/DD/forms/dd/dd1380.pdf",
        "description": "Standard form for documenting patient care in combat, used by Navy corpsmen and doctors, with fields for MIST and 9-line data",
        "priority": 3,
        "category": "forms"
    },
    {
        "name": "JTS Forms and After Action Report Submission",
        "url": "https://jts.health.mil/index.cfm/documents/forms_after_action",
        "description": "Provides fillable forms like the TCCC Card, essential for trauma care documentation",
        "priority": 4,
        "category": "forms"
    },
    {
        "name": "Valkyrie Program Presentation",
        "url": "https://health.mil/Reference-Center/Presentations/2020/02/04/Valkyrie-Combat-Whole-Blood-Transfusion-Program-Lt-McBeth",
        "description": "Focuses on the Valkyrie Whole Blood Transfusion Program, particularly relevant for 1st Marine Division initiatives",
        "priority": 5,
        "category": "specialized_programs"
    },
    {
        "name": "MCTP 3-40A: Health Service Support Operations",
        "url": "https://www.marines.mil/portals/1/Publications/MCTP%203-40A.pdf",
        "description": "Offers Marine Corps doctrine for health service support, aligning with USMC medical personnel needs",
        "priority": 6,
        "category": "specialized_programs"
    },
    {
        "name": "Hospital Corpsman Manual",
        "url": "https://www.med.navy.mil/Portals/62/Documents/BUMED/EBOOKS/Hospital%20Corpsman%20Manual.pdf",
        "description": "Navy training manual for corpsmen, covering combat casualty care and documentation",
        "priority": 7,
        "category": "training"
    },
    {
        "name": "NAVMC 3500.84B: Health Service Support Training and Readiness Manual",
        "url": "https://www.marines.mil/portals/1/Publications/NAVMC%203500.84.pdf",
        "description": "Specifies training standards for health service support, including TCCC",
        "priority": 8,
        "category": "training"
    },
    {
        "name": "USUHS Military Specific Curriculum",
        "url": "https://www.usuhs.edu/academic-programs/military-specific-curriculum",
        "description": "Includes operational medicine and TCCC-related training from the Uniformed Services University",
        "priority": 10,
        "category": "supplementary"
    }
]

def download_document(doc: Dict[str, Any], output_dir: str, force: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Download a document from the specified URL.
    
    Args:
        doc: Document information dictionary
        output_dir: Directory to save the document
        force: Whether to force download even if the file exists
        
    Returns:
        Tuple of (success status, output file path or None)
    """
    try:
        url = doc["url"]
        name = doc["name"]
        
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If filename is empty or doesn't have an extension, use a sanitized name
        if not filename or '.' not in filename:
            filename = name.lower().replace(' ', '_').replace(':', '').replace('-', '_') + '.html'
        
        # Create destination path
        category_dir = os.path.join(output_dir, doc["category"])
        os.makedirs(category_dir, exist_ok=True)
        
        output_path = os.path.join(category_dir, filename)
        
        # Check if file already exists
        if os.path.exists(output_path) and not force:
            logger.info(f"Document '{name}' already exists at {output_path}")
            return True, output_path
        
        # Download the file
        logger.info(f"Downloading '{name}' from {url}")
        
        headers = {
            "User-Agent": "TCCC.ai Document Library Builder/1.0"
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded '{name}' to {output_path}")
        
        # Small delay to avoid overwhelming servers
        time.sleep(1)
        
        return True, output_path
        
    except Exception as e:
        logger.error(f"Failed to download '{doc['name']}': {str(e)}")
        return False, None

def main() -> int:
    """
    Main function to download all documents.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Download military medicine documents for TCCC.ai RAG database")
    parser.add_argument("--output", "-o", type=str, default="data/rag_documents", 
                        help="Output directory for documents")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force download even if files already exist")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Downloading {len(DOCUMENTS)} documents to {output_dir}")
    
    # Sort documents by priority
    sorted_docs = sorted(DOCUMENTS, key=lambda x: x["priority"])
    
    # Download each document
    success_count = 0
    failure_count = 0
    
    for doc in sorted_docs:
        logger.info(f"Processing document {doc['priority']}/{len(DOCUMENTS)}: {doc['name']}")
        success, _ = download_document(doc, output_dir, args.force)
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    logger.info(f"Download complete: {success_count} successful, {failure_count} failed")
    
    if failure_count > 0:
        logger.warning("Some documents could not be downloaded. Check the logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())