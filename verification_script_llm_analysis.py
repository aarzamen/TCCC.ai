#!/usr/bin/env python3
"""
Verification script for the LLM Analysis module.

This script demonstrates the functionality of the LLM Analysis module
by processing a sample transcription and generating different types of reports.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMAnalysisVerification")

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tccc.utils import ConfigManager
from tccc.llm_analysis import LLMAnalysis
from tccc.document_library import DocumentLibrary

def create_sample_transcription():
    """Create a sample medical transcription for testing.
    
    Returns:
        Dictionary with transcription text
    """
    return {
        "text": "Medic: Applied tourniquet to right leg at 0930 hours due to bleeding."
    }

def setup_document_library():
    """Set up a document library with sample documents.
    
    Returns:
        Initialized DocumentLibrary instance
    """
    logger.info("Setting up document library...")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("document_library")
    
    # Initialize document library
    doc_library = DocumentLibrary()
    doc_library.initialize(config)
    
    # Add a sample document with medical procedures if not already present
    # In a real scenario, this would check if documents already exist
    sample_content = """
    # TCCC Medical Procedures Reference
    
    ## Airway Management
    - Chin lift/jaw thrust
    - Nasopharyngeal airway
    - Endotracheal intubation
    - Cricothyroidotomy
    
    ## Breathing Management
    - Needle decompression for tension pneumothorax
    - Chest seal for open pneumothorax
    - Supplemental oxygen administration
    
    ## Circulation Management
    - Tourniquet application for extremity hemorrhage
    - Wound packing with hemostatic gauze
    - IV/IO access
    - Fluid resuscitation
    - TXA administration for hemorrhagic shock
    
    ## Medication Administration
    - Morphine: 5-10mg IV/IO for pain
    - Antibiotics: Ceftriaxone 1-2g IV for open wounds
    - TXA: 1g IV over 10 minutes for hemorrhagic shock
    
    ## Assessment Guidelines
    - GCS scoring and monitoring
    - Vital signs interpretation in trauma
    - Signs of tension pneumothorax
    - Signs of hypovolemic shock
    """
    
    # Create a sample file
    sample_file_path = os.path.join(project_root, "data", "documents", "tccc_procedures.txt")
    os.makedirs(os.path.dirname(sample_file_path), exist_ok=True)
    
    with open(sample_file_path, "w") as f:
        f.write(sample_content)
    
    # Add document to library
    doc_id = doc_library.add_document({
        "file_path": sample_file_path,
        "metadata": {
            "category": "Medical Procedures",
            "type": "Reference",
            "source": "TCCC Guidelines"
        }
    })
    
    if doc_id:
        logger.info(f"Added document with ID: {doc_id}")
    else:
        logger.warning("Failed to add document to library")
    
    return doc_library

def verify_llm_analysis():
    """Verify the LLM Analysis functionality."""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("llm_analysis")
    
    # Initialize document library for context
    doc_library = setup_document_library()
    
    # Initialize LLM analysis module
    logger.info("Initializing LLM Analysis module")
    llm_analysis = LLMAnalysis()
    success = llm_analysis.initialize(config)
    
    if not success:
        logger.error("Failed to initialize LLM Analysis module")
        return False
    
    # Get initial status
    logger.info("LLM Analysis Module Status:")
    status = llm_analysis.get_status()
    logger.info(json.dumps(status, indent=2))
    
    # Create sample transcription
    transcription = create_sample_transcription()
    
    # Process transcription
    logger.info("Processing transcription...")
    start_time = time.time()
    events = llm_analysis.process_transcription(
        transcription,
        context={"enhance_with_context": True}
    )
    processing_time = time.time() - start_time
    
    logger.info(f"Processed transcription in {processing_time:.2f}s")
    logger.info(f"Extracted {len(events)} medical events")
    
    # Display a few extracted events
    logger.info("\nSample extracted events:")
    for i, event in enumerate(events[:3]):  # Show first 3 events
        logger.info(f"\nEvent {i+1}:")
        # Format event for display, limiting the output
        event_display = {k: v for k, v in event.items() if k not in ["context_reference"]}
        logger.info(json.dumps(event_display, indent=2))
        
        # Show context reference if available
        if "context_reference" in event and event["context_reference"]:
            logger.info("Context reference: " + event["context_reference"]["source"])
    
    # Generate different types of reports
    report_types = ["medevac", "zmist", "soap", "tccc"]
    
    for report_type in report_types:
        logger.info(f"\nGenerating {report_type.upper()} report:")
        
        start_time = time.time()
        report = llm_analysis.generate_report(report_type, events)
        report_time = time.time() - start_time
        
        logger.info(f"Generated {report_type.upper()} report in {report_time:.2f}s")
        
        # Display report content
        logger.info(f"\n{report_type.upper()} REPORT:")
        logger.info(report["content"])
    
    logger.info("\nLLM Analysis verification completed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify LLM Analysis functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        result = verify_llm_analysis()
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.exception("Verification failed with error")
        sys.exit(1)