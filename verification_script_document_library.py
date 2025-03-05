#!/usr/bin/env python3
"""
Verification script for the Document Library module.

This script demonstrates the functionality of the Document Library module
by creating sample documents, adding them to the library, and performing
various queries.
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
logger = logging.getLogger("DocumentLibraryVerification")

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tccc.utils import ConfigManager
from tccc.document_library import DocumentLibrary

def create_sample_documents(base_dir):
    """Create sample text documents for testing.
    
    Args:
        base_dir: Directory to create the sample documents in
        
    Returns:
        List of file paths for the created documents
    """
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create sample text files
    documents = []
    
    # Document 1: Medical procedures
    doc1_path = os.path.join(base_dir, "medical_procedures.txt")
    with open(doc1_path, "w") as f:
        f.write("""# Common Medical Procedures in TCCC

## Airway Management
Proper airway management is critical in tactical care. The following procedures may be necessary:

1. Chin Lift/Jaw Thrust: Basic maneuver to open airway
2. Nasopharyngeal Airway (NPA): Used for unconscious patients with intact gag reflex
3. Supraglottic Airway Device: Used when NPA is insufficient
4. Surgical Cricothyroidotomy: Used when other methods fail

## Breathing Support
Ensure adequate oxygenation and ventilation:

1. Needle Decompression: For tension pneumothorax
2. Chest Seal: For open pneumothorax
3. Oxygen Administration: When available
4. Assisted Ventilation: When breathing is inadequate

## Circulation Management
Control bleeding and maintain circulation:

1. Direct Pressure: First-line treatment for hemorrhage
2. Tourniquet Application: For life-threatening extremity bleeding
3. Hemostatic Dressings: For wounds not amenable to tourniquet
4. IV/IO Access: For fluid and medication administration
5. Fluid Resuscitation: For hypovolemic shock
""")
    documents.append(doc1_path)
    
    # Document 2: Tactical considerations
    doc2_path = os.path.join(base_dir, "tactical_considerations.txt")
    with open(doc2_path, "w") as f:
        f.write("""# Tactical Considerations in TCCC

## Care Under Fire Phase
During direct threat conditions:

1. Return fire as directed or required
2. Move casualty to cover if possible
3. Direct casualty to move to cover and apply self-aid if able
4. Control major bleeding with tourniquet
5. Maintain tactical situational awareness

## Tactical Field Care Phase
During indirect threat conditions:

1. Maintain tactical security
2. Conduct thorough assessment
3. Treat wounds according to priority
4. Consider evacuation resources and timing
5. Communicate with tactical leadership

## Tactical Evacuation Care Phase
During evacuation:

1. Continue ongoing resuscitation
2. Monitor for deterioration
3. Prepare documentation
4. Prepare for transfer of care
5. Maintain security as situation dictates

## Environmental Considerations
Adapt care based on:

1. Climate conditions (heat, cold, humidity)
2. Terrain considerations
3. Time of day/night operations
4. Available resources
5. Remote vs. urban settings
""")
    documents.append(doc2_path)
    
    # Document 3: Medical equipment
    doc3_path = os.path.join(base_dir, "medical_equipment.txt")
    with open(doc3_path, "w") as f:
        f.write("""# Medical Equipment for TCCC

## Individual First Aid Kit (IFAK)
Standard components:

1. Combat Application Tourniquet (CAT)
2. Hemostatic gauze (Combat Gauze)
3. Pressure bandage
4. Nasopharyngeal airway
5. Chest seal
6. Surgical tape
7. Trauma shears
8. Gloves
9. Casualty card

## Medical Operator Kit
Advanced components:

1. Backup tourniquets
2. Additional hemostatic agents
3. Intravenous access supplies
4. Fluid bags (Hextend, NS)
5. Chest decompression needle
6. Advanced airway equipment
7. Surgical cricothyroidotomy kit
8. Medication module
9. Hypothermia prevention kit

## Evacuation Equipment
For casualty movement:

1. Tactical litters
2. Poleless litters
3. Improvised carrying devices
4. SKEDCO rescue system
5. Casualty drag straps
""")
    documents.append(doc3_path)
    
    logger.info(f"Created {len(documents)} sample documents in {base_dir}")
    return documents

def verify_document_library():
    """Verify the Document Library functionality."""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("document_library")
    
    # Set up a data directory for this demonstration
    data_dir = os.path.join(project_root, "data")
    doc_dir = os.path.join(data_dir, "sample_documents")
    
    # Update paths in config to use our test directories
    config["storage"]["base_dir"] = os.path.join(data_dir, "documents")
    config["storage"]["index_path"] = os.path.join(data_dir, "document_index")
    config["storage"]["cache_dir"] = os.path.join(data_dir, "query_cache")
    config["embedding"]["cache_dir"] = os.path.join(data_dir, "models/embeddings")
    
    # Ensure directories exist
    for path in [config["storage"]["base_dir"], config["storage"]["index_path"], 
                config["storage"]["cache_dir"], config["embedding"]["cache_dir"]]:
        os.makedirs(path, exist_ok=True)
    
    # Create sample documents
    document_paths = create_sample_documents(doc_dir)
    
    # Initialize document library
    logger.info("Initializing Document Library")
    doc_library = DocumentLibrary()
    success = doc_library.initialize(config)
    
    if not success:
        logger.error("Failed to initialize Document Library")
        return False
    
    # Get initial status
    logger.info("Initial Document Library Status:")
    status = doc_library.get_status()
    logger.info(json.dumps(status, indent=2))
    
    # Add documents to the library
    document_ids = []
    for doc_path in document_paths:
        logger.info(f"Adding document: {os.path.basename(doc_path)}")
        document = {
            "file_path": doc_path,
            "metadata": {
                "category": "TCCC Reference",
                "type": "Text",
                "source": "Verification Script",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        doc_id = doc_library.add_document(document)
        if doc_id:
            document_ids.append(doc_id)
            logger.info(f"Document added with ID: {doc_id}")
        else:
            logger.error(f"Failed to add document: {os.path.basename(doc_path)}")
    
    # Get updated status
    logger.info("Document Library Status after adding documents:")
    status = doc_library.get_status()
    logger.info(json.dumps(status, indent=2))
    
    # Perform some sample queries
    sample_queries = [
        "How do I manage airways in tactical care?",
        "What equipment is needed for chest decompression?",
        "What are the phases of tactical combat casualty care?",
        "How do I apply a tourniquet?",
        "What environmental factors affect care?"
    ]
    
    for query in sample_queries:
        logger.info(f"\nQuery: {query}")
        
        # First query - should not be cached
        start_time = time.time()
        results = doc_library.query(query, n_results=2)
        query_time = time.time() - start_time
        
        logger.info(f"Query processing time: {results.get('processing_time', query_time):.3f}s")
        logger.info(f"Cache hit: {results.get('cache_hit', False)}")
        logger.info(f"Total results: {results.get('total_results', 0)}")
        
        # Display results
        for i, result in enumerate(results.get("results", [])):
            logger.info(f"\nResult {i+1} (score: {result['score']}):")
            logger.info(f"Document: {result['metadata'].get('file_name', 'Unknown')}")
            
            # Truncate text for display
            text = result['text']
            if len(text) > 200:
                text = text[:197] + "..."
            logger.info(f"Text: {text}")
        
        # Second query - should be cached
        logger.info(f"\nRepeating query: {query}")
        start_time = time.time()
        results = doc_library.query(query, n_results=2)
        query_time = time.time() - start_time
        
        logger.info(f"Query processing time: {results.get('processing_time', query_time):.3f}s")
        logger.info(f"Cache hit: {results.get('cache_hit', False)}")
    
    # Get document metadata
    if document_ids:
        logger.info(f"\nGetting metadata for document: {document_ids[0]}")
        metadata = doc_library.get_document_metadata(document_ids[0])
        logger.info(json.dumps(metadata, indent=2))
    
    logger.info("\nDocument Library verification completed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Document Library functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        result = verify_document_library()
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.exception("Verification failed with error")
        sys.exit(1)