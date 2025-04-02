#!/usr/bin/env python3
"""
Verification script for the TCCC.ai RAG Database.

This script tests the RAG system's components and functionality,
including document processing, vector storage, query execution,
and LLM prompt generation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import utilities
from tccc.utils import ConfigManager
from tccc.utils.logging import get_logger

# Import document library
from tccc.document_library import DocumentLibrary

# Configure logging
logger = get_logger("RAG_Verification")

def test_initialization(doc_library):
    """Test document library initialization."""
    print("\n=== Testing Document Library Initialization ===")
    status = doc_library.get_status()
    print(f"Status: {status['status']}")
    print(f"Documents: {status['documents']['count']}")
    print(f"Chunks: {status['documents']['chunks']}")
    print(f"Vectors: {status['index']['vectors']}")
    print(f"Model: {status['model']['name']}")
    
    # Check components
    if "components" in status:
        components = status["components"]
        print("\nComponents:")
        for name, avail in components.items():
            print(f"- {name}: {'Available' if avail else 'Not available'}")
    
    return status["status"] == "initialized"

def test_document_processor(doc_library):
    """Test document processor functionality."""
    print("\n=== Testing Document Processor ===")
    
    # Test if document processor is available
    if not hasattr(doc_library, 'document_processor') or doc_library.document_processor is None:
        print("Document processor not available!")
        return False
    
    # Test with a sample document
    sample_file = os.path.join(project_root, "data/sample_documents/tactical_considerations.txt")
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        return False
    
    print(f"Testing document processing with: {sample_file}")
    result = doc_library.document_processor.process_document(sample_file)
    
    if result.get("success", False):
        print("Document processing successful!")
        print(f"Extracted {len(result.get('text', ''))} characters of text")
        print(f"Metadata: {list(result.get('metadata', {}).keys())}")
        return True
    else:
        print(f"Document processing failed: {result.get('error', 'Unknown error')}")
        return False

def test_basic_query(doc_library):
    """Test basic query functionality."""
    print("\n=== Testing Basic Query ===")
    
    queries = [
        "What is TCCC?",
        "How to apply a tourniquet?",
        "What are the steps in MARCH assessment?",
        "How to complete a TCCC card?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        result = doc_library.query(query, n_results=3)
        query_time = time.time() - start_time
        
        print(f"Results: {len(result.get('results', []))}")
        print(f"Time: {query_time:.2f} seconds")
        
        # Print top result snippet
        if result.get('results'):
            top_result = result['results'][0]
            text_preview = top_result.get('text', '')[:150] + '...' if len(top_result.get('text', '')) > 150 else top_result.get('text', '')
            print(f"Top result: {text_preview}")
    
    return True

def test_advanced_query(doc_library):
    """Test advanced query functionality."""
    print("\n=== Testing Advanced Query ===")
    
    # Check if advanced query is available
    if not hasattr(doc_library, 'advanced_query'):
        print("Advanced query functionality not available!")
        return False
    
    query = "How to treat a tension pneumothorax?"
    
    # Test with different strategies
    strategies = ["semantic", "keyword", "hybrid", "expanded"]
    for strategy in strategies:
        print(f"\nQuery with {strategy} strategy: {query}")
        start_time = time.time()
        result = doc_library.advanced_query(
            query_text=query,
            strategy=strategy,
            limit=3
        )
        query_time = time.time() - start_time
        
        print(f"Results: {len(result.get('results', []))}")
        print(f"Time: {query_time:.2f} seconds")
        print(f"Strategy used: {result.get('strategy', 'unknown')}")
        
        # Print number of results for each strategy
        print(f"Found {len(result.get('results', []))} results with {strategy} strategy")
    
    return True

def test_medical_vocabulary(doc_library):
    """Test medical vocabulary functionality."""
    print("\n=== Testing Medical Vocabulary ===")
    
    # Check if medical vocabulary is available
    if not hasattr(doc_library, 'medical_vocabulary') or doc_library.medical_vocabulary is None:
        print("Medical vocabulary not available!")
        return False
    
    # Test term extraction
    test_text = """
    The medic quickly applied a tourniquet to stop the hemorrhage, 
    then checked the casualty's airway and respiration. 
    Using the MARCH protocol, they assessed circulation and checked for hypothermia.
    The patient required a needle decompression for tension pneumothorax.
    """
    
    print("Test text:")
    print(test_text.strip())
    
    # Extract medical terms
    medical_terms = doc_library.extract_medical_terms(test_text)
    print("\nExtracted medical terms:")
    for term in medical_terms:
        print(f"- {term}")
    
    # Explain medical terms
    explanations = doc_library.explain_medical_terms(test_text)
    print("\nExplanations:")
    for term, explanation in explanations.items():
        print(f"- {explanation}")
    
    return len(medical_terms) > 0

def test_llm_prompt_generation(doc_library):
    """Test LLM prompt generation."""
    print("\n=== Testing LLM Prompt Generation ===")
    
    # Check if prompt generation is available
    if not hasattr(doc_library, 'generate_llm_prompt'):
        print("LLM prompt generation not available!")
        return False
    
    queries = [
        "What is TCCC?",
        "How to apply a tourniquet?",
        "Explain the MARCH protocol"
    ]
    
    # Test with different context length limits
    context_lengths = [1500, 1000, 500, 2000]
    
    for query in queries:
        print(f"\nGenerating prompt for: {query}")
        
        # Regular prompt generation
        prompt = doc_library.generate_llm_prompt(query)
        print(f"Standard prompt length: {len(prompt)} characters")
        
        # Now test with different context length limits
        print("\nTesting with different context lengths:")
        for context_length in context_lengths:
            try:
                context_prompt = doc_library.generate_llm_prompt(
                    query=query,
                    max_context_length=context_length
                )
                print(f"- Context length {context_length}: {len(context_prompt)} characters")
            except Exception as e:
                print(f"- Context length {context_length}: Error: {str(e)}")
        
        # Print preview of standard prompt
        print("\nPrompt preview:")
        preview_lines = prompt.split('\n')[:8]
        for line in preview_lines:
            print(f"> {line}")
        print("...")
    
    return True

def test_interactive_rag_explorer():
    """Test the interactive RAG explorer."""
    print("\n=== Testing Interactive RAG Explorer ===")
    
    # Check if the required files exist
    explorer_script = os.path.join(project_root, "jetson_interactive_rag.py")
    launch_script = os.path.join(project_root, "launch_rag_explorer.sh")
    desktop_file = os.path.join(project_root, "TCCC_RAG_Explorer.desktop")
    
    print(f"Checking explorer script: {os.path.basename(explorer_script)}")
    explorer_exists = os.path.exists(explorer_script)
    print(f"  - File exists: {'✅' if explorer_exists else '❌'}")
    
    print(f"Checking launch script: {os.path.basename(launch_script)}")
    launch_exists = os.path.exists(launch_script)
    print(f"  - File exists: {'✅' if launch_exists else '❌'}")
    
    print(f"Checking desktop integration: {os.path.basename(desktop_file)}")
    desktop_exists = os.path.exists(desktop_file)
    print(f"  - File exists: {'✅' if desktop_exists else '❌'}")
    
    # Check if files are executable
    if explorer_exists:
        explorer_executable = os.access(explorer_script, os.X_OK)
        print(f"  - Explorer executable: {'✅' if explorer_executable else '❌'}")
    else:
        explorer_executable = False
    
    if launch_exists:
        launch_executable = os.access(launch_script, os.X_OK)
        print(f"  - Launch script executable: {'✅' if launch_executable else '❌'}")
    else:
        launch_executable = False
    
    # Check desktop file content
    if desktop_exists:
        with open(desktop_file, 'r') as f:
            desktop_content = f.read()
        has_correct_exec = "jetson_interactive_rag.py" in desktop_content
        has_pdf_mimetype = "MimeType=application/pdf" in desktop_content
        print(f"  - Desktop points to correct script: {'✅' if has_correct_exec else '❌'}")
        print(f"  - Desktop has PDF MIME type: {'✅' if has_pdf_mimetype else '❌'}")
    else:
        has_correct_exec = has_pdf_mimetype = False
    
    # Overall test result
    basic_files = explorer_exists and launch_exists and desktop_exists
    permissions = explorer_executable and launch_executable
    desktop_config = has_correct_exec and has_pdf_mimetype
    
    success = basic_files and permissions and desktop_config
    print(f"Overall RAG Explorer test: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify TCCC.ai RAG Database functionality")
    parser.add_argument("--config", "-c", type=str, default="config/document_library.yaml",
                      help="Configuration file path")
    parser.add_argument("--add-sample", "-a", action="store_true",
                      help="Add sample document before testing")
    parser.add_argument("--full", "-f", action="store_true",
                      help="Run full tests including interactive components")
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("document_library")
    
    # Initialize document library
    doc_library = DocumentLibrary()
    success = doc_library.initialize(config)
    
    if not success:
        print("Failed to initialize Document Library")
        return 1
    
    # Add sample document if requested
    if args.add_sample:
        sample_file = os.path.join(project_root, "data/sample_documents/tactical_considerations.txt")
        if os.path.exists(sample_file):
            print(f"Adding sample document: {sample_file}")
            document = {
                "file_path": sample_file,
                "metadata": {
                    "title": "Tactical Considerations",
                    "source": "TCCC Guidelines",
                    "category": "training"
                }
            }
            doc_id = doc_library.add_document(document)
            if doc_id:
                print(f"Added sample document with ID: {doc_id}")
            else:
                print("Failed to add sample document")
    
    # Run tests
    test_results = {}
    
    test_results["initialization"] = test_initialization(doc_library)
    test_results["document_processor"] = test_document_processor(doc_library)
    test_results["basic_query"] = test_basic_query(doc_library)
    test_results["advanced_query"] = test_advanced_query(doc_library)
    test_results["medical_vocabulary"] = test_medical_vocabulary(doc_library)
    test_results["llm_prompt_generation"] = test_llm_prompt_generation(doc_library)
    
    # Test interactive components if requested
    if args.full:
        test_results["interactive_explorer"] = test_interactive_rag_explorer()
    
    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        all_passed = all_passed and result
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())