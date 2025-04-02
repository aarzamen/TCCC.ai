#!/usr/bin/env python3
"""
Script to add a PDF file to the RAG vector database and query it.
Usage: python add_pdf_to_rag.py <pdf_file_path> [<query>]
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

def add_pdf_to_database(pdf_path):
    """Process and add a PDF to the vector database."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return False
    
    # Load configuration
    config = ConfigManager().load_config('document_library')
    
    # Create a copy in the documents directory
    docs_dir = Path("data/documents")
    docs_dir.mkdir(exist_ok=True, parents=True)
    dest_path = docs_dir / os.path.basename(pdf_path)
    shutil.copy2(pdf_path, dest_path)
    
    print(f"Processing PDF: {pdf_path}")
    
    # Process the document using existing scripts
    print("Processing document and updating vector database...")
    # Create temp input directory for the single file
    tmp_dir = Path("data/temp_documents")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_path = tmp_dir / os.path.basename(pdf_path)
    shutil.copy2(pdf_path, tmp_path)
    
    result = os.system(f"python process_rag_documents.py --input {tmp_dir}")
    
    if result != 0:
        print(f"Error: Failed to process PDF file {pdf_path}")
        return False
    
    print(f"Successfully added {pdf_path} to the vector database")
    return True

def query_database(query_text):
    """Query the vector database with the given text."""
    # Load configuration
    config = ConfigManager().load_config('document_library')
    
    # Initialize document library
    doc_lib = DocumentLibrary()
    doc_lib.initialize(config)
    
    print(f"Querying database with: '{query_text}'")
    
    # First try standard query
    print("\nStandard query results:")
    print("=" * 50)
    result = doc_lib.query(query_text)
    
    if not result or not result.get('results'):
        print("No results found.")
    else:
        for i, res in enumerate(result['results'][:5]):  # Limit to top 5
            print(f"Result {i+1} (Score: {res['score']:.4f}):")
            source = res.get('metadata', {}).get('source', 'Unknown')
            print(f"Source: {source}")
            print(f"Content: {res['text'][:200]}...")
            print("-" * 50)
    
    # Try advanced query with different strategies
    strategies = ["semantic", "keyword", "hybrid", "expanded"]
    for strategy in strategies:
        print(f"\nResults using {strategy} strategy:")
        result = doc_lib.advanced_query(query_text, strategy=strategy)
        
        print("=" * 50)
        if not result or not result.get('results'):
            print("No results found.")
            continue
        
        for i, res in enumerate(result['results'][:3]):  # Limit to top 3 for each strategy
            print(f"Result {i+1} (Score: {res['score']:.4f}):")
            source = res.get('metadata', {}).get('source', 'Unknown')
            print(f"Source: {source}")
            print(f"Content: {res['text'][:200]}...")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Add PDF to vector database and query it")
    parser.add_argument("pdf_path", help="Path to the PDF file to add")
    parser.add_argument("query", nargs="?", help="Query to search for (optional)")
    
    args = parser.parse_args()
    
    # Add PDF to database
    success = add_pdf_to_database(args.pdf_path)
    if not success:
        sys.exit(1)
    
    # Query if provided
    if args.query:
        query_database(args.query)
    else:
        print("\nPDF successfully added to database. Run with a query to search:")
        print(f"python add_pdf_to_rag.py {args.pdf_path} \"your search query\"")

if __name__ == "__main__":
    main()