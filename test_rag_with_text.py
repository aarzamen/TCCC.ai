#!/usr/bin/env python3
"""
Test RAG functionality using text files
This script provides a simple command-line interface for testing the RAG system
with text files, demonstrating direct text content input.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from tccc.utils import ConfigManager
from tccc.document_library import DocumentLibrary


def read_text_file(file_path):
    """Read text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None


def add_text_to_rag(doc_lib, text, metadata):
    """Add text directly to the RAG database."""
    # Create document data with direct text content
    doc_data = {
        'text': text,
        'metadata': metadata,
        'source': metadata.get('source', 'Direct text input')
    }
    
    # Add document to database
    return doc_lib.add_document(doc_data)


def process_text_file(doc_lib, file_path):
    """Process a text file and add it to the RAG database."""
    print(f"\nProcessing text file: {file_path}")
    
    # Read the file
    text = read_text_file(file_path)
    if not text:
        return None
    
    # Create metadata
    filename = os.path.basename(file_path)
    title = os.path.splitext(filename)[0].replace('_', ' ').title()
    
    metadata = {
        'title': title,
        'source': file_path,
        'file_name': filename,
        'format': 'text'
    }
    
    # Add to RAG database
    doc_id = add_text_to_rag(doc_lib, text, metadata)
    if doc_id:
        print(f"✅ Document added successfully with ID: {doc_id}")
        return doc_id
    else:
        print("❌ Failed to add document to RAG database")
        return None


def query_rag(doc_lib, query, n_results=3):
    """Query the RAG database."""
    print(f"\nQuerying RAG database: '{query}'")
    results = doc_lib.query(query, n_results=n_results)
    
    print(f"Found {len(results.get('results', []))} results")
    for i, res in enumerate(results.get('results', [])):
        print(f"\nResult {i+1} (Score: {res.get('score', 0):.4f}):")
        
        # Get document info
        doc_id = res.get('document_id', 'unknown')
        metadata = res.get('metadata', {})
        title = metadata.get('title', 'Untitled')
        source = metadata.get('source', 'Unknown')
        
        print(f"Document: {title} (ID: {doc_id})")
        print(f"Source: {source}")
        
        # Show text preview
        text = res.get('text', '')
        print(f"Content: {text[:300]}..." if len(text) > 300 else f"Content: {text}")
        print("-" * 50)
    
    return results


def process_directory(doc_lib, directory, extensions=None):
    """Process all text files in a directory."""
    if extensions is None:
        extensions = ['.txt', '.md']
    
    print(f"\nProcessing all text files in directory: {directory}")
    
    files_processed = 0
    files_added = 0
    
    for ext in extensions:
        for file_path in Path(directory).glob(f"*{ext}"):
            files_processed += 1
            doc_id = process_text_file(doc_lib, file_path)
            if doc_id:
                files_added += 1
    
    print(f"\nProcessed {files_processed} files, added {files_added} to the RAG database")
    return files_added


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test RAG functionality with text files")
    parser.add_argument("--file", "-f", type=str, help="Path to a text file to process")
    parser.add_argument("--directory", "-d", type=str, help="Directory containing text files to process")
    parser.add_argument("--query", "-q", type=str, help="Query to run against the RAG database")
    parser.add_argument("--results", "-r", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()
    
    # Initialize document library
    print("Initializing Document Library...")
    config = ConfigManager().load_config("document_library")
    doc_lib = DocumentLibrary()
    success = doc_lib.initialize(config)
    
    if not success:
        print("Failed to initialize Document Library")
        return 1
    
    print(f"Document Library initialized: {len(doc_lib.documents)} documents, {len(doc_lib.chunks)} chunks")
    
    # Process file or directory if provided
    doc_id = None
    if args.file:
        doc_id = process_text_file(doc_lib, args.file)
    elif args.directory:
        process_directory(doc_lib, args.directory)
    
    # Run query if provided or enter interactive mode
    if args.query:
        results = query_rag(doc_lib, args.query, args.results)
        
        # If we just added a document and it's not in the results, try a direct vector search
        if doc_id and doc_id not in [r.get('document_id') for r in results.get('results', [])]:
            print("\n🔍 Document added but not found in results. Trying direct content search...")
            
            # Get the document text
            doc_chunks = [chunk for chunk_id, chunk in doc_lib.chunks.items() 
                         if chunk.get('doc_id') == doc_id]
            
            if doc_chunks:
                # Use the document content for a more targeted query
                print(f"📄 Scanning specific content from added document...")
                doc_content = "\n".join([chunk.get('text', '') for chunk in doc_chunks])
                
                # Extract key phrases from the document that match the query
                query_terms = args.query.lower().split()
                content_lines = doc_content.split('\n')
                
                # Find lines that match query terms
                matching_lines = []
                for line in content_lines:
                    line_lower = line.lower()
                    if any(term in line_lower for term in query_terms):
                        matching_lines.append(line)
                
                if matching_lines:
                    # Use a matching line as query
                    direct_query = matching_lines[0]
                    print(f"🔍 Using content-based query: '{direct_query[:50]}...'")
                    direct_results = doc_lib.query(direct_query, n_results=args.results)
                    
                    # Check if our document is in the results
                    doc_in_results = False
                    for res in direct_results.get('results', []):
                        if res.get('document_id') == doc_id:
                            doc_in_results = True
                            break
                    
                    if doc_in_results:
                        print("✅ Found our document with content-based query!")
                    else:
                        print("❌ Still couldn't find our document in results.")
                        
                        # Print the document content for reference
                        print("\n📄 Here's what the document contains:")
                        for i, line in enumerate(matching_lines[:5]):
                            print(f"{i+1}. {line}")
    else:
        # Interactive mode if no specific query
        print("\nEntering interactive query mode (type 'exit' to quit)")
        while True:
            query = input("\nQuery> ")
            if query.lower() == 'exit':
                break
            if query.strip():
                query_rag(doc_lib, query, args.results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())