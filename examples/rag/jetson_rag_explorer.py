#!/usr/bin/env python3
"""
Jetson RAG Explorer - Process PDFs and query the knowledge base in a new terminal window.
Usage: python jetson_rag_explorer.py <pdf_file_path>
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

from tccc.utils import ConfigManager
from tccc.document_library import DocumentLibrary

def process_pdf(pdf_path):
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
    
    # Process the document using document library
    doc_lib = DocumentLibrary()
    doc_lib.initialize(config)
    
    # Process the PDF and add it to the database
    try:
        # First use the process_rag_documents.py script for comprehensive processing
        subprocess.run(["python", "process_rag_documents.py", "--input", os.path.dirname(dest_path)], 
                       check=True)
        print(f"Successfully processed {pdf_path} and added to RAG database")
        return True
    except subprocess.CalledProcessError:
        print(f"Error: Failed to process PDF file {pdf_path}")
        return False

def launch_query_terminal():
    """Launch a new terminal with the RAG explorer interface."""
    # Create a temporary script for the terminal
    query_script = """#!/bin/bash
echo "===== TCCC RAG Explorer ====="
echo "Enter your query or type 'exit' to quit"
echo ""

while true; do
    read -p "Query> " query
    if [ "$query" == "exit" ]; then
        echo "Exiting RAG Explorer"
        break
    fi
    
    if [ -n "$query" ]; then
        python -c "
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

# Initialize document library
config = ConfigManager().load_config('document_library')
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# Execute query
print(f'\\nSearching for: \\\"{query}\\\"')
print('=' * 50)
result = doc_lib.query('\"\""$query"\""')

if not result or not result.get('results'):
    print('No results found.')
else:
    for i, res in enumerate(result['results'][:5]):
        print(f'Result {i+1} (Score: {res[\"score\"]:.4f}):')
        source = res.get('metadata', {}).get('source', 'Unknown')
        print(f'Source: {source}')
        print(f'Content: {res[\"text\"][:300]}...')
        print('-' * 50)
"
    fi
    echo ""
done
"""
    
    script_path = Path("data/rag_query_terminal.sh")
    with open(script_path, 'w') as f:
        f.write(query_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    # Detect available terminal emulator on Jetson
    terminal_emulators = ["gnome-terminal", "xterm", "konsole", "lxterminal"]
    terminal_cmd = None
    
    for term in terminal_emulators:
        try:
            # Check if terminal is available on system
            subprocess.run(["which", term], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            terminal_cmd = term
            break
        except subprocess.CalledProcessError:
            continue
    
    if not terminal_cmd:
        print("No supported terminal emulator found. Using fallback xterm.")
        terminal_cmd = "xterm"
    
    # Launch new terminal with the script
    terminal_args = ["--"] if terminal_cmd == "gnome-terminal" else []
    subprocess.Popen([terminal_cmd] + terminal_args + ["bash", str(script_path)])
    print(f"RAG Explorer terminal launched using {terminal_cmd}. Enter queries to search the knowledge base.")

def main():
    parser = argparse.ArgumentParser(description="TCCC RAG Explorer for Jetson")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to process")
    
    args = parser.parse_args()
    
    # Process PDF if provided
    if args.pdf_path:
        if process_pdf(args.pdf_path):
            print("PDF successfully processed and added to the RAG database")
        else:
            print("Failed to process PDF. Check logs for details.")
    
    # Launch query terminal
    launch_query_terminal()

if __name__ == "__main__":
    main()