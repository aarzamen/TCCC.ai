#!/usr/bin/env python3
"""
Jetson Interactive RAG Explorer - Process PDFs via drag-and-drop and query in the same window.
Usage: python jetson_interactive_rag.py
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
import threading
from pathlib import Path

from tccc.utils import ConfigManager
from tccc.document_library import DocumentLibrary

class InteractiveRAGTerminal:
    """Interactive terminal for PDF processing and RAG querying."""
    
    def __init__(self):
        self.config = ConfigManager().load_config('document_library')
        self.doc_lib = DocumentLibrary()
        self.doc_lib.initialize(self.config)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tccc_rag_"))
        
    def process_pdf(self, pdf_path):
        """Process and add a PDF to the vector database."""
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return False
        
        # Create a copy in the documents directory
        docs_dir = Path("data/documents")
        docs_dir.mkdir(exist_ok=True, parents=True)
        dest_path = docs_dir / os.path.basename(pdf_path)
        shutil.copy2(pdf_path, dest_path)
        
        print(f"\n🔄 Processing PDF: {os.path.basename(pdf_path)}")
        print("Please wait, this may take a minute...")
        
        try:
            # Process the document using document library
            print("Extracting text from PDF...")
            document = self.doc_lib.document_processor.process_file(str(dest_path))
            
            if not document:
                print(f"Error: Failed to process PDF file {pdf_path}")
                return False
            
            print("Adding document to vector database...")
            self.doc_lib.add_document(document)
            
            # Get document stats
            total_docs = len(self.doc_lib.documents)
            total_chunks = len(self.doc_lib.chunks)
            
            print(f"✅ Successfully added PDF to database (Total: {total_docs} documents, {total_chunks} chunks)")
            return True
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False
            
    def query_database(self, query):
        """Query the vector database."""
        if not query:
            return
            
        print(f"\n🔍 Searching: '{query}'")
        print("=" * 50)
        
        # Execute query
        result = self.doc_lib.query(query)
        
        # Print results
        if not result or not result.get('results'):
            print("No results found.")
            return
            
        for i, res in enumerate(result['results'][:5]):
            print(f"Result {i+1} (Score: {res['score']:.4f}):")
            source = res.get('metadata', {}).get('source', 'Unknown')
            if isinstance(source, dict) and 'filename' in source:
                source = source['filename']
            print(f"Source: {source}")
            print(f"Content: {res['text'][:300]}...")
            print("-" * 50)
    
    def generate_startup_script(self):
        """Generate the bash script for the interactive terminal."""
        script = """#!/bin/bash
# TCCC RAG Explorer - Interactive Terminal
# Supports drag-and-drop PDF processing and database querying

export TCCC_PROJECT_DIR="$(pwd)"
export PYTHONPATH="$TCCC_PROJECT_DIR:$PYTHONPATH"

# Run the document processor extension to add support for additional file formats
echo "Extending document processor with additional format support..."
python -m extend_document_processor

# Function to process any document file
process_document() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "Error: File not found: $file"
        return 1
    fi
    
    # Get file extension
    ext="${file##*.}"
    ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    
    # List of supported file extensions
    local supported_exts=("pdf" "txt" "md" "docx" "py" "html" "htm" "xml" "json" "csv" "yaml" "yml")
    
    # Check if extension is supported
    local is_supported=0
    for supported in "${supported_exts[@]}"; do
        if [[ "$ext" == "$supported" ]]; then
            is_supported=1
            break
        fi
    done
    
    if [[ $is_supported -eq 0 ]]; then
        echo "Warning: File type .$ext may not be fully supported"
    fi
    
    # Process the document
    python -c "
from tccc.utils import ConfigManager
from tccc.document_library import DocumentLibrary
import os, sys, shutil
from pathlib import Path

# Load configuration
config = ConfigManager().load_config('document_library')

# Initialize document library
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# File paths
file_path = '$file'
file_name = os.path.basename(file_path)
print(f'\\n🔄 Processing document: {file_name}')
print('Please wait, this may take a moment...')

# Copy to documents directory
docs_dir = Path('data/documents')
docs_dir.mkdir(exist_ok=True, parents=True)
dest_path = docs_dir / file_name
shutil.copy2(file_path, dest_path)

# Process document
result = doc_lib.document_processor.process_document(str(dest_path))
if not result or not result.get('success', False):
    error_msg = result.get('error', 'Unknown error') if result else 'Failed to process document'
    print(f'Error: {error_msg}')
    sys.exit(1)

# Create document data
doc_data = {
    'text': result.get('text', ''),
    'metadata': result.get('metadata', {}),
    'source': str(dest_path)
}

# Add to database
doc_id = doc_lib.add_document(doc_data)
if not doc_id:
    print(f'Error: Failed to add document to database')
    sys.exit(1)

# Get document stats
total_docs = len(doc_lib.documents)
total_chunks = len(doc_lib.chunks)
file_type = result.get('format', 'unknown').upper()

print(f'✅ Successfully added {file_type} document to database')
print(f'   Document ID: {doc_id}')
print(f'   Total documents: {total_docs}')
print(f'   Total chunks: {total_chunks}')
print(f'   Content length: {len(result.get(\"text\", \"\"))} characters')
"
}

# Function to query the database
query_database() {
    query="$1"
    python -c "
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

# Initialize document library
config = ConfigManager().load_config('document_library')
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# Execute query
print(f'\\n🔍 Searching: \\'$query\\'')
print('=' * 50)
result = doc_lib.query('$query')

if not result or not result.get('results'):
    print('No results found.')
else:
    for i, res in enumerate(result['results'][:5]):
        print(f'Result {i+1} (Score: {res[\"score\"]:.4f}):')
        source = res.get('metadata', {}).get('source', 'Unknown')
        if isinstance(source, dict) and 'filename' in source:
            source = source['filename']
        print(f'Source: {source}')
        print(f'Content: {res[\"text\"][:300]}...')
        print('-' * 50)
"
}

# Welcome message
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           TCCC RAG Explorer - Interactive Terminal        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

• To process a PDF: Drag and drop a PDF file into this window
• To query the database: Type 'q: your query' and press Enter
• For help: Type 'help' and press Enter
• To exit: Type 'exit' and press Enter

EOF

# Process any document provided as argument
if [[ $# -eq 1 ]]; then
    if [[ -f "$1" ]]; then
        process_document "$1"
    elif [[ -d "$1" ]]; then
        # Process all documents in a directory
        echo "Processing all documents in directory: $1"
        
        # Find and process all supported document types
        for ext in pdf txt md py docx html htm xml json csv yaml yml; do
            for file in "$1"/*.$ext "$1"/*.${ext^^}; do
                # Check if file exists (to handle cases where no matches are found)
                if [[ -f "$file" ]]; then
                    process_document "$file"
                fi
            done
        done
    else
        echo "Error: File or directory not found: $1"
    fi
fi

# Main input loop
while true; do
    read -e -p "RAG> " input
    
    if [[ -z "$input" ]]; then
        continue
    elif [[ "$input" == "exit" ]]; then
        echo "Exiting TCCC RAG Explorer"
        break
    elif [[ "$input" == "help" ]]; then
        cat << 'EOF'
Available commands:
  • Drag and drop any document  - Process and add to database
  • q: <query>                  - Search the database
  • q! <query>                  - Deep search (uses all strategies)
  • stats                       - Show database statistics
  • formats                     - Show supported file formats
  • clear                       - Clear the screen
  • help                        - Show this help message
  • exit                        - Exit the program

Supported file types:
  • PDF files (.pdf)            - With metadata extraction
  • Text files (.txt, .md)      - Plain text and markdown
  • Code files (.py, etc)       - Treated as text documents
  • Word documents (.docx)      - With metadata extraction
  • Other formats (.html, .xml, .json, .csv, .yaml)

Tips:
  • You can drag a folder to process all documents within it
  • For medical queries, use precise terminology
  • Longer, more specific queries often yield better results
EOF
    elif [[ "$input" == "clear" ]]; then
        clear
    elif [[ "$input" =~ ^q:\ (.+)$ ]]; then
        query="${BASH_REMATCH[1]}"
        query_database "$query"
    elif [[ "$input" =~ ^q!\ (.+)$ ]]; then
        query="${BASH_REMATCH[1]}"
        python -c "
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

# Initialize document library
config = ConfigManager().load_config('document_library')
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# Execute query with all strategies
print(f'\\n🔍 Deep Search: \\'$query\\'')
print('=' * 50)

strategies = ['semantic', 'keyword', 'hybrid', 'expanded']
all_results = []

for strategy in strategies:
    print(f'\\nStrategy: {strategy.upper()}')
    result = doc_lib.advanced_query('$query', strategy=strategy)
    
    if not result or not result.get('results'):
        print('No results for this strategy.')
        continue
    
    for res in result['results'][:2]:  # Top 2 from each strategy
        print(f'Score: {res[\"score\"]:.4f} - {res[\"text\"][:150]}...')
        # Track for deduplication 
        all_results.append((res[\"score\"], res[\"text\"][:50]))
    
    print('-' * 30)

print('\\nSearch complete using all strategies.')
"
    elif [[ "$input" == "stats" ]]; then
        python -c "
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager
import os

# Initialize document library
config = ConfigManager().load_config('document_library')
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# Calculate stats
print('\\n📊 RAG Database Statistics')
print('=' * 50)
print(f'Total documents:       {len(doc_lib.documents)}')
print(f'Total chunks:          {len(doc_lib.chunks)}')
print(f'Vector dimensions:     {doc_lib.vector_store.index.d}')
print(f'Vector count:          {doc_lib.vector_store.index.ntotal}')

# Get document types
doc_types = {}
for doc_id, doc in doc_lib.documents.items():
    ext = os.path.splitext(doc.get('source', ''))[-1].lower()
    doc_types[ext] = doc_types.get(ext, 0) + 1

print('\\nDocument types:')
for ext, count in doc_types.items():
    if ext:
        print(f'  • {ext[1:]} files: {count}')
    else:
        print(f'  • unknown type: {count}')

print('\\nLatest documents:')
sorted_docs = sorted(doc_lib.documents.items(), key=lambda x: x[1].get('timestamp', 0), reverse=True)
for i, (doc_id, doc) in enumerate(sorted_docs[:3]):
    if i < 3:  # Show only the 3 most recent
        source = doc.get('source', 'Unknown')
        if isinstance(source, dict) and 'filename' in source:
            source = source['filename']
        print(f'  • {os.path.basename(source)}')
"
    # Handle all types of quotes by creating a clean version of the input
    clean_input="${input//\'/}"     # Remove single quotes
    clean_input="${clean_input//\"/}" # Remove double quotes
    
    # Now check the clean input for files or directories
    if [[ -f "$input" ]]; then
        # Process any document file (unquoted)
        process_document "$input"
    elif [[ -f "$clean_input" ]]; then
        # Handle quoted paths with spaces
        process_document "$clean_input"
    elif [[ -d "$input" ]]; then
        # Process all documents in a directory (unquoted)
        echo "Processing all documents in directory: $input"
        
        # Find and process all supported document types
        for ext in pdf txt md py docx html htm xml json csv yaml yml; do
            for file in "$input"/*.$ext "$input"/*.${ext^^}; do
                # Check if file exists (to handle cases where no matches are found)
                if [[ -f "$file" ]]; then
                    process_document "$file"
                fi
            done
        done
    elif [[ -d "$clean_input" ]]; then
        # Handle quoted directory paths with spaces
        echo "Processing all documents in directory: $clean_input"
        
        # Find and process all supported document types for quoted paths
        for ext in pdf txt md py docx html htm xml json csv yaml yml; do
            for file in "$clean_input"/*.$ext "$clean_input"/*.${ext^^}; do
                # Check if file exists (to handle cases where no matches are found)
                if [[ -f "$file" ]]; then
                    process_document "$file"
                fi
            done
        done
    elif [[ "$input" == "formats" ]]; then
        # Show supported formats
        echo -e "\n📄 Supported Document Formats:"
        echo "================================="
        echo "• PDF files (.pdf) - Full support with metadata extraction"
        echo "• Text files (.txt) - Plain text with basic metadata"
        echo "• Markdown files (.md) - Treated as text with markdown formatting"
        echo "• Word documents (.docx) - Full support with metadata extraction"
        echo "• HTML files (.html, .htm) - Extracts text content from HTML"
        echo "• Python files (.py) - Code files treated as text documents"
        echo "• Data files (.json, .xml, .csv, .yaml, .yml) - Structured data as text"
        echo ""
        echo "Note: All text-based files will be properly processed and added to the knowledge base."
    else
        echo "Unknown command or file not found. Type 'help' for available commands."
    fi
done
"""
        
        script_path = self.temp_dir / "interactive_rag.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        return script_path
        
    def launch_terminal(self, pdf_path=None):
        """Launch the interactive terminal."""
        script_path = self.generate_startup_script()
        
        # Determine the terminal emulator with better Jetson compatibility
        terminal_emulators = ["gnome-terminal", "xterm", "konsole", "lxterminal", "terminator"]
        terminal_cmd = None
        
        for term in terminal_emulators:
            try:
                subprocess.run(["which", term], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                terminal_cmd = term
                break
            except subprocess.CalledProcessError:
                continue
        
        if not terminal_cmd:
            print("No supported terminal emulator found. Using fallback xterm.")
            terminal_cmd = "xterm"
            # Check if xterm is installed, if not use current terminal
            try:
                subprocess.run(["which", "xterm"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError:
                # Fallback to executing directly in current terminal
                print("xterm not found. Running in current terminal.")
                if pdf_path:
                    os.system(f"bash {script_path} '{pdf_path}'")
                else:
                    os.system(f"bash {script_path}")
                return
        
        # Handle terminal-specific parameters
        try:
            # Launch terminal with appropriate command
            if terminal_cmd == "gnome-terminal":
                if pdf_path:
                    cmd = [terminal_cmd, "--", "bash", str(script_path), pdf_path]
                else:
                    cmd = [terminal_cmd, "--", "bash", str(script_path)]
            elif terminal_cmd == "terminator":
                if pdf_path:
                    cmd = [terminal_cmd, "-e", f"bash {script_path} '{pdf_path}'"]
                else:
                    cmd = [terminal_cmd, "-e", f"bash {script_path}"]
            else:
                if pdf_path:
                    cmd = [terminal_cmd, "-e", f"bash {script_path} '{pdf_path}'"]
                else:
                    cmd = [terminal_cmd, "-e", f"bash {script_path}"]
            
            subprocess.Popen(cmd)
            print(f"RAG Explorer launched in a new {terminal_cmd} window.")
        except Exception as e:
            print(f"Error launching terminal: {str(e)}")
            print("Falling back to current terminal.")
            if pdf_path:
                os.system(f"bash {script_path} '{pdf_path}'")
            else:
                os.system(f"bash {script_path}")

def main():
    """Main function."""
    rag_terminal = InteractiveRAGTerminal()
    
    # Parse command line arguments
    pdf_path = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) and sys.argv[1].lower().endswith('.pdf'):
        pdf_path = sys.argv[1]
    
    # Launch terminal
    rag_terminal.launch_terminal(pdf_path)

if __name__ == "__main__":
    main()