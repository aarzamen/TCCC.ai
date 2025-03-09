#!/bin/bash
# Launch the TCCC RAG Explorer with text file support

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check if a text file was provided
if [ $# -eq 1 ] && [ -f "$1" ]; then
    echo "Processing file: $1"
    python test_rag_with_text.py --file "$1"
else
    # Launch in interactive mode
    echo "Starting RAG explorer in interactive mode..."
    python test_rag_with_text.py
fi