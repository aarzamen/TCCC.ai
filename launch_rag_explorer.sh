#!/bin/bash
# Launch the TCCC RAG Explorer on Jetson

# Change to script directory
cd "$(dirname "$0")"

# Check if PDF file path was provided
if [ $# -eq 1 ]; then
    PDF_PATH="$1"
    # Check if file exists
    if [ -f "$PDF_PATH" ]; then
        echo "Processing PDF: $PDF_PATH"
        # Activate virtual environment and run with PDF
        source venv/bin/activate
        python jetson_interactive_rag.py "$PDF_PATH"
    else
        echo "Error: PDF file not found at $PDF_PATH"
        exit 1
    fi
else
    # Just launch the explorer without PDF processing
    source venv/bin/activate
    python jetson_interactive_rag.py
fi