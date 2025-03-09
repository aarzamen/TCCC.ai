#!/bin/bash
# Launch the TCCC RAG Explorer on Jetson

# Change to script directory
cd "$(dirname "$0")"

# Check if a file or directory path was provided
if [ $# -eq 1 ]; then
    INPUT_PATH="$1"
    
    # Check if input is a file
    if [ -f "$INPUT_PATH" ]; then
        echo "Processing file: $INPUT_PATH"
        # Activate virtual environment and run with file
        source venv/bin/activate
        python jetson_interactive_rag.py "$INPUT_PATH"
    # Check if input is a directory
    elif [ -d "$INPUT_PATH" ]; then
        echo "Processing directory: $INPUT_PATH"
        # Activate virtual environment and run with directory
        source venv/bin/activate
        python jetson_interactive_rag.py "$INPUT_PATH"
    else
        echo "Error: File or directory not found at $INPUT_PATH"
        exit 1
    fi
else
    # Just launch the explorer without processing
    source venv/bin/activate
    python jetson_interactive_rag.py
fi