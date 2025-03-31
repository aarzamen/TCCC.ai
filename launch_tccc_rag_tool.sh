#!/bin/bash
# Launch TCCC RAG Tool
# This script launches the comprehensive RAG testing and demo tool

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH for proper module resolution
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Optional Jetson optimization flag
JETSON_OPT=""
if [ "$1" == "--jetson" ]; then
    JETSON_OPT="-j"
    shift
fi

# Run the tool in interactive mode by default
python tccc_rag_tool.py -i $JETSON_OPT "$@"

# Deactivate virtual environment if it was activated
if [ -d "venv" ]; then
    deactivate
fi