#!/bin/bash
#
# Test script for the optimized RAG system
#

set -e

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"

# Print header
echo "===================================="
echo "      TCCC RAG Optimization Test    "
echo "===================================="

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Run optimization script
echo "Step 1: Running RAG optimization..."
python "$SCRIPT_DIR/optimize_rag_performance.py" --optimize-all

# Run medical terminology test
echo -e "\nStep 2: Testing medical terminology handling..."
python "$SCRIPT_DIR/test_rag_medical_terms.py" --skip-query-test

# Run final benchmark
echo -e "\nStep 3: Running performance benchmark..."
python "$SCRIPT_DIR/optimize_rag_performance.py" --benchmark

echo -e "\nOptimization and testing complete!"
echo "To launch the optimized RAG system, run:"
echo "  ./launch_rag_on_jetson.sh"
echo "Or use the desktop shortcut: TCCC_RAG_Query.desktop"

exit 0