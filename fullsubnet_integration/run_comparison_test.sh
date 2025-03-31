#\!/bin/bash
# Run comparison test between FullSubNet and Battlefield audio enhancers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory
OUTPUT_DIR="$SCRIPT_DIR/comparison_results"
mkdir -p "$OUTPUT_DIR"

# Check if input file was provided
if [ $# -lt 1 ]; then
    # No input file provided, generate test audio
    echo "No input file provided. Generating test audio..."
    python3 test_fullsubnet.py --generate --output "$OUTPUT_DIR"
    INPUT_FILE=$(find "$OUTPUT_DIR" -name "test_audio_*.wav" | sort -r | head -n 1)
else
    INPUT_FILE="$1"
fi

echo "=== FullSubNet vs Battlefield Audio Enhancer Comparison ==="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

# Run benchmark comparison
python3 test_fullsubnet.py --benchmark --input "$INPUT_FILE" --output "$OUTPUT_DIR"

# Test transcription accuracy
echo -e "\nTesting transcription accuracy..."
python3 test_fullsubnet.py --transcribe --input "$INPUT_FILE" --output "$OUTPUT_DIR"

echo -e "\nComparison complete. Results saved to $OUTPUT_DIR"
echo "Visualizations are in $OUTPUT_DIR/visualizations"

# Open visualizations if available
if [ -d "$OUTPUT_DIR/visualizations" ]; then
    if command -v xdg-open &> /dev/null; then
        find "$OUTPUT_DIR/visualizations" -name "*.png" | sort | head -n 1 | xargs xdg-open
    else
        echo "Visualizations available at $OUTPUT_DIR/visualizations"
    fi
fi
