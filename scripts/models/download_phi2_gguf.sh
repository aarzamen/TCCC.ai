#!/bin/bash
# Script to download Phi-2 GGUF model for TCCC project

# Set output directory
OUTPUT_DIR="models/phi-2-gguf"
MODEL_FILENAME="phi-2.Q4_K_M.gguf"
MODEL_REPO="TheBloke/phi-2-GGUF"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Downloading Phi-2 GGUF model from Hugging Face..."
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_REPO/$MODEL_FILENAME"

# Check if Python and huggingface_hub are available
if ! python -c "import huggingface_hub" &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Download model using huggingface_hub
echo "Starting download..."
MODEL_PATH=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='$MODEL_REPO', filename='$MODEL_FILENAME'))")

# Copy to output directory
if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
    echo "Model downloaded to: $MODEL_PATH"
    echo "Copying to: $OUTPUT_DIR/$MODEL_FILENAME"
    cp "$MODEL_PATH" "$OUTPUT_DIR/$MODEL_FILENAME"
    
    # Verify file size
    FILE_SIZE=$(ls -lh "$OUTPUT_DIR/$MODEL_FILENAME" | awk '{print $5}')
    echo "File size: $FILE_SIZE"
    
    echo "Download complete!"
    echo "To use this model, update your config in llm_analysis.yaml:"
    echo ""
    echo "model:"
    echo "  primary:"
    echo "    provider: \"local\""
    echo "    name: \"phi-2-gguf\""
    echo "    gguf_model_path: \"$OUTPUT_DIR/$MODEL_FILENAME\""
    echo "    use_gguf: true"
    echo ""
    echo "Then test with: python test_phi_gguf.py"
else
    echo "Error: Model download failed or file not found."
    exit 1
fi