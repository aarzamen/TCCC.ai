# PHI-2 Model Download Instructions

The PHI-2 model requires access to the Hugging Face model repository. Due to the large size of the model weights (~6-7GB) and potential authentication requirements, we've provided detailed instructions for downloading the model files properly.

## Prerequisites

1. A Hugging Face account with access to the `microsoft/phi-2` model
2. At least 10GB of free disk space
3. Python environment with required dependencies

## Download Instructions

### Option 1: Using Hugging Face CLI (Recommended)

1. Install the Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   ```

2. Login to Hugging Face:
   ```bash
   huggingface-cli login
   ```

3. Download the model:
   ```bash
   huggingface-cli download microsoft/phi-2 --local-dir models/phi-2-instruct --local-dir-use-symlinks False
   ```

### Option 2: Using Python API

1. Create a download script:
   ```python
   from huggingface_hub import snapshot_download
   
   # Login if needed
   from huggingface_hub import login
   login()  # Will prompt for token
   
   # Download the model
   snapshot_download(
       repo_id="microsoft/phi-2",
       local_dir="models/phi-2-instruct",
       local_dir_use_symlinks=False
   )
   ```

2. Run the script:
   ```bash
   python download_script.py
   ```

### Option 3: Manual Download

1. Visit the model page: https://huggingface.co/microsoft/phi-2
2. Download all model files, especially:
   - `config.json`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - Model weights (safetensors files)
   - `model.safetensors.index.json`
3. Place all files in the `models/phi-2-instruct` directory

## Verification

After downloading, verify that the model files are complete:

```bash
ls -la models/phi-2-instruct
```

The `model-*.safetensors` files should be several gigabytes in size, not just a few bytes.

## Troubleshooting

- If you encounter "404 Client Error: Not Found", you may need to login with your Hugging Face account
- If the download is slow or interrupted, use the `--resume-download` flag with the Hugging Face CLI
- If you're still having issues, download directly from the Hugging Face website

## Next Steps

After downloading the complete model files, update the configuration in `config/llm_analysis.yaml` to ensure it points to the correct model path.

---

For more assistance, refer to the [Hugging Face documentation](https://huggingface.co/docs/huggingface_hub/guides/download)