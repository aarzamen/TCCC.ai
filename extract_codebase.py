#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import mimetypes
import chardet

# Initialize the output file
output_file = "codebase_document.txt"

# Directories to exclude
excluded_dirs = [
    "__pycache__", 
    ".git", 
    "venv", 
    "node_modules", 
    "large_model_chunks"
]

# Extensions to exclude (binary files and other non-text formats)
excluded_extensions = [
    '.pyc', '.pyo', '.so', '.o', '.a', '.lib', '.dll', '.exe', '.obj',
    '.pyd', '.db', '.dat', '.bin', '.pkl', '.pickle', '.h5', '.hdf5',
    '.npy', '.npz', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
    '.wav', '.mp3', '.mp4', '.avi', '.mov', '.mkv', '.flac', '.ogg',
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
    '.pth', '.onnx', '.pt', '.gguf', '.faiss'
]

# Function to detect if a file is binary
def is_binary(file_path):
    try:
        # Check extension first
        if any(file_path.endswith(ext) for ext in excluded_extensions):
            return True
            
        mime = mimetypes.guess_type(file_path)[0]
        if mime and not mime.startswith('text/'):
            return True
            
        # Read a small chunk to check for binary content
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\x00' in chunk
    except Exception:
        return True

def process_file(file_path, output):
    try:
        # Skip binary files
        if is_binary(file_path):
            return
            
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            if not raw_data:
                return
            
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            
        # Read and write the file content
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
            
        output.write(f"\n\n# File: {file_path}\n\n")
        output.write(content)
    except Exception as e:
        output.write(f"\n\n# Error processing file: {file_path}\n# Error: {str(e)}\n\n")

def walk_directory(dir_path, output):
    try:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            
            # Skip excluded directories
            if os.path.isdir(item_path) and (item in excluded_dirs or any(excl in item_path for excl in excluded_dirs)):
                continue
                
            if os.path.isdir(item_path):
                walk_directory(item_path, output)
            elif os.path.isfile(item_path):
                process_file(item_path, output)
    except Exception as e:
        output.write(f"\n\n# Error processing directory: {dir_path}\n# Error: {str(e)}\n\n")

def main():
    try:
        project_dir = os.getcwd()
        processed_files = 0
        
        print(f"Starting to process files in {project_dir}")
        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(f"# Project Codebase Document\n# Generated from: {project_dir}\n\n")
            walk_directory(project_dir, output)
        
        # Get file size
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"Successfully created {os.path.abspath(output_file)}")
        print(f"File size: {file_size_mb:.2f} MB")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())