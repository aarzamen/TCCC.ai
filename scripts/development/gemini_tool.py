#!/usr/bin/env python3
"""
Gemini Tool - Utility for leveraging Gemini 2.5 Pro's 1M token context window
for codebase analysis and other large-context tasks.

Usage:
    ./gemini_tool.py --query "Analyze the relationship between audio_pipeline and stt_engine"
    ./gemini_tool.py --files "src/tccc/audio_pipeline/*.py src/tccc/stt_engine/*.py" --query "Find integration issues"
    ./gemini_tool.py --dir "src/tccc" --query "Identify all event handlers" --recursive
    ./gemini_tool.py --glob "**/*.py" --query "Find all uses of the config manager"
"""

import argparse
import glob
import os
import json
import sys
from pathlib import Path
import google.generativeai as genai

def setup_argparse():
    parser = argparse.ArgumentParser(description="Invoke Gemini 2.5 Pro for codebase analysis")
    parser.add_argument("--query", type=str, required=True, help="The query to send to Gemini")
    
    # Input sources (use only one)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--files", type=str, help="Space-separated list of files to analyze")
    input_group.add_argument("--dir", type=str, help="Directory to analyze")
    input_group.add_argument("--glob", type=str, help="Glob pattern to match files")
    input_group.add_argument("--text", type=str, help="Direct text input to send")
    
    # Optional arguments
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument("--output", type=str, help="Output file for Gemini's response")
    parser.add_argument("--max-files", type=int, default=50, help="Maximum number of files to process")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro-latest", help="Gemini model to use")
    
    return parser

def get_api_key():
    """Get the Gemini API key from environment variable or config file."""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        # Try to read from a config file
        config_paths = [
            Path.home() / ".gemini_config",
            Path.home() / ".config" / "gemini" / "config", 
            Path(__file__).parent / "gemini_config.json"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        api_key = config.get("api_key")
                        if api_key:
                            break
                except:
                    pass
    
    if not api_key:
        print("Error: Gemini API key not found. Please set the GEMINI_API_KEY environment variable "
              "or create a config file at ~/.gemini_config or ~/.config/gemini/config")
        sys.exit(1)
        
    return api_key

def get_file_contents(file_paths, max_files=50):
    """Read and return the contents of multiple files."""
    contents = []
    
    for i, file_path in enumerate(file_paths):
        if i >= max_files:
            print(f"Warning: Reached maximum file limit ({max_files}). Truncating.")
            break
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                contents.append(f"File: {file_path}\n\n{text}\n\n{'='*80}\n")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return "\n".join(contents)

def invoke_gemini(text, query, model="gemini-1.5-pro-latest"):
    """Send the text and query to Gemini and return the response."""
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(model)
    
    full_prompt = f"""
Task: Analyze code and respond to the following query.

Query: {query}

Code to analyze:

{text}

Please provide a detailed analysis based on the query above. Focus on being accurate, specific, and helpful.
"""
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error invoking Gemini API: {e}"

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    file_paths = []
    
    # Process different input sources
    if args.files:
        file_paths = args.files.split()
    elif args.dir:
        if args.recursive:
            file_paths = glob.glob(os.path.join(args.dir, "**"), recursive=True)
        else:
            file_paths = glob.glob(os.path.join(args.dir, "*"))
        file_paths = [p for p in file_paths if os.path.isfile(p)]
    elif args.glob:
        file_paths = glob.glob(args.glob, recursive=True)
    
    # Prepare the input text
    if args.text:
        text = args.text
    elif file_paths:
        print(f"Processing {len(file_paths)} files...")
        text = get_file_contents(file_paths, args.max_files)
    else:
        text = ""
    
    if not text.strip() and not args.query.strip():
        parser.print_help()
        sys.exit(1)
    
    # Invoke Gemini
    print(f"Sending query to Gemini: {args.query}")
    print(f"Text length: {len(text)} characters")
    
    response = invoke_gemini(text, args.query, args.model)
    
    # Output handling
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Response saved to {args.output}")
    else:
        print("\n" + "="*80 + "\n")
        print(response)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()