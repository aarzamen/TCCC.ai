#!/usr/bin/env python3
"""
Test script to generate LLM prompts with the RAG system
"""

import os
import sys

from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

def main():
    # Load configuration
    config = ConfigManager().load_config('document_library')
    
    # Initialize document library
    dl = DocumentLibrary()
    dl.initialize(config)
    
    # Test query
    query = "How to treat a tension pneumothorax?"
    print(f"Generating prompt for: {query}")
    
    # Generate standard prompt
    prompt = dl.generate_llm_prompt(query)
    print(f"\nPrompt length: {len(prompt)} characters")
    print("\n--- Prompt Preview ---")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    # Test with different context lengths
    context_lengths = [500, 1000, 2000]
    for length in context_lengths:
        print(f"\n\nPrompt with context length {length}:")
        try:
            prompt = dl.generate_llm_prompt(query, max_context_length=length)
            print(f"Length: {len(prompt)} characters")
            print("\n--- Preview ---")
            print(prompt[:200] + "...")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Try with different templates based on query type
    query_types = [
        "What is tension pneumothorax?",  # explanation
        "How to perform needle decompression?",  # procedure
        "How to document a tension pneumothorax on a TCCC card?"  # form
    ]
    
    for q in query_types:
        print(f"\n\nPrompt for query type test: {q}")
        prompt = dl.generate_llm_prompt(q)
        # Extract just the first few lines to see the template type
        preview = "\n".join(prompt.split("\n")[:8])
        print(preview + "...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())