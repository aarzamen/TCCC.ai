#!/usr/bin/env python3
"""
Simple tool for testing LLM analysis with custom input and saving results to a file.
This allows direct testing of the LLM analysis module with user-provided input.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Set up project path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Import LLM analysis module
from tccc.llm_analysis import LLMAnalysis
from tccc.utils.config_manager import ConfigManager
from tccc.document_library import DocumentLibrary

def main():
    """Run direct LLM analysis test with user input."""
    parser = argparse.ArgumentParser(description='Test LLM analysis with user input')
    parser.add_argument('--input', '-i', help='Input text file', default=None)
    parser.add_argument('--output', '-o', help='Output file for results', default='llm_analysis_output.json')
    parser.add_argument('--report', '-r', help='Generate report (medevac, zmist, soap, tccc)', default=None)
    parser.add_argument('--mock', action='store_true', help='Use mock LLM engine')
    args = parser.parse_args()
    
    # Set mock mode if requested
    if args.mock:
        os.environ["TCCC_USE_MOCK_LLM"] = "1"
    else:
        os.environ["TCCC_USE_MOCK_LLM"] = "0"
    
    # Get input text (from file, stdin, or interactively)
    input_text = ""
    if args.input:
        # Read from file
        with open(args.input, 'r') as f:
            input_text = f.read()
        print(f"Read input from file: {args.input}")
    else:
        # First check if we have stdin content
        if not sys.stdin.isatty():
            input_text = sys.stdin.read()
            print("Read input from stdin")
        else:
            # Otherwise, get interactive input
            print("Enter text to analyze (press Ctrl+D on empty line to finish):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            input_text = "\n".join(lines)
            print("\nInput received.")
    
    if not input_text.strip():
        print("Error: No input provided")
        return 1
    
    # Initialize the LLM analysis module
    print("\nInitializing LLM analysis module...")
    config_manager = ConfigManager()
    llm_config = config_manager.load_config("llm_analysis")
    
    llm_analysis = LLMAnalysis()
    doc_lib = DocumentLibrary()
    doc_lib_config = config_manager.load_config("document_library")
    doc_lib.initialize(doc_lib_config)
    
    init_success = llm_analysis.initialize(llm_config)
    if not init_success:
        print("Failed to initialize LLM analysis module")
        return 1
    
    llm_analysis.set_document_library(doc_lib)
    
    # Process the input text
    print("\nProcessing input text...")
    transcription = {'text': input_text}
    results = llm_analysis.process_transcription(transcription)
    
    # Generate report if requested
    if args.report:
        print(f"\nGenerating {args.report} report...")
        report = llm_analysis.generate_report(args.report, results)
        # Add report to results
        results = {
            'events': results,
            'report': report
        }
    
    # Save results to output file
    output_file = args.output
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Also display the results in the terminal
    print("\n===== ANALYSIS RESULTS =====")
    if args.report:
        print("\nExtracted Events:")
        for idx, event in enumerate(results['events']):
            print(f"{idx+1}. {event.get('value', event.get('name', event.get('event', 'Event')))}")
        
        print("\nGenerated Report:")
        print(results['report']['content'])
    else:
        for idx, event in enumerate(results):
            print(f"{idx+1}. {event.get('value', event.get('name', event.get('event', 'Event')))}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())