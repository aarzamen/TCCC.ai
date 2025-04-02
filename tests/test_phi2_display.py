#!/usr/bin/env python3
"""
Test script for LLM analysis with Jetson terminal display.

This script demonstrates LLM analysis capabilities by using either the real Phi-2 model
or the mock implementation and displays the results directly in the terminal.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Set up project path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Import required modules
from tccc.llm_analysis import LLMAnalysis
from tccc.utils.config_manager import ConfigManager

def clear_terminal():
    """Clear the terminal screen."""
    os.system('clear')

def print_header(use_mock=True):
    """Print a header for the application."""
    print("=" * 80)
    print("LLM ANALYSIS TEST - TCCC PROJECT".center(80))
    print("=" * 80)
    if use_mock:
        print("Using MOCK Phi-2 model - Real model not available or incomplete".center(80))
    else:
        print("Using real Phi-2 model (Microsoft)".center(80))
    print("=" * 80)
    print()

def main():
    """Main function to test LLM analysis."""
    parser = argparse.ArgumentParser(description='Test LLM analysis with terminal display')
    parser.add_argument('--input', '-i', help='Input text file', default=None)
    parser.add_argument('--mock', action='store_true', help='Force use of mock model')
    parser.add_argument('--real', action='store_true', help='Force use of real model (will fail if not available)')
    args = parser.parse_args()
    
    # Set mock mode based on command line args
    use_mock = args.mock
    if args.real:
        use_mock = False  # Force real model if explicitly requested
    
    # Configure environment for model choice
    os.environ["TCCC_USE_MOCK_LLM"] = "1" if use_mock else "0"
    
    # Clear terminal and print header
    clear_terminal()
    print_header(use_mock)
    
    print("Initializing LLM analysis module...")
    
    # Load configuration
    config_manager = ConfigManager()
    llm_config = config_manager.load_config("llm_analysis")
    
    # Force real model implementation if requested
    if not use_mock:
        if "model" in llm_config and "primary" in llm_config["model"]:
            llm_config["model"]["primary"]["force_real"] = True
    
    # Initialize LLM analysis
    llm_analysis = LLMAnalysis()
    
    try:
        # Initialize module
        init_success = llm_analysis.initialize(llm_config)
        if not init_success:
            print("Failed to initialize LLM analysis module")
            return 1
        
        # Get system status to check model info
        status = llm_analysis.get_status()
        
        # Check if we're using mock model
        using_mock = True
        if "llm_engine" in status and "models" in status["llm_engine"]:
            if "primary" in status["llm_engine"]["models"]:
                primary_model = status["llm_engine"]["models"]["primary"]
                if "name" in primary_model and "mock" not in primary_model["name"].lower():
                    using_mock = False
        
        if using_mock and not use_mock:
            # We wanted real but got mock
            print("\nWARNING: Requested real model but using mock implementation instead")
            print("LLM module was initialized with mock model")
        
        print("\nLLM analysis module initialized successfully!")
        print(f"Using {'mock' if using_mock else 'real'} model implementation")
        print(f"Model status: {json.dumps(status['llm_engine']['models'], indent=2)}")
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM analysis: {str(e)}")
        return 1
    
    # Get input text
    input_text = ""
    if args.input:
        # Read from file
        with open(args.input, 'r') as f:
            input_text = f.read()
        print(f"Read input from file: {args.input}")
    else:
        # Get interactive input
        print("\nEnter text for analysis (press Ctrl+D on empty line to finish):")
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
    
    # Process the input text
    print("\nProcessing input with LLM analysis...")
    start_time = time.time()
    
    try:
        # Create transcription object
        transcription = {'text': input_text}
        
        # Process transcription
        results = llm_analysis.process_transcription(transcription)
        
        # Generate report
        report = llm_analysis.generate_report("tccc", results)
        
        elapsed = time.time() - start_time
        
        # Display the results
        print("\n" + "=" * 80)
        print("MEDICAL ENTITY EXTRACTION RESULTS".center(80))
        print("=" * 80)
        
        # Print extracted entities
        for idx, event in enumerate(results):
            event_type = event.get('type', event.get('name', ''))
            event_value = event.get('value', event.get('name', event.get('event', 'Unknown')))
            event_time = event.get('time', '')
            print(f"{idx+1}. {event_type}: {event_value} {event_time}")
        
        print("\n" + "=" * 80)
        print("GENERATED TCCC REPORT".center(80))
        print("=" * 80)
        print(report['content'])
        
        # Print metrics
        print("\n" + "-" * 80)
        print(f"Processing time: {elapsed:.2f} seconds")
        print(f"Entities extracted: {len(results)}")
        print("-" * 80)
        
        # Save the results
        output_file = "llm_analysis_output.json"
        with open(output_file, 'w') as f:
            json.dump({
                "input": input_text,
                "entities": results,
                "report": report,
                "metrics": {
                    "elapsed_time": elapsed,
                    "entities_count": len(results)
                }
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())