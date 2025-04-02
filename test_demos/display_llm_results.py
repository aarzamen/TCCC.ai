#!/usr/bin/env python3
"""
Display LLM analysis results on Jetson display.

This script displays the results of LLM analysis on the Jetson's terminal/display.
It works with either the real or mock Phi-2 model.
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
from tccc.display.display_interface import DisplayInterface

def clear_terminal():
    """Clear the terminal screen."""
    os.system('clear')

def print_header(use_mock=True):
    """Print a header for the application."""
    print("=" * 80)
    print("TCCC LLM ANALYSIS - JETSON DISPLAY".center(80))
    print("=" * 80)
    if use_mock:
        print("Using MOCK Phi-2 model - Deterministic rules engine".center(80))
    else:
        print("Using real Phi-2 model (Microsoft)".center(80))
    print("=" * 80)
    print()

def main():
    """Main function to display LLM analysis results."""
    parser = argparse.ArgumentParser(description='Display LLM analysis results on Jetson')
    parser.add_argument('--input', '-i', help='Input text file', default=None)
    parser.add_argument('--json-file', '-j', help='JSON results file to display', default=None)
    parser.add_argument('--mock', action='store_true', help='Force use of mock model')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (terminal only)')
    args = parser.parse_args()
    
    # Set mock mode based on command line args
    use_mock = args.mock
    
    # Configure environment for model choice
    os.environ["TCCC_USE_MOCK_LLM"] = "1" if use_mock else "0"
    
    # Clear terminal and print header
    clear_terminal()
    print_header(use_mock)
    
    # Initialize display
    print("Initializing display...")
    display = None
    
    if not args.headless:
        try:
            from tccc.display.display_interface import DisplayInterface
            display = DisplayInterface()
            display_config = ConfigManager().load_config("display")
            display.initialize(display_config)
            print("Display initialized successfully")
        except Exception as e:
            print(f"WARNING: Could not initialize display: {str(e)}")
            print("Continuing in terminal-only mode")
    
    # Process input or load results
    results = None
    input_text = ""
    
    if args.json_file:
        # Load results from JSON file
        print(f"Loading results from {args.json_file}...")
        try:
            with open(args.json_file, 'r') as f:
                results = json.load(f)
            
            # Get input text from results
            if "input" in results:
                input_text = results["input"]
                
            print("Results loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load results: {str(e)}")
            return 1
    else:
        # We need to process new input
        # Initialize LLM analysis module
        print("\nInitializing LLM analysis module...")
        llm_analysis = LLMAnalysis()
        llm_config = ConfigManager().load_config("llm_analysis")
        
        # Force real model implementation if requested
        if not use_mock:
            if "model" in llm_config and "primary" in llm_config["model"]:
                llm_config["model"]["primary"]["force_real"] = True
        
        # Initialize LLM analysis
        init_success = llm_analysis.initialize(llm_config)
        if not init_success:
            print("Failed to initialize LLM analysis module")
            return 1
            
        # Get input text
        if args.input:
            # Read from file
            with open(args.input, 'r') as f:
                input_text = f.read()
            print(f"Read input from file: {args.input}")
        else:
            # Get interactive input
            print("\nEnter medical text for analysis (press Ctrl+D on empty line to finish):")
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
        
        # Create transcription object
        transcription = {'text': input_text}
        
        # Process transcription
        extracted_entities = llm_analysis.process_transcription(transcription)
        
        # Generate report
        report = llm_analysis.generate_report("tccc", extracted_entities)
        
        elapsed = time.time() - start_time
        
        # Create results object
        results = {
            "input": input_text,
            "entities": extracted_entities,
            "report": report,
            "metrics": {
                "elapsed_time": elapsed,
                "entities_count": len(extracted_entities)
            }
        }
        
        # Save results
        output_file = "llm_analysis_output.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    # Display results
    if display and not args.headless:
        # Show results on display
        try:
            # Set up display with TCCC header
            display.clear()
            display.set_header("TCCC MEDICAL ANALYSIS")
            
            # Display input summary
            display.add_text_block("PATIENT STATUS", input_text[:150] + "...")
            
            # Display vital signs
            vitals = [e for e in results["entities"] if e.get("type") == "vital_sign"][:5]
            vitals_text = "\n".join([f"{v.get('value')} {v.get('unit', '')}" for v in vitals])
            display.add_text_block("VITAL SIGNS", vitals_text)
            
            # Display procedures
            procedures = [e for e in results["entities"] if e.get("type") == "procedure"][:5]
            proc_text = "\n".join([f"{p.get('value')} ({p.get('time', 'unknown')})" for p in procedures])
            display.add_text_block("PROCEDURES", proc_text)
            
            # Display medications
            medications = [e for e in results["entities"] if e.get("type") == "medication"][:5]
            med_text = "\n".join([f"{m.get('value')} {m.get('dosage', '')}" for m in medications])
            display.add_text_block("MEDICATIONS", med_text)
            
            # Display report excerpt
            display.add_text_block("TCCC REPORT", results["report"]["content"][:200] + "...")
            
            # Update display
            display.update()
            print("Results displayed on Jetson display")
            
        except Exception as e:
            print(f"ERROR displaying results: {str(e)}")
    
    # Always display in terminal too
    print("\n" + "=" * 80)
    print("PATIENT MEDICAL ANALYSIS".center(80))
    print("=" * 80)
    
    print("\nINPUT TEXT:")
    print("-" * 80)
    print(input_text[:300] + ("..." if len(input_text) > 300 else ""))
    
    print("\nEXTRACTED ENTITIES:")
    print("-" * 80)
    categories = {}
    for entity in results["entities"]:
        category = entity.get("type", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(entity)
    
    # Print by category, limiting to 3 items per category
    for category, entities in categories.items():
        print(f"\n{category.upper()}:")
        for idx, entity in enumerate(entities[:3]):
            value = entity.get("value", entity.get("name", "Unknown"))
            details = []
            if "time" in entity:
                details.append(f"Time: {entity['time']}")
            if "dosage" in entity:
                details.append(f"Dosage: {entity['dosage']}")
            if "location" in entity:
                details.append(f"Location: {entity['location']}")
            detail_str = ", ".join(details)
            print(f"  - {value} {detail_str}")
        if len(entities) > 3:
            print(f"  - ... and {len(entities) - 3} more")
    
    print("\nTCCC REPORT:")
    print("-" * 80)
    if "report" in results and "content" in results["report"]:
        report_content = results["report"]["content"]
        # Split by newlines and limit to 10 lines
        report_lines = report_content.split("\n")
        print("\n".join(report_lines[:10]))
        if len(report_lines) > 10:
            print(f"... and {len(report_lines) - 10} more lines")
    
    print("\n" + "=" * 80)
    print(f"Processing time: {results['metrics']['elapsed_time']:.2f} seconds")
    print(f"Entities extracted: {results['metrics']['entities_count']}")
    print("=" * 80)
    
    # Wait for user to press enter before exiting (if using display)
    if display and not args.headless:
        input("\nPress Enter to exit...")
        display.clear()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())