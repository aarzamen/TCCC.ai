#!/usr/bin/env python3
"""
Jetson RAG Explorer - A lightweight interface for the TCCC RAG system

This script provides a terminal-based interface for querying the TCCC RAG system,
optimized for Jetson hardware. It supports:
- PDF processing and indexing
- Medical terminology querying
- Multiple search strategies
- Performance optimization for resource-constrained environments
"""

import os
import sys
import time
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import TCCC modules
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/rag_explorer.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Common TCCC-related queries for suggestions
COMMON_QUERIES = [
    "What is the MARCH algorithm?",
    "How to treat a tension pneumothorax?",
    "Tourniquet application procedure",
    "TCCC guidelines for airway management",
    "Signs and symptoms of hypovolemic shock",
    "Battlefield assessment steps",
    "Proper use of hemostatic agents",
    "NPA insertion technique",
    "Treatment for junctional hemorrhage",
    "Triage categories in TCCC"
]

def print_header():
    """Print the application header."""
    print("\n" + "=" * 80)
    print("  TCCC RAG EXPLORER - Medical Knowledge Query System".center(80))
    print("=" * 80)
    print("  Optimized for Jetson Hardware | Type 'help' for commands".center(80))
    print("-" * 80 + "\n")

def print_help():
    """Print help information."""
    print("\nAvailable commands:")
    print("  help              - Show this help message")
    print("  query <text>      - Query the RAG system with the specified text")
    print("  strategy <name>   - Set query strategy (semantic, keyword, hybrid, expanded)")
    print("  explain <term>    - Get explanation for a medical term")
    print("  suggest           - Show query suggestions")
    print("  stats             - Show system statistics")
    print("  add <pdf_path>    - Add a PDF document to the knowledge base")
    print("  clear             - Clear the screen")
    print("  exit              - Exit the application\n")

def show_query_suggestions():
    """Show query suggestions."""
    print("\nQuery suggestions:")
    for i, query in enumerate(COMMON_QUERIES, 1):
        print(f"  {i}. {query}")
    print()

def format_query_result(result: Dict[str, Any]) -> str:
    """Format query result for display."""
    output = []
    
    # Add query info
    query = result.get("query", "")
    strategy = result.get("strategy", "")
    processing_time = result.get("processing_time", 0)
    cache_hit = result.get("cache_hit", False)
    
    output.append(f"Query: '{query}'")
    output.append(f"Strategy: {strategy}")
    output.append(f"Time: {processing_time:.4f}s {'(cached)' if cache_hit else ''}")
    
    # Add results
    results = result.get("results", [])
    output.append(f"Found {len(results)} results:\n")
    
    for i, res in enumerate(results, 1):
        # Get result info
        doc_id = res.get("id", res.get("document_id", "unknown"))
        score = res.get("score", 0)
        metadata = res.get("metadata", {})
        text = res.get("text", "")
        
        # Format result
        output.append(f"Result {i} (Score: {score:.4f}):")
        
        # Add metadata if available
        if metadata:
            title = metadata.get("title", "")
            source = metadata.get("source", metadata.get("file_name", ""))
            if title:
                output.append(f"Title: {title}")
            if source:
                output.append(f"Source: {source}")
        
        # Add text preview (truncated for display)
        max_text_length = 300
        if text:
            preview = text[:max_text_length] + "..." if len(text) > max_text_length else text
            output.append(f"Content: {preview}")
        
        # Add separator
        if i < len(results):
            output.append("-" * 40)
    
    return "\n".join(output)

def explain_medical_term(doc_lib, term: str) -> str:
    """Get explanation for a medical term."""
    if not hasattr(doc_lib, "medical_vocabulary") or not doc_lib.medical_vocabulary:
        return "Medical vocabulary not available"
    
    info = doc_lib.medical_vocabulary.get_term_info(term)
    if not info:
        return f"Term '{term}' not found in medical vocabulary"
    
    # Format explanation based on term type
    term_type = info.get("type", "unknown")
    if term_type == "abbreviation":
        explanation = f"{term} - {info.get('expansion', '')}"
    elif "synonyms" in info and info["synonyms"]:
        synonyms = ", ".join(info["synonyms"])
        explanation = f"{term} - related terms: {synonyms}"
    elif "definition" in info:
        explanation = f"{term} - {info.get('definition', '')}"
    else:
        explanation = f"{term} - No detailed information available"
    
    # Add category if available
    category = info.get("category", "")
    if category:
        explanation += f"\nCategory: {category}"
    
    return explanation

def get_system_stats(doc_lib) -> str:
    """Get system statistics."""
    output = []
    output.append("\nSystem Statistics:")
    
    # Get document library status
    status = doc_lib.get_status()
    output.append(f"Documents: {status['documents']['count']}")
    output.append(f"Chunks: {status['documents']['chunks']}")
    output.append(f"Vectors: {status['index']['vectors']}")
    output.append(f"Model: {status['model']['name']}")
    
    # Get cache statistics if available
    if "cache" in status:
        cache = status["cache"]
        output.append("\nCache:")
        output.append(f"  Memory entries: {cache.get('memory_entries', 0)}")
        output.append(f"  Disk entries: {cache.get('disk_entries', 0)}")
    
    # Get component status
    if "components" in status:
        components = status["components"]
        output.append("\nComponents:")
        for name, comp_status in components.items():
            if isinstance(comp_status, bool):
                status_str = "✓" if comp_status else "✗"
            else:
                status_str = comp_status
            output.append(f"  {name}: {status_str}")
    
    return "\n".join(output)

def process_pdf(doc_lib, pdf_path):
    """Process and add a PDF to the RAG database."""
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"
    
    # Create a copy in the documents directory
    docs_dir = Path("data/documents")
    docs_dir.mkdir(exist_ok=True, parents=True)
    dest_path = docs_dir / os.path.basename(pdf_path)
    shutil.copy2(pdf_path, dest_path)
    
    print(f"\nProcessing PDF: {pdf_path}")
    
    # Try to use process_rag_documents.py if it exists
    if os.path.exists("process_rag_documents.py"):
        try:
            subprocess.run(["python", "process_rag_documents.py", "--input", str(dest_path)], 
                          check=True)
            return f"Successfully processed {pdf_path} and added to RAG database"
        except subprocess.CalledProcessError:
            print("External processor failed, falling back to direct processing...")
    
    # Fall back to direct processing using document library
    try:
        # Check if PDF processing is supported
        if hasattr(doc_lib, "document_processor") and doc_lib.document_processor:
            doc_data = {
                "file_path": str(dest_path),
                "metadata": {
                    "source": str(dest_path),
                    "title": os.path.basename(pdf_path)
                }
            }
            
            doc_id = doc_lib.add_document(doc_data)
            if doc_id:
                return f"Successfully added document with ID: {doc_id}"
            else:
                return f"Failed to add document to RAG database"
        else:
            return "PDF processing not supported in this configuration"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def launch_terminal_interface(config):
    """Launch the terminal interface."""
    # Initialize document library
    print("Initializing Document Library...")
    doc_lib = DocumentLibrary()
    success = doc_lib.initialize(config)
    
    if not success:
        print("Failed to initialize Document Library")
        return 1
    
    print("Document Library initialized successfully!")
    
    # Get default query strategy
    default_strategy = config.get("search", {}).get("default_strategy", "hybrid")
    current_strategy = default_strategy
    
    # Main application loop
    print_header()
    print("Type a query or command. Use 'help' for assistance.")
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            # Process commands
            if not user_input:
                continue
            
            if user_input.lower() == "exit":
                print("Exiting RAG Explorer. Goodbye!")
                break
            
            elif user_input.lower() == "help":
                print_help()
            
            elif user_input.lower() == "suggest":
                show_query_suggestions()
            
            elif user_input.lower() == "stats":
                print(get_system_stats(doc_lib))
            
            elif user_input.lower() == "clear":
                os.system("clear" if os.name == "posix" else "cls")
                print_header()
            
            elif user_input.lower().startswith("strategy "):
                new_strategy = user_input[9:].strip().lower()
                if new_strategy in ["semantic", "keyword", "hybrid", "expanded"]:
                    current_strategy = new_strategy
                    print(f"Query strategy set to: {current_strategy}")
                else:
                    print(f"Invalid strategy: {new_strategy}")
                    print("Valid strategies: semantic, keyword, hybrid, expanded")
            
            elif user_input.lower().startswith("explain "):
                term = user_input[8:].strip()
                if term:
                    explanation = explain_medical_term(doc_lib, term)
                    print(f"\n{explanation}")
                else:
                    print("Please specify a term to explain")
            
            elif user_input.lower().startswith("add "):
                pdf_path = user_input[4:].strip()
                if pdf_path:
                    result = process_pdf(doc_lib, pdf_path)
                    print(f"\n{result}")
                else:
                    print("Please specify a PDF file path")
            
            elif user_input.lower().startswith("query "):
                query_text = user_input[6:].strip()
                if query_text:
                    print(f"\nExecuting query with strategy '{current_strategy}'...")
                    start_time = time.time()
                    result = doc_lib.advanced_query(query_text, strategy=current_strategy)
                    print(format_query_result(result))
                else:
                    print("Please specify a query")
            
            else:
                # Treat as a query
                print(f"\nExecuting query with strategy '{current_strategy}'...")
                start_time = time.time()
                result = doc_lib.advanced_query(user_input, strategy=current_strategy)
                print(format_query_result(result))
        
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Jetson RAG Explorer")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--low-memory", action="store_true", help="Enable low memory mode")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file to process")
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        logger.info(f"Loading custom config from {args.config}")
        config = config_manager.load_config_from_file(args.config)
    else:
        optimized_config = "config/optimized_jetson_rag.yaml"
        if os.path.exists(optimized_config):
            logger.info(f"Loading optimized config from {optimized_config}")
            config = config_manager.load_config_from_file(optimized_config)
        else:
            logger.info("Loading default document library config")
            config = config_manager.load_config("document_library")
    
    # Apply low memory mode if specified
    if args.low_memory:
        logger.info("Low memory mode enabled")
        if "cache" not in config:
            config["cache"] = {}
        
        config["cache"]["max_memory_entries"] = 20
        
        if "embedding" not in config:
            config["embedding"] = {}
        
        config["embedding"]["batch_size"] = 8
        config["embedding"]["use_gpu"] = False
    
    # Initialize document library for processing PDF if needed
    if args.pdf_path:
        print("Initializing Document Library...")
        doc_lib = DocumentLibrary()
        success = doc_lib.initialize(config)
        
        if not success:
            print("Failed to initialize Document Library")
            return 1
        
        # Process the PDF
        result = process_pdf(doc_lib, args.pdf_path)
        print(result)
    
    # Launch the terminal interface
    return launch_terminal_interface(config)

if __name__ == "__main__":
    sys.exit(main())