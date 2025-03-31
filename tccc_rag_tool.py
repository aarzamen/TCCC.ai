#!/usr/bin/env python3
"""
TCCC RAG Testing and Demo Tool

A comprehensive tool for testing and demonstrating the TCCC RAG system capabilities,
optimized for medical terminology and Jetson deployment.

Features:
- Multiple query strategies (semantic, keyword, hybrid, expanded)
- Medical terminology enhancement and testing
- Performance benchmarking for Jetson optimization
- Query caching and memory efficiency
- Rich visualization of results
- Export capabilities for offline analysis
"""

import os
import sys
import time
import json
import argparse
import sqlite3
import datetime
import textwrap
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'src'))

# TCCC imports
from tccc.document_library import DocumentLibrary
from tccc.document_library.medical_vocabulary import MedicalVocabularyManager
from tccc.utils import ConfigManager

# For report generation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Check if rich is available for enhanced output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Initialize console if rich is available
if HAS_RICH:
    console = Console()
else:
    # Fallback to standard print functions
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
            
        def rule(self, title=None):
            if title:
                print(f"\n{'=' * 10} {title} {'=' * 10}")
            else:
                print(f"\n{'=' * 30}")
    
    console = FallbackConsole()

# TCCC-specific medical queries for testing
TCCC_MEDICAL_QUERIES = [
    # Hemorrhage control queries
    "How do I apply a CAT tourniquet?",
    "Treatment for junctional hemorrhage",
    "When should hemostatic agents be used?",
    
    # Airway management queries
    "NPA insertion procedure",
    "Signs of airway obstruction in combat",
    "TCCC recommendations for airway management",
    
    # Respiratory queries
    "How to recognize tension pneumothorax",
    "Needle decompression procedure",
    "Battlefield assessment of respiration",
    
    # Circulation queries
    "Hypovolemic shock treatment",
    "How to assess circulation in TCCC",
    "Field blood transfusion protocol",
    
    # Head injury / hypothermia queries
    "Combat TBI assessment guidelines",
    "Preventing hypothermia in casualties",
    "Head injury treatment in tactical field care",
    
    # General TCCC queries
    "MARCH algorithm explanation",
    "Principles of tactical field care",
    "TCCC card documentation guidelines"
]

class TcccRagTool:
    """
    TCCC RAG Testing and Demo Tool implementation.
    
    This class provides:
    - Document library initialization with various configurations
    - Multiple query strategies
    - Medical terminology testing
    - Performance benchmarking
    - Result visualization
    """
    
    def __init__(self, args):
        """Initialize the tool with command line arguments."""
        self.args = args
        self.config = None
        self.doc_lib = None
        self.vocab_manager = None
        self.initialized = False
        self.results_cache = {}
        
        # Colors for terminal output
        self.colors = {
            "header": "\033[95m",
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "reset": "\033[0m",
            "bold": "\033[1m",
            "underline": "\033[4m"
        }
        
    def initialize(self):
        """Initialize the RAG components."""
        try:
            # Load configuration
            self._load_configuration()
            
            # Initialize document library
            self._initialize_document_library()
            
            # Initialize vocabulary manager
            self._initialize_vocabulary_manager()
            
            self.initialized = True
            return True
        except Exception as e:
            console.print(f"[bold red]Error initializing RAG tool: {str(e)}[/bold red]")
            return False
    
    def _load_configuration(self):
        """Load the appropriate configuration."""
        config_manager = ConfigManager()
        
        if self.args.config:
            # Load custom config file
            config_path = self.args.config
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            self.config = config_manager.load_config_from_file(config_path)
            console.print(f"Loaded custom config from: [bold]{config_path}[/bold]")
        else:
            # Try to find the best available config
            config_options = [
                # Optimized config for Jetson
                ("config/optimized_jetson_rag.yaml", "Optimized Jetson config"),
                ("config/jetson_mvp.yaml", "Jetson MVP config"),
                # Standard document library config
                ("config/document_library.yaml", "Standard document library config")
            ]
            
            for config_path, desc in config_options:
                if os.path.exists(config_path):
                    self.config = config_manager.load_config_from_file(config_path)
                    console.print(f"Loaded {desc} from: [bold]{config_path}[/bold]")
                    break
            
            # If no config files found, use the default
            if not self.config:
                self.config = config_manager.load_config("document_library")
                console.print("Loaded default document library configuration")
        
        # Apply optimization settings if requested
        if self.args.optimize_for_jetson:
            self._apply_jetson_optimizations()
    
    def _apply_jetson_optimizations(self):
        """Apply Jetson-specific optimizations to the configuration."""
        if not self.config:
            return
        
        # Set optimized embedding parameters
        if "embedding" in self.config:
            console.print("[bold green]Applying Jetson optimizations...[/bold green]")
            
            # Update embedding model settings
            self.config["embedding"]["normalize"] = True
            self.config["embedding"]["batch_size"] = 8  # Smaller batch size for Jetson
            
            # Enable quantization if supported
            if "quantization" not in self.config["embedding"]:
                self.config["embedding"]["quantization"] = {
                    "enabled": True,
                    "bits": 8,  # int8 quantization
                    "optimize_for_gpu": True
                }
            
            # Add TensorRT acceleration if not present
            if "tensorrt" not in self.config["embedding"]:
                self.config["embedding"]["tensorrt"] = {
                    "enabled": True,
                    "precision": "fp16"
                }
        
        # Optimize cache settings
        if "cache" not in self.config:
            self.config["cache"] = {}
            
        self.config["cache"]["memory_limit"] = "256MB"  # Limit memory cache size for Jetson
        self.config["cache"]["disk_enabled"] = True
        self.config["cache"]["ttl"] = 86400  # 24 hours cache lifetime
        
        console.print("[bold green]Jetson optimizations applied[/bold green]")
    
    def _initialize_document_library(self):
        """Initialize the document library component."""
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Initializing document library...[/bold green]"),
                transient=True,
            ) as progress:
                progress.add_task("init", total=1)
                
                # Initialize document library
                self.doc_lib = DocumentLibrary()
                success = self.doc_lib.initialize(self.config)
        else:
            console.print("Initializing document library...")
            self.doc_lib = DocumentLibrary()
            success = self.doc_lib.initialize(self.config)
        
        if not success:
            raise RuntimeError("Failed to initialize document library")
        
        # Display library status
        status = self.doc_lib.get_status()
        
        if HAS_RICH:
            table = Table(title="Document Library Status")
            table.add_column("Component", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Documents", str(status.get("documents", {}).get("count", 0)))
            table.add_row("Chunks", str(status.get("documents", {}).get("chunks", 0)))
            table.add_row("Vectors", str(status.get("index", {}).get("vectors", 0)))
            table.add_row("Model", status.get("model", {}).get("name", "Unknown"))
            
            console.print(table)
        else:
            console.print("\nDocument Library Status:")
            console.print(f"Documents: {status.get('documents', {}).get('count', 0)}")
            console.print(f"Chunks: {status.get('documents', {}).get('chunks', 0)}")
            console.print(f"Vectors: {status.get('index', {}).get('vectors', 0)}")
            console.print(f"Model: {status.get('model', {}).get('name', 'Unknown')}")
    
    def _initialize_vocabulary_manager(self):
        """Initialize the medical vocabulary manager."""
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Initializing medical vocabulary...[/bold green]"),
                transient=True,
            ) as progress:
                progress.add_task("init", total=1)
                
                # Initialize vocabulary manager
                self.vocab_manager = MedicalVocabularyManager(self.config)
                success = self.vocab_manager.initialize()
        else:
            console.print("\nInitializing medical vocabulary...")
            self.vocab_manager = MedicalVocabularyManager(self.config)
            success = self.vocab_manager.initialize()
        
        if not success:
            console.print("[yellow]Warning: Failed to initialize medical vocabulary. Some features will be limited.[/yellow]")
    
    def run_query(self, query_text, strategy="hybrid", limit=5, timer=True):
        """
        Execute a query against the document library.
        
        Args:
            query_text: Query text
            strategy: Query strategy (semantic, keyword, hybrid, expanded)
            limit: Maximum number of results
            timer: Whether to time the query execution
            
        Returns:
            Query results dictionary
        """
        if not self.initialized or not self.doc_lib:
            console.print("[bold red]Error: RAG tool not initialized[/bold red]")
            return {"error": "RAG tool not initialized"}
        
        # Check cache
        cache_key = f"{query_text}_{strategy}_{limit}"
        if cache_key in self.results_cache:
            results = self.results_cache[cache_key]
            results["cache_hit"] = True
            return results
        
        start_time = time.time() if timer else None
        
        # Execute query
        try:
            results = self.doc_lib.advanced_query(
                query_text=query_text,
                strategy=strategy,
                limit=limit
            )
            
            if timer:
                execution_time = time.time() - start_time
                results["execution_time"] = execution_time
            
            # Cache results
            self.results_cache[cache_key] = results
            results["cache_hit"] = False
            
            return results
        except Exception as e:
            console.print(f"[bold red]Error executing query: {str(e)}[/bold red]")
            return {
                "error": str(e),
                "results": []
            }
    
    def display_results(self, results, detailed=False):
        """
        Display query results.
        
        Args:
            results: Query results dictionary
            detailed: Whether to show detailed results
        """
        if "error" in results:
            console.print(f"[bold red]Error: {results['error']}[/bold red]")
            return
        
        query = results.get("query", "Unknown query")
        strategy = results.get("strategy", "default")
        result_count = len(results.get("results", []))
        
        # Display query information
        if HAS_RICH:
            console.print(f"[bold]Query:[/bold] {query}")
            console.print(f"[bold]Strategy:[/bold] {strategy}")
            console.print(f"[bold]Results:[/bold] {result_count}")
            
            if "execution_time" in results:
                console.print(f"[bold]Execution time:[/bold] {results['execution_time']:.4f}s")
            
            if results.get("cache_hit", False):
                console.print("[bold yellow]Results from cache[/bold yellow]")
            
            console.rule()
        else:
            console.print(f"\nQuery: {query}")
            console.print(f"Strategy: {strategy}")
            console.print(f"Results: {result_count}")
            
            if "execution_time" in results:
                console.print(f"Execution time: {results['execution_time']:.4f}s")
            
            if results.get("cache_hit", False):
                console.print("Results from cache")
            
            console.print("\n" + "=" * 40)
        
        # Display results
        if result_count == 0:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Process medical terms if vocabulary manager is available
        medical_terms = {}
        if self.vocab_manager and query:
            try:
                medical_terms = self.vocab_manager.explain_medical_terms(query)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to process medical terms: {str(e)}[/yellow]")
        
        # Display medical terms if found
        if medical_terms and HAS_RICH:
            medical_panel = Panel(
                "\n".join([f"• {term} - {explanation}" for term, explanation in medical_terms.items()]),
                title="Medical Terms Detected",
                border_style="blue"
            )
            console.print(medical_panel)
            console.print()
        elif medical_terms:
            console.print("\nMedical Terms Detected:")
            for term, explanation in medical_terms.items():
                console.print(f"• {term} - {explanation}")
            console.print()
        
        # Display results
        for i, result in enumerate(results.get("results", [])):
            if HAS_RICH:
                result_num = f"[bold cyan]Result {i+1}[/bold cyan]"
                score = f"[bold green]Score: {result.get('score', 0):.4f}[/bold green]"
                console.print(f"{result_num} ({score})")
                
                # Display metadata
                if "metadata" in result and result["metadata"]:
                    meta = result["metadata"]
                    meta_text = []
                    
                    for key, value in meta.items():
                        if key in ["text", "content"]:  # Skip large text fields
                            continue
                        meta_text.append(f"{key}: {value}")
                    
                    if meta_text:
                        console.print("[bold]Metadata:[/bold]")
                        for line in meta_text:
                            console.print(f"  {line}")
                
                # Display text content
                if "text" in result:
                    text = result["text"]
                    if not detailed and len(text) > 500:
                        text = text[:497] + "..."
                    
                    # Display as markdown if it looks like formatted text
                    if "# " in text or "**" in text or "- " in text:
                        try:
                            console.print(Markdown(text))
                        except:
                            console.print(text)
                    else:
                        console.print(text)
                
                console.rule()
            else:
                # Plain text display
                console.print(f"\nResult {i+1} (Score: {result.get('score', 0):.4f})")
                
                # Display metadata
                if "metadata" in result and result["metadata"]:
                    meta = result["metadata"]
                    console.print("Metadata:")
                    for key, value in meta.items():
                        if key in ["text", "content"]:  # Skip large text fields
                            continue
                        console.print(f"  {key}: {value}")
                
                # Display text content
                if "text" in result:
                    text = result["text"]
                    if not detailed and len(text) > 500:
                        text = text[:497] + "..."
                    
                    console.print("\nContent:")
                    console.print(textwrap.fill(text, width=100))
                
                console.print("\n" + "-" * 40)
    
    def generate_llm_prompt(self, query, strategy="hybrid", limit=3):
        """
        Generate a prompt for LLM with RAG context.
        
        Args:
            query: Query text
            strategy: Query strategy
            limit: Maximum number of results
            
        Returns:
            Formatted prompt string
        """
        if not self.initialized or not self.doc_lib:
            return f"""
            You are TCCC.ai, an expert in Tactical Combat Casualty Care.
            Please answer the following query based on your general knowledge:
            
            {query}
            
            Note: Unable to retrieve specific context from the Document Library.
            """
        
        try:
            # Generate prompt using DocumentLibrary's method
            prompt = self.doc_lib.generate_llm_prompt(
                query=query,
                strategy=strategy,
                limit=limit
            )
            
            return prompt
        except Exception as e:
            console.print(f"[bold red]Error generating prompt: {str(e)}[/bold red]")
            
            # Fallback minimal prompt
            return f"""
            You are TCCC.ai, an expert in Tactical Combat Casualty Care.
            Please answer the following query based on your general knowledge:
            
            {query}
            
            Note: Unable to retrieve context from the Document Library due to an error.
            """
    
    def test_medical_vocabulary(self):
        """Test medical vocabulary functionality and coverage."""
        if not self.initialized or not self.vocab_manager:
            console.print("[bold red]Error: Medical vocabulary manager not initialized[/bold red]")
            return
        
        console.rule("Medical Vocabulary Test")
        
        # Test key TCCC terms
        test_terms = [
            "TCCC", "TQ", "CAT", "Hemorrhage", "Tourniquet", "NPA",
            "Tension pneumothorax", "Needle decompression", "Hemostatic",
            "MARCH", "Airway", "Respiration", "Circulation", "Hypothermia"
        ]
        
        if HAS_RICH:
            table = Table(title="Medical Term Coverage")
            table.add_column("Term", style="cyan")
            table.add_column("Found", style="green")
            table.add_column("Type", style="blue")
            table.add_column("Information", style="yellow")
            
            found_terms = 0
            for term in test_terms:
                info = self.vocab_manager.get_term_info(term)
                if info:
                    found_terms += 1
                    term_type = info.get("type", "unknown")
                    
                    if term_type == "abbreviation":
                        expansion = info.get("expansion", "")
                        table.add_row(term, "✓", term_type, expansion)
                    elif "synonyms" in info and info["synonyms"]:
                        synonyms = ", ".join(info["synonyms"][:3])
                        table.add_row(term, "✓", term_type, f"Synonyms: {synonyms}...")
                    else:
                        definition = info.get("definition", "")
                        table.add_row(term, "✓", term_type, definition)
                else:
                    table.add_row(term, "✗", "", "")
            
            console.print(table)
            
            coverage = found_terms / len(test_terms) * 100
            console.print(f"Vocabulary coverage: [bold]{coverage:.1f}%[/bold] ({found_terms}/{len(test_terms)} terms found)")
        else:
            # Plain text output
            console.print("\nMedical Term Coverage:")
            
            found_terms = 0
            for term in test_terms:
                info = self.vocab_manager.get_term_info(term)
                if info:
                    found_terms += 1
                    term_type = info.get("type", "unknown")
                    
                    if term_type == "abbreviation":
                        expansion = info.get("expansion", "")
                        console.print(f"✓ {term} ({term_type}): {expansion}")
                    elif "synonyms" in info and info["synonyms"]:
                        synonyms = ", ".join(info["synonyms"][:3])
                        console.print(f"✓ {term} ({term_type}): Synonyms: {synonyms}...")
                    else:
                        definition = info.get("definition", "")
                        console.print(f"✓ {term} ({term_type}): {definition}")
                else:
                    console.print(f"✗ {term}: Not found in vocabulary")
            
            coverage = found_terms / len(test_terms) * 100
            console.print(f"\nVocabulary coverage: {coverage:.1f}% ({found_terms}/{len(test_terms)} terms found)")
        
        # Test query expansion
        console.rule("Query Expansion Test")
        
        test_queries = [
            "How do I apply a tourniquet?",
            "Treatment for tension pneumothorax",
            "MARCH algorithm for tactical field care"
        ]
        
        for query in test_queries:
            expansions = self.vocab_manager.expand_query(query)
            
            if HAS_RICH:
                console.print(f"[bold]Original query:[/bold] {query}")
                console.print(f"[bold]Generated {len(expansions)} expansions:[/bold]")
                
                for i, exp in enumerate(expansions, 1):
                    console.print(f"  {i}. {exp}")
                console.print()
            else:
                console.print(f"\nOriginal query: '{query}'")
                console.print(f"Generated {len(expansions)} expansions:")
                
                for i, exp in enumerate(expansions, 1):
                    console.print(f"  {i}. {exp}")
        
        # Test term extraction
        console.rule("Medical Term Extraction Test")
        
        test_texts = [
            "The patient presented with a tension pneumothorax requiring immediate needle decompression.",
            "Applied a CAT tourniquet to control hemorrhage from the lower extremity.",
            "Following the MARCH algorithm, first address massive hemorrhage, then airway, respiration, circulation."
        ]
        
        for i, text in enumerate(test_texts, 1):
            terms = self.vocab_manager.extract_medical_terms(text)
            explanations = self.vocab_manager.explain_medical_terms(text)
            
            if HAS_RICH:
                console.print(f"[bold]Text {i}:[/bold] {text}")
                console.print(f"[bold]Extracted {len(terms)} medical terms:[/bold]")
                
                for term in terms:
                    console.print(f"  • {term}")
                
                if explanations:
                    console.print("[bold]Explanations:[/bold]")
                    for term, explanation in explanations.items():
                        console.print(f"  • {explanation}")
                console.print()
            else:
                console.print(f"\nText {i}: {text}")
                console.print(f"Extracted {len(terms)} medical terms:")
                
                for term in terms:
                    console.print(f"  • {term}")
                
                if explanations:
                    console.print("Explanations:")
                    for term, explanation in explanations.items():
                        console.print(f"  • {explanation}")
    
    def benchmark_query_strategies(self):
        """Benchmark different query strategies with medical terminology."""
        if not self.initialized or not self.doc_lib:
            console.print("[bold red]Error: RAG tool not initialized[/bold red]")
            return
        
        console.rule("Query Strategy Benchmark")
        
        strategies = ["semantic", "keyword", "hybrid", "expanded"]
        results = {}
        
        # Select benchmark queries
        benchmark_queries = TCCC_MEDICAL_QUERIES[:5]  # Use first 5 queries for benchmarking
        
        for strategy in strategies:
            total_time = 0
            total_results = 0
            query_times = []
            
            if HAS_RICH:
                console.print(f"[bold]Testing strategy:[/bold] {strategy}")
                
                with Progress() as progress:
                    task = progress.add_task(f"[cyan]Running queries with {strategy} strategy...", total=len(benchmark_queries))
                    
                    for i, query in enumerate(benchmark_queries, 1):
                        # Run query and measure time
                        start_time = time.time()
                        result = self.doc_lib.advanced_query(query, strategy=strategy, limit=3)
                        query_time = time.time() - start_time
                        
                        query_times.append(query_time)
                        total_time += query_time
                        total_results += len(result.get('results', []))
                        
                        # Update progress
                        progress.update(task, advance=1)
                        progress.print(f"Query {i}: '{query}'")
                        progress.print(f"  Time: {query_time:.4f}s, Results: {len(result.get('results', []))}")
                        
                        # Show first result if available
                        if result.get('results'):
                            score = result['results'][0].get('score', 0)
                            progress.print(f"  Top score: {score:.4f}")
            else:
                console.print(f"\nTesting strategy: {strategy}")
                
                for i, query in enumerate(benchmark_queries, 1):
                    # Run query and measure time
                    start_time = time.time()
                    result = self.doc_lib.advanced_query(query, strategy=strategy, limit=3)
                    query_time = time.time() - start_time
                    
                    query_times.append(query_time)
                    total_time += query_time
                    total_results += len(result.get('results', []))
                    
                    console.print(f"Query {i}: '{query}'")
                    console.print(f"  Time: {query_time:.4f}s, Results: {len(result.get('results', []))}")
                    
                    # Show first result if available
                    if result.get('results'):
                        score = result['results'][0].get('score', 0)
                        console.print(f"  Top score: {score:.4f}")
            
            avg_time = total_time / len(benchmark_queries)
            avg_results = total_results / len(benchmark_queries)
            results[strategy] = (avg_time, avg_results)
            
            console.print(f"Strategy {strategy}: Avg time = {avg_time:.4f}s, Avg results = {avg_results:.1f}")
        
        # Determine best strategy based on query time
        best_strategy = min(results.items(), key=lambda x: x[1][0])[0]
        
        if HAS_RICH:
            console.print(f"\n[bold green]Best performing strategy:[/bold green] {best_strategy} ({results[best_strategy][0]:.4f}s)")
            
            # Create results table
            table = Table(title="Query Strategy Performance")
            table.add_column("Strategy", style="cyan")
            table.add_column("Avg Time (s)", style="green")
            table.add_column("Avg Results", style="blue")
            
            for strategy, (avg_time, avg_results) in results.items():
                table.add_row(
                    strategy,
                    f"{avg_time:.4f}",
                    f"{avg_results:.1f}"
                )
            
            console.print(table)
        else:
            console.print(f"\nBest performing strategy: {best_strategy} ({results[best_strategy][0]:.4f}s)")
            console.print("\nQuery Strategy Performance:")
            
            for strategy, (avg_time, avg_results) in results.items():
                console.print(f"  {strategy}: {avg_time:.4f}s, {avg_results:.1f} results")
        
        return best_strategy, results
    
    def export_results(self, results, filename=None):
        """
        Export query results to a file.
        
        Args:
            results: Query results dictionary
            filename: Output filename (optional)
        """
        if not results:
            console.print("[bold red]Error: No results to export[/bold red]")
            return
        
        # Generate default filename if not provided
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            query_text = results.get("query", "unknown")
            query_slug = "".join(c if c.isalnum() else "_" for c in query_text[:20])
            filename = f"rag_results_{query_slug}_{timestamp}.json"
        
        try:
            # Ensure directory exists
            output_dir = os.path.join(script_dir, "rag_results")
            os.makedirs(output_dir, exist_ok=True)
            
            # Full output path
            output_path = os.path.join(output_dir, filename)
            
            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            console.print(f"[bold green]Results exported to:[/bold green] {output_path}")
            return output_path
        except Exception as e:
            console.print(f"[bold red]Error exporting results: {str(e)}[/bold red]")
            return None

    def interactive_mode(self):
        """Run in interactive query mode."""
        if not self.initialized:
            console.print("[bold red]Error: RAG tool not initialized[/bold red]")
            return
        
        console.rule("TCCC RAG Interactive Mode")
        console.print("Enter queries to search the RAG database. Type 'exit' or 'quit' to exit.")
        console.print("Commands:")
        console.print("  'strategy:semantic|keyword|hybrid|expanded' - Change search strategy")
        console.print("  'limit:N' - Change result limit")
        console.print("  'prompt' - Show LLM prompt for the last query")
        console.print("  'export' - Export last results to file")
        console.print("  'report' - Generate comprehensive system report")
        console.print("  'help' - Show this help")
        
        # Default settings
        strategy = "hybrid"
        limit = 5
        last_query = None
        last_results = None
        
        while True:
            try:
                if HAS_RICH:
                    user_input = input(f"\n[Strategy: {strategy}, Limit: {limit}] Query> ")
                else:
                    user_input = input(f"\nQuery [{strategy}, limit={limit}]> ")
                
                user_input = user_input.strip()
                if not user_input:
                    continue
                
                # Check for exit command
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                # Check for help command
                if user_input.lower() in ['help', '?']:
                    console.print("Commands:")
                    console.print("  'strategy:semantic|keyword|hybrid|expanded' - Change search strategy")
                    console.print("  'limit:N' - Change result limit")
                    console.print("  'prompt' - Show LLM prompt for the last query")
                    console.print("  'export' - Export last results to file")
                    console.print("  'report' - Generate comprehensive system report")
                    console.print("  'benchmark' - Run query strategy benchmark")
                    console.print("  'vocab' - Test medical vocabulary")
                    console.print("  'help' - Show this help")
                    console.print("  'exit' or 'quit' - Exit interactive mode")
                    continue
                
                # Check for strategy command
                if user_input.lower().startswith('strategy:'):
                    new_strategy = user_input.split(':', 1)[1].strip().lower()
                    if new_strategy in ['semantic', 'keyword', 'hybrid', 'expanded']:
                        strategy = new_strategy
                        console.print(f"[bold green]Strategy changed to: {strategy}[/bold green]")
                    else:
                        console.print("[bold red]Invalid strategy. Use semantic, keyword, hybrid, or expanded.[/bold red]")
                    continue
                
                # Check for limit command
                if user_input.lower().startswith('limit:'):
                    try:
                        new_limit = int(user_input.split(':', 1)[1].strip())
                        if new_limit > 0:
                            limit = new_limit
                            console.print(f"[bold green]Result limit changed to: {limit}[/bold green]")
                        else:
                            console.print("[bold red]Limit must be a positive number.[/bold red]")
                    except ValueError:
                        console.print("[bold red]Invalid limit. Use a positive number.[/bold red]")
                    continue
                
                # Check for LLM prompt command
                if user_input.lower() == 'prompt':
                    if last_query:
                        prompt = self.generate_llm_prompt(last_query, strategy, limit)
                        
                        if HAS_RICH:
                            console.print(Panel(prompt, title="LLM Prompt", border_style="green"))
                        else:
                            console.print("\n--- LLM Prompt ---")
                            console.print(prompt)
                            console.print("------------------")
                    else:
                        console.print("[bold yellow]No query has been executed yet.[/bold yellow]")
                    continue
                
                # Check for export command
                if user_input.lower() == 'export':
                    if last_results:
                        self.export_results(last_results)
                    else:
                        console.print("[bold yellow]No results to export yet.[/bold yellow]")
                    continue
                
                # Check for benchmark command
                if user_input.lower() == 'benchmark':
                    self.benchmark_query_strategies()
                    continue
                
                # Check for vocabulary test command
                if user_input.lower() == 'vocab':
                    self.test_medical_vocabulary()
                    continue
                
                # Check for report generation command
                if user_input.lower() == 'report':
                    report_path = self.generate_system_report()
                    # Try to open the report PDF if on a graphical system
                    if report_path:
                        try:
                            import subprocess
                            if sys.platform == "linux":
                                subprocess.run(["xdg-open", report_path], check=False)
                            elif sys.platform == "darwin":
                                subprocess.run(["open", report_path], check=False)
                            elif sys.platform == "win32":
                                os.startfile(report_path)
                        except:
                            pass
                    continue
                
                # Execute query
                console.print(f"[bold]Executing query with {strategy} strategy...[/bold]")
                results = self.run_query(user_input, strategy=strategy, limit=limit)
                
                # Display results
                self.display_results(results)
                
                # Store last query and results
                last_query = user_input
                last_results = results
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Operation cancelled. Type 'exit' to quit.[/bold yellow]")
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
    
    def generate_system_report(self):
        """Generate a comprehensive report on the RAG system."""
        if not self.initialized:
            console.print("[bold red]Error: RAG tool not initialized[/bold red]")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"tccc_rag_report_{timestamp}.pdf"
        report_path = os.path.join(script_dir, "reports", report_filename)
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        console.print(f"[bold green]Generating RAG system report: {report_path}[/bold green]")
        
        # Run benchmark on standard medical queries
        console.print("[bold]Running benchmark for report...[/bold]")
        best_strategy, results = self.benchmark_query_strategies()
        
        # Test vocabulary and collect metrics
        console.print("[bold]Testing medical vocabulary for report...[/bold]")
        vocab_metrics = self._collect_vocabulary_metrics()
        
        # Get library status
        status = self.doc_lib.get_status() if self.doc_lib else {}
        
        # Create PDF report
        with PdfPages(report_path) as pdf:
            # Title page
            plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.text(0.5, 0.9, "TCCC RAG System Report", fontsize=24, ha='center')
            plt.text(0.5, 0.85, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                     fontsize=14, ha='center')
            
            # System overview
            plt.text(0.5, 0.75, "System Overview", fontsize=18, ha='center')
            overview_text = [
                f"Documents: {status.get('documents', {}).get('count', 0)}",
                f"Chunks: {status.get('documents', {}).get('chunks', 0)}",
                f"Vectors: {status.get('index', {}).get('vectors', 0)}",
                f"Embedding Model: {status.get('model', {}).get('name', 'Unknown')}",
                f"Vector Dimension: {status.get('index', {}).get('dimension', 0)}",
                f"Medical Vocabulary: {'Available' if self.vocab_manager else 'Not Available'}"
            ]
            
            y_pos = 0.7
            for line in overview_text:
                plt.text(0.5, y_pos, line, fontsize=12, ha='center')
                y_pos -= 0.05
            
            pdf.savefig()
            plt.close()
            
            # Performance metrics
            plt.figure(figsize=(11, 8.5))
            plt.title("Query Strategy Performance", fontsize=18)
            
            strategies = list(results.keys())
            times = [results[s][0] for s in strategies]
            result_counts = [results[s][1] for s in strategies]
            
            # Create subplot for query times
            plt.subplot(2, 1, 1)
            bars = plt.bar(strategies, times, color='skyblue')
            plt.ylabel('Average Query Time (s)')
            plt.title('Query Time by Strategy')
            
            # Add values above bars
            for bar, time_val in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{time_val:.4f}s', ha='center', fontsize=10)
            
            # Create subplot for result counts
            plt.subplot(2, 1, 2)
            bars = plt.bar(strategies, result_counts, color='lightgreen')
            plt.ylabel('Average Results Count')
            plt.title('Results Count by Strategy')
            
            # Add values above bars
            for bar, count in zip(bars, result_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f'{count:.1f}', ha='center', fontsize=10)
            
            plt.tight_layout(pad=3.0)
            pdf.savefig()
            plt.close()
            
            # Medical vocabulary analysis
            if vocab_metrics:
                plt.figure(figsize=(11, 8.5))
                plt.title("Medical Vocabulary Analysis", fontsize=18)
                
                # Coverage chart
                plt.subplot(2, 2, 1)
                labels = ['Found', 'Not Found']
                sizes = [vocab_metrics['found_terms'], vocab_metrics['total_terms'] - vocab_metrics['found_terms']]
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
                plt.axis('equal')
                plt.title('Term Coverage')
                
                # Category distribution
                plt.subplot(2, 2, 2)
                categories = list(vocab_metrics['category_counts'].keys())
                counts = list(vocab_metrics['category_counts'].values())
                plt.barh(categories, counts, color='skyblue')
                plt.xlabel('Term Count')
                plt.title('Terms by Category')
                
                # Term types
                plt.subplot(2, 1, 2)
                types = list(vocab_metrics['type_counts'].keys())
                type_counts = list(vocab_metrics['type_counts'].values())
                plt.bar(types, type_counts, color='lightgreen')
                plt.ylabel('Count')
                plt.title('Term Types')
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            
            # Query examples
            plt.figure(figsize=(11, 8.5))
            plt.title("Example Queries and Response Times", fontsize=18)
            
            # Test some example queries
            example_queries = [
                "How to treat a tension pneumothorax?",
                "MARCH algorithm explanation",
                "Combat application tourniquet procedure"
            ]
            
            query_times = {}
            for strategy in strategies:
                query_times[strategy] = []
                for query in example_queries:
                    start_time = time.time()
                    self.run_query(query, strategy=strategy, limit=3, timer=False)
                    query_times[strategy].append(time.time() - start_time)
            
            # Plot query time comparison
            plt.subplot(1, 1, 1)
            x = np.arange(len(example_queries))
            width = 0.2
            multiplier = 0
            
            for strategy, times in query_times.items():
                offset = width * multiplier
                plt.bar(x + offset, times, width, label=strategy)
                multiplier += 1
            
            plt.xlabel('Query')
            plt.ylabel('Time (seconds)')
            plt.title('Query Response Times by Strategy')
            plt.xticks(x + width, [f"Query {i+1}" for i in range(len(example_queries))])
            plt.legend(loc='upper left')
            
            # Add query text below
            plt.figtext(0.1, 0.01, f"Query 1: {example_queries[0]}", fontsize=10)
            plt.figtext(0.1, 0.04, f"Query 2: {example_queries[1]}", fontsize=10)
            plt.figtext(0.1, 0.07, f"Query 3: {example_queries[2]}", fontsize=10)
            
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            pdf.savefig()
            plt.close()
            
            # System recommendations
            plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.text(0.5, 0.95, "System Recommendations", fontsize=20, ha='center')
            
            recommendations = [
                f"Recommended Query Strategy: {best_strategy}",
                f"Estimated Optimal Batch Size: {8 if self.args.optimize_for_jetson else 16}",
                "Consider adding more documents to the collection" if status.get('documents', {}).get('count', 0) < 10 else "Document collection is adequate",
                "Consider expanding medical vocabulary" if not vocab_metrics or vocab_metrics['coverage'] < 70 else "Medical vocabulary coverage is good"
            ]
            
            y_pos = 0.85
            for line in recommendations:
                plt.text(0.5, y_pos, line, fontsize=14, ha='center')
                y_pos -= 0.07
            
            pdf.savefig()
            plt.close()
        
        console.print(f"[bold green]Report generated successfully: {report_path}[/bold green]")
        return report_path
    
    def _collect_vocabulary_metrics(self):
        """Collect metrics about the medical vocabulary for reporting."""
        if not self.vocab_manager:
            return None
        
        # Test key TCCC terms for coverage calculation
        test_terms = [
            "TCCC", "TQ", "CAT", "Hemorrhage", "Tourniquet", "NPA",
            "Tension pneumothorax", "Needle decompression", "Hemostatic",
            "MARCH", "Airway", "Respiration", "Circulation", "Hypothermia"
        ]
        
        found_terms = 0
        type_counts = {"term": 0, "abbreviation": 0, "other": 0}
        category_counts = {}
        
        for term in test_terms:
            info = self.vocab_manager.get_term_info(term)
            if info:
                found_terms += 1
                term_type = info.get("type", "other")
                type_counts[term_type] = type_counts.get(term_type, 0) + 1
                
                category = info.get("category", "uncategorized")
                category_counts[category] = category_counts.get(category, 0) + 1
        
        coverage = (found_terms / len(test_terms)) * 100
        
        return {
            "total_terms": len(test_terms),
            "found_terms": found_terms,
            "coverage": coverage,
            "type_counts": type_counts,
            "category_counts": category_counts
        }


def main():
    """Main entry point for the TCCC RAG Tool."""
    parser = argparse.ArgumentParser(
        description="TCCC RAG Testing and Demo Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Run in interactive mode
          python tccc_rag_tool.py -i
          
          # Execute a specific query with hybrid strategy
          python tccc_rag_tool.py -q "How to treat tension pneumothorax?" -s hybrid
          
          # Generate an LLM prompt for a query
          python tccc_rag_tool.py -q "MARCH algorithm" -p
          
          # Test medical vocabulary capabilities
          python tccc_rag_tool.py -v
          
          # Benchmark query strategies
          python tccc_rag_tool.py -b
          
          # Run with Jetson optimizations
          python tccc_rag_tool.py -i -j
        """)
    )
    
    # Mode selection arguments
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('-i', '--interactive', action='store_true',
                          help='Run in interactive mode')
    mode_group.add_argument('-q', '--query', type=str,
                          help='Execute a single query')
    mode_group.add_argument('-v', '--test-vocabulary', action='store_true',
                          help='Test medical vocabulary')
    mode_group.add_argument('-b', '--benchmark', action='store_true',
                          help='Benchmark query strategies')
    mode_group.add_argument('-r', '--report', action='store_true',
                          help='Generate comprehensive RAG system report')
    
    # Query options
    query_group = parser.add_argument_group('Query Options')
    query_group.add_argument('-s', '--strategy', type=str, choices=['semantic', 'keyword', 'hybrid', 'expanded'],
                           default='hybrid',
                           help='Query strategy (default: hybrid)')
    query_group.add_argument('-l', '--limit', type=int, default=5,
                           help='Maximum number of results (default: 5)')
    query_group.add_argument('-d', '--detailed', action='store_true',
                           help='Show detailed results')
    query_group.add_argument('-p', '--prompt', action='store_true',
                           help='Generate LLM prompt for the query')
    query_group.add_argument('-e', '--export', action='store_true',
                           help='Export results to file')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('-c', '--config', type=str,
                            help='Path to custom config file')
    config_group.add_argument('-j', '--optimize-for-jetson', action='store_true',
                            help='Apply optimizations for Jetson hardware')
    
    args = parser.parse_args()
    
    # Initialize the tool
    tool = TcccRagTool(args)
    if not tool.initialize():
        console.print("[bold red]Failed to initialize RAG tool. Exiting.[/bold red]")
        return 1
    
    # Determine operation mode
    if args.interactive:
        tool.interactive_mode()
    elif args.benchmark:
        tool.benchmark_query_strategies()
    elif args.test_vocabulary:
        tool.test_medical_vocabulary()
    elif args.report:
        # Generate comprehensive system report
        report_path = tool.generate_system_report()
        if report_path:
            console.print(f"[bold green]System report generated:[/bold green] {report_path}")
            # Try to open the report PDF if on a graphical system
            try:
                import subprocess
                if sys.platform == "linux":
                    subprocess.run(["xdg-open", report_path], check=False)
                elif sys.platform == "darwin":
                    subprocess.run(["open", report_path], check=False)
                elif sys.platform == "win32":
                    os.startfile(report_path)
            except:
                pass
    elif args.query:
        # Execute query
        results = tool.run_query(args.query, strategy=args.strategy, limit=args.limit)
        
        # Display results
        tool.display_results(results, detailed=args.detailed)
        
        # Generate prompt if requested
        if args.prompt:
            prompt = tool.generate_llm_prompt(args.query, strategy=args.strategy, limit=args.limit)
            
            if HAS_RICH:
                console.print(Panel(prompt, title="LLM Prompt", border_style="green"))
            else:
                console.print("\n--- LLM Prompt ---")
                console.print(prompt)
                console.print("------------------")
        
        # Export results if requested
        if args.export and results:
            tool.export_results(results)
    else:
        # No mode specified, default to interactive
        console.print("[bold yellow]No operation mode specified. Starting interactive mode...[/bold yellow]")
        tool.interactive_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())