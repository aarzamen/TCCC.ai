#!/usr/bin/env python3
"""
RAG Performance Optimization Script for Jetson Hardware

This script optimizes the TCCC Document Library for better performance
on Jetson hardware, including:
1. Database optimization
2. Medical vocabulary enhancement
3. Vector storage optimization
4. Query caching improvements
5. Benchmarking
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Import TCCC modules
from tccc.document_library import DocumentLibrary
from tccc.document_library.medical_vocabulary import MedicalVocabularyManager
from tccc.utils import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Sample medical queries for benchmarking
MEDICAL_QUERIES = [
    "How to treat a tension pneumothorax?",
    "What is the proper procedure for tourniquet application?",
    "Signs and symptoms of hypovolemic shock",
    "When should I use a hemostatic dressing?",
    "MARCH PAWS algorithm steps",
    "Nasopharyngeal airway insertion technique",
    "How to perform needle decompression",
    "TQ placement for junctional bleeding",
    "Combat casualty assessment",
    "Tactical field care principles"
]

def optimize_medical_vocabulary(config):
    """Optimize the medical vocabulary for better query handling."""
    logger.info("Optimizing medical vocabulary...")
    
    # Initialize vocabulary manager
    vocab_manager = MedicalVocabularyManager(config)
    
    # Initialize and build the database
    success = vocab_manager.initialize()
    
    if success:
        # Test query expansion
        logger.info("Testing medical query expansion...")
        for query in MEDICAL_QUERIES[:3]:  # Test with first few queries
            expansions = vocab_manager.expand_query(query)
            logger.info(f"Query: '{query}' -> {len(expansions)} expansions")
            
        # Extract and explain terms from a sample text
        sample_text = "The patient presented with a tension pneumothorax requiring immediate needle decompression. " \
                     "A 14G needle was inserted at the 2nd intercostal space, mid-clavicular line according to TCCC guidelines."
        
        terms = vocab_manager.extract_medical_terms(sample_text)
        explanations = vocab_manager.explain_medical_terms(sample_text)
        
        logger.info(f"Extracted {len(terms)} medical terms from sample text")
        logger.info(f"Generated {len(explanations)} term explanations")
        
        return True
    else:
        logger.error("Failed to initialize medical vocabulary")
        return False

def optimize_vector_storage(doc_lib):
    """Optimize vector storage for better performance on Jetson."""
    logger.info("Optimizing vector storage...")
    
    # Get vector store status
    status = doc_lib.get_status()
    logger.info(f"Current vector store status: {status['index']['vectors']} vectors, dimension: {status['index']['dimension']}")
    
    # Optimize vector store if it exists
    if hasattr(doc_lib, 'vector_store'):
        vector_store = doc_lib.vector_store
        
        # Perform optimization based on vector store type
        if hasattr(vector_store, 'get_status'):
            vs_status = vector_store.get_status()
            logger.info(f"Vector store status: {vs_status}")
    
    return True

def optimize_cache(doc_lib):
    """Optimize the query cache for better performance."""
    logger.info("Optimizing query cache...")
    
    # Check if cache manager exists
    if hasattr(doc_lib, 'cache_manager') and doc_lib.cache_manager:
        cache_manager = doc_lib.cache_manager
        
        # Get cache stats
        if hasattr(cache_manager, 'get_stats'):
            stats = cache_manager.get_stats()
            logger.info(f"Current cache stats: {stats}")
        
        # Clear old cache entries
        if hasattr(cache_manager, 'clear'):
            # Clear entries older than 1 week
            one_week_seconds = 7 * 24 * 60 * 60
            cleared = cache_manager.clear(older_than_seconds=one_week_seconds)
            logger.info(f"Cleared {cleared} old cache entries")
        
        # Optimize SQLite database if available
        if hasattr(cache_manager, 'vacuum_db'):
            success = cache_manager.vacuum_db()
            if success:
                logger.info("Successfully vacuumed cache database")
            else:
                logger.warning("Failed to vacuum cache database")
    
    return True

def benchmark_queries(doc_lib):
    """Benchmark query performance with different strategies."""
    logger.info("Benchmarking query performance...")
    
    results = {}
    strategies = ["semantic", "keyword", "hybrid", "expanded"]
    
    # Warm up
    doc_lib.advanced_query("test query", strategy="hybrid", limit=3)
    
    # Test each strategy with medical queries
    for strategy in strategies:
        strategy_times = []
        
        for query in MEDICAL_QUERIES:
            start_time = time.time()
            result = doc_lib.advanced_query(query, strategy=strategy, limit=3)
            query_time = time.time() - start_time
            
            strategy_times.append(query_time)
            
            # Log first query details
            if len(strategy_times) == 1:
                logger.info(f"Strategy: {strategy}, Time: {query_time:.4f}s, Results: {len(result.get('results', []))}")
        
        # Calculate average time
        avg_time = sum(strategy_times) / len(strategy_times)
        results[strategy] = avg_time
        
        logger.info(f"Strategy {strategy}: Average time = {avg_time:.4f}s")
    
    # Determine best strategy
    best_strategy = min(results.items(), key=lambda x: x[1])[0]
    logger.info(f"Best performing strategy: {best_strategy} ({results[best_strategy]:.4f}s)")
    
    return best_strategy, results

def optimize_for_jetson(doc_lib, config):
    """Apply Jetson-specific optimizations."""
    logger.info("Applying Jetson-specific optimizations...")
    
    # Update configuration for better Jetson performance
    if "embedding" in config:
        # Limit batch size for embedding to reduce memory usage
        config["embedding"]["batch_size"] = min(config["embedding"].get("batch_size", 32), 16)
        logger.info(f"Set embedding batch size to {config['embedding']['batch_size']}")
        
        # Use CPU for embeddings by default on Jetson (unless explicitly configured)
        if "use_gpu" not in config["embedding"]:
            config["embedding"]["use_gpu"] = False
            logger.info("Set default embedding to use CPU for Jetson compatibility")
    
    # Limit memory cache size
    if "cache" not in config:
        config["cache"] = {}
    
    config["cache"]["max_memory_entries"] = 50
    logger.info(f"Limited cache memory entries to {config['cache']['max_memory_entries']}")
    
    # Set optimal search parameters
    if "search" not in config:
        config["search"] = {}
    
    config["search"]["default_results"] = 3
    config["search"]["max_results"] = 10
    config["search"]["min_similarity"] = 0.65
    logger.info("Set optimal search parameters for Jetson")
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Optimize RAG performance for Jetson hardware")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--optimize-all", action="store_true", help="Apply all optimizations")
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        logger.info(f"Loading custom config from {args.config}")
        config = config_manager.load_config_from_file(args.config)
    else:
        logger.info("Loading default document library config")
        config = config_manager.load_config("document_library")
    
    # Apply Jetson optimizations to config
    config = optimize_for_jetson(None, config)
    
    # Save optimized config
    config_path = "config/optimized_jetson_rag.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config_manager.save_config(config, config_path)
    logger.info(f"Saved optimized config to {config_path}")
    
    # Initialize document library with optimized config
    logger.info("Initializing Document Library with optimized config")
    doc_lib = DocumentLibrary()
    success = doc_lib.initialize(config)
    
    if not success:
        logger.error("Failed to initialize Document Library")
        return 1
    
    # Get initial status
    status = doc_lib.get_status()
    logger.info(f"Document Library initialized: {status}")
    
    # Optimize medical vocabulary
    vocab_success = optimize_medical_vocabulary(config)
    logger.info(f"Medical vocabulary optimization {'successful' if vocab_success else 'failed'}")
    
    # Optimize vector storage
    vector_success = optimize_vector_storage(doc_lib)
    logger.info(f"Vector storage optimization {'successful' if vector_success else 'failed'}")
    
    # Optimize cache
    cache_success = optimize_cache(doc_lib)
    logger.info(f"Cache optimization {'successful' if cache_success else 'failed'}")
    
    # Run benchmarks if requested
    if args.benchmark:
        logger.info("Running performance benchmarks...")
        best_strategy, benchmark_results = benchmark_queries(doc_lib)
        
        # Update config with optimal strategy if available
        if "search" in config:
            config["search"]["default_strategy"] = best_strategy
            logger.info(f"Set default strategy to {best_strategy} based on benchmarks")
            
            # Save updated config
            config_manager.save_config(config, config_path)
            logger.info(f"Updated optimized config with benchmark results")
    
    logger.info("RAG optimization complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())