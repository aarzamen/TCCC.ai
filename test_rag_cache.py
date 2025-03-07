#!/usr/bin/env python3
"""
Test script to clear the RAG cache and verify cache behavior
"""

import os
import sys
import shutil
import time

from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

def main():
    # Load configuration
    config = ConfigManager().load_config('document_library')
    
    # Clear cache
    cache_dir = config.get('cache', {}).get('directory', 'data/query_cache')
    print(f"Clearing cache directory: {cache_dir}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        print("Cache cleared.")
    
    # Initialize document library
    dl = DocumentLibrary()
    dl.initialize(config)
    
    # Test queries
    query = "tension pneumothorax"
    print(f"First query - should miss cache: {query}")
    start_time = time.time()
    result1 = dl.query(query)
    query_time1 = time.time() - start_time
    print(f"Time: {query_time1:.4f}s, Cache hit: {result1.get('cache_hit', False)}")
    print(f"Results: {len(result1['results'])}")
    
    # Second query - should hit cache
    print(f"\nSecond query - should hit cache: {query}")
    start_time = time.time()
    result2 = dl.query(query)
    query_time2 = time.time() - start_time
    print(f"Time: {query_time2:.4f}s, Cache hit: {result2.get('cache_hit', False)}")
    print(f"Results: {len(result2['results'])}")
    print(f"Speed improvement: {query_time1/query_time2:.1f}x faster")
    
    # Try a query with expanded strategy
    print(f"\nAdvanced query with expanded strategy: {query}")
    start_time = time.time()
    result3 = dl.advanced_query(query, strategy="expanded")
    query_time3 = time.time() - start_time
    print(f"Time: {query_time3:.4f}s, Cache hit: {result3.get('cache_hit', False)}")
    print(f"Results: {len(result3['results'])}")
    
    # Check vector store status
    print("\nVector store status:")
    status = dl.get_status()
    print(f"Documents: {status['documents']['count']}")
    print(f"Chunks: {status['documents']['chunks']}")
    print(f"Vectors: {status['index']['vectors']}")
    print(f"Model: {status['model']['name']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())