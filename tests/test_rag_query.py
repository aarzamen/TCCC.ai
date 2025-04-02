#!/usr/bin/env python3
"""
Test RAG query for tension pneumothorax treatment
"""

from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

def main():
    # Load configuration
    config = ConfigManager().load_config('document_library')
    
    # Initialize document library
    dl = DocumentLibrary()
    dl.initialize(config)
    
    # Query using command line argument or default
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else 'How to treat a tension pneumothorax?'
    print(f"Executing query: {query}")
    
    # Try standard query
    result = dl.query(query)
    print(f"Results found: {len(result['results'])}")
    for r in result['results']:
        print(f"Score: {r['score']:.3f} - {r['text'][:100]}...")
    
    # Try advanced query with all strategies
    strategies = ["semantic", "keyword", "hybrid", "expanded"]
    for strategy in strategies:
        print(f"\nTrying {strategy} strategy:")
        result = dl.advanced_query(query, strategy=strategy)
        print(f"Results found: {len(result['results'])}")
        for r in result['results']:
            print(f"Score: {r['score']:.3f} - {r['text'][:100]}...")

if __name__ == "__main__":
    main()