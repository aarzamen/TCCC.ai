#!/usr/bin/env python3
"""Test RAG query results"""

from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

# Initialize document library
config = ConfigManager().load_config('document_library')
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# Test query
query = "tension pneumothorax treatment"
print(f"Query: {query}")
result = doc_lib.query(query)

print(f"Results found: {len(result['results'])}")
for i, res in enumerate(result['results'][:3]):
    print(f"Result {i+1} (Score: {res['score']:.4f}):")
    source = res.get('metadata', {}).get('source', 'Unknown')
    print(f"Source: {source}")
    print(f"Content: {res['text'][:150]}...")
    print("-" * 50)

# Test advanced query with multiple strategies
strategies = ["semantic", "keyword", "hybrid", "expanded"]
for strategy in strategies:
    print(f"\nTrying {strategy} strategy:")
    result = doc_lib.advanced_query(query, strategy=strategy)
    print(f"Results found: {len(result['results'])}")
    if result['results']:
        top_result = result['results'][0]
        print(f"Top result (Score: {top_result['score']:.4f}):")
        print(f"Content: {top_result['text'][:100]}...")