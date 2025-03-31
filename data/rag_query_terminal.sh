#!/bin/bash
echo "===== TCCC RAG Explorer ====="
echo "Enter your query or type 'exit' to quit"
echo ""

while true; do
    read -p "Query> " query
    if [ "$query" == "exit" ]; then
        echo "Exiting RAG Explorer"
        break
    fi
    
    if [ -n "$query" ]; then
        python -c "
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

# Initialize document library
config = ConfigManager().load_config('document_library')
doc_lib = DocumentLibrary()
doc_lib.initialize(config)

# Execute query
print(f'\nSearching for: \"{query}\"')
print('=' * 50)
result = doc_lib.query('"""$query"""')

if not result or not result.get('results'):
    print('No results found.')
else:
    for i, res in enumerate(result['results'][:5]):
        print(f'Result {i+1} (Score: {res["score"]:.4f}):')
        source = res.get('metadata', {}).get('source', 'Unknown')
        print(f'Source: {source}')
        print(f'Content: {res["text"][:300]}...')
        print('-' * 50)
"
    fi
    echo ""
done
