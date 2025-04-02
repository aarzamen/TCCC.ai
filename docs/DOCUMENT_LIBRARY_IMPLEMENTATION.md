# Document Library Implementation Guide

This document describes the implementation of the Document Library module for the TCCC.ai system, focusing on the integration of Nexa AI's all-MiniLM-L12-v2 embedding model for semantic search capabilities.

## Overview

The Document Library provides:

1. **Document storage and retrieval**: Efficiently store documents and their metadata
2. **Semantic search**: Find relevant documents based on meaning, not just keywords
3. **Vector embeddings**: Convert text chunks into numerical vectors for similarity search
4. **Chunking**: Split larger documents into manageable chunks for more precise retrieval
5. **Caching**: Store query results to improve response times for repeated queries

## Architecture

The Document Library consists of the following components:

1. **Document Storage**: Filesystem-based storage for original documents and metadata
2. **Vector Index**: FAISS-based vector database for efficient similarity search
3. **Embedding Model**: Nexa AI's all-MiniLM-L12-v2 for generating text embeddings
4. **Query Cache**: In-memory and disk-based caching for improved performance
5. **Metadata Store**: JSON-based storage for document metadata

## Dependencies

- `sentence-transformers`: Python framework for sentence, paragraph, and image embeddings
- `faiss-cpu`: Facebook AI Similarity Search library for efficient vector similarity search
- `numpy`: Scientific computing library for array operations
- Standard Python libraries: `os`, `json`, `time`, `datetime`, `pickle`, `hashlib`

## Configuration

The Document Library is configured through a YAML file (`config/document_library.yaml`), which includes:

### Storage Settings

```yaml
storage:
  type: "file"
  base_dir: "data/documents"
  index_path: "data/document_index"
  cache_dir: "data/query_cache"
  retention_days: 0
  max_size_mb: 1024
  compression_level: 6
```

### Embedding Settings

```yaml
embedding:
  model_name: "all-MiniLM-L12-v2"
  cache_dir: "data/models/embeddings"
  dimension: 384
  max_seq_length: 512
  use_gpu: true
  batch_size: 32
  normalize: true
```

### Search Settings

```yaml
search:
  provider: "faiss"
  semantic_search: true
  embedding_model: "all-MiniLM-L12-v2"
  min_similarity: 0.7
  default_results: 5
  max_results: 20
  fuzzy_matching: true
  cache_timeout: 3600
```

### Indexing Settings

```yaml
indexing:
  auto_index: true
  update_frequency: 300
  max_document_size_kb: 5120
  chunk_size: 1000
  chunk_overlap: 200
```

## Implementation Details

### Document Ingestion Process

1. **Document Addition**: When a document is added to the library, it is assigned a unique ID and its metadata is stored.
2. **Chunking**: Documents are split into smaller chunks to improve search precision.
3. **Embedding**: Each chunk is converted into a numerical vector using the embedding model.
4. **Indexing**: Vectors are added to the FAISS index for similarity search.
5. **Storage**: Document metadata and chunk information are saved to disk.

### Query Process

1. **Cache Check**: Check if the query has been executed recently and is in cache.
2. **Query Embedding**: Convert the query text to a vector using the same embedding model.
3. **Vector Similarity Search**: Use FAISS to find the most similar document chunks.
4. **Result Processing**: Group results by document and return the most relevant ones.
5. **Caching**: Cache the results for future use.

### Caching Mechanism

The Document Library implements a two-tier caching system:

1. **In-Memory Cache**: Fast access for recently used queries
2. **Disk Cache**: Persistent storage for query results
3. **Cache Invalidation**: Automatic invalidation based on configurable timeout

## Model Details

### Nexa AI's all-MiniLM-L12-v2

This model is based on the MiniLM architecture and has been fine-tuned for semantic search applications. It provides:

- **Embedding Dimension**: 384
- **Sequence Length**: Up to 512 tokens
- **Performance**: Optimized for edge devices like the Jetson Orin Nano
- **Size**: Smaller than BERT but with comparable performance for embedding tasks
- **Language Support**: Primarily English, with limited multilingual capabilities

The model generates dense vector representations (embeddings) of text that capture semantic meaning, allowing for similarity-based search rather than just keyword matching.

## Usage Examples

### Adding a Document

```python
from tccc.document_library import DocumentLibrary
from tccc.utils import ConfigManager

# Initialize
config_manager = ConfigManager()
config = config_manager.load_config("document_library")
doc_library = DocumentLibrary()
doc_library.initialize(config)

# Add document
document = {
    "file_path": "/path/to/document.txt",
    "metadata": {
        "category": "TCCC Reference",
        "type": "Text",
        "source": "Training Manual"
    }
}
doc_id = doc_library.add_document(document)
```

### Querying Documents

```python
# Search for relevant documents
results = doc_library.query(
    "What are the steps for airway management in tactical care?",
    n_results=3
)

# Process results
for result in results.get("results", []):
    print(f"Document: {result['metadata'].get('file_name')}")
    print(f"Score: {result['score']}")
    print(f"Text: {result['text'][:200]}...")
    print()
```

## Performance Considerations

### Memory Usage

- The FAISS index and document chunks are kept in memory for performance
- For large document collections, consider:
  - Reducing chunk size
  - Using disk-based FAISS indexes
  - Implementing document partitioning

### Optimization for Jetson Orin Nano

- GPU acceleration can be enabled for embedding generation
- Batch processing is used for efficient embedding
- The embedding model is cached to avoid repeated loading
- Index size grows with document count, monitor memory usage

## Future Enhancements

1. **PDF Support**: Add support for PDF documents with proper text extraction
2. **Multiple Languages**: Extend embedding model support for multilingual documents
3. **Hybrid Search**: Combine vector search with keyword-based search for improved accuracy
4. **Document Clustering**: Automatically organize documents by topic
5. **Incremental Indexing**: Update only changed documents for faster reindexing

## Troubleshooting

### Common Issues

1. **Memory Issues**: If experiencing memory issues, reduce batch size and chunk size in configuration
2. **Slow Indexing**: For large documents, increase chunk size to reduce the number of vectors
3. **Model Loading Failures**: Ensure internet connectivity for first-time model downloads
4. **Query Performance**: Adjust cache timeout for frequently accessed queries

### Logs to Monitor

The Document Library logs important events to the application logger:

- Initialization events
- Document addition
- Query processing
- Cache operations
- Error conditions

Monitor these logs to identify performance issues or errors in the document library.

## Conclusion

The Document Library provides a powerful semantic search capability to the TCCC.ai system using Nexa AI's all-MiniLM-L12-v2 embedding model. The implementation is optimized for the Jetson Orin Nano platform while providing flexible configuration options to adapt to different document collections and usage patterns.