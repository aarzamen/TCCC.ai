# all-MiniLM-L12-v2 Embedding Model

This directory contains the all-MiniLM-L12-v2 embedding model used for document retrieval and semantic search in the TCCC.ai Document Library.

## Model Description

all-MiniLM-L12-v2 is a sentence embedding model that maps sentences to a 384-dimensional dense vector space. It is based on the MiniLM architecture and is optimized for edge deployment. The model is used for semantic similarity search in the Document Library.

## Technical Specifications

- **Architecture**: MiniLM (distilled from BERT)
- **Layers**: 12 transformer layers
- **Embedding Dimension**: 384
- **Model Size**: ~130MB
- **Vocabulary Size**: 30,522 tokens
- **Max Sequence Length**: 512 tokens
- **Parameters**: ~33 million
- **Training Data**: MS MARCO, Natural Questions, and additional web data

## Performance

- **Speed**: ~1ms per embedding on GPU, ~10ms on CPU
- **Memory Usage**: ~500MB during inference
- **Semantic Textual Similarity (STS) Benchmark**: 78.6% (Spearman's rank correlation)
- **Retrieval Performance**: Strong performance on retrieval tasks, especially for short to medium length texts

## Integration with TCCC.ai

The model is used by the Document Library module for the following tasks:

1. **Document Chunking**: Documents are split into manageable chunks
2. **Chunk Embedding**: Each chunk is embedded into a 384-dimensional vector
3. **Vector Indexing**: Embeddings are stored in a FAISS vector index for similarity search
4. **Query Embedding**: User queries are embedded in the same vector space
5. **Semantic Search**: Finding the most relevant document chunks for a query

## Usage in TCCC.ai

The model is loaded and managed by the Document Library module:

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Generate embeddings for text
embeddings = model.encode(["Your text here", "Another text"])
```

## Optimization for Jetson

For optimal performance on Jetson Orin Nano:

1. The model is cached to disk after first load
2. Batch processing is used to maximize throughput
3. The model can be converted to ONNX format for further optimization
4. GPU acceleration is used when available

## Source and License

- **Source**: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
- **License**: Apache License 2.0
- **Citation**: Reimers, Nils, and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. 2019.

## Model Files

The model files are not stored in this repository due to size constraints. They are downloaded automatically when the Document Library is initialized for the first time, and stored in the configured cache directory (`data/models/embeddings` by default).

## Alternatives

If this model is not suitable for your use case, consider these alternatives:

- **all-MiniLM-L6-v2**: Smaller (6 layers) version, faster but slightly less accurate
- **all-mpnet-base-v2**: Higher quality but larger model (768 dimensions)
- **paraphrase-multilingual-MiniLM-L12-v2**: Multilingual version supporting 50+ languages