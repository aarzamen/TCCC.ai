# RAG Database Implementation Guide

This document describes the implementation of the Retrieval-Augmented Generation (RAG) database for the TCCC.ai system, using the Document Library with Nexa AI's all-MiniLM-L12-v2 embeddings.

## Overview

The RAG Database provides:

1. **Document Storage**: A repository of military medicine documents
2. **Vector Embeddings**: Semantic representation of document chunks
3. **Semantic Search**: Finding relevant documents based on meaning
4. **Context Enhancement**: Providing relevant context to the LLM
5. **Military Medicine Focus**: Specialized for combat casualty care
6. **Form Automation**: Supporting automated completion of critical military medical forms

## Implementation Structure

The RAG Database consists of:

1. **Document Library**: Core component for document storage and retrieval
2. **Document Downloader**: Script to download key military medicine documents
3. **Document Processor**: Script to process and index documents
4. **Embedding Model**: Nexa AI's all-MiniLM-L12-v2 for vector embeddings
5. **FAISS Index**: Efficient similarity search for vectors

## Document Sources

The RAG database includes key military medicine documents focused on:

1. **TCCC Casualty Card (DD Form 1380)**: PRIMARY SOURCE document for casualty documentation
2. **TCCC Guidelines**: Official Tactical Combat Casualty Care protocols
3. **9-Line MEDEVAC**: Medical evacuation request procedures
4. **MIST Report Format**: Military standardized patient hand-off format
5. **Valkyrie Program**: Whole blood transfusion procedures
6. **Medical Forms**: Standard military medical documentation

These documents are sourced from official military sources such as:
- Joint Trauma System (JTS)
- Department of Defense (DoD)
- United States Marine Corps (USMC)
- United States Navy (USN)
- Uniformed Services University (USUHS)

## Implementation Details

### Document Download Process

The document download process is implemented in `download_rag_documents.py`:

1. **Document Definition**: Each document has metadata including URL, priority, and category
2. **Categorized Storage**: Documents are stored in category-specific folders
3. **Error Handling**: Robust handling of download failures
4. **User-Agent**: Proper identification for document requests
5. **Incremental Updates**: Only downloads new or changed documents

### Document Processing Pipeline

The document processing pipeline in `process_rag_documents.py` includes:

1. **Format Handling**: Support for PDF, DOCX, HTML, and plain text formats
2. **Text Extraction**: Clean extraction of text content from various formats
3. **Metadata Enrichment**: Adding source, category, and timestamp information
4. **Document Chunking**: Splitting documents into manageable segments
5. **Vector Embedding**: Converting text to semantic vectors
6. **Index Storage**: Saving vectors to FAISS for efficient retrieval

### Vector Embedding Details

The vector embedding process uses Nexa AI's all-MiniLM-L12-v2 model:

1. **Embedding Dimension**: 384-dimensional vectors
2. **Chunking Strategy**: 1000-character chunks with 200-character overlap
3. **Normalization**: Vectors are normalized for cosine similarity
4. **Batch Processing**: Efficient processing of multiple chunks
5. **Incremental Updates**: Ability to add new documents without reindexing

### FAISS Index Configuration

The FAISS index is configured for optimal similarity search:

1. **Index Type**: Flat index for maximum accuracy
2. **Similarity Metric**: Inner product (for normalized vectors)
3. **In-Memory Storage**: Fast access to vector database
4. **Persistence**: Index is saved to disk for persistence
5. **Search Parameters**: Configuration for result count and threshold

## Usage

### Downloading Documents

```bash
# Download all documents to default location
python download_rag_documents.py

# Download to specific location
python download_rag_documents.py --output /path/to/documents

# Force re-download of all documents
python download_rag_documents.py --force
```

### Processing Documents

```bash
# Process downloaded documents
python process_rag_documents.py

# Process documents in specific location
python process_rag_documents.py --input /path/to/documents

# Use specific configuration
python process_rag_documents.py --config custom_config.yaml
```

### Integrating with LLM Analysis

The Document Library is integrated with the LLM Analysis module to provide context-enhanced responses:

```python
from tccc.document_library import DocumentLibrary
from tccc.llm_analysis import LLMAnalysis

# Initialize components
doc_library = DocumentLibrary()
doc_library.initialize(config)

llm = LLMAnalysis()
llm.initialize(config, doc_library=doc_library)

# Analyze text with document context
analysis_result = llm.analyze_text("What is the proper sequence for MARCH assessment?", 
                                   use_documents=True)
```

## Document Categories

The RAG database organizes documents into the following categories:

1. **core_guidelines**: Core TCCC and medical procedures
2. **forms**: Standard military medical forms and documentation
3. **specialized_programs**: Specialized medical programs like Valkyrie
4. **training**: Training manuals and educational resources
5. **supplementary**: Additional supporting materials

## Performance Considerations

### Vector Search Performance

- The all-MiniLM-L12-v2 model provides an optimal balance of accuracy and speed
- FAISS enables millisecond-level search even with thousands of documents
- In-memory index provides fastest performance for Jetson deployment
- Query caching reduces repeated search overhead

### Memory Usage

- PDF extraction requires ~200MB per large document during processing
- FAISS index size grows linearly with document count (~4KB per chunk)
- Model requires ~500MB for embedding generation
- Overall RAM usage approximately 1-2GB for typical document set

### Storage Requirements

- Original documents: ~50-100MB for core collection
- Processed text: ~10-20MB for extracted text
- FAISS index: ~1-5MB for vector storage
- Cache files: Variable, typically <10MB

## Maintenance

To keep the RAG database current:

1. **Regular Updates**: Run the download script periodically to get updated documents
2. **Validation**: Verify document integrity and completeness
3. **Cleaning**: Remove obsolete or redundant documents
4. **Reindexing**: Periodically reindex to incorporate new document versions
5. **Performance Monitoring**: Monitor query performance and resource usage

## Critical Form Implementation

### TCCC Casualty Card (DD Form 1380)

The DD Form 1380 TCCC Casualty Card is a priority implementation for the system:

1. **Reference Template**: https://tccc.org.ua/files/downloads/tccc-cpp-skill-card-55-dd-1380-tccc-casualty-card-en.pdf
2. **Implementation Status**: Planned (high priority)
3. **Purpose**: Document battlefield casualties and treatments
4. **Integration Points**:
   - Document Library provides form template and field schemas
   - LLM Analysis extracts relevant information from audio transcripts
   - Processing Core maps extracted data to form fields
   - Data Store persists completed forms

For detailed implementation specifications, refer to the [TCCC Casualty Card specification](/references/module_specs/tccc_casualty_card.md).

## Future Enhancements

1. **Medical OCR**: Improved handling of scanned documents
2. **Multilingual Support**: Adding support for coalition force documentation
3. **Document Summarization**: Automatic summary generation for long documents
4. **Metadata Extraction**: Improved automatic metadata extraction
5. **Incremental Updates**: Smarter handling of document changes
6. **Form Automation Pipeline**: Enhanced processing of medical forms using structured field extraction

## Conclusion

The RAG Database provides the TCCC.ai system with a comprehensive knowledge base of military medicine documents, enhancing the LLM's ability to provide accurate and contextually relevant responses for combat casualty care scenarios.