# Multi-Format Document Support for TCCC RAG System

## Overview

The TCCC RAG system has been enhanced to support a wide range of document formats beyond PDFs. This implementation enables users to add various document types to the RAG knowledge base, including code files, plain text, and structured data.

## Supported Document Types

### Text Documents
- Plain text files (`.txt`)
- Markdown documents (`.md`)
- HTML files (`.html`, `.htm`)
- XML files (`.xml`)

### Code Files
- Python (`.py`)
- JavaScript (`.js`)
- C/C++ (`.c`, `.cpp`, `.h`)
- Shell scripts (`.sh`, `.bash`)
- Go (`.go`)
- Rust (`.rs`)
- SQL (`.sql`)

### Structured Data
- JSON (`.json`)
- YAML (`.yaml`, `.yml`)
- CSV (`.csv`)

### Office Documents
- PDF (`.pdf`)
- Word documents (`.docx`)

## Implementation Details

### Document Processing Pipeline

1. **Format Detection**: The system automatically detects the document format based on file extension and content analysis.

2. **Format-Specific Processing**:
   - Code files: Special handling to preserve structure and extract documentation
   - Text files: General text extraction with format-specific normalization
   - Structured data: Parsing and conversion to searchable text
   - Office documents: Extraction of text and metadata

3. **Text Normalization**: Format-specific text normalization to optimize for search

4. **Chunking**: Documents are split into optimal-sized chunks for embedding

5. **Embedding**: Text chunks are converted to vectors using all-MiniLM-L12-v2

6. **Indexing**: Vectors are stored in FAISS for efficient similarity search

### Code File Processing Enhancement

Code files receive special handling to improve search and retrieval:

1. **Structure Preservation**: Maintains code structure while extracting content
2. **Comment Emphasis**: Emphasizes comments and documentation strings
3. **Metadata Extraction**: Captures language-specific metadata
4. **Code Pattern Recognition**: Identifies functions, classes, and key patterns

### Directory Batch Processing

The system can process entire directories, automatically handling all supported file formats:

```bash
./launch_rag_explorer.sh /path/to/directory
```

This will:
1. Scan the directory for all supported file types
2. Process each file with the appropriate handler
3. Add all documents to the knowledge base
4. Provide a summary of processed documents

## User Interface Enhancements

The interactive RAG explorer now includes:

1. **Format Support Command**: `formats` - Shows all supported file formats
2. **Statistics Command**: `stats` - Shows document statistics by type
3. **Drag and Drop Support**: For any supported file type

## Testing

The implementation has been tested with:
- Multiple code files with different languages
- Various structured data formats
- Markdown documents with medical terminology
- Plain text formats

Search results correctly identify content across all formats, with appropriate weighting for code documentation vs. implementation details.

## Usage Examples

### Add a Python File
```
RAG> /path/to/file.py
```

### Query Code Examples
```
RAG> q: how to implement hemorrhage control
```

### Process Directory
```
RAG> /path/to/code/repository
```

## Future Enhancements

1. Support for additional formats:
   - Binary file metadata extraction
   - Image OCR integration
   - Audio transcription

2. Format-specific query optimization:
   - Code-specific query handlers
   - Structured data query optimization