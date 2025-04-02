# TCCC RAG Explorer User Guide

## Overview
The TCCC RAG Explorer is a tool for processing various document types and querying the knowledge base. The system uses advanced language models to understand and retrieve relevant information from documents of many formats.

## Features
- Process multiple document types through drag-and-drop
- Query the knowledge base with natural language questions
- View results in real-time in the same terminal window
- All processing happens on the Jetson device (no internet required)

## How to Use

### Launch the Explorer
1. Double-click the "TCCC RAG Explorer" desktop icon
2. Or run `./launch_rag_explorer.sh` from the command line

### Process Documents
1. Drag and drop any supported file directly into the terminal window:
   - PDF files (.pdf)
   - Text documents (.txt, .md)
   - Code files (.py, .js, .c, etc.)
   - Word documents (.docx)
   - Structured data (.json, .xml, .yaml, .csv)
   - HTML files (.html, .htm)
2. The system will process and add the document to the knowledge base
3. Processing time varies depending on document size and complexity

### Query the Knowledge Base
1. Type `q: your question here` and press Enter
   - Example: `q: how to treat tension pneumothorax`
   - For deeper search: `q! how to treat tension pneumothorax`
2. Results will display in the terminal with relevance scores

### Other Commands
- `help` - Show available commands
- `formats` - Display all supported file formats
- `stats` - Show database statistics
- `clear` - Clear the screen
- `exit` - Exit the program

## Advanced Use Cases

### Process Multiple Documents
You can drag and drop an entire directory to process all supported documents within it at once.

### Process Code Repositories
Python, JavaScript, C/C++, and other code files are supported with special handling for code structure.

### Medical Terminology
The system recognizes medical terminology and can handle specialized queries related to TCCC procedures.

## Support
For technical support, contact your system administrator or refer to the TCCC documentation.

## Technical Details
- Built on production-ready embedding model (all-MiniLM-L12-v2)
- Uses FAISS vector database for efficient similarity search
- Special handling for code files that preserves structure
- Optimized for Jetson Orin Nano platform
- No mock functionality - all features are fully implemented