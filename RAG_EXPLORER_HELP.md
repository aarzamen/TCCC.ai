# TCCC RAG Explorer User Guide

## Overview
The TCCC RAG Explorer is a tool for processing PDF documents and querying the knowledge base. The system uses advanced language models to understand and retrieve relevant information from medical documents.

## Features
- Process PDF documents through drag-and-drop
- Query the knowledge base with natural language questions
- View results in real-time in the same terminal window
- All processing happens on the Jetson device (no internet required)

## How to Use

### Launch the Explorer
1. Double-click the "TCCC RAG Explorer" desktop icon
2. Or run `./launch_rag_explorer.sh` from the command line

### Process a PDF Document
1. Drag and drop a PDF file directly into the terminal window
2. The system will process and add the document to the knowledge base
3. Processing may take a minute depending on document size

### Query the Knowledge Base
1. Type `q: your question here` and press Enter
   - Example: `q: how to treat tension pneumothorax`
2. Results will display in the terminal with relevance scores

### Other Commands
- `help` - Show available commands
- `clear` - Clear the screen
- `exit` - Exit the program

## Advanced Use Cases

### Process Multiple Documents
You can drag and drop multiple PDFs in sequence to build a comprehensive knowledge base.

### Medical Terminology
The system recognizes medical terminology and can handle specialized queries related to TCCC procedures.

## Support
For technical support, contact your system administrator or refer to the TCCC documentation.

## Technical Details
- Built on production-ready embedding model (all-MiniLM-L12-v2)
- Uses FAISS vector database for efficient similarity search
- Optimized for Jetson Orin Nano platform
- No mock functionality - all features are fully implemented