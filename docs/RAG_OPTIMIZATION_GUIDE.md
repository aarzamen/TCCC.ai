# TCCC RAG System Optimization Guide

This guide explains the optimizations made to the TCCC RAG (Retrieval-Augmented Generation) system for improved performance, particularly on resource-constrained Jetson hardware.

## Key Optimizations

### 1. Medical Vocabulary Enhancement

The medical vocabulary system has been significantly enhanced to better handle TCCC-specific terminology:

- **SQLite Database Storage**: Replaced in-memory storage with a SQLite database for efficient term lookup and reduced memory footprint.
- **Category-Based Organization**: Organized medical terms into functional categories (hemorrhage, airway, respiration, etc.) for improved query expansion.
- **Specialized TCCC Terminology**: Added comprehensive support for military medical abbreviations and terms.
- **Spelling Correction**: Added support for common misspellings of medical terms.
- **Multi-Word Phrase Detection**: Improved detection of medical phrases spanning multiple words.

### 2. Query Processing Improvements

Query processing has been enhanced with several techniques:

- **Intelligent Query Expansion**: The system now expands queries with relevant medical terminology but limits expansions to prevent query explosion.
- **Category Detection**: Automatically detects the medical category a query belongs to for more targeted results.
- **Strategy Optimization**: Added benchmarking to determine the optimal search strategy (semantic, keyword, hybrid, expanded) based on performance.
- **Non-Exact Matching**: Improved matching for medical terms that may be expressed in different ways.

### 3. Performance Optimization for Jetson

Several changes were made specifically for better performance on Jetson hardware:

- **Memory Usage Reduction**: Reduced batch sizes and limited in-memory cache entries to decrease RAM usage.
- **CPU-Based Embedding**: Default to CPU for embedding generation to avoid GPU memory limitations.
- **Disk-Based Caching**: Enhanced the caching system to rely more on disk storage than memory.
- **Result Limiting**: Optimized default result limits to prevent excessive processing.
- **Deferred Loading**: Implemented smarter component initialization and loading.

### 4. Usability Improvements

The user interface has been enhanced:

- **Terminal-Based Explorer**: Created a lightweight terminal interface for querying the RAG system.
- **Medical Term Explanation**: Added ability to explain TCCC-specific medical terms.
- **PDF Document Processing**: Improved handling of PDF documents for knowledge base expansion.
- **Query Suggestions**: Added common TCCC queries for quick reference.

## Usage Instructions

### Starting the Optimized RAG System

1. Run the optimization script:
   ```
   ./optimize_rag_performance.py --optimize-all
   ```

2. Launch the RAG explorer:
   ```
   ./launch_rag_on_jetson.sh
   ```

3. Or use the desktop shortcut: `TCCC_RAG_Query.desktop`

### Using the RAG Explorer

1. **Basic Querying**:
   - Type any medical question to search the knowledge base
   - Example: `How do I apply a tourniquet?`

2. **Commands**:
   - `help` - Show available commands
   - `strategy <name>` - Change search strategy (semantic, keyword, hybrid, expanded)
   - `explain <term>` - Get explanation for a medical term
   - `add <pdf_path>` - Add a PDF document to the knowledge base
   - `suggest` - Show query suggestions
   - `stats` - Show system statistics
   - `clear` - Clear the screen
   - `exit` - Exit the application

3. **Explaining Medical Terms**:
   - Use `explain` to understand abbreviations and specialized terms
   - Example: `explain TCCC` or `explain tension pneumothorax`

## Performance Benchmarks

Typical performance metrics on Jetson hardware:

| Strategy | Avg. Query Time | Notes |
|----------|----------------|-------|
| Semantic | ~0.8s | Best for concept-level queries |
| Keyword | ~0.5s | Fast but less accurate |
| Hybrid | ~1.0s | Good balance of accuracy and speed |
| Expanded | ~1.2s | Best for medical terminology handling |

Memory usage has been optimized to stay under 2GB during operation, suitable for even the most constrained Jetson configurations.

## Troubleshooting

If you encounter issues:

1. **High Memory Usage**: Use the `--low-memory` flag with both the optimizer and the explorer.
2. **Slow Performance**: Consider using `strategy keyword` for faster results.
3. **Missing Results**: Try different query formulations or use the expanded strategy.
4. **Loading Errors**: Ensure the config directory exists and contains valid configurations.

## Future Improvements

Planned enhancements:

1. Model quantization for even lower memory usage
2. Integration with the Jetson CUDA runtime for improved performance
3. Additional medical vocabulary specializations
4. Expansion of the document processing capabilities