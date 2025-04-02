# TCCC RAG Tool Usage Guide

## Overview

The TCCC RAG (Retrieval-Augmented Generation) Tool is a comprehensive solution for testing and demonstrating the capabilities of the TCCC document library system. It provides various features including:

- Multiple query strategies (semantic, keyword, hybrid, expanded)
- Medical terminology enhancement and testing
- Performance benchmarking for Jetson optimization
- Query caching and memory efficiency
- Rich visualization of results
- Export capabilities for offline analysis

## Quick Start

### Using the Desktop Shortcut

1. Double-click the `TCCC_RAG_Tool.desktop` icon
2. The tool will open in a terminal window in interactive mode

### Using the Command Line

To launch the tool with basic settings:

```bash
./launch_tccc_rag_tool.sh
```

For Jetson-optimized mode:

```bash
./launch_tccc_rag_tool.sh --jetson
```

## Command Line Options

The RAG tool can be used directly with various command-line options:

```bash
python tccc_rag_tool.py [OPTIONS]
```

### Mode Selection

- `-i, --interactive`: Run in interactive mode
- `-q QUERY, --query QUERY`: Execute a single query
- `-v, --test-vocabulary`: Test medical vocabulary
- `-b, --benchmark`: Benchmark query strategies
- `-r, --report`: Generate comprehensive RAG system report

### Query Options

- `-s {semantic,keyword,hybrid,expanded}, --strategy {semantic,keyword,hybrid,expanded}`: Query strategy (default: hybrid)
- `-l LIMIT, --limit LIMIT`: Maximum number of results (default: 5)
- `-d, --detailed`: Show detailed results
- `-p, --prompt`: Generate LLM prompt for the query
- `-e, --export`: Export results to file

### Configuration

- `-c CONFIG, --config CONFIG`: Path to custom config file
- `-j, --optimize-for-jetson`: Apply optimizations for Jetson hardware

## Interactive Mode Commands

When running in interactive mode, the following commands are available:

- `strategy:semantic|keyword|hybrid|expanded`: Change search strategy
- `limit:N`: Change result limit
- `prompt`: Show LLM prompt for the last query
- `export`: Export last results to file
- `report`: Generate comprehensive system report with visualizations
- `benchmark`: Run query strategy benchmark
- `vocab`: Test medical vocabulary
- `help`: Show help
- `exit` or `quit`: Exit interactive mode

## Query Strategies

The tool supports multiple query strategies:

1. **Semantic**: Uses vector similarity search based on embedding meaning
2. **Keyword**: Uses traditional keyword-based search
3. **Hybrid**: Combines semantic and keyword searches for best results
4. **Expanded**: Enhances queries with medical terminology expansion

## Medical Vocabulary Features

The tool includes specialized medical vocabulary handling:

- TCCC-specific terminology
- Medical term detection and expansion
- Abbreviation expansion (e.g., CAT → Combat Application Tourniquet)
- Categorized medical terms (hemorrhage, airway, respiration, etc.)
- Synonym recognition

## Jetson Optimization

When running on Jetson hardware, the tool can apply several optimizations:

- Model quantization (int8) for reduced memory usage
- TensorRT acceleration for faster inference
- Optimized batch sizes for GPU processing
- Memory-efficient caching strategies

Use the `--optimize-for-jetson` flag or the `--jetson` parameter with the launch script to enable these optimizations.

## Examples

### Run Interactive Query Session
```bash
./launch_tccc_rag_tool.sh
```

### Execute a Specific Query with Detailed Results
```bash
python tccc_rag_tool.py -q "How to treat tension pneumothorax?" -s hybrid -d
```

### Generate an LLM Prompt for a Query
```bash
python tccc_rag_tool.py -q "MARCH algorithm" -p
```

### Benchmark Query Strategies
```bash
python tccc_rag_tool.py -b
```

### Generate Comprehensive System Report
```bash
python tccc_rag_tool.py -r
```

### Run with Jetson Optimizations
```bash
./launch_tccc_rag_tool.sh --jetson
```

## Troubleshooting

- If the tool fails to initialize, check that the document library configuration is valid
- For database errors, ensure the index files in `data/document_index/` are not corrupted
- If medical vocabulary features are unavailable, check the vocabulary files in `config/vocabulary/`
- For performance issues on Jetson, try using the optimization flag and reducing the result limit