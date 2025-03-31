#!/bin/bash

# efficient_search.sh - Script to perform efficient code searches in the TCCC project
# This script helps avoid expensive API costs by intelligently searching the codebase

# Default search locations
DEFAULT_SEARCH_PATHS="src tests examples scripts"

# Directories to always exclude
EXCLUDED_DIRS=(
  "venv"
  "gguf_download/models"
  "models"
  "data/document_index"
  "fullsubnet_integration/models"
  "cache"
  ".git"
)

# File patterns to exclude
EXCLUDED_PATTERNS=(
  "*.faiss"
  "*.pth"
  "*.gguf"
  "*.wav"
  "transcript_*.txt"
  "*.log"
  "*.ipynb"
)

function show_help() {
  echo "TCCC Project Code Search Utility"
  echo ""
  echo "Usage: ./efficient_search.sh [OPTIONS] PATTERN"
  echo ""
  echo "Options:"
  echo "  -p, --path PATH   Specify search path (default: ${DEFAULT_SEARCH_PATHS})"
  echo "  -i, --include PATTERN   Only search files matching PATTERN (e.g. '*.py')"
  echo "  -l, --list-files   Only list filenames, not matched lines"
  echo "  -h, --help   Display this help message"
  echo ""
  echo "Examples:"
  echo "  ./efficient_search.sh 'def process_audio'"
  echo "  ./efficient_search.sh -i '*.py' 'AudioPipeline'"
  echo "  ./efficient_search.sh -p 'src/tccc/audio_pipeline' 'initialize'"
  echo ""
}

# Default values
SEARCH_PATH=$DEFAULT_SEARCH_PATHS
INCLUDE_PATTERN=""
LIST_FILES_ONLY=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--path)
      SEARCH_PATH="$2"
      shift 2
      ;;
    -i|--include)
      INCLUDE_PATTERN="$2"
      shift 2
      ;;
    -l|--list-files)
      LIST_FILES_ONLY=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      SEARCH_PATTERN="$1"
      shift
      ;;
  esac
done

if [ -z "$SEARCH_PATTERN" ]; then
  echo "Error: No search pattern provided."
  show_help
  exit 1
fi

# Build the exclude arguments for find command
EXCLUDE_ARGS=""
for dir in "${EXCLUDED_DIRS[@]}"; do
  EXCLUDE_ARGS="$EXCLUDE_ARGS -not -path '*/$dir/*'"
done

for pattern in "${EXCLUDED_PATTERNS[@]}"; do
  EXCLUDE_ARGS="$EXCLUDE_ARGS -not -name '$pattern'"
done

# Build the include argument if specified
INCLUDE_ARG=""
if [ -n "$INCLUDE_PATTERN" ]; then
  INCLUDE_ARG="-name '$INCLUDE_PATTERN'"
else
  # Default to common code files if no pattern specified
  INCLUDE_ARG="-name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json'"
fi

# Construct and execute the search command
if [ $LIST_FILES_ONLY -eq 1 ]; then
  GREP_OPTIONS="-l"
else
  GREP_OPTIONS="-n --color=always"
fi

# Print a notice about what we're searching
echo "Searching for: '$SEARCH_PATTERN'"
echo "In directories: $SEARCH_PATH"
echo "File pattern: ${INCLUDE_PATTERN:-'(code files)'}"
echo "Excluding: ${EXCLUDED_DIRS[*]} and ${EXCLUDED_PATTERNS[*]}"
echo "-------------------------------------------------------------"

# Use eval to properly handle the complex command with variables containing spaces
eval "find $SEARCH_PATH $INCLUDE_ARG $EXCLUDE_ARGS -type f -print0 | xargs -0 grep $GREP_OPTIONS \"$SEARCH_PATTERN\""

exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "-------------------------------------------------------------"
  echo "Search completed successfully."
elif [ $exit_code -eq 1 ]; then
  echo "-------------------------------------------------------------"
  echo "No matches found."
else
  echo "-------------------------------------------------------------"
  echo "Search encountered an error (code: $exit_code)."
fi