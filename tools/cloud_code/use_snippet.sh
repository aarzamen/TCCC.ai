#!/bin/bash
# Enhanced script to extract and use snippets from CLOUDCODE_SNIPPETS.txt

SCRIPT_DIR="$(dirname "$0")"
SNIPPETS_PATH="$SCRIPT_DIR/CLOUDCODE_SNIPPETS.txt"
TEMP_DIR="/tmp/claude_code"
OUTPUT_FILE=""

# Function to display usage information
show_usage() {
  echo "Usage: $0 [options] <snippet_number> [module_name] [output_file]"
  echo "Extract and prepare snippets from CLOUDCODE_SNIPPETS.txt for Claude Code CLI"
  echo ""
  echo "Options:"
  echo "  -h, --help      Show this help message"
  echo "  -l, --list      List available snippets"
  echo "  -c, --context   Include CORE context automatically"
  echo "  -p, --preview   Preview snippet without saving"
  echo "  -e, --edit      Open in editor (default behavior)"
  echo "  -o FILE         Write output to FILE instead of default location"
  echo "  -n, --no-edit   Don't open editor, just save to file"
  echo ""
  echo "Examples:"
  echo "  $0 1                           # Extract snippet 1 (Implement Module Function)"
  echo "  $0 3 \"Audio Pipeline\"          # Extract snippet 3 with module context"
  echo "  $0 -c 5                        # Extract snippet 5 with CORE sections"
  echo "  $0 -o my_prompt.txt 2          # Save snippet 2 to my_prompt.txt"
  exit 1
}

# Function to list available snippets
list_snippets() {
  echo "Available snippets in CLOUDCODE_SNIPPETS.txt:"
  echo ""
  grep -n "^## [0-9]" "$SNIPPETS_PATH" | sed 's/^[0-9]*://' | sed 's/^## /  /'
  exit 0
}

# Function to extract a specific snippet
extract_snippet() {
  local snippet_num="$1"
  
  awk -v num="^## $snippet_num\\. " '
    $0 ~ num {
      flag=1
      next
    }
    flag && $0 ~ "^## [0-9]" {
      flag=0
    }
    flag {
      print
    }
  ' "$SNIPPETS_PATH"
}

# Function to get CORE context using extract_cloudcode_section.sh
get_core_context() {
  "$SCRIPT_DIR/extract_cloudcode_section.sh" -c
}

# Function to get module context
get_module_context() {
  local module_name="$1"
  "$SCRIPT_DIR/extract_cloudcode_section.sh" MODULE "$module_name"
}

# Parse command line arguments
INCLUDE_CORE=false
PREVIEW_ONLY=false
EDIT_FILE=true
SNIPPET_NUM=""
MODULE_NAME=""

# Create temp directory if it doesn't exist
mkdir -p "$TEMP_DIR"

# Default output file
DEFAULT_OUTPUT="$TEMP_DIR/snippet_$(date +%s).txt"

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_usage
      ;;
    -l|--list)
      list_snippets
      ;;
    -c|--context)
      INCLUDE_CORE=true
      shift
      ;;
    -p|--preview)
      PREVIEW_ONLY=true
      shift
      ;;
    -e|--edit)
      EDIT_FILE=true
      shift
      ;;
    -n|--no-edit)
      EDIT_FILE=false
      shift
      ;;
    -o)
      shift
      OUTPUT_FILE="$1"
      shift
      ;;
    *)
      if [ -z "$SNIPPET_NUM" ]; then
        SNIPPET_NUM="$1"
      elif [ -z "$MODULE_NAME" ]; then
        MODULE_NAME="$1"
      elif [ -z "$OUTPUT_FILE" ]; then
        OUTPUT_FILE="$1"
      else
        echo "Error: Too many arguments"
        show_usage
      fi
      shift
      ;;
  esac
done

# Set output file if not specified
if [ -z "$OUTPUT_FILE" ]; then
  OUTPUT_FILE="$DEFAULT_OUTPUT"
fi

# Validate input
if [ -z "$SNIPPET_NUM" ]; then
  echo "Error: Snippet number is required"
  show_usage
fi

# Check if snippets file exists
if [ ! -f "$SNIPPETS_PATH" ]; then
  echo "Error: CLOUDCODE_SNIPPETS.txt not found at $SNIPPETS_PATH"
  exit 1
fi

# Extract snippet content
echo "Extracting snippet $SNIPPET_NUM..."
SNIPPET_CONTENT=$(extract_snippet "$SNIPPET_NUM")

# Check if snippet was found
if [ -z "$SNIPPET_CONTENT" ]; then
  echo "Error: Snippet number $SNIPPET_NUM not found in CLOUDCODE_SNIPPETS.txt"
  echo "Run '$0 -l' to see available snippets"
  exit 1
fi

# If preview only, display and exit
if [ "$PREVIEW_ONLY" = true ]; then
  echo "====== SNIPPET $SNIPPET_NUM PREVIEW ======"
  echo ""
  echo "$SNIPPET_CONTENT"
  echo ""
  echo "====== END PREVIEW ======"
  exit 0
fi

# Build complete prompt
PROMPT=""

# Add CORE context if requested
if [ "$INCLUDE_CORE" = true ]; then
  echo "Including CORE context sections..."
  CORE_CONTEXT=$(get_core_context)
  PROMPT+="$CORE_CONTEXT"
  PROMPT+=$'\n\n---\n\n'
fi

# Add module context if provided
if [ -n "$MODULE_NAME" ]; then
  echo "Including context for $MODULE_NAME module..."
  MODULE_CONTEXT=$(get_module_context "$MODULE_NAME")
  
  if [ -n "$MODULE_CONTEXT" ]; then
    PROMPT+="$MODULE_CONTEXT"
    PROMPT+=$'\n\n---\n\n'
  else
    echo "Warning: No context found for module '$MODULE_NAME'"
  fi
fi

# Add the snippet to the prompt
PROMPT+="$SNIPPET_CONTENT"

# Write prompt to output file
echo "$PROMPT" > "$OUTPUT_FILE"

echo "Prompt created at: $OUTPUT_FILE"

# Open in editor if requested
if [ "$EDIT_FILE" = true ]; then
  echo "Opening in editor for customization..."
  echo "Replace all placeholders with your specific details."

  if [ -n "$EDITOR" ]; then
    $EDITOR "$OUTPUT_FILE"
  elif command -v nano >/dev/null 2>&1; then
    nano "$OUTPUT_FILE"
  elif command -v vim >/dev/null 2>&1; then
    vim "$OUTPUT_FILE"
  else
    echo "No editor found. Please edit $OUTPUT_FILE manually."
  fi
fi

echo ""
echo "To use with Claude Code CLI:"
echo "  cat $OUTPUT_FILE | claude"