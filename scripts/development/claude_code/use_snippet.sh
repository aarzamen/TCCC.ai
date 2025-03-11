#!/bin/bash
# Script to extract and use snippets from CLOUDCODE_SNIPPETS.txt

SNIPPETS_PATH="/home/ama/tccc-project/CLOUDCODE_SNIPPETS.txt"
SNIPPET_NUM="$1"
MODULE_NAME="$2"
OUTPUT_FILE="${3:-/tmp/claude_snippet.txt}"

# Check if snippet number was provided
if [ -z "$SNIPPET_NUM" ]; then
  echo "Usage: $0 <snippet_number> [module_name] [output_file]"
  echo "Example: $0 1 'Audio Pipeline' my_snippet.txt"
  echo ""
  echo "Available snippets:"
  grep -n "^## [0-9]" "$SNIPPETS_PATH" | sed 's/^[0-9]*://' | sed 's/^## /  /'
  exit 1
fi

# Extract the snippet
echo "Extracting snippet $SNIPPET_NUM..."
snippet_content=$(awk -v num="^## $SNIPPET_NUM\\. " '
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
' "$SNIPPETS_PATH")

if [ -z "$snippet_content" ]; then
  echo "Error: Snippet number $SNIPPET_NUM not found in $SNIPPETS_PATH"
  exit 1
fi

# Create temporary output file
> "$OUTPUT_FILE"

# If module name is provided, prepare a session with context
if [ -n "$MODULE_NAME" ]; then
  echo "Preparing context for $MODULE_NAME module..."
  ./prepare_claude_session.sh "$MODULE_NAME" "$OUTPUT_FILE"
  
  # Add separator
  echo "" >> "$OUTPUT_FILE"
  echo "---" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
fi

# Add the snippet to the output file
echo "$snippet_content" >> "$OUTPUT_FILE"

# Open the file in the default editor for customization
echo "Opening snippet in editor for customization..."
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

echo ""
echo "Snippet saved to $OUTPUT_FILE"
echo ""
echo "To use with Claude Code CLI:"
echo "  cat $OUTPUT_FILE | claude"