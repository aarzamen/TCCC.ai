#!/bin/bash
# Script to prepare a Claude Code session with the appropriate context

# Check if a module was specified
if [ -z "$1" ]; then
  echo "Usage: $0 <module_name> [output_file]"
  echo "Example: $0 'Audio Pipeline' session.txt"
  echo ""
  echo "Available modules:"
  grep -E "^## .* \[MODULE\]" /home/ama/tccc-project/CLOUDCODE.md | sed 's/## /  /' | sed 's/ \[MODULE\]//'
  exit 1
fi

MODULE_NAME="$1"
OUTPUT_FILE="${2:-/tmp/claude_session.txt}"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Always include core sections
echo "Preparing Claude session for $MODULE_NAME module..."

# Project Context
echo "Adding Project Context section..."
./extract_cloudcode_section.sh CORE "Project Context" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Development Environment
echo "Adding Development Environment section..."
./extract_cloudcode_section.sh CORE "Development Environment" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Claude Code CLI Workflow
echo "Adding Claude Code CLI Workflow section..."
./extract_cloudcode_section.sh CORE "Claude Code CLI Workflow" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Add the specific module section
echo "Adding $MODULE_NAME module section..."
./extract_cloudcode_section.sh MODULE "$MODULE_NAME" >> "$OUTPUT_FILE" 2>/dev/null

if [ $? -ne 0 ]; then
  echo "Warning: Module '$MODULE_NAME' not found. Continuing without module-specific context."
fi

echo "" >> "$OUTPUT_FILE"

# Add instructions for the user
cat >> "$OUTPUT_FILE" << EOF
I'm now ready to help you with the $MODULE_NAME module. Please provide:

1. The specific task or functionality you want to implement
2. Any relevant interface requirements or constraints
3. Performance considerations for the Jetson platform

Or, if you're using a snippet from CLOUDCODE_SNIPPETS.txt, paste it below with your specific details filled in.
EOF

echo "Session prepared and saved to $OUTPUT_FILE"
echo "To use with Claude Code CLI:"
echo "  cat $OUTPUT_FILE | claude"
echo ""
echo "Or use the file as a starting point for your prompt"