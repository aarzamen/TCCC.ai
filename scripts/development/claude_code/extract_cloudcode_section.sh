#!/bin/bash
# Simple script to extract sections from CLOUDCODE.md for use with Claude Code CLI

CLOUDCODE_PATH="/home/ama/tccc-project/CLOUDCODE.md"
SECTION_TYPE="$1"
SECTION_NAME="$2"

if [ -z "$SECTION_TYPE" ] || [ -z "$SECTION_NAME" ]; then
  echo "Usage: $0 <section_type> <section_name>"
  echo "Example: $0 MODULE 'Audio Pipeline'"
  echo "Example: $0 CORE 'Project Context'"
  echo ""
  echo "Available section types:"
  echo "  CORE - Core sections (always include)"
  echo "  MODULE - Module-specific sections"
  echo "  WORKFLOW - Development workflow sections"
  echo "  REFERENCE - Technical reference sections"
  echo "  META - Document maintenance sections"
  echo ""
  echo "Available sections:"
  grep -E "^## .* \[(CORE|MODULE|WORKFLOW|REFERENCE|META)\]" "$CLOUDCODE_PATH" | sed 's/## /  /' | sed 's/ \[.*\]//'
  exit 1
fi

# Extract section using pattern matching
section_content=$(awk -v section="$SECTION_NAME.*\\[$SECTION_TYPE\\]" '
  $0 ~ "^## " section {
    flag=1
    print
    next
  }
  flag && $0 ~ "^## " {
    flag=0
  }
  flag {
    print
  }
' "$CLOUDCODE_PATH")

if [ -z "$section_content" ]; then
  echo "Error: Section '$SECTION_NAME [$SECTION_TYPE]' not found in $CLOUDCODE_PATH"
  exit 1
fi

echo "$section_content"