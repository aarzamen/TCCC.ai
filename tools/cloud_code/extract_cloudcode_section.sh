#!/bin/bash
# Enhanced script to extract sections from CLOUDCODE.md for use with Claude Code CLI

CLOUDCODE_PATH="$(dirname "$0")/CLOUDCODE.md"
OUTPUT_FILE=""

# Function to display usage information
show_usage() {
  echo "Usage: $0 [options] <section_type> [section_name] [output_file]"
  echo "Extract sections from CLOUDCODE.md for Claude Code CLI sessions"
  echo ""
  echo "Options:"
  echo "  -h, --help    Show this help message"
  echo "  -l, --list    List available sections"
  echo "  -a, --all     Extract all sections of specified type"
  echo "  -o FILE       Write output to FILE instead of stdout"
  echo "  -c, --core    Extract all CORE sections (shortcut)"
  echo ""
  echo "Available section types:"
  echo "  CORE          Core sections (always include)"
  echo "  MODULE        Module-specific sections"
  echo "  WORKFLOW      Development workflow sections"
  echo "  REFERENCE     Technical reference sections"
  echo "  META          Document maintenance sections"
  echo ""
  echo "Examples:"
  echo "  $0 MODULE 'Audio Pipeline'       # Extract Audio Pipeline module section"
  echo "  $0 -a CORE                       # Extract all CORE sections"
  echo "  $0 CORE 'Project Context' -o context.md  # Save to file"
  echo "  $0 -c                            # Extract all CORE sections (shortcut)"
  exit 1
}

# Function to list available sections
list_sections() {
  echo "Available sections in CLOUDCODE.md:"
  echo ""
  grep -E "^## .* \[(CORE|MODULE|WORKFLOW|REFERENCE|META)\]" "$CLOUDCODE_PATH" | sed 's/## /  /' | sed 's/ \[/\t\[/'
  exit 0
}

# Function to extract a specific section
extract_section() {
  local section_type="$1"
  local section_name="$2"
  
  if [ -z "$section_name" ]; then
    # If no section name provided, get all sections of this type
    awk -v type="\\[$section_type\\]" '
      $0 ~ "^## .* " type {
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
    ' "$CLOUDCODE_PATH"
  else
    # Extract specific named section
    awk -v section="$section_name.*\\[$section_type\\]" '
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
    ' "$CLOUDCODE_PATH"
  fi
}

# Function to extract all CORE sections
extract_all_core() {
  sections=("Project Context" "Development Environment" "Claude Code CLI Workflow")
  
  for section in "${sections[@]}"; do
    echo "## $section Section"
    echo ""
    extract_section "CORE" "$section"
    echo ""
    echo "---"
    echo ""
  done
}

# Parse command line arguments
EXTRACT_ALL=false
SECTION_TYPE=""
SECTION_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_usage
      ;;
    -l|--list)
      list_sections
      ;;
    -a|--all)
      EXTRACT_ALL=true
      shift
      SECTION_TYPE="$1"
      shift
      ;;
    -o)
      shift
      OUTPUT_FILE="$1"
      shift
      ;;
    -c|--core)
      extract_all_core > "${OUTPUT_FILE:-/dev/stdout}"
      exit 0
      ;;
    *)
      if [ -z "$SECTION_TYPE" ]; then
        SECTION_TYPE="$1"
      elif [ -z "$SECTION_NAME" ]; then
        SECTION_NAME="$1"
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

# Validate input
if [ -z "$SECTION_TYPE" ]; then
  echo "Error: Section type is required"
  show_usage
fi

# Check if the file exists
if [ ! -f "$CLOUDCODE_PATH" ]; then
  echo "Error: CLOUDCODE.md not found at $CLOUDCODE_PATH"
  exit 1
fi

# Extract content based on arguments
if [ "$EXTRACT_ALL" = true ]; then
  content=$(extract_section "$SECTION_TYPE")
else
  content=$(extract_section "$SECTION_TYPE" "$SECTION_NAME")
fi

# Check if any content was found
if [ -z "$content" ]; then
  if [ -z "$SECTION_NAME" ]; then
    echo "Error: No sections found with type [$SECTION_TYPE]"
  else
    echo "Error: Section '$SECTION_NAME [$SECTION_TYPE]' not found in CLOUDCODE.md"
  fi
  exit 1
fi

# Output content
if [ -n "$OUTPUT_FILE" ]; then
  echo "$content" > "$OUTPUT_FILE"
  echo "Section(s) extracted to $OUTPUT_FILE"
else
  echo "$content"
fi