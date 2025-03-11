#!/bin/bash
# Enhanced script to prepare a Claude Code session with the appropriate context

SCRIPT_DIR="$(dirname "$0")"
CLOUDCODE_PATH="$SCRIPT_DIR/CLOUDCODE.md"
TEMP_DIR="/tmp/claude_code"
OUTPUT_FILE=""

# Function to display usage information
show_usage() {
  echo "Usage: $0 [options] <module_name> [output_file]"
  echo "Prepare a Claude Code CLI session with appropriate context from CLOUDCODE.md"
  echo ""
  echo "Options:"
  echo "  -h, --help          Show this help message"
  echo "  -l, --list          List available modules"
  echo "  -w, --workflow      Include specific workflow section"
  echo "  -r, --reference     Include specific reference section"
  echo "  -a, --all-core      Include all CORE sections (default behavior)"
  echo "  -m, --minimal       Include only Project Context from CORE sections"
  echo "  -o FILE             Write output to FILE instead of default location"
  echo ""
  echo "Examples:"
  echo "  $0 'Audio Pipeline'               # Prepare session for Audio Pipeline module"
  echo "  $0 -w 'Testing' 'STT Engine'      # Include Testing workflow with STT Engine"
  echo "  $0 -m 'Document Library'          # Minimal context with Document Library"
  echo "  $0 -o session.txt 'Data Store'    # Save to session.txt"
  exit 1
}

# Function to list available modules
list_modules() {
  echo "Available modules in CLOUDCODE.md:"
  echo ""
  grep -E "^## .* \[MODULE\]" "$CLOUDCODE_PATH" | sed 's/## /  /' | sed 's/ \[MODULE\]//'
  echo ""
  echo "Available workflows:"
  grep -E "^## .* \[WORKFLOW\]" "$CLOUDCODE_PATH" | sed 's/## /  /' | sed 's/ \[WORKFLOW\]//'
  echo ""
  echo "Available reference sections:"
  grep -E "^## .* \[REFERENCE\]" "$CLOUDCODE_PATH" | sed 's/## /  /' | sed 's/ \[REFERENCE\]//'
  exit 0
}

# Create temp directory if it doesn't exist
mkdir -p "$TEMP_DIR"

# Default output file
DEFAULT_OUTPUT="$TEMP_DIR/session_$(date +%s).txt"

# Parse command line arguments
INCLUDE_ALL_CORE=true
INCLUDE_MINIMAL=false
WORKFLOW_SECTION=""
REFERENCE_SECTION=""
MODULE_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_usage
      ;;
    -l|--list)
      list_modules
      ;;
    -w|--workflow)
      shift
      WORKFLOW_SECTION="$1"
      shift
      ;;
    -r|--reference)
      shift
      REFERENCE_SECTION="$1"
      shift
      ;;
    -a|--all-core)
      INCLUDE_ALL_CORE=true
      INCLUDE_MINIMAL=false
      shift
      ;;
    -m|--minimal)
      INCLUDE_ALL_CORE=false
      INCLUDE_MINIMAL=true
      shift
      ;;
    -o)
      shift
      OUTPUT_FILE="$1"
      shift
      ;;
    *)
      if [ -z "$MODULE_NAME" ]; then
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
if [ -z "$MODULE_NAME" ]; then
  echo "Error: Module name is required"
  show_usage
fi

# Check if CloudCode file exists
if [ ! -f "$CLOUDCODE_PATH" ]; then
  echo "Error: CLOUDCODE.md not found at $CLOUDCODE_PATH"
  exit 1
fi

# Clear the output file
> "$OUTPUT_FILE"

echo "Preparing Claude session for $MODULE_NAME module..."

# Include CORE sections based on options
if [ "$INCLUDE_ALL_CORE" = true ]; then
  echo "Adding all CORE sections..."
  
  # Project Context
  echo "- Project Context"
  "$SCRIPT_DIR/extract_cloudcode_section.sh" CORE "Project Context" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  echo "---" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  
  # Development Environment
  echo "- Development Environment"
  "$SCRIPT_DIR/extract_cloudcode_section.sh" CORE "Development Environment" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  echo "---" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  
  # Claude Code CLI Workflow
  echo "- Claude Code CLI Workflow"
  "$SCRIPT_DIR/extract_cloudcode_section.sh" CORE "Claude Code CLI Workflow" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  echo "---" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  
elif [ "$INCLUDE_MINIMAL" = true ]; then
  echo "Adding minimal CORE context (Project Context only)..."
  "$SCRIPT_DIR/extract_cloudcode_section.sh" CORE "Project Context" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  echo "---" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
fi

# Add the specific module section
echo "Adding $MODULE_NAME module section..."
"$SCRIPT_DIR/extract_cloudcode_section.sh" MODULE "$MODULE_NAME" >> "$OUTPUT_FILE" 2>/dev/null

if [ $? -ne 0 ]; then
  echo "Warning: Module '$MODULE_NAME' not found. Continuing without module-specific context."
fi

echo "" >> "$OUTPUT_FILE"

# Add workflow section if specified
if [ -n "$WORKFLOW_SECTION" ]; then
  echo "Adding $WORKFLOW_SECTION workflow section..."
  "$SCRIPT_DIR/extract_cloudcode_section.sh" WORKFLOW "$WORKFLOW_SECTION" >> "$OUTPUT_FILE" 2>/dev/null
  
  if [ $? -ne 0 ]; then
    echo "Warning: Workflow '$WORKFLOW_SECTION' not found."
  else
    echo "" >> "$OUTPUT_FILE"
    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
  fi
fi

# Add reference section if specified
if [ -n "$REFERENCE_SECTION" ]; then
  echo "Adding $REFERENCE_SECTION reference section..."
  "$SCRIPT_DIR/extract_cloudcode_section.sh" REFERENCE "$REFERENCE_SECTION" >> "$OUTPUT_FILE" 2>/dev/null
  
  if [ $? -ne 0 ]; then
    echo "Warning: Reference section '$REFERENCE_SECTION' not found."
  else
    echo "" >> "$OUTPUT_FILE"
    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
  fi
fi

# Add instructions for the user
cat >> "$OUTPUT_FILE" << EOF
I'm now ready to help you with the $MODULE_NAME module. Please provide:

1. The specific task or functionality you want to implement
2. Any relevant interface requirements or constraints
3. Performance considerations for the Jetson platform

Or, if you're using a snippet from CLOUDCODE_SNIPPETS.txt, paste it below with your specific details filled in.
EOF

echo ""
echo "Session prepared and saved to: $OUTPUT_FILE"
echo ""
echo "To use with Claude Code CLI:"
echo "  cat $OUTPUT_FILE | claude"