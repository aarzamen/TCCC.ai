#!/bin/bash
# gemini_analyze.sh - User-friendly wrapper for the Gemini tool

# Ensure we're in the project root directory
cd "$(dirname "$0")/../.."

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    source venv/bin/activate
fi

# Check if required packages are installed
if ! pip list | grep -q "google-generativeai"; then
    echo "Installing required package google-generativeai..."
    pip install google-generativeai
fi

# Make the Python script executable if it isn't already
chmod +x scripts/development/gemini_tool.py

# Parse arguments
OUTPUT_FILE=""
VERBOSE=false

print_usage() {
    echo "Usage: ./gemini_analyze.sh [options] <command> [command_args]"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE    Save output to file"
    echo "  -v, --verbose        Show verbose output"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Commands:"
    echo "  query TEXT              Run a direct query with no additional context"
    echo "  files QUERY FILE1 ...   Analyze specific files with a query"
    echo "  dir QUERY DIR           Analyze a directory with a query"
    echo "  glob QUERY PATTERN      Analyze files matching a glob pattern with a query"
    echo "  module QUERY MODULE     Analyze a Python module (e.g., audio_pipeline)"
    echo "  component QUERY COMP    Analyze a component across multiple modules"
    echo ""
    echo "Examples:"
    echo "  ./gemini_analyze.sh query \"How does the TCCC system work?\""
    echo "  ./gemini_analyze.sh files \"Find integration issues\" src/tccc/audio_pipeline/*.py src/tccc/stt_engine/*.py"
    echo "  ./gemini_analyze.sh dir \"List all classes\" src/tccc/document_library"
    echo "  ./gemini_analyze.sh glob \"Find event handlers\" \"src/tccc/**/*.py\""
    echo "  ./gemini_analyze.sh module \"Explain the pipeline\" audio_pipeline"
    echo "  ./gemini_analyze.sh component \"How are events used?\" event_bus"
    echo "  ./gemini_analyze.sh -o analysis.txt module \"Explain architecture\" system"
}

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Check if any command was provided
if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift

# Common output handling
output_opts=""
if [[ -n "$OUTPUT_FILE" ]]; then
    output_opts="--output $OUTPUT_FILE"
fi

# Process commands
case "$COMMAND" in
    query)
        if [[ $# -lt 1 ]]; then
            echo "Error: query command requires a query text"
            print_usage
            exit 1
        fi
        
        QUERY="$1"
        ./scripts/development/gemini_tool.py --query "$QUERY" --text "" $output_opts
        ;;
        
    files)
        if [[ $# -lt 2 ]]; then
            echo "Error: files command requires a query and at least one file"
            print_usage
            exit 1
        fi
        
        QUERY="$1"
        shift
        FILES="$*"
        
        if $VERBOSE; then
            echo "Analyzing files: $FILES"
        fi
        
        ./scripts/development/gemini_tool.py --query "$QUERY" --files "$FILES" $output_opts
        ;;
        
    dir)
        if [[ $# -lt 2 ]]; then
            echo "Error: dir command requires a query and a directory path"
            print_usage
            exit 1
        fi
        
        QUERY="$1"
        DIR="$2"
        RECURSIVE=""
        
        if [[ "$3" == "--recursive" ]]; then
            RECURSIVE="--recursive"
        fi
        
        if $VERBOSE; then
            echo "Analyzing directory: $DIR"
        fi
        
        ./scripts/development/gemini_tool.py --query "$QUERY" --dir "$DIR" $RECURSIVE $output_opts
        ;;
        
    glob)
        if [[ $# -lt 2 ]]; then
            echo "Error: glob command requires a query and a glob pattern"
            print_usage
            exit 1
        fi
        
        QUERY="$1"
        PATTERN="$2"
        
        if $VERBOSE; then
            echo "Analyzing files matching: $PATTERN"
        fi
        
        ./scripts/development/gemini_tool.py --query "$QUERY" --glob "$PATTERN" $output_opts
        ;;
        
    module)
        if [[ $# -lt 2 ]]; then
            echo "Error: module command requires a query and a module name"
            print_usage
            exit 1
        fi
        
        QUERY="$1"
        MODULE="$2"
        
        if $VERBOSE; then
            echo "Analyzing module: $MODULE"
        fi
        
        MODULE_PATH="src/tccc/$MODULE"
        if [ -d "$MODULE_PATH" ]; then
            ./scripts/development/gemini_tool.py --query "$QUERY" --dir "$MODULE_PATH" --recursive $output_opts
        else
            echo "Error: Module directory not found at $MODULE_PATH"
            exit 1
        fi
        ;;
        
    component)
        if [[ $# -lt 2 ]]; then
            echo "Error: component command requires a query and a component name"
            print_usage
            exit 1
        fi
        
        QUERY="$1"
        COMPONENT="$2"
        
        if $VERBOSE; then
            echo "Analyzing component: $COMPONENT"
        fi
        
        # Look for the component across multiple modules
        PATTERN="src/tccc/**/*${COMPONENT}*.py"
        ./scripts/development/gemini_tool.py --query "$QUERY" --glob "$PATTERN" $output_opts
        ;;
        
    *)
        echo "Error: Unknown command '$COMMAND'"
        print_usage
        exit 1
        ;;
esac