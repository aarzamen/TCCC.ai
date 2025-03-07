#!/bin/bash
# Script to check for potential conflicts in multi-agent changes

echo "===================================================="
echo "TCCC Agent Conflict Detection Tool"
echo "===================================================="

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get list of agent task files
AGENT_FILES=$(ls AGENT*_TASK.md 2>/dev/null)
if [ -z "$AGENT_FILES" ]; then
    echo -e "${YELLOW}No agent task files found. Create files named AGENT<N>_TASK.md${NC}"
    echo "Example: AGENT1_TASK.md, AGENT2_TASK.md, etc."
    exit 1
fi

# Extract file paths from agent task files
echo "Analyzing agent task files for potential conflicts..."
declare -A AGENT_PATHS
declare -A FILE_AGENTS

for agent_file in $AGENT_FILES; do
    agent_num=$(echo $agent_file | sed -n 's/AGENT\([0-9]\+\)_TASK.md/\1/p')
    echo -e "${YELLOW}Checking $agent_file (Agent $agent_num)${NC}"
    
    # Extract file paths mentioned in task file
    # Look for common file path patterns like src/tccc/...
    paths=$(grep -oE "src/tccc/[a-zA-Z0-9_/]+\.py" $agent_file | sort | uniq)
    
    if [ -z "$paths" ]; then
        echo "  No file paths found in $agent_file"
        continue
    fi
    
    echo "  Files mentioned:"
    for path in $paths; do
        echo "    - $path"
        AGENT_PATHS[$agent_num]+="$path "
        
        # Record which agents are working on each file
        if [ -z "${FILE_AGENTS[$path]}" ]; then
            FILE_AGENTS[$path]="$agent_num"
        else
            FILE_AGENTS[$path]+=" $agent_num"
        fi
    done
    echo ""
done

# Check for conflicts
echo -e "\n${YELLOW}Checking for potential conflicts...${NC}"
echo "----------------------------------------------------"

CONFLICT_FOUND=0

for path in "${!FILE_AGENTS[@]}"; do
    agents="${FILE_AGENTS[$path]}"
    
    # Count number of agents working on this file
    agent_count=$(echo $agents | wc -w)
    
    if [ $agent_count -gt 1 ]; then
        CONFLICT_FOUND=1
        echo -e "${RED}CONFLICT RISK: $path${NC}"
        echo -e "  ${RED}Multiple agents assigned: $agents${NC}"
        
        # Check if file exists and show its content type
        if [ -f "$path" ]; then
            lines=$(wc -l < "$path")
            echo "  File exists: $lines lines"
        else
            echo "  File does not exist yet"
        fi
        echo ""
    fi
done

if [ $CONFLICT_FOUND -eq 0 ]; then
    echo -e "${GREEN}No conflicts detected!${NC}"
    echo -e "Each file is assigned to only one agent."
else
    echo -e "\n${RED}Conflicts detected!${NC}"
    echo -e "Please coordinate agent assignments to avoid conflicts."
    echo -e "Options:"
    echo -e "  1. Assign each file to a single agent"
    echo -e "  2. Split the file into smaller files"
    echo -e "  3. Create clear sections for each agent with comments"
    echo -e "     <!-- AGENT1: Start section -->"
    echo -e "     <!-- AGENT1: End section -->"
fi

echo -e "\n${YELLOW}Analysis complete!${NC}"