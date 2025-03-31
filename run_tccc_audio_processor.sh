#!/bin/bash
# TCCC Audio Processor - Combined script for recording and processing tactical audio

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

clear
echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}     TCCC TACTICAL AUDIO PROCESSOR     ${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo

echo -e "${BLUE}What would you like to do?${NC}"
echo "1. Record new tactical audio"
echo "2. Process existing tactical audio"
echo "3. View previous analysis results"
echo "4. Exit"
read -p "> " CHOICE

# Default to 1 if no input provided
if [ -z "$CHOICE" ]; then
    CHOICE=1
fi

case $CHOICE in
    1)
        ./capture_tactical_audio.sh
        echo
        echo -e "${BLUE}Would you like to process this recording now? (y/n)${NC}"
        read -p "> " PROCESS_NOW
        if [[ $PROCESS_NOW == "y" || $PROCESS_NOW == "Y" ]]; then
            ./process_tactical_audio.sh
        fi
        ;;
    2)
        ./process_tactical_audio.sh
        ;;
    3)
        # Find all analysis results
        RESULTS=($(find tactical_processing_results -name "*.txt" -type f))
        
        if [ ${#RESULTS[@]} -eq 0 ]; then
            echo -e "${RED}No analysis results found.${NC}"
            echo "Please record and process tactical audio first."
            exit 0
        fi
        
        echo -e "${BLUE}Select a result file to view:${NC}"
        for i in "${!RESULTS[@]}"; do
            # Extract timestamp from filename for better display
            FILENAME=$(basename "${RESULTS[$i]}")
            TIMESTAMP=$(echo $FILENAME | sed 's/tactical_analysis_\(.*\)\.txt/\1/')
            echo "$((i+1)). Analysis from $TIMESTAMP"
        done
        
        read -p "> " RESULT_CHOICE
        # Default to 1 if empty
        if [ -z "$RESULT_CHOICE" ]; then
            RESULT_CHOICE=1
        fi
        
        if [ "$RESULT_CHOICE" -gt 0 ] && [ "$RESULT_CHOICE" -le "${#RESULTS[@]}" ]; then
            SELECTED=${RESULTS[$((RESULT_CHOICE-1))]}
            echo -e "${GREEN}Displaying: ${SELECTED}${NC}"
            echo
            # Simply use cat for display in terminal
            cat "$SELECTED"
            echo
            echo -e "${BLUE}Press Enter to continue...${NC}"
            read
        else
            echo -e "${RED}Invalid selection.${NC}"
        fi
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Please run again.${NC}"
        exit 1
        ;;
esac

echo -e "${YELLOW}=================================================${NC}"
echo -e "${GREEN}Thank you for using the TCCC Tactical Audio Processor${NC}"
echo -e "${YELLOW}=================================================${NC}"