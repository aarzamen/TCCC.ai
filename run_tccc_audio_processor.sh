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
            echo "$((i+1)). ${RESULTS[$i]}"
        done
        
        read -p "> " RESULT_CHOICE
        if [ "$RESULT_CHOICE" -gt 0 ] && [ "$RESULT_CHOICE" -le "${#RESULTS[@]}" ]; then
            SELECTED=${RESULTS[$((RESULT_CHOICE-1))]}
            if command -v less &> /dev/null; then
                less "$SELECTED"
            else
                cat "$SELECTED"
            fi
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