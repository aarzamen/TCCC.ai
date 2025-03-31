#!/bin/bash
#
# TCCC.ai Installation Verification Script
#
# This script verifies that the TCCC.ai system was installed correctly.

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================\033[0m"
echo -e "${GREEN}   TCCC.ai Installation Verification  \033[0m"
echo -e "${GREEN}=======================================\033[0m"

# Check Python installation
echo -e "${YELLOW}Checking Python installation...\033[0m"
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version)
    echo -e "${GREEN}Python installed: $python_version\033[0m"
else
    echo -e "${RED}Python 3 not found!\033[0m"
    exit 1
fi

# Check virtual environment
echo -e "${YELLOW}Checking virtual environment...\033[0m"
if [ -d "venv" ]; then
    echo -e "${GREEN}Virtual environment found\033[0m"
else
    echo -e "${RED}Virtual environment not found!\033[0m"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check TCCC package installation
echo -e "${YELLOW}Checking TCCC package installation...\033[0m"
if python3 -c "import tccc" 2>/dev/null; then
    echo -e "${GREEN}TCCC package installed\033[0m"
else
    echo -e "${RED}TCCC package not installed!\033[0m"
    exit 1
fi

# Check for required data files
echo -e "${YELLOW}Checking for required data files...\033[0m"
if [ -d "data/documents" ]; then
    echo -e "${GREEN}Documents directory found\033[0m"
else
    echo -e "${YELLOW}Documents directory not found, please run download_rag_documents.py\033[0m"
fi

# Run a basic system check
echo -e "${YELLOW}Running basic system check...\033[0m"
if [ -f "scripts/run_all_verifications.sh" ]; then
    bash scripts/run_all_verifications.sh --quick
else
    echo -e "${RED}Verification script not found!\033[0m"
    exit 1
fi

echo -e "${GREEN}Installation verification completed!\033[0m"
