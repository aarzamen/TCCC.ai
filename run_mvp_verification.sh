#!/bin/bash
#
# TCCC MVP Verification Script
# Runs a comprehensive verification of the TCCC MVP
#

# Move to project directory
cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}TCCC MVP Verification${NC}"
echo "===============================
This script will verify all core components of the TCCC MVP
to ensure it meets the requirements for a working system.

The verification will check:
- Core components (Audio, STT, Event System, Display)
- Integration points between components
- End-to-end functionality
"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please create a virtual environment with required dependencies first."
    exit 1
fi

# Source virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    echo "Please ensure Python is installed and available in the virtual environment."
    exit 1
fi

# Check if verification script exists
if [ ! -f "verify_tccc_system.py" ]; then
    echo -e "${RED}Error: Verification script not found${NC}"
    echo "The verification script 'verify_tccc_system.py' is missing."
    exit 1
fi

# Make verification script executable
chmod +x verify_tccc_system.py

# Run the comprehensive MVP verification
echo -e "${YELLOW}Starting MVP verification...${NC}"
echo "This may take a few minutes to complete."
echo "---------------------------------------"

./verify_tccc_system.py --mvp

# Check the exit code
STATUS=$?

echo "---------------------------------------"

if [ $STATUS -eq 0 ]; then
    echo -e "${GREEN}MVP verification completed successfully!${NC}"
    echo "The TCCC system meets all critical requirements for the MVP."
    echo "See verification_status.txt for detailed results."
else
    echo -e "${RED}MVP verification failed.${NC}"
    echo "Some critical components did not pass verification."
    echo "Please check verification_status.txt for details on which components failed."
fi

# Deactivate virtual environment
deactivate

exit $STATUS