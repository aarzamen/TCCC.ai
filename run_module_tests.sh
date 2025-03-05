#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/ama/tccc-project

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Header
echo -e "${CYAN}Running TCCC.ai Module Tests${NC}"
echo -e "${CYAN}============================${NC}"

# Function to run tests for a specific module
run_test() {
    module_name=$1
    test_path=$2
    
    echo -e "\n${YELLOW}Testing module: ${module_name}${NC}"
    echo -e "${YELLOW}------------------------------${NC}"
    
    if [ -f "$test_path" ]; then
        python -m pytest "$test_path" -v
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ $module_name tests passed${NC}"
            return 0
        else
            echo -e "${RED}✗ $module_name tests failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}No tests found for $module_name${NC}"
        return 2
    fi
}

# Run tests for each module
failed_modules=()
passed_modules=()
skipped_modules=()

# Utils
if run_test "Utils - Logging" "tests/utils/test_logging.py"; then
    passed_modules+=("Utils - Logging")
else
    failed_modules+=("Utils - Logging")
fi

# Audio Pipeline
if run_test "Audio Pipeline" "tests/audio_pipeline/test_audio_pipeline.py"; then
    passed_modules+=("Audio Pipeline")
else
    failed_modules+=("Audio Pipeline")
fi

# Data Store
if run_test "Data Store" "tests/data_store/test_data_store.py"; then
    passed_modules+=("Data Store")
else
    failed_modules+=("Data Store")
fi

# Processing Core (skip due to missing dependencies)
echo -e "\n${YELLOW}Testing module: Processing Core${NC}"
echo -e "${YELLOW}------------------------------${NC}"
echo -e "${CYAN}Skipping due to missing dependencies (spacy)${NC}"
skipped_modules+=("Processing Core")

# LLM Analysis (skip due to missing dependencies)
echo -e "\n${YELLOW}Testing module: LLM Analysis${NC}"
echo -e "${YELLOW}------------------------------${NC}"
echo -e "${CYAN}Skipping due to missing dependencies (fitz)${NC}"
skipped_modules+=("LLM Analysis")

# STT Engine (if tests exist)
if [ -f "tests/stt_engine/test_stt_engine.py" ]; then
    if run_test "STT Engine" "tests/stt_engine/test_stt_engine.py"; then
        passed_modules+=("STT Engine")
    else
        failed_modules+=("STT Engine")
    fi
else
    echo -e "\n${YELLOW}Testing module: STT Engine${NC}"
    echo -e "${YELLOW}------------------------------${NC}"
    echo -e "${CYAN}No tests found${NC}"
    skipped_modules+=("STT Engine")
fi

# Document Library (skip due to missing dependencies)
echo -e "\n${YELLOW}Testing module: Document Library${NC}"
echo -e "${YELLOW}------------------------------${NC}"
echo -e "${CYAN}Skipping due to missing dependencies (fitz)${NC}"
skipped_modules+=("Document Library")

# Summary
echo -e "\n${CYAN}Test Summary${NC}"
echo -e "${CYAN}===========${NC}"

echo -e "${GREEN}Passed (${#passed_modules[@]})${NC}:"
for module in "${passed_modules[@]}"; do
    echo -e "  ${GREEN}✓ $module${NC}"
done

if [ ${#failed_modules[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed (${#failed_modules[@]})${NC}:"
    for module in "${failed_modules[@]}"; do
        echo -e "  ${RED}✗ $module${NC}"
    done
fi

if [ ${#skipped_modules[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Skipped (${#skipped_modules[@]})${NC}:"
    for module in "${skipped_modules[@]}"; do
        echo -e "  ${YELLOW}- $module${NC}"
    done
fi

# Exit with error if any tests failed
if [ ${#failed_modules[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi