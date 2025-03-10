#!/bin/bash
# Run all verification scripts for TCCC.ai system

echo "=================================================="
echo "TCCC.ai System Verification Suite"
echo "=================================================="

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Record start time
START_TIME=$(date +%s)

# Array to track results
declare -A RESULTS

# Function to run a verification script and record result
run_verification() {
    local script="$1"
    local name="$2"
    local args="${3:-}"
    
    echo -e "\n${YELLOW}Running verification for: ${name}${NC}"
    echo "------------------------------------------------"
    
    if python "$script" $args; then
        RESULTS["$name"]="PASSED"
        echo -e "\n${GREEN}Verification PASSED: ${name}${NC}"
    else
        RESULTS["$name"]="FAILED"
        echo -e "\n${RED}Verification FAILED: ${name}${NC}"
    fi
}

# Run individual module verifications
echo -e "\n${YELLOW}RUNNING MODULE-LEVEL VERIFICATIONS${NC}"
echo "=================================================="

run_verification "./verification_script_processing_core.py" "Processing Core"
run_verification "./verification_script_data_store.py" "Data Store"
run_verification "./verification_script_document_library.py" "Document Library"
run_verification "./verification_script_audio_pipeline.py" "Audio Pipeline"
run_verification "./verification_script_stt_engine.py" "STT Engine"
run_verification "./verification_script_llm_analysis.py" "LLM Analysis"

# Run system-level verification with mock implementations
echo -e "\n${YELLOW}RUNNING SYSTEM-LEVEL VERIFICATION${NC}"
echo "=================================================="

run_verification "./test_system_integration.py" "Basic System Integration" 
run_verification "./verification_script_system_enhanced.py" "System Integration" "--mock all"

# Run interface and integration verifications
echo -e "\n${YELLOW}RUNNING INTERFACE VERIFICATIONS${NC}"
echo "=================================================="

run_verification "./verification_script_async_modules.py" "Async/Sync Interface"
run_verification "./verification_script_audio_pipeline_integration.py" "Audio Pipeline/STT Integration"
run_verification "./verification_script_audio_chunk_management.py" "Audio Chunk Management"

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Print summary report
echo -e "\n${YELLOW}VERIFICATION SUMMARY${NC}"
echo "=================================================="
echo "Total time: $ELAPSED seconds"
echo ""
echo "Results:"

PASSED=0
FAILED=0
for name in "${!RESULTS[@]}"; do
    result=${RESULTS["$name"]}
    if [ "$result" == "PASSED" ]; then
        echo -e "${GREEN}✓ $name: $result${NC}"
        PASSED=$((PASSED+1))
    else
        echo -e "${RED}✗ $name: $result${NC}"
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "Total Passed: $PASSED"
echo "Total Failed: $FAILED"

# Return exit code based on whether all tests passed
if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}ALL VERIFICATIONS PASSED!${NC}"
    exit 0
else
    echo -e "\n${RED}SOME VERIFICATIONS FAILED!${NC}"
    exit 1
fi