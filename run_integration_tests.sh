#!/bin/bash

# TCCC System Integration Test Script
# This script runs all integration tests for the TCCC system

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== TCCC System Integration Tests ===${NC}"
echo "Running tests to verify all system components are properly integrated."
echo "Each test verifies a specific integration point between components."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Create results directory
RESULTS_DIR="integration_test_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/integration_results_$TIMESTAMP.log"

echo -e "${YELLOW}Test results will be saved to:${NC} $RESULTS_FILE"
echo ""

# Function to run a test and report result
run_test() {
    TEST_NAME=$1
    TEST_CMD=$2
    
    echo -e "${YELLOW}Running $TEST_NAME...${NC}"
    echo "Command: $TEST_CMD"
    echo ""
    
    echo "=== $TEST_NAME ===" >> "$RESULTS_FILE"
    echo "Command: $TEST_CMD" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # Run the test and capture output and return code
    OUTPUT=$(eval "$TEST_CMD 2>&1")
    RC=$?
    
    # Save output to results file
    echo "$OUTPUT" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # Report result
    if [ $RC -eq 0 ]; then
        echo -e "${GREEN}✓ $TEST_NAME PASSED${NC}"
        echo "Test PASSED ($RC)" >> "$RESULTS_FILE"
    else
        echo -e "${RED}✗ $TEST_NAME FAILED${NC}"
        echo "Test FAILED ($RC)" >> "$RESULTS_FILE"
        echo "Error details:"
        echo "$OUTPUT" | grep -i "error\|fail\|exception" | head -5
    fi
    
    echo "" >> "$RESULTS_FILE"
    echo "----------------------------------------" >> "$RESULTS_FILE"
    echo ""
    
    return $RC
}

# Store test results
TEST_RESULTS=()
TEST_NAMES=()

# Run component verification tests first
echo -e "${YELLOW}=== Running Component Verification Tests ===${NC}"

# Audio Pipeline verification
TEST_NAME="Audio Pipeline Verification"
TEST_CMD="python verification_script_audio_pipeline.py"
run_test "$TEST_NAME" "$TEST_CMD"
TEST_RESULTS+=($?)
TEST_NAMES+=("$TEST_NAME")

# STT Engine verification
TEST_NAME="STT Engine Verification"
TEST_CMD="python verification_script_stt_engine.py"
run_test "$TEST_NAME" "$TEST_CMD"
TEST_RESULTS+=($?)
TEST_NAMES+=("$TEST_NAME")

# Processing Core verification
TEST_NAME="Processing Core Verification"
TEST_CMD="python verification_script_processing_core.py"
run_test "$TEST_NAME" "$TEST_CMD"
TEST_RESULTS+=($?)
TEST_NAMES+=("$TEST_NAME")

# LLM Analysis verification
TEST_NAME="LLM Analysis Verification"
TEST_CMD="python verification_script_llm_analysis.py"
run_test "$TEST_NAME" "$TEST_CMD"
TEST_RESULTS+=($?)
TEST_NAMES+=("$TEST_NAME")

# Document Library verification
TEST_NAME="Document Library Verification"
TEST_CMD="python verification_script_document_library.py"
run_test "$TEST_NAME" "$TEST_CMD"
TEST_RESULTS+=($?)
TEST_NAMES+=("$TEST_NAME")

# Run the system integration test in mock mode
echo -e "${YELLOW}=== Running System Integration Tests ===${NC}"

# System Integration test (with mock components)
TEST_NAME="System Integration Test (Mock)"
TEST_CMD="python test_system_integration.py --mock"
run_test "$TEST_NAME" "$TEST_CMD"
TEST_RESULTS+=($?)
TEST_NAMES+=("$TEST_NAME")

# Run with real components (if available)
if [ ! -z "$RUN_REAL_COMPONENTS" ]; then
    # System Integration test (with real components)
    TEST_NAME="System Integration Test (Real)"
    TEST_CMD="python test_system_integration.py"
    run_test "$TEST_NAME" "$TEST_CMD"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("$TEST_NAME")
fi

# Run the Jetson hardware test if on Jetson device and flag is set
if [ ! -z "$RUN_JETSON_TESTS" ] && [ -f "/etc/nv_tegra_release" ]; then
    TEST_NAME="Jetson Hardware Integration"
    TEST_CMD="python test_jetson_integration.py"
    run_test "$TEST_NAME" "$TEST_CMD"
    TEST_RESULTS+=($?)
    TEST_NAMES+=("$TEST_NAME")
fi

# Print summary of all test results
echo -e "${YELLOW}=== Integration Test Summary ===${NC}"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for i in "${!TEST_RESULTS[@]}"; do
    if [ ${TEST_RESULTS[$i]} -eq 0 ]; then
        echo -e "${GREEN}✓ ${TEST_NAMES[$i]} PASSED${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}✗ ${TEST_NAMES[$i]} FAILED${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "Tests Passed: $PASS_COUNT"
echo "Tests Failed: $FAIL_COUNT"
echo "Total Tests:  ${#TEST_RESULTS[@]}"
echo ""

# Write summary to results file
echo "=== Integration Test Summary ===" >> "$RESULTS_FILE"
echo "Tests Passed: $PASS_COUNT" >> "$RESULTS_FILE"
echo "Tests Failed: $FAIL_COUNT" >> "$RESULTS_FILE"
echo "Total Tests:  ${#TEST_RESULTS[@]}" >> "$RESULTS_FILE"

echo -e "${YELLOW}Full test results saved to:${NC} $RESULTS_FILE"

# Exit with error if any tests failed
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Some tests failed. Please check the logs for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All integration tests passed successfully!${NC}"
    exit 0
fi