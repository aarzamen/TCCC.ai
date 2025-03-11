#!/bin/bash
# Verify all dependencies for the TCCC.ai project

set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Checking Python version..."
python --version

echo "Checking installed packages..."
pip list

echo "Verifying required packages..."
pip check

echo "Checking for security vulnerabilities..."
pip-audit

echo "All dependency checks completed!"