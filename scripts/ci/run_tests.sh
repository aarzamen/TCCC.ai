#!/bin/bash
# Run all tests for the TCCC.ai project

set -e

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run style checks
echo "Running style checks..."
black --check src tests
isort --check src tests
flake8 src tests

# Run type checks
echo "Running type checks..."
mypy src

# Run unit tests
echo "Running unit tests..."
pytest tests/unit/

# Run integration tests if requested
if [ "$1" == "--all" ]; then
    echo "Running integration tests..."
    pytest tests/integration/
fi

# Run coverage report
echo "Generating coverage report..."
pytest --cov=src --cov-report=term --cov-report=html tests/

echo "All tests completed successfully!"