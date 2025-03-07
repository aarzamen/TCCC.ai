#!/bin/bash

# TCCC.ai Git Hooks Setup Script
# This script configures local git hooks for the project

echo "Setting up TCCC.ai Git hooks..."

# Configure git to use our custom hooks directory
git config core.hooksPath .githooks

# Ensure hooks are executable
chmod +x .githooks/pre-commit

echo "Git hooks installed successfully!"
echo "Pre-commit hooks will now run automatically on each commit."
echo "To bypass hooks temporarily, use: git commit --no-verify"