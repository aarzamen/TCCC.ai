# Repository Migration Guide

This document provides guidance for migrating from the old repository structure to the new one.

## Overview of Changes

The TCCC.ai repository has been reorganized to follow modern Python project standards and to make it more suitable for public hosting. Key changes include:

- Organized scripts into logical directories
- Created a structured documentation system
- Consolidated tests into a standardized test suite
- Added proper GitHub configuration files
- Implemented modern Python project tooling

## Directory Structure Changes

| Old Location | New Location |
|--------------|--------------|
| Root directory scripts | `scripts/`, `examples/`, or `tools/` directories |
| Root directory tests | `tests/unit/` or `tests/integration/` |
| Documentation markdown files | `docs/` directory |
| Claude Code tools | `tools/cloud_code/` |
| Verification scripts | `tools/verification/` |

## How to Find Files

1. **Scripts**: Scripts have been categorized and moved to subdirectories:
   - Deployment scripts: `scripts/deployment/`
   - Development scripts: `scripts/development/`
   - Display configuration: `scripts/display/`
   - Hardware configuration: `scripts/hardware/`
   - Model downloads: `scripts/models/`

2. **Examples**: Example code has been organized by functionality:
   - Audio processing: `examples/audio_processing/`
   - Full system demos: `examples/full_system/`
   - Microphone tools: `examples/microphone/`
   - RAG examples: `examples/rag/`
   - STT examples: `examples/stt/`

3. **Documentation**: All documentation is now in the `docs/` directory:
   - Architecture docs: `docs/architecture/`
   - Development guides: `docs/development/`
   - User guides: `docs/guides/`
   - Module documentation: `docs/modules/`

4. **Tests**: Tests are now organized in the `tests/` directory:
   - Unit tests: `tests/unit/`
   - Integration tests: `tests/integration/`
   - Test resources: `tests/resources/`

## Common Commands

Here are the updated commands for common tasks:

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with test script
./scripts/ci/run_tests.sh
```

### Development Setup

```bash
# Set up development environment
./scripts/development/setup_workspace.sh

# Set up git hooks
./scripts/development/setup_git_hooks.sh
```

### Building Documentation

```bash
# Build documentation
cd docs
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Repository Setup for Contributors

If you're a new contributor, follow these steps:

1. Clone the repository
   ```bash
   git clone https://github.com/tccc-ai/tccc-project.git
   cd tccc-project
   ```

2. Set up your development environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

4. Run tests to make sure everything is working
   ```bash
   pytest
   ```

## Troubleshooting

If you're having trouble finding files or running commands in the new structure:

1. Use the repository search functionality to locate files by name or content
2. Check the `scripts/` directory for scripts that used to be in the root
3. Look for example code in the `examples/` directory
4. Check if your file has been renamed or split into multiple files