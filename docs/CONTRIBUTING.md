# Contributing to TCCC.ai

Thank you for your interest in contributing to TCCC.ai! This document provides guidelines and instructions for contributing to the project.

## Development Philosophy

TCCC.ai is designed with these core principles:

1. **Edge-First Development**: All components must run efficiently on Jetson devices
2. **Minimal Dependencies**: Only essential dependencies to maintain small footprint
3. **Performance Focus**: Optimize for low latency and resource efficiency
4. **Component Modularity**: Each module should have clear interfaces and be independently testable

## Getting Started

1. **Fork the Repository**
   
   Start by forking the repository on GitHub, then clone your fork:
   
   ```bash
   git clone https://github.com/YOUR-USERNAME/tccc-project.git
   cd tccc-project
   ```

2. **Set Up Development Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"  # Installs development dependencies
   ```

3. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- We follow PEP 8 style guidelines
- Use type annotations for all function definitions
- Maximum line length is 100 characters
- Use descriptive variable and function names

### Testing

- All new code should include appropriate tests
- Run tests before submitting PRs:
  ```bash
  pytest
  ```
- For performance-critical code, include benchmarks

### Documentation

- Update documentation to reflect your changes
- Document all public functions and classes
- Include docstrings in Google style format
- Update README.md if needed

### Commits

- Use clear, descriptive commit messages
- Reference issue numbers in commit messages when applicable
- Keep commits focused on single changes where possible

## Edge Deployment Considerations

When contributing to TCCC.ai, please keep these Jetson-specific considerations in mind:

- **Memory Efficiency**: Minimize memory footprint, especially for models
- **CPU/GPU Balance**: Design for appropriate CPU/GPU workload distribution
- **Power Awareness**: Consider power consumption impacts of your code
- **Model Optimization**: Use quantization-aware design patterns
- **Startup Time**: Optimize for fast module initialization

## Pull Request Process

1. Update the README.md with details of your changes if appropriate
2. Run all tests and ensure they pass
3. If you've added code that should be tested, add tests
4. Ensure your code follows style guidelines
5. Create a pull request with a clear description of the changes

## Code Review

All submissions require review. We use GitHub pull requests for this purpose:

1. Submit a pull request to the repository
2. Maintainers will review your changes
3. Address any requested changes
4. Once approved, your submission will be merged

## License

By contributing to TCCC.ai, you agree that your contributions will be licensed under the same license as the project (MIT License).