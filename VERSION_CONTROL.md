# TCCC.ai Version Control Guide

This document explains the version control standards and practices used in the TCCC.ai project.

## Version Control Standards

### File Tracking

- **All project code must be properly tracked in version control**
- Every source file, configuration file, and critical documentation must be committed
- No placeholder or incomplete code should be committed without explicit TODO markers
- Temporary or generated files should be excluded via `.gitignore`

### Using Git LFS

For larger binary files, we use Git Large File Storage (LFS):

```bash
# Install Git LFS (one-time)
git lfs install

# Track specific file patterns
git lfs track "*.model"
git lfs track "*.bin"
git lfs track "*.pt"
git lfs track "data/document_index/document_index.faiss"

# Add the .gitattributes to track these patterns
git add .gitattributes

# Commit and push as normal
git add your-large-file.model
git commit -m "feat(model): add quantized model for edge deployment"
git push
```

### Branch Strategy

We follow a structured branching strategy:

- `main` - The primary branch that must always be deployable
- `feature/*` - For new features (e.g., `feature/battlefield-audio`)
- `fix/*` - For bug fixes (e.g., `fix/audio-pipeline-crash`)
- `release/v*` - For release preparation (e.g., `release/v1.0.0`)

### Commit Message Format

We use semantic commit messages:

```
type(scope): description
```

Types include:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation updates
- `style`: Formatting changes
- `refactor`: Code restructuring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Routine tasks, dependency updates, etc.

Examples:
```
feat(audio-pipeline): implement battlefield noise reduction
fix(stt-engine): resolve crash on long audio inputs
docs(jetson): update deployment guide with latest Orin info
perf(llm-analysis): quantize Phi-2 model to 4-bit precision
```

## Development Workflow

1. **Start Fresh**
   ```bash
   git checkout main
   git pull
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Keep commits small and focused
   - Add tests for new functionality
   - Run unit tests before committing

4. **Validate Changes**
   ```bash
   # Validate version control status
   ./validate_version_control.py
   
   # Run linting and tests
   flake8 src/ tests/
   mypy src/
   pytest
   ```

5. **Create Pull Request**
   - Ensure CI workflow passes
   - Get at least one code review
   - Address feedback

## Version Control Validation

We enforce version control standards through:

1. **GitHub Workflows**
   - Automated checks on pull requests
   - Integration with code quality tools

2. **Local Validation**
   - Use `validate_version_control.py` script

3. **Pre-commit Hooks**
   - Available in development environment

## Common Issues and Resolution

### Untracked Files

**Problem**: Critical files not being tracked in version control.

**Solution**: Run `git status` to identify untracked files and add them with `git add`.

### Large Binary Files

**Problem**: Large files causing repository bloat.

**Solution**: Configure Git LFS for appropriate file types.

### Incomplete Features

**Problem**: Committing partial implementations.

**Solution**: Use feature flags or postpone commits until implementation is complete.