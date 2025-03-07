#!/usr/bin/env python3
"""
TCCC.ai Version Control Validator

This script validates that all critical project files are properly tracked in git
and reports any issues that need to be addressed before deployment.
"""

import os
import subprocess
import sys
from typing import List, Dict, Set, Tuple


def run_git_command(cmd: List[str]) -> str:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(
            ["git"] + cmd, 
            capture_output=True, 
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


def get_untracked_files() -> Set[str]:
    """Get all untracked files in the repository."""
    result = run_git_command(["ls-files", "--others", "--exclude-standard"])
    return set(result.split("\n")) if result else set()


def get_modified_files() -> Set[str]:
    """Get all modified files in the repository."""
    result = run_git_command(["ls-files", "--modified"])
    return set(result.split("\n")) if result else set()


def get_critical_directories() -> List[str]:
    """List of directories that must have all files tracked."""
    return [
        "src/tccc",
        "config",
        "tests",
        "scripts",
    ]


def get_allowed_untracked_patterns() -> List[str]:
    """Patterns for files that are allowed to be untracked."""
    return [
        "venv/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache/",
        "*.egg-info/",
        "data/backups/",
        "logs/",
    ]


def is_pattern_match(file_path: str, patterns: List[str]) -> bool:
    """Check if a file path matches any of the given patterns."""
    for pattern in patterns:
        # Simple pattern matching - could be enhanced with fnmatch
        if pattern.endswith('/') and file_path.startswith(pattern):
            return True
        if pattern.startswith('*') and file_path.endswith(pattern[1:]):
            return True
        if pattern == file_path:
            return True
    return False


def categorize_untracked(untracked: Set[str]) -> Dict[str, List[str]]:
    """Categorize untracked files by type."""
    categories = {
        "documentation": [],
        "code": [],
        "configuration": [],
        "tests": [],
        "scripts": [],
        "data": [],
        "other": [],
    }
    
    for file in untracked:
        if file.endswith(('.md', '.txt')):
            categories["documentation"].append(file)
        elif file.endswith(('.py')):
            if file.startswith(('test_', 'tests/')):
                categories["tests"].append(file)
            else:
                categories["code"].append(file)
        elif file.endswith(('.yaml', '.yml', '.json')):
            categories["configuration"].append(file)
        elif file.endswith(('.sh')):
            categories["scripts"].append(file)
        elif file.startswith(('data/')):
            categories["data"].append(file)
        else:
            categories["other"].append(file)
    
    return categories


def validate_critical_directories(untracked: Set[str]) -> List[str]:
    """Check if any files in critical directories are untracked."""
    violations = []
    critical_dirs = get_critical_directories()
    allowed_patterns = get_allowed_untracked_patterns()
    
    for file in untracked:
        for critical_dir in critical_dirs:
            if file.startswith(critical_dir):
                if not is_pattern_match(file, allowed_patterns):
                    violations.append(file)
                break
    
    return violations


def validate_deployment_readiness() -> Tuple[bool, List[str]]:
    """Validate if the project is ready for deployment from a VC perspective."""
    issues = []
    
    # Check for uncommitted changes
    if get_modified_files():
        issues.append("There are modified files that need to be committed")
    
    # Check for critical untracked files
    untracked = get_untracked_files()
    critical_untracked = validate_critical_directories(untracked)
    if critical_untracked:
        issues.append(f"Found {len(critical_untracked)} untracked files in critical directories")
    
    return len(issues) == 0, issues


def main():
    """Main function to run the validation."""
    print("TCCC.ai Version Control Validator")
    print("=================================\n")
    
    untracked = get_untracked_files()
    modified = get_modified_files()
    
    # Display statistics
    print(f"Total untracked files: {len(untracked)}")
    print(f"Total modified files: {len(modified)}\n")
    
    # Categorize untracked files
    if untracked:
        print("Untracked Files by Category:")
        categories = categorize_untracked(untracked)
        for category, files in categories.items():
            if files:
                print(f"\n{category.capitalize()} ({len(files)}):")
                for file in sorted(files):
                    print(f"  - {file}")
    
    # Check critical directories
    critical_untracked = validate_critical_directories(untracked)
    if critical_untracked:
        print("\nCRITICAL: The following files in core directories must be tracked:")
        for file in sorted(critical_untracked):
            print(f"  - {file}")
    
    # Deployment readiness check
    ready, issues = validate_deployment_readiness()
    print("\nDeployment Readiness:")
    if ready:
        print("✅ Project is ready for deployment from a version control perspective")
    else:
        print("❌ Project is NOT ready for deployment. Issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Final recommendations
    if not ready:
        print("\nRecommended Actions:")
        print("1. Review and commit all files in critical directories")
        print("2. Update .gitignore for any files that should be excluded")
        print("3. Use Git LFS for large files (>10MB)")
        print("4. Run the VERSION_CONTROL_CHECKLIST.md process")
    
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())