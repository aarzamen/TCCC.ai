import os
import pathspec
import argparse
from pathlib import Path
import platform
import subprocess

# --- Configuration ---

# Define common text file extensions (add more if needed for your project)
TEXT_EXTENSIONS = {
    '.py', '.yaml', '.yml', '.sh', '.md', '.txt', '.json', '.xml', '.html', '.css', '.js',
    '.gitignore', '.dockerignore', 'Dockerfile', '.cfg', '.ini', '.toml', '.env_example'
}

# Define extensions for files often large or binary, to skip by default.
BINARY_LIKE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico',
    '.mp3', '.wav', '.ogg', '.flac', # Audio files
    '.mp4', '.avi', '.mov', '.mkv', # Video files
    '.pdf',
    '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z',
    '.so', '.dll', '.exe', '.dylib',
    '.o', '.a', '.obj',
    '.class', '.jar',
    '.db', '.sqlite', '.sqlite3',
    '.pkl', '.pickle', '.joblib',
    '.bin', '.dat',
    '.pth', '.pt', '.onnx', '.gguf', '.safetensors', # Model files
    '.npy', '.npz',
    '.ipynb',
    '.DS_Store',
    '.log'
}

# Define specific directory names to always ignore
ALWAYS_IGNORE_DIRS = {
    '.git', '__pycache__', '.vscode', '.idea', 'build', 'dist', '.eggs',
    '*.egg-info', 'venv', '.venv', 'env', '.env', 'node_modules',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', '.hypothesis',
    'data', 'logs', # Often large/irrelevant for code context
    # 'models', # Usually handled by extension or gitignore
}

# --- Functions ---

def read_gitignore(gitignore_path):
    patterns = []
    if gitignore_path.is_file():
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = f.read().splitlines()
        except Exception as e:
            print(f"Warning: Could not read {gitignore_path}: {e}")
    return patterns

def get_system_info():
    """Gathers basic system information."""
    info = {}
    info['platform'] = platform.platform()
    info['python_version'] = platform.python_version()
    try:
        # Use subprocess for potentially more detailed Linux info
        lsb_info = subprocess.run(['lsb_release', '-a'], capture_output=True, text=True, check=False, encoding='utf-8').stdout
        info['lsb_release'] = lsb_info.strip() if lsb_info else "lsb_release not found"
    except FileNotFoundError:
        info['lsb_release'] = "lsb_release command not found."
    except Exception as e:
         info['lsb_release'] = f"Error running lsb_release: {e}"
    try:
        uname_info = subprocess.run(['uname', '-a'], capture_output=True, text=True, check=False, encoding='utf-8').stdout
        info['uname'] = uname_info.strip() if uname_info else "uname failed"
    except FileNotFoundError:
        info['uname'] = "uname command not found."
    except Exception as e:
         info['uname'] = f"Error running uname: {e}"

    # Check for NVIDIA tools (basic check)
    try:
        jetson_release_info = subprocess.run(['dpkg-query', '-W', 'nvidia-jetpack'], capture_output=True, text=True, check=False, encoding='utf-8').stdout
        info['jetpack_version'] = jetson_release_info.strip() if jetson_release_info else "nvidia-jetpack package not found"
    except FileNotFoundError:
         info['jetpack_version'] = "dpkg-query command not found."
    except Exception as e:
         info['jetpack_version'] = f"Error checking jetpack version: {e}"


    return info


def create_code_snapshot(project_dir, output_dir, max_size_kb=1024):
    project_path = Path(project_dir).resolve()
    output_path = Path(output_dir).resolve()
    max_size_bytes = max_size_kb * 1024

    if not project_path.is_dir():
        print(f"Error: Project directory '{project_path}' not found.")
        return None # Return None on error

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating snapshot of '{project_path}' in '{output_path}'...")

    # --- Load .gitignore ---
    gitignore_path = project_path / '.gitignore'
    gitignore_patterns = read_gitignore(gitignore_path)
    always_ignore_patterns = [f"{d}/" for d in ALWAYS_IGNORE_DIRS if '*' not in d and '.' not in d[-4:]] + \
                             [d for d in ALWAYS_IGNORE_DIRS if '*' in d or '.' in d[-4:]]
    all_patterns = gitignore_patterns + list(always_ignore_patterns)
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, all_patterns)
    print(f"Loaded {len(gitignore_patterns)} patterns from .gitignore.")
    print(f"Total ignore patterns (including always ignored): {len(all_patterns)}")

    # --- Walk through the project directory ---
    included_files = 0
    skipped_gitignore = 0
    skipped_binary = 0
    skipped_size = 0
    skipped_decode = 0
    ignored_dirs_count = 0
    total_size_included_bytes = 0

    for root, dirs, files in os.walk(project_path, topdown=True):
        current_path = Path(root)
        relative_root = current_path.relative_to(project_path)

        # Prune ignored directories
        original_dirs = list(dirs)
        dirs[:] = []
        for d in original_dirs:
            dir_rel_path = relative_root / d
            dir_rel_path_str_for_match = str(dir_rel_path).replace(os.sep, '/')
            if not spec.match_file(dir_rel_path_str_for_match) and \
               not spec.match_file(dir_rel_path_str_for_match + '/'):
                dirs.append(d)
            else:
                ignored_dirs_count += 1

        # Process files
        for filename in files:
            file_path = current_path / filename
            relative_file_path = file_path.relative_to(project_path)
            relative_file_path_str_for_match = str(relative_file_path).replace(os.sep, '/')

            # Checks (gitignore, extension, size)
            if spec.match_file(relative_file_path_str_for_match):
                skipped_gitignore += 1; continue
            file_ext = file_path.suffix.lower()
            if file_ext in BINARY_LIKE_EXTENSIONS:
                 skipped_binary += 1; continue
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size_bytes:
                    skipped_size += 1; continue
                if file_size == 0 and file_ext != '.py':
                    continue
            except OSError as e:
                print(f"Warning: Could not stat file {relative_file_path}: {e}"); continue

            # Construct output path
            output_file_dir = output_path / relative_root
            output_file_path = output_file_dir / f"{filename}.txt"
            output_file_dir.mkdir(parents=True, exist_ok=True)

            # Read/Write content
            try:
                with open(file_path, 'rb') as infile_b:
                    raw_content = infile_b.read()
                content = raw_content.decode('utf-8', errors='ignore')
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(content)
                included_files += 1
                total_size_included_bytes += file_size
            except Exception as e:
                print(f"Warning: Error processing file {relative_file_path}: {e}")
                skipped_decode += 1

    # --- Reporting ---
    report = {}
    report["status"] = "Success"
    report["included_files"] = included_files
    report["total_size_bytes"] = total_size_included_bytes
    report["total_size_kb"] = total_size_included_bytes / 1024
    report["total_size_mb"] = report["total_size_kb"] / 1024
    report["ignored_dirs_count"] = ignored_dirs_count
    report["skipped_gitignore"] = skipped_gitignore
    report["skipped_binary"] = skipped_binary
    report["skipped_size"] = skipped_size
    report["skipped_decode"] = skipped_decode
    report["output_path"] = str(output_path)
    report["max_size_kb_limit"] = max_size_kb

    print("-" * 30)
    print("Snapshot creation complete.")
    print(f"Included files written: {report['included_files']}")
    print(f"Total size included: {report['total_size_kb']:.2f} KB ({report['total_size_mb']:.2f} MB)")
    print(f"Ignored directories (pruned): {report['ignored_dirs_count']}")
    print(f"Skipped files (gitignore/rules): {report['skipped_gitignore']}")
    print(f"Skipped files (binary-like ext): {report['skipped_binary']}")
    print(f"Skipped files (too large >{max_size_kb}KB): {report['skipped_size']}")
    print(f"Skipped files (read/decode error): {report['skipped_decode']}")
    print(f"Snapshot written to: {report['output_path']}")

    return report # Return summary dictionary

# --- Main Execution ---

if __name__ == "__main__":
    from datetime import datetime
    # Determine project directory assuming the script is run from the project root or within tools/
    script_path = Path(__file__).resolve()
    # Go up one level if script is in tools/, otherwise assume current dir is project root
    if script_path.parent.name == 'tools':
        default_project_dir = str(script_path.parent.parent) # tccc-project/
    else:
        default_project_dir = str(Path.cwd()) # Assume run from project root

    # Define the base directory for snapshots - one level above project dir (e.g., home dir)
    snapshots_base_dir = str(Path(default_project_dir).parent / "tccc_snapshots")

    # Create a dated snapshot folder name
    current_date = datetime.now().strftime("%Y%m%d")
    # Create the specific output directory path within the base snapshots dir
    default_output_dir = os.path.join(snapshots_base_dir, "snapshot_" + current_date)

    parser = argparse.ArgumentParser(description="Create a text snapshot of a project, respecting .gitignore.")
    # Update help texts to reflect new defaults
    parser.add_argument("--project-dir", default=default_project_dir, help=f"Path to the project directory to snapshot (default: {default_project_dir})")
    parser.add_argument("--output-dir", default=default_output_dir, help=f"Path to the output directory for the snapshot (default: {default_output_dir})")
    parser.add_argument("--max-size", type=int, default=1024, help="Maximum individual file size in KB to include (default: 1024)")
    parser.add_argument("--add-sysinfo", action='store_true', help="Add a system_info.txt file to the snapshot root.")


    args = parser.parse_args()

    snapshot_report = create_code_snapshot(args.project_dir, args.output_dir, args.max_size)

    if snapshot_report and args.add_sysinfo:
        print("Gathering system info...")
        sys_info = get_system_info()
        sys_info_path = Path(args.output_dir) / "system_info.txt"
        try:
            with open(sys_info_path, 'w', encoding='utf-8') as f:
                for key, value in sys_info.items():
                    f.write(f"{key}: {value}\n")
            print(f"System info saved to: {sys_info_path}")
        except Exception as e:
            print(f"Warning: Failed to save system info: {e}")