#!/usr/bin/env python3
"""
TCCC.ai Demo Launcher

Scans the current directory for Python scripts (demos), sorts them by
modification time (most recent first), and provides a TUI list
to easily select and run a demo.

When a demo is run, its complete terminal output is also captured to a
log file named `[demo_script_name]_YYYYMMDD_HHMMSS.log` in this directory.

-----------------------------------------------------------------------
Metadata:
    Created By:     Cascade (via Windsurf)
    Date Created:   2025-04-01
    Project:        TCCC.ai
    Relevant Task:  Managing and running demo scripts, capturing output for review
    Origin:         User requested a tool to easily list and run accumulating
                    demo scripts from a TUI, avoiding manual filename typing
                    and providing persistence/discoverability for demos.
                    Also requested a desktop shortcut for easy launching on Jetson.
                    User requested automatic logging of demo output to text files.
-----------------------------------------------------------------------

Dependencies:
  - inquirerpy (install via: pip install -r requirements.txt)

Usage:
  - Run this script from within the test_demos directory:
    python demo_launcher.py
  - Alternatively, use the generated desktop shortcut on the Jetson.
  - Use arrow keys to navigate, Enter to select and run.
  - Output logs will be saved in this directory.
"""

import os
import sys
import stat
import subprocess # Added for running scripts
from datetime import datetime # Added for log file timestamp
from typing import List, Dict, Optional

# Workaround for package capitalization issue on Linux:
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

def get_demo_scripts(directory: str = ".") -> List[Dict[str, any]]:
    """Scans the directory for Python files and returns their details."""
    scripts = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != os.path.basename(__file__):
                filepath = os.path.join(directory, filename)
                try:
                    mtime = os.path.getmtime(filepath)
                    scripts.append({"name": filename, "path": filepath, "mtime": mtime})
                except OSError:
                    print(f"Warning: Could not access {filename}, skipping.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        return []
    except PermissionError:
        print(f"Error: Permission denied for directory: {directory}", file=sys.stderr)
        return []

    # Sort by modification time, newest first
    scripts.sort(key=lambda x: x["mtime"], reverse=True)
    return scripts

def run_demo(script_path: str):
    """Runs the selected demo script using 'python' and logs output."""
    script_name = os.path.basename(script_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{os.path.splitext(script_name)[0]}_{timestamp}.log"
    # Use the 'script' command to capture output. -q (quiet), -c (command)
    # We assume 'python' is the correct interpreter based on desktop shortcut testing.
    # If issues arise, we might need to pass the full path to python.
    command = ['script', '-q', '-c', f"python '{script_path}'", log_filename]

    print(f"\n--- Running: {script_name} ---")
    print(f"--- Logging output to: {log_filename} ---")
    print("-" * (len(script_name) + 14))

    try:
        # Run the script command
        process = subprocess.run(command, check=False) # Don't check=True, let script handle errors

        if process.returncode != 0:
            print(f"\n--- {script_name} finished with exit code: {process.returncode} ---")
        else:
            print(f"\n--- {script_name} finished successfully ---")

    except FileNotFoundError:
        print(f"\nError: 'script' command not found. Cannot log output.", file=sys.stderr)
        print("Attempting to run demo without logging...")
        try:
            # Fallback: run directly if 'script' is missing
            subprocess.run(['python', script_path], check=False)
        except Exception as e:
            print(f"Error running script directly: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to run {script_name}: {e}", file=sys.stderr)

    # Keep terminal open briefly or wait for user
    input("\nPress Enter to return to the launcher menu...")

def main():
    """Main function to display menu and handle selection."""
    print("--- TCCC.ai Demo Launcher ---")
    while True: # Loop to allow running multiple demos
        print("\nScanning for demos...")
        demo_scripts = get_demo_scripts()

        if not demo_scripts:
            print("No demo scripts found in the current directory.")
            input("Press Enter to exit.")
            return

        choices = [
            Choice(value=script["path"], name=f"{script['name']} (Modified: {datetime.fromtimestamp(script['mtime']).strftime('%Y-%m-%d %H:%M')})")
            for script in demo_scripts
        ]
        choices.append(Separator())
        choices.append(Choice(value=None, name="Exit"))

        try:
            selected_script_path: Optional[str] = inquirer.select(
                message="Select a demo to run:",
                choices=choices,
                default=None,
                vi_mode=True,
                border=True,
                max_height="50%" # Adjust height as needed
            ).execute()

            if selected_script_path is None:
                print("Exiting launcher.")
                break # Exit the while loop
            else:
                # Clear screen slightly before running (optional)
                # os.system('cls' if os.name == 'nt' else 'clear')
                run_demo(selected_script_path)
                # After demo finishes, loop will restart and show menu again

        except KeyboardInterrupt:
            print("\nExiting launcher.")
            break
        except Exception as e:
            print(f"\nAn error occurred in the launcher: {e}", file=sys.stderr)
            input("Press Enter to try again...") # Allow user to see error


if __name__ == "__main__":
    main()
