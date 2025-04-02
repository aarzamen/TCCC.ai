#!/usr/bin/env python3
"""
TCCC Dashboard - A simple terminal dashboard for monitoring verification tests
"""

import os
import sys
import time
import subprocess
import re
import curses
from datetime import datetime
import signal

def get_verification_status():
    """Parse verification_results.txt if it exists"""
    results = {"passed": [], "failed": [], "running": []}
    if os.path.exists("verification_results.txt"):
        with open("verification_results.txt", "r") as f:
            content = f.read()
            passed = re.findall(r"Verification PASSED: ([^(]+)", content)
            failed = re.findall(r"Verification FAILED: ([^(]+)", content)
            results["passed"] = [p.strip() for p in passed]
            results["failed"] = [f.strip() for f in failed]
    return results

def get_running_verification():
    """Check if any verification scripts are running"""
    try:
        output = subprocess.check_output(
            "ps -ef | grep verification_script | grep -v grep", 
            shell=True, 
            stderr=subprocess.DEVNULL
        ).decode('utf-8')
        running = re.findall(r"verification_script_([^\s\.]+)", output)
        return running
    except subprocess.CalledProcessError:
        return []

def get_system_stats():
    """Get CPU, memory, and disk usage"""
    stats = {}
    
    # CPU usage
    try:
        cpu = subprocess.check_output("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'", shell=True).decode().strip()
        stats["cpu"] = f"{cpu}%"
    except:
        stats["cpu"] = "N/A"
    
    # Memory usage
    try:
        mem = subprocess.check_output("free -m | grep Mem | awk '{print $3/$2 * 100.0}'", shell=True).decode().strip()
        stats["mem"] = f"{float(mem):.1f}%"
    except:
        stats["mem"] = "N/A"
    
    # Disk usage
    try:
        disk = subprocess.check_output("df -h / | grep / | awk '{print $5}'", shell=True).decode().strip()
        stats["disk"] = disk
    except:
        stats["disk"] = "N/A"
        
    return stats

def get_recent_logs(lines=5):
    """Get recent log entries"""
    logs = []
    if os.path.exists("logs"):
        log_files = sorted([f for f in os.listdir("logs") if f.endswith(".log")], reverse=True)
        if log_files:
            try:
                output = subprocess.check_output(
                    f"tail -n {lines} logs/{log_files[0]}", 
                    shell=True
                ).decode('utf-8')
                logs = output.strip().split("\n")
            except:
                logs = ["Error reading logs"]
    return logs

def main(stdscr):
    # Setup the screen
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Passed
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Failed
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Running
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Headers
    
    # Handle window resizing
    def handle_resize(signum, frame):
        curses.endwin()
        curses.initscr()
    
    signal.signal(signal.SIGWINCH, handle_resize)
    
    while True:
        try:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Title
            title = "TCCC Verification Dashboard"
            stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
            stdscr.addstr(1, 0, "=" * width)
            
            # Time
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stdscr.addstr(0, width - len(time_str) - 1, time_str)
            
            # Get status information
            status = get_verification_status()
            running = get_running_verification()
            stats = get_system_stats()
            logs = get_recent_logs()
            
            # System stats
            stdscr.addstr(2, 2, "System Stats:", curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(3, 4, f"CPU: {stats['cpu']} | Memory: {stats['mem']} | Disk: {stats['disk']}")
            
            # Verification status
            row = 5
            stdscr.addstr(row, 2, "Verification Status:", curses.color_pair(4) | curses.A_BOLD)
            row += 1
            
            # Passed tests
            stdscr.addstr(row, 4, f"Passed: ", curses.A_BOLD)
            if status["passed"]:
                status_text = ", ".join(status["passed"][:3])
                if len(status["passed"]) > 3:
                    status_text += f" (+{len(status['passed']) - 3} more)"
                stdscr.addstr(status_text, curses.color_pair(1))
            else:
                stdscr.addstr("None", curses.color_pair(1))
            row += 1
            
            # Failed tests
            stdscr.addstr(row, 4, f"Failed: ", curses.A_BOLD)
            if status["failed"]:
                status_text = ", ".join(status["failed"][:3])
                if len(status["failed"]) > 3:
                    status_text += f" (+{len(status['failed']) - 3} more)"
                stdscr.addstr(status_text, curses.color_pair(2))
            else:
                stdscr.addstr("None", curses.color_pair(2))
            row += 1
            
            # Running tests
            stdscr.addstr(row, 4, f"Running: ", curses.A_BOLD)
            if running:
                stdscr.addstr(", ".join(running), curses.color_pair(3))
            else:
                stdscr.addstr("None", curses.color_pair(3))
            row += 2
            
            # Recent logs
            stdscr.addstr(row, 2, "Recent Logs:", curses.color_pair(4) | curses.A_BOLD)
            row += 1
            for log in logs:
                if row < height - 2:
                    # Truncate long log lines
                    if len(log) > width - 6:
                        log = log[:width - 9] + "..."
                    stdscr.addstr(row, 4, log)
                    row += 1
            
            # Footer
            if height > 15:
                stdscr.addstr(height-2, 0, "=" * width)
                footer = "Press 'q' to quit | 'r' to run all verifications"
                stdscr.addstr(height-1, (width - len(footer)) // 2, footer)
            
            stdscr.refresh()
            
            # Check for user input (with timeout)
            stdscr.timeout(1000)  # Update every second
            key = stdscr.getch()
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                subprocess.Popen(["./run_all_verifications.sh"], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            # In case of error, wait a bit and try again
            time.sleep(1)

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("Dashboard stopped.")