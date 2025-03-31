#\!/usr/bin/env python3
"""
Real-time status monitor for the TCCC audio enhancement pipeline.
Displays metrics, stats, and performance indicators for the audio enhancers.
"""

import os
import sys
import time
import json
import argparse
import curses
import threading
import queue
import socket
import signal

class AudioStatusMonitor:
    """Real-time status monitor for audio enhancement pipeline."""
    
    def __init__(self, update_interval=0.5):
        """Initialize status monitor."""
        self.update_interval = update_interval
        self.status_data = {}
        self.running = False
        self.stats_queue = queue.Queue()
        
        # Set up status server
        self.server_socket = None
        self.monitor_thread = None
    
    def start_socket_server(self, port=9876):
        """Start socket server to receive status updates."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind(('localhost', port))
            self.server_socket.settimeout(0.5)
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitor_socket,
                daemon=True
            )
            self.monitor_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting status server: {e}")
            return False
    
    def _monitor_socket(self):
        """Monitor socket for status updates."""
        while self.running:
            try:
                data, addr = self.server_socket.recvfrom(4096)
                try:
                    status = json.loads(data.decode('utf-8'))
                    self.stats_queue.put(status)
                except json.JSONDecodeError:
                    pass
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Error in socket monitor: {e}")
                time.sleep(1)
    
    def start_ui(self, stdscr):
        """Start the curses UI."""
        # Initialize curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
        curses.init_pair(6, curses.COLOR_BLUE, -1)
        
        # Colors
        GREEN = curses.color_pair(1)
        YELLOW = curses.color_pair(2)
        RED = curses.color_pair(3)
        CYAN = curses.color_pair(4)
        MAGENTA = curses.color_pair(5)
        BLUE = curses.color_pair(6)
        
        # Start status server
        self.running = True
        self.start_socket_server()
        
        # Main UI loop
        try:
            while self.running:
                # Check for new stats
                try:
                    while not self.stats_queue.empty():
                        status = self.stats_queue.get_nowait()
                        self.status_data.update(status)
                except queue.Empty:
                    pass
                
                # Clear screen
                stdscr.clear()
                
                # Get window dimensions
                height, width = stdscr.getmaxyx()
                
                # Draw header
                header = "TCCC Audio Enhancement Status Monitor"
                stdscr.addstr(0, (width - len(header)) // 2, header, CYAN | curses.A_BOLD)
                stdscr.addstr(1, 0, "=" * width, CYAN)
                
                # Draw status information
                row = 3
                
                # Audio pipeline status
                stdscr.addstr(row, 2, "Audio Pipeline Status:", YELLOW | curses.A_BOLD)
                row += 1
                
                # Enhancement mode
                mode = self.status_data.get('enhancement_mode', 'unknown')
                stdscr.addstr(row, 4, f"Enhancement mode: ", BLUE)
                stdscr.addstr(f"{mode}", GREEN if mode \!= 'none' else YELLOW)
                row += 1
                
                # Audio levels
                current_level = self.status_data.get('current_level', 0.0) * 100
                peak_level = self.status_data.get('peak_level', 0.0) * 100
                stdscr.addstr(row, 4, f"Current level: ", BLUE)
                level_color = GREEN if current_level < 70 else (YELLOW if current_level < 90 else RED)
                stdscr.addstr(f"{current_level:.1f}%", level_color)
                stdscr.addstr(f" (Peak: {peak_level:.1f}%)", YELLOW)
                row += 1
                
                # Speech detection
                is_speech = self.status_data.get('is_speech', False)
                stdscr.addstr(row, 4, f"Speech detected: ", BLUE)
                stdscr.addstr(f"{'YES' if is_speech else 'NO'}", GREEN if is_speech else YELLOW)
                row += 1
                
                # Processing stats
                row += 1
                stdscr.addstr(row, 2, "Processing Statistics:", YELLOW | curses.A_BOLD)
                row += 1
                
                # FullSubNet stats
                fullsubnet_active = self.status_data.get('fullsubnet_active', False)
                if fullsubnet_active:
                    stdscr.addstr(row, 4, "FullSubNet Enhancer:", GREEN | curses.A_BOLD)
                    row += 1
                    
                    proc_time = self.status_data.get('fullsubnet_processing_time', 0.0)
                    stdscr.addstr(row, 6, f"Processing time: ", BLUE)
                    time_color = GREEN if proc_time < 30 else (YELLOW if proc_time < 60 else RED)
                    stdscr.addstr(f"{proc_time:.2f} ms/chunk", time_color)
                    row += 1
                    
                    snr_improve = self.status_data.get('fullsubnet_snr_improvement', 0.0)
                    stdscr.addstr(row, 6, f"SNR improvement: ", BLUE)
                    stdscr.addstr(f"{snr_improve:.1f} dB", GREEN)
                    row += 1
                    
                    gpu_status = "ENABLED" if self.status_data.get('fullsubnet_gpu', False) else "DISABLED"
                    stdscr.addstr(row, 6, f"GPU acceleration: ", BLUE)
                    stdscr.addstr(gpu_status, GREEN if gpu_status == "ENABLED" else YELLOW)
                    row += 1
                    
                    row += 1
                
                # Battlefield stats
                battlefield_active = self.status_data.get('battlefield_active', False)
                if battlefield_active:
                    stdscr.addstr(row, 4, "Battlefield Enhancer:", GREEN | curses.A_BOLD)
                    row += 1
                    
                    proc_time = self.status_data.get('battlefield_processing_time', 0.0)
                    stdscr.addstr(row, 6, f"Processing time: ", BLUE)
                    time_color = GREEN if proc_time < 20 else (YELLOW if proc_time < 40 else RED)
                    stdscr.addstr(f"{proc_time:.2f} ms/chunk", time_color)
                    row += 1
                    
                    env_type = self.status_data.get('battlefield_environment', 'unknown').upper()
                    stdscr.addstr(row, 6, f"Environment: ", BLUE)
                    stdscr.addstr(env_type, GREEN)
                    row += 1
                    
                    distance = self.status_data.get('battlefield_distance', 1.0)
                    stdscr.addstr(row, 6, f"Distance factor: ", BLUE)
                    stdscr.addstr(f"{distance:.1f}x", GREEN if distance < 2.0 else YELLOW)
                    row += 1
                    
                    row += 1
                
                # System load
                stdscr.addstr(row, 2, "System Resources:", YELLOW | curses.A_BOLD)
                row += 1
                
                cpu_load = self.status_data.get('cpu_load', 0.0)
                stdscr.addstr(row, 4, f"CPU usage: ", BLUE)
                cpu_color = GREEN if cpu_load < 50 else (YELLOW if cpu_load < 80 else RED)
                stdscr.addstr(f"{cpu_load:.1f}%", cpu_color)
                row += 1
                
                gpu_load = self.status_data.get('gpu_load', 0.0)
                if gpu_load > 0:
                    stdscr.addstr(row, 4, f"GPU usage: ", BLUE)
                    gpu_color = GREEN if gpu_load < 50 else (YELLOW if gpu_load < 80 else RED)
                    stdscr.addstr(f"{gpu_load:.1f}%", gpu_color)
                    row += 1
                    
                    gpu_mem = self.status_data.get('gpu_memory', 0.0)
                    stdscr.addstr(row, 4, f"GPU memory: ", BLUE)
                    mem_color = GREEN if gpu_mem < 50 else (YELLOW if gpu_mem < 80 else RED)
                    stdscr.addstr(f"{gpu_mem:.1f}%", mem_color)
                    row += 1
                
                # Runtime stats
                uptime = self.status_data.get('uptime', 0.0)
                if uptime > 0:
                    row += 1
                    stdscr.addstr(row, 2, f"Running time: ", BLUE)
                    minutes, seconds = divmod(int(uptime), 60)
                    hours, minutes = divmod(minutes, 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    stdscr.addstr(time_str, GREEN)
                
                # Footer with instructions
                footer = "Press 'q' to quit, 'r' to reset stats"
                stdscr.addstr(height - 2, (width - len(footer)) // 2, footer, CYAN)
                stdscr.addstr(height - 1, 0, "=" * width, CYAN)
                
                # Refresh screen
                stdscr.refresh()
                
                # Check for key input
                stdscr.timeout(int(self.update_interval * 1000))
                key = stdscr.getch()
                
                # Handle key input
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('r'):
                    self.status_data = {}
                
        except KeyboardInterrupt:
            self.running = False
        
        # Clean up
        if self.server_socket:
            self.server_socket.close()
    
    def start(self):
        """Start the status monitor."""
        # Initialize curses
        curses.wrapper(self.start_ui)
    
    def stop(self):
        """Stop the status monitor."""
        self.running = False
        
        # Close socket server
        if self.server_socket:
            self.server_socket.close()
        
        # Wait for monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

def send_status_update(status_data, port=9876):
    """Send status update to monitor."""
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Convert status data to JSON
        data = json.dumps(status_data).encode('utf-8')
        
        # Send data
        sock.sendto(data, ('localhost', port))
        
        # Close socket
        sock.close()
        
        return True
    except Exception as e:
        print(f"Error sending status update: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Audio Enhancement Status Monitor")
    parser.add_argument("--update-interval", type=float, default=0.5,
                      help="Update interval in seconds")
    parser.add_argument("--simulate", action="store_true",
                      help="Simulate status updates for testing")
    args = parser.parse_args()
    
    # Initialize status monitor
    monitor = AudioStatusMonitor(update_interval=args.update_interval)
    
    # Set up signal handler
    def signal_handler(sig, frame):
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start simulation if requested
    if args.simulate:
        def simulate_updates():
            """Simulate status updates."""
            import random
            import psutil
            
            start_time = time.time()
            
            while monitor.running:
                # Generate random status data
                status = {
                    'enhancement_mode': random.choice(['auto', 'fullsubnet', 'battlefield', 'both', 'none']),
                    'current_level': random.random() * 0.8,
                    'peak_level': random.random() * 0.9,
                    'is_speech': random.choice([True, False, False, True, True]),
                    'uptime': time.time() - start_time,
                    'cpu_load': psutil.cpu_percent(),
                }
                
                # Add FullSubNet stats
                if random.random() > 0.3:
                    status.update({
                        'fullsubnet_active': True,
                        'fullsubnet_processing_time': random.uniform(10, 60),
                        'fullsubnet_snr_improvement': random.uniform(3, 12),
                        'fullsubnet_gpu': random.choice([True, True, False]),
                    })
                
                # Add Battlefield stats
                if random.random() > 0.3:
                    status.update({
                        'battlefield_active': True,
                        'battlefield_processing_time': random.uniform(5, 35),
                        'battlefield_environment': random.choice(['indoor', 'outdoor', 'mixed']),
                        'battlefield_distance': random.uniform(1.0, 3.0),
                    })
                
                # Add GPU stats
                if random.random() > 0.5:
                    status.update({
                        'gpu_load': random.uniform(10, 90),
                        'gpu_memory': random.uniform(20, 70),
                    })
                
                # Send update
                send_status_update(status)
                
                # Wait for next update
                time.sleep(args.update_interval)
        
        # Start simulation thread
        simulation_thread = threading.Thread(target=simulate_updates, daemon=True)
        simulation_thread.start()
    
    # Start monitor
    try:
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()

if __name__ == "__main__":
    main()
