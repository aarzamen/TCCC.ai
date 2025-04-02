#!/usr/bin/env python3
"""
Test script for the Battlefield Audio Enhancer.

This script provides a test harness for the enhanced audio pipeline 
with a visible display on the Jetson's monitor to show audio levels
and speech detection in real-time.
"""

import os
import sys
import time
import numpy as np
import threading
import argparse
import curses
import subprocess
import atexit
import logging
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import battlefield enhancer
from battlefield_audio_enhancer import BattlefieldAudioEnhancer, MicrophoneProcessor
from tccc.utils.logging import get_logger
from tccc.utils.config_manager import ConfigManager

# Configure logging
logger = get_logger("battlefield_enhancer_test")

class VisualMonitor:
    """
    Visual audio monitor for displaying audio levels and processing stats
    in a visible terminal window on the Jetson's monitor.
    """
    
    def __init__(self, processor, visible_on_display=True):
        """
        Initialize the visual monitor.
        
        Args:
            processor: MicrophoneProcessor instance
            visible_on_display: Whether to launch visible window on display
        """
        self.processor = processor
        self.visible_on_display = visible_on_display
        self.running = False
        self.window = None
        self.thread = None
        
        # Display buffer
        self.level_history = [0.0] * 100  # Store recent levels for display
        self.speech_history = [False] * 100  # Store speech detection
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def launch_visible_window(self):
        """
        Launch a visible terminal window on the Jetson's display.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.visible_on_display:
            return False
        
        try:
            # Find the X display
            if not os.environ.get('DISPLAY'):
                # Check common X socket location
                if os.path.exists('/tmp/.X11-unix/X0'):
                    os.environ['DISPLAY'] = ':0'
                    logger.info("Set DISPLAY to :0 based on X socket detection")
                else:
                    logger.warning("No X display available")
                    return False
            
            # Build command to re-launch this script in a visible terminal
            script_path = os.path.abspath(__file__)
            device_arg = f"--device {self.processor.device_id}" if hasattr(self.processor, 'device_id') else ""
            terminal_cmd = f"DISPLAY={os.environ.get('DISPLAY', ':0')} xterm -T 'Battlefield Audio Monitor' -geometry 100x40+0+0 -fa 'Monospace' -fs 14 -bg black -fg green -e python3 {script_path} --terminal {device_arg}"
            
            logger.info(f"Launching terminal with: {terminal_cmd}")
            subprocess.Popen(terminal_cmd, shell=True)
            
            # Allow time for terminal to launch
            time.sleep(1)
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch terminal: {e}")
            return False
    
    def start(self):
        """Start the visual monitor."""
        if self.running:
            return
        
        self.running = True
        
        # If in terminal mode, initialize curses
        if '--terminal' in sys.argv:
            self.thread = threading.Thread(target=self._curses_loop, daemon=True)
            self.thread.start()
        else:
            # Launch visible window on display
            if self.launch_visible_window():
                logger.info("Terminal window launched on Jetson's display")
                return
            
            # Fallback to simple terminal output
            self.thread = threading.Thread(target=self._simple_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the visual monitor."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
    
    def cleanup(self):
        """Clean up resources."""
        if self.window:
            try:
                curses.endwin()
            except:
                pass
            self.window = None
        
        self.running = False
    
    def _curses_loop(self):
        """Main curses display loop for graphical terminal output."""
        try:
            # Initialize curses
            self.window = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.curs_set(0)  # Hide cursor
            self.window.timeout(100)  # Non-blocking input
            
            # Define color pairs
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Normal levels
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Mid levels
            curses.init_pair(3, curses.COLOR_RED, -1)    # High levels
            curses.init_pair(4, curses.COLOR_CYAN, -1)   # Info
            curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_GREEN)  # Speech
            curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Headers
            
            # Main display loop
            while self.running:
                try:
                    # Handle input
                    key = self.window.getch()
                    if key == ord('q'):
                        break
                    
                    # Get current stats
                    stats = self.processor.get_stats()
                    
                    # Get current audio data
                    audio_data = self.processor.get_audio()
                    if len(audio_data) > 0:
                        # Calculate level (RMS)
                        level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2 / 32767.0 ** 2))
                        
                        # Update history
                        self.level_history.append(level)
                        self.level_history = self.level_history[-100:]
                        
                        self.speech_history.append(stats.get('speech_detected', False))
                        self.speech_history = self.speech_history[-100:]
                    
                    # Clear screen
                    self.window.clear()
                    
                    # Draw title
                    self.window.addstr(0, 0, " TCCC BATTLEFIELD AUDIO ENHANCER ", curses.color_pair(6) | curses.A_BOLD)
                    
                    # Draw status
                    running_time = time.time() - stats.get('start_time', time.time())
                    self.window.addstr(2, 0, f"Runtime: {running_time:.1f}s   Chunks: {stats.get('chunks_processed', 0)}", curses.color_pair(4))
                    self.window.addstr(3, 0, f"Speech chunks: {stats.get('speech_chunks', 0)} ({stats.get('speech_chunks', 0)/max(1, stats.get('chunks_processed', 1))*100:.1f}%)", curses.color_pair(4))
                    
                    # Draw environment info
                    env_type = stats.get('adaptive_settings', {}).get('environment_type', 'unknown')
                    distance = stats.get('adaptive_settings', {}).get('distance_factor', 1.0)
                    self.window.addstr(5, 0, f"Environment: {env_type.upper()}   Distance factor: {distance:.1f}x", curses.color_pair(4))
                    
                    # Draw speech status
                    if stats.get('speech_detected', False):
                        self.window.addstr(7, 0, " SPEECH DETECTED ", curses.color_pair(5) | curses.A_BOLD)
                    else:
                        self.window.addstr(7, 0, " SILENCE ", curses.color_pair(4))
                    
                    # Draw level meter (current)
                    current_level = self.level_history[-1] if self.level_history else 0
                    level_db = 20 * np.log10(current_level + 1e-6)
                    self.window.addstr(9, 0, f"Level: {level_db:.1f} dB", curses.color_pair(4))
                    
                    # Draw bar meter
                    meter_width = 50
                    level_scaled = min(1.0, current_level * 10)  # Scale for better visibility
                    meter_fill = int(level_scaled * meter_width)
                    
                    # Choose color based on level
                    if level_scaled < 0.3:
                        color = curses.color_pair(1)  # Green
                    elif level_scaled < 0.7:
                        color = curses.color_pair(2)  # Yellow
                    else:
                        color = curses.color_pair(3)  # Red
                    
                    meter_str = "█" * meter_fill + "░" * (meter_width - meter_fill)
                    self.window.addstr(10, 0, f"|{meter_str}|", color)
                    
                    # Draw level history graph
                    self.window.addstr(12, 0, "Audio Level History:", curses.color_pair(4))
                    
                    graph_height = 10
                    graph_width = min(80, curses.COLS - 2)
                    
                    for i in range(min(len(self.level_history), graph_width)):
                        level = self.level_history[-(i+1)]
                        level_scaled = min(1.0, level * 10)  # Scale for better visibility
                        bar_height = int(level_scaled * graph_height)
                        
                        # Choose color based on level
                        if level_scaled < 0.3:
                            color = curses.color_pair(1)  # Green
                        elif level_scaled < 0.7:
                            color = curses.color_pair(2)  # Yellow
                        else:
                            color = curses.color_pair(3)  # Red
                        
                        # Add speech indicator
                        is_speech = self.speech_history[-(i+1)]
                        
                        # Draw from bottom up
                        x_pos = graph_width - i
                        for j in range(bar_height):
                            y_pos = 13 + graph_height - j
                            self.window.addstr(y_pos, x_pos, "█", color)
                        
                        # Draw speech indicator at bottom
                        if is_speech:
                            self.window.addstr(13 + graph_height + 1, x_pos, "▲", curses.color_pair(5))
                        else:
                            self.window.addstr(13 + graph_height + 1, x_pos, "·", curses.color_pair(4))
                    
                    # Draw performance stats
                    proc_time = stats.get('average_processing_time_ms', 0)
                    y_pos = 13 + graph_height + 3
                    self.window.addstr(y_pos, 0, f"Processing time: {proc_time:.2f}ms per chunk", curses.color_pair(4))
                    
                    # Draw audio quality stats
                    est_snr = stats.get('estimated_snr_db', 0)
                    self.window.addstr(y_pos + 1, 0, f"Estimated SNR: {est_snr:.1f} dB", curses.color_pair(4))
                    
                    # Draw noise floor and threshold
                    noise_floor = stats.get('adaptive_settings', {}).get('noise_floor', 0) * 100
                    speech_threshold = stats.get('adaptive_settings', {}).get('speech_threshold', 0) * 100
                    self.window.addstr(y_pos + 2, 0, 
                                     f"Adaptive thresholds: Noise floor: {noise_floor:.2f}%  Speech: {speech_threshold:.2f}%", 
                                     curses.color_pair(4))
                    
                    # Draw controls
                    self.window.addstr(curses.LINES-2, 0, "Press 'q' to quit", curses.color_pair(4))
                    
                    # Refresh display
                    self.window.refresh()
                    
                    # Sleep briefly
                    time.sleep(0.05)
                    
                except Exception as e:
                    logger.error(f"Error in curses loop: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Curses error: {e}")
        finally:
            # Restore terminal
            if self.window:
                curses.endwin()
                self.window = None
    
    def _simple_loop(self):
        """Simple text-based display loop for SSH sessions."""
        try:
            # Main display loop
            while self.running:
                try:
                    # Get current stats
                    stats = self.processor.get_stats()
                    
                    # Get current audio data
                    audio_data = self.processor.get_audio()
                    if len(audio_data) > 0:
                        # Calculate level (RMS)
                        level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2 / 32767.0 ** 2))
                        
                        # Update history
                        self.level_history.append(level)
                        self.level_history = self.level_history[-20:]  # Keep last 20 samples
                        
                        self.speech_history.append(stats.get('speech_detected', False))
                        self.speech_history = self.speech_history[-20:]
                    
                    # Calculate level in dB
                    current_level = self.level_history[-1] if self.level_history else 0
                    level_db = 20 * np.log10(current_level + 1e-6)
                    
                    # Draw meter
                    meter_width = 50
                    level_scaled = min(1.0, current_level * 10)  # Scale for better visibility
                    meter = '█' * int(level_scaled * meter_width) + '░' * (meter_width - int(level_scaled * meter_width))
                    
                    # Format with colors based on speech detection
                    if stats.get('speech_detected', False):
                        status = "\033[1;92mSPEAKING\033[0m"  # Green
                    else:
                        status = "\033[1;90mSILENCE\033[0m"   # Gray
                    
                    # Get environment info
                    env = stats.get('adaptive_settings', {}).get('environment_type', 'unknown')
                    dist = stats.get('adaptive_settings', {}).get('distance_factor', 1.0)
                    
                    # Clear line and display status
                    sys.stdout.write(f"\r\033[K|{meter}| {level_db:.1f} dB - {status} - Env: {env} - Dist: {dist:.1f}x")
                    sys.stdout.flush()
                    
                    # Sleep briefly
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in simple display loop: {e}")
                    time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Display error: {e}")

def run_test(args):
    """
    Run the battlefield audio enhancer test.
    
    Args:
        args: Command line arguments
    """
    try:
        # Load configuration
        config = None
        if args.config:
            config_manager = ConfigManager()
            config = config_manager.load_config_from_file(args.config)
        
        # Create microphone processor
        processor = MicrophoneProcessor(config)
        
        # Set device ID if specified
        if args.device is not None:
            processor.device_id = args.device
        
        # Determine output file if saving
        output_file = args.output
        if args.save and not output_file:
            output_file = f"battlefield_enhanced_{int(time.time())}.wav"
        
        # Create visual monitor
        monitor = VisualMonitor(processor, visible_on_display=not args.terminal)
        
        print("\nStarting battlefield audio enhancer test...")
        print(f"Recording from device {processor.device_id} for {args.duration} seconds")
        if args.save:
            print(f"Saving processed audio to: {output_file}")
        
        # Start monitoring
        monitor.start()
        
        # Start processing
        processor.start(save_output=args.save, output_file=output_file, show_visual=False)
        
        # Run for specified duration or until interrupted
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        # Stop processing
        processor.stop()
        monitor.stop()
        
        # Print final stats
        if not args.terminal:  # Don't print stats in terminal mode (already shown in visual)
            stats = processor.get_stats()
            print("\nTest completed!")
            print(f"Processed {stats['chunks_processed']} chunks in {stats['duration']:.1f}s")
            print(f"Speech detected in {stats['speech_chunks']} chunks ({stats['speech_chunks']/max(1, stats['chunks_processed'])*100:.1f}%)")
            print(f"Average processing time: {stats['average_processing_time_ms']:.2f}ms per chunk")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Battlefield Audio Enhancer Test")
    
    # Terminal mode flag (for launched window)
    parser.add_argument("--terminal", action="store_true",
                      help="Run in terminal mode (for launched window)")
    
    # Microphone options
    parser.add_argument("--device", "-d", type=int, 
                      help="Audio input device ID")
    parser.add_argument("--duration", "-t", type=int, default=30,
                      help="Recording duration in seconds")
    parser.add_argument("--save", "-s", action="store_true",
                      help="Save processed audio to file")
    parser.add_argument("--output", "-o", type=str,
                      help="Output audio file")
    
    # Configuration options
    parser.add_argument("--config", "-c", type=str,
                      help="Path to custom configuration file")
    
    # Display options
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Show detailed processing information")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run test
    return run_test(args)

if __name__ == "__main__":
    sys.exit(main())