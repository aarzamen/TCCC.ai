#!/usr/bin/env python3
"""
Test script for the TCCC Audio Pipeline.

This script tests the audio capture functionality using the configured device.
It displays real-time audio levels and diagnostics to verify proper operation.
"""

import os
import sys
import time
import numpy as np
import argparse
from pathlib import Path
import logging
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add project source to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioPipelineTest")

# Import TCCC modules
from tccc.audio_pipeline.audio_pipeline import AudioPipeline, MicrophoneSource
from tccc.utils.config import Config

class AudioVisualizer:
    def __init__(self, buffer_size=100):
        self.audio_data_queue = queue.Queue()
        self.buffer_size = buffer_size
        self.levels = np.zeros(buffer_size)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot(np.arange(buffer_size), self.levels)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, buffer_size)
        self.ax.set_title('Real-time Audio Levels')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Normalized Audio Level')
        self.ax.grid(True)
        
    def update_plot(self, frame):
        while not self.audio_data_queue.empty():
            try:
                level = self.audio_data_queue.get_nowait()
                self.levels = np.roll(self.levels, -1)
                self.levels[-1] = level
            except queue.Empty:
                break
        self.line.set_ydata(self.levels)
        return self.line,
    
    def add_audio_level(self, level):
        self.audio_data_queue.put(level)
    
    def start(self):
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        plt.show()

def audio_callback(audio_data, metadata, visualizer=None):
    """Process incoming audio data and display diagnostics"""
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Calculate audio level (normalized)
    if len(audio_array) > 0:
        level = np.abs(audio_array).mean() / 32768.0  # Normalize to 0-1
        
        # Display audio statistics
        peak = np.abs(audio_array).max() / 32768.0
        logger.info(f"Audio level: {level:.6f} (peak: {peak:.6f}) - Size: {len(audio_array)} samples")
        
        # Show clipping warning if needed
        if peak > 0.95:
            logger.warning("AUDIO CLIPPING DETECTED!")
        
        # Add to visualizer if available
        if visualizer:
            visualizer.add_audio_level(level)
    else:
        logger.warning("Received empty audio buffer")

def test_audio_pipeline():
    """Main test function for the audio pipeline"""
    parser = argparse.ArgumentParser(description="Test TCCC Audio Pipeline")
    parser.add_argument("--config", default=str(project_root / "config/jetson_mvp.yaml"), 
                        help="Path to config file")
    parser.add_argument("--device", type=int, default=0,
                        help="Audio device index to use for testing")
    parser.add_argument("--duration", type=int, default=30,
                        help="Test duration in seconds")
    parser.add_argument("--visualize", action="store_true",
                        help="Show real-time audio level visualization")
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = Config.from_yaml(args.config)
        audio_config = config.get('audio_pipeline', {})
        
        # Override device index if specified
        if args.device is not None:
            audio_config['microphone']['device_index'] = args.device
            logger.info(f"Using device index: {args.device}")
        
        logger.info(f"Audio configuration: {audio_config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1

    # Initialize audio pipeline
    logger.info("Initializing Audio Pipeline...")
    try:
        audio_pipeline = AudioPipeline()
        audio_pipeline.initialize(audio_config)
        logger.info("Audio Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Audio Pipeline: {e}")
        return 1

    # Set up visualizer if requested
    visualizer = None
    visualizer_thread = None
    if args.visualize:
        visualizer = AudioVisualizer()
        visualizer_thread = threading.Thread(target=visualizer.start)
        visualizer_thread.daemon = True
        visualizer_thread.start()

    # Register callback
    def callback_wrapper(audio_data):
        audio_callback(audio_data, {}, visualizer)
    
    audio_pipeline.register_audio_callback(callback_wrapper)

    # Start audio capture
    logger.info("Starting audio capture...")
    try:
        audio_pipeline.start_capture()
        logger.info(f"Audio capture started. Running for {args.duration} seconds...")
        
        # Display device info
        try:
            source = audio_pipeline.active_source
            if source and hasattr(source, 'device_info'):
                logger.info(f"Active device info: {source.device_info}")
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")
        
        # Run for specified duration
        end_time = time.time() + args.duration
        try:
            while time.time() < end_time:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        
        # Stop capture
        logger.info("Stopping audio capture...")
        audio_pipeline.stop_capture()
        logger.info("Audio capture stopped")
        
    except Exception as e:
        logger.error(f"Error during audio capture: {e}")
        return 1
    
    # If visualization is active, wait for user to close window
    if args.visualize and visualizer_thread and visualizer_thread.is_alive():
        logger.info("Close the visualization window to exit")
        visualizer_thread.join()
    
    logger.info("Audio Pipeline test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(test_audio_pipeline())
