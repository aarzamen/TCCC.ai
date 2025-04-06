#!/usr/bin/env python3
"""
Test script for the TCCC MVP Flow.

This script tests the complete minimum viable product (MVP) flow:
1. Audio capture from microphone
2. Speech-to-text processing
3. LLM analysis of transcription
4. Result display and logging (without relying on DataStore)

Note: This test bypasses the DataStore module which is commented out in system.py for MVP testing.
"""

import os
import sys
import time
import argparse
import asyncio
import signal
import threading
import json
from pathlib import Path
import logging
from datetime import datetime

# Add project source to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MVPFlowTest")

# Import TCCC modules
from tccc.system.system import TCCCSystem
from tccc.utils.config import Config

class MVPTester:
    """Tests the complete MVP flow of the TCCC system"""
    def __init__(self, config_path, output_dir=None):
        self.config_path = config_path
        self.output_dir = output_dir or Path("./tccc_mvp_results")
        self.system = None
        self.events = []
        self.running = False
        self.event_lock = threading.Lock()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            config = Config.from_yaml(self.config_path)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    async def initialize(self):
        """Initialize the TCCC system"""
        try:
            # Load configuration
            config = self.load_config()
            
            # Initialize system
            logger.info("Initializing TCCC System...")
            self.system = TCCCSystem()
            
            # Monkey patch the DataStore functionality to avoid database issues
            if hasattr(self.system, "data_store"):
                logger.info("Bypassing DataStore functionality for MVP test")
                self.system.data_store = None
            
            # Register event callback
            self.system.register_event_callback(self.event_callback)
            
            # Initialize the system
            result = await self.system.initialize(config)
            if result:
                logger.info("TCCC System initialized successfully")
                return True
            else:
                logger.error("Failed to initialize TCCC System")
                return False
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def event_callback(self, event_type, event_data):
        """Callback for system events"""
        try:
            with self.event_lock:
                self.events.append({
                    'timestamp': time.time(),
                    'event_type': event_type,
                    'event_data': event_data
                })
            
            # Log the event
            logger.info(f"Event received: {event_type}")
            
            # Special handling for transcription events
            if event_type == 'transcription' and event_data.get('data', {}).get('text'):
                text = event_data.get('data', {}).get('text')
                logger.info(f"Transcription: {text}")
                
            # Special handling for LLM analysis events
            if event_type == 'llm_analysis':
                logger.info(f"LLM Analysis received: {event_data.get('type', 'unknown')}")
                
        except Exception as e:
            logger.error(f"Error in event callback: {e}")
    
    async def start(self, duration=60):
        """Start the MVP test"""
        try:
            # Create output directory if needed
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Output directory: {self.output_dir}")
            
            # Start audio capture
            logger.info("Starting audio capture...")
            await self.system.start_audio_capture()
            self.running = True
            
            # Record test start
            test_start = {
                'test_id': f"mvp_{int(time.time())}",
                'start_time': datetime.now().isoformat(),
                'config_file': self.config_path,
                'duration': duration
            }
            
            with open(Path(self.output_dir) / "test_info.json", 'w') as f:
                json.dump(test_start, f, indent=2)
            
            logger.info(f"MVP test running for {duration} seconds...")
            logger.info("Speak clearly into the microphone to test the complete pipeline.")
            
            # Set up signal handler for clean shutdown
            def signal_handler(sig, frame):
                logger.info("Test interrupted. Shutting down...")
                self.running = False
            
            original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal_handler)
            
            # Run for the specified duration
            try:
                for remaining in range(duration, 0, -1):
                    if not self.running:
                        break
                    
                    if remaining % 10 == 0 or remaining <= 5:
                        logger.info(f"{remaining} seconds remaining...")
                    
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Test cancelled")
            
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            
            # Stop the system
            logger.info("Stopping audio capture...")
            await self.system.stop_audio_capture()
            self.running = False
            
            # Save collected events
            await self.save_results()
            
            return True
        except Exception as e:
            logger.error(f"Error during test: {e}")
            self.running = False
            return False
    
    async def save_results(self):
        """Save the test results to the output directory"""
        try:
            if not self.output_dir:
                return
                
            # Gather results
            with self.event_lock:
                event_count = len(self.events)
                
                # Count event types
                event_types = {}
                for event in self.events:
                    event_type = event['event_type']
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                # Extract transcriptions
                transcriptions = []
                for event in self.events:
                    if event['event_type'] == 'transcription':
                        text = event['event_data'].get('data', {}).get('text')
                        if text:
                            transcriptions.append({
                                'timestamp': event['timestamp'],
                                'text': text,
                                'is_partial': event['event_data'].get('data', {}).get('is_partial', False)
                            })
                
                # Extract LLM analyses
                analyses = []
                for event in self.events:
                    if event['event_type'] == 'llm_analysis':
                        analyses.append({
                            'timestamp': event['timestamp'],
                            'analysis_type': event['event_data'].get('type'),
                            'data': event['event_data']
                        })
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save event statistics
            stats = {
                'timestamp': timestamp,
                'total_events': event_count,
                'event_types': event_types,
                'transcription_count': len(transcriptions),
                'analysis_count': len(analyses)
            }
            
            with open(Path(self.output_dir) / f"event_stats_{timestamp}.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Save transcriptions
            if transcriptions:
                with open(Path(self.output_dir) / f"transcriptions_{timestamp}.json", 'w') as f:
                    json.dump(transcriptions, f, indent=2)
                    
                # Also save as text for easy viewing
                with open(Path(self.output_dir) / f"transcriptions_{timestamp}.txt", 'w') as f:
                    for t in transcriptions:
                        if not t.get('is_partial'):
                            f.write(f"{t['text']}\n")
            
            # Save LLM analyses
            if analyses:
                with open(Path(self.output_dir) / f"analyses_{timestamp}.json", 'w') as f:
                    json.dump(analyses, f, indent=2)
            
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    async def shutdown(self):
        """Shutdown the TCCC system"""
        try:
            if self.system:
                await self.system.shutdown()
            logger.info("TCCC System shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def generate_summary(self):
        """Generate a summary of the test results"""
        try:
            with self.event_lock:
                event_count = len(self.events)
                
                # Count by type
                transcription_events = sum(1 for e in self.events if e['event_type'] == 'transcription')
                llm_events = sum(1 for e in self.events if e['event_type'] == 'llm_analysis')
                
                # Extract unique transcriptions (non-partial)
                unique_transcriptions = set()
                for event in self.events:
                    if event['event_type'] == 'transcription' and not event['event_data'].get('data', {}).get('is_partial', True):
                        text = event['event_data'].get('data', {}).get('text', '')
                        if text and len(text) > 3:  # Ignore very short transcriptions
                            unique_transcriptions.add(text)
            
            summary = []
            summary.append("=== TCCC MVP Flow Test Summary ===")
            summary.append(f"Total events processed: {event_count}")
            summary.append(f"Transcription events: {transcription_events}")
            summary.append(f"LLM analysis events: {llm_events}")
            summary.append("")
            
            if unique_transcriptions:
                summary.append("=== Unique Transcriptions Captured ===")
                for i, text in enumerate(unique_transcriptions, 1):
                    summary.append(f"{i}. {text}")
                summary.append("")
                
            if llm_events > 0:
                summary.append(f"LLM analyses completed: {llm_events}")
                summary.append("Check the output directory for detailed analysis results.")
                summary.append("")
                
            success = transcription_events > 0 and llm_events > 0
            if success:
                summary.append("MVP FLOW TEST: PASSED ✓")
                summary.append("All components are working together successfully.")
            else:
                summary.append("MVP FLOW TEST: INCOMPLETE ✗")
                if transcription_events == 0:
                    summary.append("No transcriptions were generated during the test.")
                if llm_events == 0:
                    summary.append("No LLM analyses were performed during the test.")
            
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary"
            
async def run_test(args):
    """Run the MVP flow test"""
    tester = MVPTester(args.config, args.output_dir)
    
    try:
        # Initialize components
        if not await tester.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Run the test
        logger.info("Starting MVP flow test...")
        result = await tester.start(args.duration)
        
        # Generate and display summary
        summary = tester.generate_summary()
        print("\n" + summary)
        
        # Shut down
        await tester.shutdown()
        
        if result:
            logger.info("MVP flow test completed successfully")
            return 0
        else:
            logger.error("MVP flow test failed")
            return 1
    except Exception as e:
        logger.error(f"Test failed: {e}")
        await tester.shutdown()
        return 1

def main():
    """Main entry point for the MVP flow test"""
    parser = argparse.ArgumentParser(description="Test TCCC MVP Flow")
    parser.add_argument("--config", default=str(project_root / "config/jetson_mvp.yaml"), 
                      help="Path to config file")
    parser.add_argument("--duration", type=int, default=60,
                      help="Test duration in seconds")
    parser.add_argument("--output-dir", type=str, default="./tccc_mvp_results",
                      help="Directory to save test results")
    args = parser.parse_args()
    
    # Create and run the event loop
    try:
        asyncio.run(run_test(args))
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
