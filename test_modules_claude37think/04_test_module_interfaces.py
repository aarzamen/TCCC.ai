#!/usr/bin/env python3
"""
Test script for the TCCC module interfaces.

This script tests the integration between modules:
1. Audio Pipeline -> STT Engine -> LLM Analysis
2. Ensures proper data flow between components
3. Validates event-based communication
"""

import os
import sys
import time
import argparse
import asyncio
import threading
import queue
import json
from pathlib import Path
import logging
import numpy as np

# Add project source to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModuleInterfaceTest")

# Import TCCC modules
from tccc.audio_pipeline.audio_pipeline import AudioPipeline
from tccc.stt_engine.stt_engine import STTEngine
from tccc.llm_analysis.llm_analysis import LLMAnalysis
from tccc.utils.config import Config
from tccc.utils.event_bus import EventBus, EventType

class ResultCollector:
    """Collects and stores results from the pipeline processing"""
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()
        
    def add_result(self, stage, data):
        """Add a result from any stage of the pipeline"""
        with self.lock:
            self.results.append({
                'timestamp': time.time(),
                'stage': stage,
                'data': data
            })
            
    def get_results(self):
        """Get a copy of all collected results"""
        with self.lock:
            return self.results.copy()
    
    def get_results_by_stage(self, stage):
        """Get results for a specific stage"""
        with self.lock:
            return [r for r in self.results if r['stage'] == stage]

class SystemEventBus(EventBus):
    """Event bus implementation for testing module interfaces"""
    def __init__(self, result_collector):
        super().__init__()
        self.collector = result_collector
        
    async def publish(self, event_type, event_data):
        """Publish an event to subscribers and collect it"""
        # Store the event
        if isinstance(event_type, EventType):
            stage = event_type.value
        else:
            stage = str(event_type)
            
        self.collector.add_result(stage, event_data)
        logger.info(f"Event published: {stage}")
        
        # Call parent method to dispatch to subscribers
        await super().publish(event_type, event_data)

class InterfaceTester:
    """Tests the interfaces between TCCC modules"""
    def __init__(self, config_path, device_index=None):
        self.config_path = config_path
        self.device_index = device_index
        self.audio_pipeline = None
        self.stt_engine = None
        self.llm_analysis = None
        self.collector = ResultCollector()
        self.event_bus = SystemEventBus(self.collector)
        self.running = False
        self.sequence_num = 0
        
    def load_config(self):
        """Load configuration from file"""
        try:
            config = Config.from_yaml(self.config_path)
            audio_config = config.get('audio_pipeline', {})
            stt_config = config.get('stt_engine', {})
            llm_config = config.get('llm_analysis', {})
            
            # Override device index if specified
            if self.device_index is not None:
                audio_config['microphone']['device_index'] = self.device_index
                logger.info(f"Using device index: {self.device_index}")
            
            return audio_config, stt_config, llm_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    async def initialize(self):
        """Initialize all modules and their connections"""
        try:
            audio_config, stt_config, llm_config = self.load_config()
            
            # Initialize audio pipeline
            logger.info("Initializing Audio Pipeline...")
            self.audio_pipeline = AudioPipeline()
            await self.audio_pipeline.initialize(audio_config)
            logger.info("Audio Pipeline initialized successfully")
            
            # Initialize STT engine
            logger.info("Initializing STT Engine...")
            self.stt_engine = STTEngine()
            await self.stt_engine.initialize(stt_config)
            # Set system reference for event handling
            self.stt_engine.set_system_reference(self)
            logger.info("STT Engine initialized successfully")
            
            # Initialize LLM Analysis
            logger.info("Initializing LLM Analysis...")
            self.llm_analysis = LLMAnalysis()
            await self.llm_analysis.initialize(llm_config)
            logger.info("LLM Analysis initialized successfully")
            
            # Connect modules
            self.audio_pipeline.register_audio_callback(self.process_audio)
            
            # Subscribe LLM Analysis to transcription events
            await self.event_bus.subscribe(self.llm_analysis, [EventType.TRANSCRIPTION])
            logger.info("LLM Analysis subscribed to transcription events")
            
            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def process_audio(self, audio_data):
        """Process audio data from pipeline and forward to STT Engine"""
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = audio_data
                
            # Add sequence number
            self.sequence_num += 1
            
            # Calculate audio level
            level = np.abs(audio_array).mean() / 32768.0
            
            # Record the audio event
            self.collector.add_result('audio', {
                'sequence': self.sequence_num,
                'timestamp': time.time(),
                'level': float(level),
                'sample_count': len(audio_array)
            })
            
            if level > 0.01:  # Only process if audio level is above threshold
                # Prepare metadata
                audio_metadata = {
                    'sequence': self.sequence_num,
                    'timestamp': time.time(),
                    'format': 'int16',
                    'channels': 1,  # Assuming mono
                    'sample_rate': 16000  # Assuming 16kHz 
                }
                
                # Enqueue audio for processing
                self.stt_engine.enqueue_audio(audio_array, audio_metadata)
                logger.debug(f"Enqueued audio chunk {self.sequence_num}, level: {level:.4f}")
            else:
                logger.debug(f"Skipping low-level audio: {level:.4f}")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    async def process_event(self, event_data):
        """Process events from STT engine"""
        try:
            if not event_data:
                return
            
            # Forward to event bus
            event_type = event_data.get('type')
            if event_type:
                # Convert string to EventType if needed
                if isinstance(event_type, str):
                    try:
                        event_type = EventType(event_type)
                    except ValueError:
                        logger.warning(f"Unknown event type: {event_type}")
                
                # Publish the event
                await self.event_bus.publish(event_type, event_data)
        except Exception as e:
            logger.error(f"Error processing event: {e}")
    
    async def start(self, duration=30):
        """Start the interface test"""
        try:
            logger.info("Starting audio capture...")
            self.running = True
            self.audio_pipeline.start_capture()
            
            # Record test parameters
            self.collector.add_result('test_info', {
                'started_at': time.time(),
                'duration': duration,
                'device_index': self.device_index
            })
            
            logger.info(f"Test running for {duration} seconds...")
            start_time = time.time()
            
            # Run for specified duration
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                logger.info("Test cancelled")
            
            # Stop capture
            logger.info("Stopping audio capture...")
            self.audio_pipeline.stop_capture()
            self.running = False
            
            # Record end time
            end_time = time.time()
            self.collector.add_result('test_info', {
                'ended_at': end_time,
                'actual_duration': end_time - start_time
            })
            
            return self.collector.get_results()
        except Exception as e:
            logger.error(f"Error during test: {e}")
            self.running = False
            return []
    
    async def shutdown(self):
        """Clean shutdown of all components"""
        try:
            if self.audio_pipeline:
                self.audio_pipeline.stop_capture()
            
            if self.stt_engine:
                await self.stt_engine.shutdown()
                
            if self.llm_analysis:
                await self.llm_analysis.shutdown()
                
            logger.info("All components shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def generate_report(self, results):
        """Generate a report of the interface test results"""
        if not results:
            return "No test results collected"
            
        # Extract events by type
        audio_events = [r for r in results if r['stage'] == 'audio']
        transcription_events = [r for r in results if r['stage'] == 'transcription']
        llm_events = [r for r in results if r['stage'] == 'llm_analysis']
        
        report = []
        report.append("=== TCCC Module Interface Test Report ===")
        report.append(f"Total events: {len(results)}")
        report.append(f"Audio chunks processed: {len(audio_events)}")
        report.append(f"Transcriptions generated: {len(transcription_events)}")
        report.append(f"LLM analyses performed: {len(llm_events)}")
        report.append("")
        
        # Show data flow timing
        if transcription_events:
            report.append("=== Transcription Results ===")
            for i, event in enumerate(transcription_events[:5], 1):  # Show first 5
                text = event['data'].get('data', {}).get('text', 'N/A')
                report.append(f"{i}. {text}")
            
            if len(transcription_events) > 5:
                report.append(f"... and {len(transcription_events) - 5} more")
            report.append("")
            
        if llm_events:
            report.append("=== LLM Analysis Results ===")
            for i, event in enumerate(llm_events[:3], 1):  # Show first 3
                analysis = event['data']
                if isinstance(analysis, dict):
                    report.append(f"{i}. Analysis type: {analysis.get('type', 'N/A')}")
                    for key, value in analysis.items():
                        if key != 'type' and key != 'raw_text':
                            report.append(f"   - {key}: {value}")
                else:
                    report.append(f"{i}. {analysis}")
            
            if len(llm_events) > 3:
                report.append(f"... and {len(llm_events) - 3} more")
                
        return "\n".join(report)
            
async def run_test(args):
    """Run the module interface test"""
    tester = InterfaceTester(args.config, args.device)
    
    try:
        # Initialize components
        if not await tester.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Run the test
        logger.info("Starting module interface test...")
        results = await tester.start(args.duration)
        
        # Shut down
        await tester.shutdown()
        
        # Process and display results
        if results:
            logger.info(f"Test completed with {len(results)} total events")
            
            # Save results if requested
            if args.output_file:
                try:
                    with open(args.output_file, 'w') as f:
                        # Convert numpy values to Python native types for JSON serialization
                        serializable_results = []
                        for r in results:
                            if isinstance(r['data'], dict):
                                data_copy = {}
                                for k, v in r['data'].items():
                                    if isinstance(v, np.ndarray):
                                        data_copy[k] = v.tolist()
                                    elif isinstance(v, np.number):
                                        data_copy[k] = v.item()
                                    else:
                                        data_copy[k] = v
                                r_copy = {'timestamp': r['timestamp'], 'stage': r['stage'], 'data': data_copy}
                            else:
                                r_copy = r
                            serializable_results.append(r_copy)
                            
                        json.dump(serializable_results, f, indent=2)
                    logger.info(f"Results saved to {args.output_file}")
                except Exception as e:
                    logger.error(f"Error saving results: {e}")
            
            # Generate and display report
            report = tester.generate_report(results)
            print("\n" + report)
            
            # Determine success based on presence of transcriptions
            transcriptions = [r for r in results if r['stage'] == 'transcription']
            if transcriptions:
                logger.info("Interface test PASSED - All modules working together")
                return 0
            else:
                logger.error("Interface test FAILED - No transcriptions generated")
                return 1
        else:
            logger.error("Test completed without any events recorded")
            return 1
    except Exception as e:
        logger.error(f"Test failed: {e}")
        await tester.shutdown()
        return 1

def main():
    """Main entry point for the module interface test"""
    parser = argparse.ArgumentParser(description="Test TCCC Module Interfaces")
    parser.add_argument("--config", default=str(project_root / "config/jetson_mvp.yaml"), 
                      help="Path to config file")
    parser.add_argument("--device", type=int, default=0,
                      help="Audio device index to use for testing")
    parser.add_argument("--duration", type=int, default=30,
                      help="Test duration in seconds")
    parser.add_argument("--output-file", type=str, default=None,
                      help="File to save JSON results")
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
