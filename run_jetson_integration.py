#!/usr/bin/env python3
"""
TCCC Jetson Integration Test

This script runs the integrated TCCC system on Jetson hardware,
using actual components and real-time processing.
"""

import os
import sys
import time
import json
import asyncio
import argparse
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
from src.tccc.system.system import TCCCSystem, SystemState
from src.tccc.utils.event_schema import (
    BaseEvent, 
    AudioSegmentEvent, 
    TranscriptionEvent,
    ProcessedTextEvent, 
    LLMAnalysisEvent,
    ErrorEvent,
    EventType
)
from src.tccc.utils.logging import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/jetson_integration.log")
    ]
)
logger = get_logger("jetson_integration")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

class JetsonStatusMonitor:
    """Monitor Jetson hardware status during testing."""
    
    def __init__(self, interval_sec: float = 5.0):
        """
        Initialize the monitor.
        
        Args:
            interval_sec: Monitoring interval in seconds
        """
        self.interval = interval_sec
        self.running = False
        self.monitor_thread = None
        self.stats = {}
        self.jetson_tegra = os.path.exists('/etc/nv_tegra_release')
        
        # Import psutil if available
        try:
            import psutil
            self.psutil_available = True
        except ImportError:
            self.psutil_available = False
            logger.warning("psutil not available, some hardware stats will be limited")
    
    def start(self):
        """Start the monitoring thread."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Hardware monitoring started")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Hardware monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_stats()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Shorter interval on error
    
    def _collect_stats(self):
        """Collect hardware statistics."""
        try:
            stats = {
                "timestamp": time.time(),
                "cpu": {},
                "memory": {},
                "temperature": {},
                "gpu": {},
                "power": {}
            }
            
            # Collect CPU and memory stats with psutil
            if self.psutil_available:
                import psutil
                
                # CPU stats
                stats["cpu"]["percent"] = psutil.cpu_percent(interval=0.5, percpu=True)
                stats["cpu"]["load_avg"] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
                stats["cpu"]["frequency"] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
                
                # Memory stats
                memory = psutil.virtual_memory()
                stats["memory"]["total"] = memory.total
                stats["memory"]["available"] = memory.available
                stats["memory"]["used"] = memory.used
                stats["memory"]["percent"] = memory.percent
                
                # Disk stats
                disk = psutil.disk_usage('/')
                stats["disk"] = {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                }
            
            # Jetson-specific stats
            if self.jetson_tegra:
                # GPU utilization
                try:
                    if os.path.exists('/sys/devices/gpu.0/load'):
                        with open('/sys/devices/gpu.0/load', 'r') as f:
                            stats["gpu"]["utilization"] = int(f.read().strip())
                except Exception as e:
                    logger.error(f"Error reading GPU utilization: {e}")
                
                # Temperature sensors
                try:
                    for i in range(10):  # Check thermal zones 0-9
                        thermal_path = f'/sys/devices/virtual/thermal/thermal_zone{i}/temp'
                        if os.path.exists(thermal_path):
                            with open(thermal_path, 'r') as f:
                                temp = int(f.read().strip()) / 1000  # Convert to degrees Celsius
                                stats["temperature"][f"zone{i}"] = temp
                except Exception as e:
                    logger.error(f"Error reading temperatures: {e}")
                
                # Power consumption
                try:
                    power_path = '/sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/'
                    if os.path.exists(power_path):
                        for i in range(3):  # Check channels 0-2
                            curr_path = f'{power_path}in_current{i}_input'
                            volt_path = f'{power_path}in_voltage{i}_input'
                            
                            if os.path.exists(curr_path) and os.path.exists(volt_path):
                                with open(curr_path, 'r') as f:
                                    current = int(f.read().strip()) / 1000  # mA to A
                                with open(volt_path, 'r') as f:
                                    voltage = int(f.read().strip()) / 1000  # mV to V
                                
                                power = voltage * current  # Watts
                                stats["power"][f"channel{i}"] = {
                                    "current": current,
                                    "voltage": voltage,
                                    "power": power
                                }
                except Exception as e:
                    logger.error(f"Error reading power stats: {e}")
            
            # Update stats and log summary
            self.stats = stats
            self._log_stats_summary()
            
        except Exception as e:
            logger.error(f"Error collecting hardware stats: {e}")
    
    def _log_stats_summary(self):
        """Log a summary of hardware stats."""
        try:
            summary = []
            
            # CPU summary
            if "cpu" in self.stats and "percent" in self.stats["cpu"]:
                cpu_avg = sum(self.stats["cpu"]["percent"]) / len(self.stats["cpu"]["percent"])
                summary.append(f"CPU: {cpu_avg:.1f}%")
            
            # Memory summary
            if "memory" in self.stats and "percent" in self.stats["memory"]:
                summary.append(f"Memory: {self.stats['memory']['percent']:.1f}%")
            
            # GPU summary
            if "gpu" in self.stats and "utilization" in self.stats["gpu"]:
                summary.append(f"GPU: {self.stats['gpu']['utilization']}%")
            
            # Temperature summary
            if "temperature" in self.stats and self.stats["temperature"]:
                temps = self.stats["temperature"].values()
                if temps:
                    avg_temp = sum(temps) / len(temps)
                    max_temp = max(temps)
                    summary.append(f"Temp: {avg_temp:.1f}°C (max: {max_temp:.1f}°C)")
            
            # Power summary
            if "power" in self.stats and self.stats["power"]:
                total_power = sum(data.get("power", 0) for data in self.stats["power"].values())
                summary.append(f"Power: {total_power:.2f}W")
            
            if summary:
                logger.info("Hardware stats: " + " | ".join(summary))
                
        except Exception as e:
            logger.error(f"Error logging stats summary: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the current hardware stats."""
        return self.stats


class EventTracker:
    """Track and analyze events during system operation."""
    
    def __init__(self):
        """Initialize the event tracker."""
        self.events = []
        self.events_by_type = {}
        self.event_counts = {}
        self.start_time = time.time()
        self.last_event_time = None
        self.sessions = set()
    
    def add_event(self, event: Dict[str, Any]):
        """
        Add an event to the tracker.
        
        Args:
            event: Event to add
        """
        try:
            # Add event to list
            self.events.append(event)
            
            # Update last event time
            self.last_event_time = time.time()
            
            # Get event type
            event_type = event.get("type", "unknown")
            
            # Update event counts
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
            
            # Add to type-specific list
            if event_type not in self.events_by_type:
                self.events_by_type[event_type] = []
            self.events_by_type[event_type].append(event)
            
            # Track session IDs
            session_id = event.get("session_id")
            if session_id:
                self.sessions.add(session_id)
                
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        try:
            runtime = time.time() - self.start_time
            event_rate = len(self.events) / runtime if runtime > 0 else 0
            
            return {
                "total_events": len(self.events),
                "event_types": list(self.event_counts.keys()),
                "event_counts": self.event_counts,
                "runtime_seconds": runtime,
                "event_rate": event_rate,
                "sessions": len(self.sessions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating event stats: {e}")
            return {"error": str(e)}
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get events of a specific type."""
        return self.events_by_type.get(event_type, [])
    
    def log_summary(self):
        """Log a summary of tracked events."""
        try:
            stats = self.get_stats()
            
            summary = [
                f"Events processed: {stats['total_events']}",
                f"Runtime: {stats['runtime_seconds']:.1f}s",
                f"Event rate: {stats['event_rate']:.2f} events/sec"
            ]
            
            # Add counts by type
            type_counts = []
            for event_type, count in stats["event_counts"].items():
                type_counts.append(f"{event_type}: {count}")
            
            if type_counts:
                summary.append("Event counts: " + ", ".join(type_counts))
            
            logger.info("Event summary: " + " | ".join(summary))
            
        except Exception as e:
            logger.error(f"Error logging event summary: {e}")


class TCCCIntegrationTest:
    """Integration test for the TCCC system."""
    
    def __init__(self, config_path: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize the integration test.
        
        Args:
            config_path: Path to configuration file
            mock_mode: Whether to use mock components
        """
        self.config_path = config_path
        self.mock_mode = mock_mode
        self.config = self._load_config()
        self.system = None
        self.monitor = JetsonStatusMonitor()
        self.event_tracker = EventTracker()
        self.running = False
        self.event_loop = None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Configuration dictionary
        """
        # Default configuration
        config = {
            "system": {
                "debug": True
            },
            "audio_pipeline": {
                "sample_rate": 16000,
                "chunk_size": 1024,
                "channels": 1,
                "vad_enabled": True
            },
            "stt_engine": {
                "model": "tiny",
                "device": "cuda",
                "compute_type": "float16",
                "language": "en",
                "beam_size": 1
            },
            "llm_analysis": {
                "model": "phi-2",
                "device": "cuda",
                "compute_type": "float16",
                "max_new_tokens": 512
            },
            "document_library": {
                "documents_path": "./data/documents",
                "index_path": "./data/document_index",
                "cache_dir": "./data/query_cache"
            },
            "data_store": {
                "db_path": "./data/processing_core/state/events.db"
            },
            "processing_core": {
                "plugin_dir": "./plugins"
            }
        }
        
        # Load configuration from file if specified
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                    # Recursively update default config with loaded values
                    def update_nested_dict(d, u):
                        for k, v in u.items():
                            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                                d[k] = update_nested_dict(d[k], v)
                            else:
                                d[k] = v
                        return d
                    
                    config = update_nested_dict(config, loaded_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {self.config_path}: {e}")
        
        # In mock mode, override some settings
        if self.mock_mode:
            logger.info("Mock mode enabled, overriding configuration")
            config["stt_engine"]["model"] = "tiny"
            config["stt_engine"]["device"] = "cpu"
            config["llm_analysis"]["model"] = "phi-tiny"
            config["llm_analysis"]["device"] = "cpu"
        
        return config
    
    async def initialize(self) -> bool:
        """
        Initialize the TCCC system.
        
        Returns:
            Success status
        """
        try:
            logger.info("Initializing TCCC Integration Test")
            
            # Create event loop
            if not self.event_loop:
                self.event_loop = asyncio.get_event_loop()
            
            # Create system instance
            self.system = TCCCSystem()
            
            # Determine mock modules
            mock_modules = []
            if self.mock_mode:
                mock_modules = ["audio_pipeline", "stt_engine", "llm_analysis"]
            
            # Initialize system
            init_success = await self.system.initialize(self.config, mock_modules=mock_modules)
            
            if not init_success:
                logger.error("Failed to initialize TCCC system")
                return False
                
            logger.info("TCCC system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TCCC Integration Test: {e}")
            return False
    
    async def start(self, duration_sec: int = 60) -> bool:
        """
        Start the integration test.
        
        Args:
            duration_sec: Test duration in seconds
            
        Returns:
            Success status
        """
        if not self.system or not self.system.initialized:
            logger.error("System not initialized")
            return False
        
        try:
            logger.info(f"Starting TCCC Integration Test for {duration_sec} seconds")
            
            # Start hardware monitoring
            self.monitor.start()
            
            # Hook into system event processing
            self._hook_event_processing()
            
            # Start the system
            start_result = await self.system.start()
            
            if not start_result:
                logger.error("Failed to start TCCC system")
                self.monitor.stop()
                return False
            
            logger.info("TCCC system started successfully")
            
            self.running = True
            
            # Run for specified duration
            try:
                logger.info(f"Running for {duration_sec} seconds...")
                
                # Wait with regular status updates
                start_time = time.time()
                update_interval = 10  # seconds
                next_update = start_time + update_interval
                
                while time.time() - start_time < duration_sec and self.running:
                    # Sleep in short intervals to allow for clean interrupt
                    await asyncio.sleep(0.5)
                    
                    # Log status update at regular intervals
                    current_time = time.time()
                    if current_time >= next_update:
                        elapsed = current_time - start_time
                        remaining = duration_sec - elapsed
                        
                        # Get system status
                        system_status = self.system.get_status()
                        state = system_status.get("state", "unknown")
                        
                        # Log status update
                        logger.info(f"Status: {state} | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
                        
                        # Log event statistics
                        self.event_tracker.log_summary()
                        
                        # Schedule next update
                        next_update = current_time + update_interval
                
                logger.info("Test duration completed")
                
            except KeyboardInterrupt:
                logger.info("Test interrupted by user")
            finally:
                # Stop the system
                await self.system.stop()
                
                # Stop monitoring
                self.monitor.stop()
                
                self.running = False
                
                # Log final event statistics
                self.event_tracker.log_summary()
                
                logger.info("TCCC Integration Test completed")
                return True
            
        except Exception as e:
            logger.error(f"Error in TCCC Integration Test: {e}")
            
            # Ensure monitor is stopped
            self.monitor.stop()
            
            self.running = False
            return False
    
    def _hook_event_processing(self):
        """Hook into system event processing to track events."""
        try:
            # Save original process_event method
            original_process_event = self.system.process_event
            
            # Create wrapper method
            async def process_event_wrapper(event_data):
                # Call original method
                result = await original_process_event(event_data)
                
                # Track the event
                self.event_tracker.add_event(event_data)
                
                return result
            
            # Replace method with wrapper
            self.system.process_event = process_event_wrapper
            
            logger.debug("Hooked event processing for tracking")
            
        except Exception as e:
            logger.error(f"Error hooking event processing: {e}")
    
    async def test_document_query(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Test document query functionality.
        
        Args:
            query: Query text
            n_results: Number of results
            
        Returns:
            Query results
        """
        if not self.system or not self.system.initialized:
            logger.error("System not initialized")
            return {"error": "System not initialized"}
        
        try:
            logger.info(f"Testing document query: '{query}'")
            
            # Create document query event
            query_event = BaseEvent(
                type=EventType.DOCUMENT_QUERY,
                source="integration_test",
                data={
                    "query": query,
                    "limit": n_results
                }
            ).to_dict()
            
            # Process the event
            await self.system.process_event(query_event)
            
            # Extract results from event tracker
            results_events = self.event_tracker.get_events_by_type(EventType.DOCUMENT_RESULTS.value)
            
            if results_events:
                result = results_events[-1]  # Get most recent result
                logger.info(f"Document query returned {len(result.get('data', {}).get('results', []))} results")
                return result
            else:
                logger.warning("No document query results found")
                return {"error": "No results"}
            
        except Exception as e:
            logger.error(f"Error testing document query: {e}")
            return {"error": str(e)}
    
    async def test_audio_processing(self, audio_file: str) -> Dict[str, Any]:
        """
        Test audio processing functionality.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Processing results
        """
        if not self.system or not self.system.initialized:
            logger.error("System not initialized")
            return {"error": "System not initialized"}
        
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return {"error": "Audio file not found"}
        
        try:
            logger.info(f"Testing audio processing with file: {audio_file}")
            
            # Read audio file
            import wave
            import numpy as np
            
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
                audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)
                
                logger.info(f"Loaded audio file: {duration:.2f} seconds at {rate} Hz")
            
            # Create audio segment event
            audio_event = AudioSegmentEvent(
                source="integration_test",
                audio_data=audio_data,
                sample_rate=rate,
                format_type="PCM16",
                channels=1,
                duration_ms=duration * 1000,
                is_speech=True,
                start_time=time.time(),
                metadata={"file": audio_file}
            ).to_dict()
            
            # Process the event
            await self.system.process_event(audio_event)
            
            # Wait for processing to complete
            await asyncio.sleep(2.0)
            
            # Extract results from event tracker
            transcription_events = self.event_tracker.get_events_by_type(EventType.TRANSCRIPTION.value)
            processed_events = self.event_tracker.get_events_by_type(EventType.PROCESSED_TEXT.value)
            llm_events = self.event_tracker.get_events_by_type(EventType.LLM_ANALYSIS.value)
            
            # Return the results
            results = {
                "transcription": transcription_events[-1] if transcription_events else None,
                "processed": processed_events[-1] if processed_events else None,
                "llm_analysis": llm_events[-1] if llm_events else None
            }
            
            # Log results summary
            if transcription_events:
                text = transcription_events[-1].get("data", {}).get("text", "")
                logger.info(f"Transcription: {text[:100]}..." if len(text) > 100 else f"Transcription: {text}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing audio processing: {e}")
            return {"error": str(e)}


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TCCC Jetson Integration Test")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mock", action="store_true", help="Use mock components")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--query", help="Test document query with specified text")
    parser.add_argument("--audio", help="Test audio processing with specified file")
    
    args = parser.parse_args()
    
    # Create and initialize test
    test = TCCCIntegrationTest(config_path=args.config, mock_mode=args.mock)
    
    if not await test.initialize():
        logger.error("Failed to initialize test")
        return 1
    
    # Run specific tests if requested
    if args.query:
        result = await test.test_document_query(args.query)
        logger.info(f"Document query result: {json.dumps(result, indent=2)}")
    
    if args.audio:
        result = await test.test_audio_processing(args.audio)
        if "error" in result:
            logger.error(f"Audio processing error: {result['error']}")
        else:
            logger.info("Audio processing completed successfully")
    
    # Run full integration test if no specific tests
    if not args.query and not args.audio:
        # Start the integration test
        await test.start(duration_sec=args.duration)
    
    return 0


if __name__ == "__main__":
    try:
        # Run main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)