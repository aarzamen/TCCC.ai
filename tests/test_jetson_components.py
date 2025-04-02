#!/usr/bin/env python3
"""
TCCC Component Test for Jetson Hardware

This script performs functional testing of core TCCC components on Jetson hardware.
Tests audio pipeline, STT engine, and document retrieval with real code.
"""

import os
import sys
import time
import argparse
import numpy as np
import threading
import queue
import json
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("jetson_test")

try:
    from src.tccc.audio_pipeline.audio_pipeline import AudioPipeline
    from src.tccc.stt_engine.stt_engine import STTEngine 
    from src.tccc.document_library.document_library import DocumentLibrary
    from src.tccc.llm_analysis.llm_analysis import LLMAnalysis
except ImportError as e:
    logger.error(f"Error importing TCCC components: {e}")
    logger.error("Make sure you're running from the project root with the virtual environment activated")
    sys.exit(1)

def load_config(config_path=None):
    """Load configuration for components."""
    # Default config with values appropriate for Jetson hardware
    config = {
        "audio_pipeline": {
            "sample_rate": 16000,
            "chunk_size": 1024,
            "channels": 1,
            "device_index": 0,
            "max_recording_time": 30,
            "vad_enabled": True,
            "noise_reduction": True
        },
        "stt_engine": {
            "model": "tiny",
            "device": "cuda",  # Use CUDA on Jetson
            "compute_type": "float16",  # Use mixed precision for Jetson
            "language": "en",
            "beam_size": 1,  # Lower beam size for faster processing
            "vad_filter": True
        },
        "document_library": {
            "documents_path": "./data/documents",
            "index_path": "./data/document_index",
            "embedding_model": "local",
            "cache_dir": "./data/query_cache",
            "max_cache_size": 1000
        },
        "llm_analysis": {
            "model": "phi-2",
            "model_path": "./models/llm/phi-2",
            "device": "cuda",
            "compute_type": "float16",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    # Load external config if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Update default config with loaded values
                for component, component_config in loaded_config.items():
                    if component in config:
                        config[component].update(component_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    return config

class STTBenchmark:
    """Benchmark the STT engine performance."""
    
    def __init__(self, config):
        """Initialize the benchmark."""
        self.config = config
        self.stt_engine = None
        self.results = []
    
    def initialize(self):
        """Initialize the STT engine."""
        try:
            self.stt_engine = STTEngine()
            success = self.stt_engine.initialize(self.config["stt_engine"])
            if not success:
                logger.error("Failed to initialize STT engine")
                return False
            
            logger.info("STT engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing STT engine: {e}")
            return False
    
    def benchmark_audio_file(self, audio_path):
        """Benchmark STT performance on an audio file."""
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        try:
            import wave
            import numpy as np
            
            logger.info(f"Processing audio file: {audio_path}")
            
            # Read the audio file
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
                audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)
                
                logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Process the audio data
            start_time = time.time()
            
            # Use the STT engine to transcribe
            result = self.stt_engine.transcribe_audio(audio_data)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            realtime_factor = processing_time / duration if duration > 0 else 0
            
            # Log results
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Realtime factor: {realtime_factor:.2f}x")
            logger.info(f"Transcription: {result.get('text', '')[:100]}...")
            
            # Store results
            benchmark_result = {
                "audio_file": audio_path,
                "duration": duration,
                "processing_time": processing_time,
                "realtime_factor": realtime_factor,
                "text_length": len(result.get("text", "")),
                "success": True
            }
            
            self.results.append(benchmark_result)
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Error benchmarking audio file: {e}")
            benchmark_result = {
                "audio_file": audio_path,
                "success": False,
                "error": str(e)
            }
            self.results.append(benchmark_result)
            return benchmark_result
    
    def run_benchmark_suite(self, audio_dir="./test_data"):
        """Run benchmark on all audio files in directory."""
        if not os.path.exists(audio_dir):
            logger.error(f"Audio directory not found: {audio_dir}")
            return False
        
        audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                      if f.endswith(('.wav', '.WAV'))]
        
        if not audio_files:
            logger.error(f"No audio files found in {audio_dir}")
            return False
        
        logger.info(f"Running benchmark on {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            self.benchmark_audio_file(audio_file)
        
        # Calculate aggregate statistics
        if self.results:
            success_results = [r for r in self.results if r.get("success", False)]
            if success_results:
                avg_realtime = sum(r["realtime_factor"] for r in success_results) / len(success_results)
                logger.info(f"Average realtime factor: {avg_realtime:.2f}x")
                
                # Memory usage reporting
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
                except ImportError:
                    logger.warning("psutil not available, skipping memory usage reporting")
            
            return True
        
        return False

class DocumentBenchmark:
    """Benchmark the Document Library performance."""
    
    def __init__(self, config):
        """Initialize the benchmark."""
        self.config = config
        self.doc_library = None
        self.results = []
    
    def initialize(self):
        """Initialize the Document Library."""
        try:
            self.doc_library = DocumentLibrary()
            success = self.doc_library.initialize(self.config["document_library"])
            if not success:
                logger.error("Failed to initialize Document Library")
                return False
            
            logger.info("Document Library initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing Document Library: {e}")
            return False
    
    def benchmark_query(self, query):
        """Benchmark document retrieval for a query."""
        try:
            logger.info(f"Processing query: {query}")
            
            # Process the query
            start_time = time.time()
            
            # Use the document library to retrieve documents
            results = self.doc_library.query(query, limit=5)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            # Log results
            logger.info(f"Query processing time: {processing_time:.4f} seconds")
            logger.info(f"Number of results: {len(results)}")
            if results:
                logger.info(f"Top result: {results[0].get('document', '')[:100]}...")
            
            # Store results
            benchmark_result = {
                "query": query,
                "processing_time": processing_time,
                "num_results": len(results),
                "success": True
            }
            
            self.results.append(benchmark_result)
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Error benchmarking query: {e}")
            benchmark_result = {
                "query": query,
                "success": False,
                "error": str(e)
            }
            self.results.append(benchmark_result)
            return benchmark_result
    
    def run_benchmark_suite(self, queries=None):
        """Run benchmark on a set of queries."""
        if queries is None:
            # Default medical queries
            queries = [
                "how to treat a tension pneumothorax",
                "symptoms of hemorrhagic shock",
                "battlefield tourniquet application",
                "chest decompression procedure",
                "managing sucking chest wound",
                "hemostatic agent application",
                "treating hypovolemic shock",
                "combat casualty assessment",
                "airway management in tactical setting",
                "battlefield triage priorities"
            ]
        
        logger.info(f"Running benchmark on {len(queries)} queries")
        
        for query in queries:
            self.benchmark_query(query)
        
        # Calculate aggregate statistics
        if self.results:
            success_results = [r for r in self.results if r.get("success", False)]
            if success_results:
                avg_time = sum(r["processing_time"] for r in success_results) / len(success_results)
                logger.info(f"Average query processing time: {avg_time:.4f} seconds")
                
                # Memory usage reporting
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
                except ImportError:
                    logger.warning("psutil not available, skipping memory usage reporting")
            
            return True
        
        return False

class LLMBenchmark:
    """Benchmark the LLM Analysis performance."""
    
    def __init__(self, config):
        """Initialize the benchmark."""
        self.config = config
        self.llm = None
        self.results = []
    
    def initialize(self):
        """Initialize the LLM."""
        try:
            self.llm = LLMAnalysis()
            success = self.llm.initialize(self.config["llm_analysis"])
            if not success:
                logger.error("Failed to initialize LLM Analysis")
                return False
            
            logger.info("LLM Analysis initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing LLM Analysis: {e}")
            return False
    
    def benchmark_prompt(self, prompt):
        """Benchmark LLM processing for a prompt."""
        try:
            logger.info(f"Processing prompt: {prompt}")
            
            # Process the prompt
            start_time = time.time()
            
            # Use the LLM to analyze the prompt
            result = self.llm.analyze_text(prompt)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            # Log results
            logger.info(f"LLM processing time: {processing_time:.4f} seconds")
            if result and "summary" in result:
                logger.info(f"Summary: {result['summary'][:100]}...")
            
            # Store results
            benchmark_result = {
                "prompt": prompt,
                "processing_time": processing_time,
                "output_length": len(result.get("summary", "")),
                "success": True
            }
            
            self.results.append(benchmark_result)
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Error benchmarking prompt: {e}")
            benchmark_result = {
                "prompt": prompt,
                "success": False,
                "error": str(e)
            }
            self.results.append(benchmark_result)
            return benchmark_result
    
    def run_benchmark_suite(self, prompts=None):
        """Run benchmark on a set of prompts."""
        if prompts is None:
            # Default medical prompts
            prompts = [
                "Patient presents with shortness of breath, decreased breath sounds on right side, and tracheal deviation to the left.",
                "Soldier has sustained a gunshot wound to the leg with bright red pulsating blood and is becoming confused.",
                "Casualty has burns covering approximately 30% of body surface area and is reporting severe pain.",
                "Field medic reports multiple patients with blast injuries, requesting triage priorities.",
                "Patient has an open chest wound that makes a sucking sound when breathing."
            ]
        
        logger.info(f"Running benchmark on {len(prompts)} prompts")
        
        for prompt in prompts:
            self.benchmark_prompt(prompt)
        
        # Calculate aggregate statistics
        if self.results:
            success_results = [r for r in self.results if r.get("success", False)]
            if success_results:
                avg_time = sum(r["processing_time"] for r in success_results) / len(success_results)
                logger.info(f"Average LLM processing time: {avg_time:.4f} seconds")
                
                # Memory usage reporting
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
                except ImportError:
                    logger.warning("psutil not available, skipping memory usage reporting")
            
            return True
        
        return False

class LiveMicrophoneTest:
    """Test live microphone capture and transcription."""
    
    def __init__(self, config):
        """Initialize the test."""
        self.config = config
        self.audio_pipeline = None
        self.stt_engine = None
        self.running = False
        self.audio_thread = None
        self.processing_thread = None
        self.audio_queue = queue.Queue()
    
    def initialize(self):
        """Initialize components."""
        try:
            # Initialize audio pipeline
            self.audio_pipeline = AudioPipeline()
            audio_success = self.audio_pipeline.initialize(self.config["audio_pipeline"])
            if not audio_success:
                logger.error("Failed to initialize Audio Pipeline")
                return False
            
            # Initialize STT engine
            self.stt_engine = STTEngine()
            stt_success = self.stt_engine.initialize(self.config["stt_engine"])
            if not stt_success:
                logger.error("Failed to initialize STT Engine")
                return False
            
            logger.info("Live microphone test initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing live microphone test: {e}")
            return False
    
    def audio_capture_thread(self):
        """Thread for capturing audio from the microphone."""
        try:
            logger.info("Starting audio capture")
            self.audio_pipeline.start()
            
            while self.running:
                # Get audio chunk
                audio_chunk = self.audio_pipeline.get_audio_segment(timeout_ms=100)
                if audio_chunk is not None:
                    self.audio_queue.put(audio_chunk)
                time.sleep(0.01)  # Small sleep to prevent tight loop
                
        except Exception as e:
            logger.error(f"Error in audio capture thread: {e}")
        finally:
            # Ensure audio pipeline is stopped
            if self.audio_pipeline:
                self.audio_pipeline.stop()
    
    def processing_thread_func(self):
        """Thread for processing audio and running STT."""
        try:
            logger.info("Starting audio processing")
            
            while self.running:
                # Get audio chunk from queue
                try:
                    audio_chunk = self.audio_queue.get(block=True, timeout=0.5)
                    
                    # Process with STT
                    if audio_chunk and "audio_data" in audio_chunk:
                        start_time = time.time()
                        result = self.stt_engine.transcribe_audio(audio_chunk["audio_data"])
                        processing_time = time.time() - start_time
                        
                        # Log results
                        if result and "text" in result and result["text"].strip():
                            logger.info(f"Transcription ({processing_time:.2f}s): {result['text']}")
                    
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # No audio data available
                    pass
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
    
    def start(self, duration_seconds=30):
        """Start the live microphone test."""
        if self.running:
            logger.warning("Live microphone test already running")
            return False
        
        if not self.audio_pipeline or not self.stt_engine:
            logger.error("Components not initialized")
            return False
        
        self.running = True
        
        # Start threads
        self.audio_thread = threading.Thread(target=self.audio_capture_thread)
        self.processing_thread = threading.Thread(target=self.processing_thread_func)
        
        self.audio_thread.start()
        self.processing_thread.start()
        
        logger.info(f"Live microphone test started (will run for {duration_seconds} seconds)")
        
        # Run for specified duration
        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            logger.info("Live microphone test interrupted by user")
        
        # Stop test
        self.stop()
        return True
    
    def stop(self):
        """Stop the live microphone test."""
        self.running = False
        
        # Wait for threads to terminate
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Ensure audio pipeline is stopped
        if self.audio_pipeline:
            self.audio_pipeline.stop()
        
        logger.info("Live microphone test stopped")
        return True

def report_jetson_stats():
    """Report Jetson-specific hardware stats."""
    try:
        # Check if running on Jetson
        if not os.path.exists('/etc/nv_tegra_release'):
            logger.info("Not running on Jetson hardware")
            return False
        
        logger.info("Collecting Jetson hardware stats")
        
        # GPU utilization
        gpu_util = "N/A"
        try:
            import subprocess
            result = subprocess.run(['cat', '/sys/devices/gpu.0/load'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util = result.stdout.strip()
                logger.info(f"GPU utilization: {gpu_util}")
        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")
        
        # CPU usage
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            logger.info(f"CPU utilization: {cpu_percent}")
            
            # Memory usage
            memory = psutil.virtual_memory()
            logger.info(f"Memory: {memory.percent}% used ({memory.used / (1024**2):.1f}MB / {memory.total / (1024**2):.1f}MB)")
        except ImportError:
            logger.warning("psutil not available, skipping CPU/memory stats")
        
        # Temperature
        try:
            temps = {}
            for i in range(10):  # Check thermal zones 0-9
                thermal_path = f'/sys/devices/virtual/thermal/thermal_zone{i}/temp'
                if os.path.exists(thermal_path):
                    with open(thermal_path, 'r') as f:
                        temp = int(f.read().strip()) / 1000  # Convert to degrees Celsius
                        temps[f'thermal_zone{i}'] = temp
            
            if temps:
                logger.info(f"Temperatures: {temps}")
        except Exception as e:
            logger.error(f"Error getting temperature: {e}")
        
        # Power consumption
        try:
            power_data = {}
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
                        power_data[f'channel{i}'] = {
                            'current': current,
                            'voltage': voltage,
                            'power': power
                        }
                
                if power_data:
                    logger.info(f"Power consumption: {power_data}")
        except Exception as e:
            logger.error(f"Error getting power consumption: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error reporting Jetson stats: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='TCCC Component Test for Jetson Hardware')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--stt', action='store_true', help='Run STT benchmark')
    parser.add_argument('--doc', action='store_true', help='Run Document Library benchmark')
    parser.add_argument('--llm', action='store_true', help='Run LLM benchmark')
    parser.add_argument('--mic', action='store_true', help='Run live microphone test')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--audio-dir', default='./test_data', help='Directory containing audio files for STT test')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds for live microphone test')
    parser.add_argument('--jetson-stats', action='store_true', help='Report Jetson hardware statistics')
    
    args = parser.parse_args()
    
    # If no tests specified, default to --all
    if not (args.stt or args.doc or args.llm or args.mic or args.all or args.jetson_stats):
        args.all = True
    
    # Load configuration
    config = load_config(args.config)
    
    # Report Jetson stats if requested
    if args.all or args.jetson_stats:
        report_jetson_stats()
    
    # Run STT benchmark
    if args.all or args.stt:
        logger.info("=== Running STT Benchmark ===")
        stt_benchmark = STTBenchmark(config)
        if stt_benchmark.initialize():
            stt_benchmark.run_benchmark_suite(audio_dir=args.audio_dir)
    
    # Run Document Library benchmark
    if args.all or args.doc:
        logger.info("=== Running Document Library Benchmark ===")
        doc_benchmark = DocumentBenchmark(config)
        if doc_benchmark.initialize():
            doc_benchmark.run_benchmark_suite()
    
    # Run LLM benchmark
    if args.all or args.llm:
        logger.info("=== Running LLM Benchmark ===")
        llm_benchmark = LLMBenchmark(config)
        if llm_benchmark.initialize():
            llm_benchmark.run_benchmark_suite()
    
    # Run live microphone test
    if args.all or args.mic:
        logger.info("=== Running Live Microphone Test ===")
        mic_test = LiveMicrophoneTest(config)
        if mic_test.initialize():
            mic_test.start(duration_seconds=args.duration)
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main()