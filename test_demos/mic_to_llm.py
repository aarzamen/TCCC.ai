#!/usr/bin/env python3
"""
Microphone to LLM pipeline - minimal implementation.

This script establishes a direct pipeline from microphone input to LLM processing,
showing the complete architecture without any demo components.
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path

# Add the src directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Import TCCC modules
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine
from tccc.llm_analysis import LLMAnalysis
from tccc.document_library import DocumentLibrary
from tccc.data_store import DataStore

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TCCC.Microphone-LLM")

class MicrophoneToLLMPipeline:
    """Establishes a direct pipeline from microphone to LLM."""
    
    def __init__(self, stt_engine_type="faster-whisper", microphone_device=0):
        """Initialize the pipeline components."""
        self.microphone_device = microphone_device
        self.stt_engine_type = stt_engine_type
        
        # Component references
        self.audio_pipeline = None
        self.stt_engine = None
        self.llm_analysis = None
        self.document_library = None
        self.data_store = None
        
        # State tracking
        self.running = False
        self.transcriptions = []
        
    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing microphone-to-LLM pipeline...")
        
        # Load configurations
        import yaml
        with open(os.path.join(project_dir, 'config', 'audio_pipeline.yaml'), 'r') as f:
            audio_config = yaml.safe_load(f)
        
        with open(os.path.join(project_dir, 'config', 'stt_engine.yaml'), 'r') as f:
            stt_config = yaml.safe_load(f)
            
        with open(os.path.join(project_dir, 'config', 'llm_analysis.yaml'), 'r') as f:
            llm_config = yaml.safe_load(f)
            
        with open(os.path.join(project_dir, 'config', 'document_library.yaml'), 'r') as f:
            doc_config = yaml.safe_load(f)
            
        with open(os.path.join(project_dir, 'config', 'data_store.yaml'), 'r') as f:
            store_config = yaml.safe_load(f)
        
        # Configure microphone source
        if 'io' not in audio_config:
            audio_config['io'] = {}
        if 'input_sources' not in audio_config['io']:
            audio_config['io']['input_sources'] = []
        
        # Add or update microphone source
        mic_source_found = False
        for source in audio_config['io']['input_sources']:
            if source.get('type') == 'microphone':
                source['device_id'] = self.microphone_device
                mic_source_found = True
                break
        
        if not mic_source_found:
            audio_config['io']['input_sources'].append({
                'name': 'microphone',
                'type': 'microphone',
                'device_id': self.microphone_device
            })
        
        # Set default input to microphone
        audio_config['io']['default_input'] = 'microphone'
        
        # Initialize components
        logger.info("Initializing Audio Pipeline...")
        self.audio_pipeline = AudioPipeline()
        if not self.audio_pipeline.initialize(audio_config):
            logger.error("Failed to initialize Audio Pipeline")
            return False
        
        logger.info("Initializing STT Engine...")
        self.stt_engine = create_stt_engine(self.stt_engine_type, stt_config)
        if not self.stt_engine.initialize(stt_config):
            logger.error("Failed to initialize STT Engine")
            return False
        
        logger.info("Initializing Document Library...")
        self.document_library = DocumentLibrary()
        if not self.document_library.initialize(doc_config):
            logger.error("Failed to initialize Document Library")
            return False
        
        logger.info("Initializing Data Store...")
        self.data_store = DataStore()
        if not self.data_store.initialize(store_config):
            logger.error("Failed to initialize Data Store")
            return False
        
        logger.info("Initializing LLM Analysis...")
        self.llm_analysis = LLMAnalysis()
        if not self.llm_analysis.initialize(llm_config):
            logger.error("Failed to initialize LLM Analysis")
            return False
        
        # Link document library to LLM analysis
        if hasattr(self.llm_analysis, 'set_document_library'):
            self.llm_analysis.set_document_library(self.document_library)
        
        logger.info("All components initialized successfully")
        return True
    
    async def start(self):
        """Start the pipeline."""
        if not self.audio_pipeline or not self.stt_engine or not self.llm_analysis:
            logger.error("Pipeline components not initialized")
            return False
        
        # Display available audio sources
        sources = self.audio_pipeline.get_available_sources()
        logger.info("Available audio sources:")
        for source in sources:
            logger.info(f"  - {source['name']} ({source['type']})")
        
        # Find microphone source
        mic_source = None
        for source in sources:
            if source['type'] == 'microphone':
                mic_source = source['name']
                break
        
        if not mic_source:
            logger.error("No microphone source found")
            return False
            
        logger.info(f"Using microphone source: {mic_source}")
        
        # Start audio capture
        if not self.audio_pipeline.start_capture(mic_source):
            logger.error("Failed to start audio capture")
            return False
        
        self.running = True
        logger.info("Microphone-to-LLM pipeline started")
        
        # Main processing loop
        try:
            while self.running:
                # Get audio from pipeline
                audio_stream = self.audio_pipeline.get_audio_stream()
                if audio_stream:
                    audio_data = audio_stream.read()
                    if audio_data is not None and len(audio_data) > 0:
                        # Process the audio with STT
                        await self.process_audio(audio_data)
                
                # Sleep to prevent high CPU usage
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Stopping pipeline due to keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
        finally:
            await self.stop()
        
        return True
    
    async def process_audio(self, audio_data):
        """Process audio data through the pipeline."""
        # Transcribe audio
        result = self.stt_engine.transcribe_segment(audio_data)
        
        # Process only if there's text
        if result and 'text' in result and result['text'].strip():
            text = result['text']
            logger.info(f"Transcription: {text}")
            
            # Store transcription
            self.transcriptions.append(text)
            
            try:
                # Process with LLM
                analysis = self.llm_analysis.analyze_transcription(text)
                
                # Store in data store
                if analysis:
                    event_data = {
                        "type": "audio_transcription",
                        "text": text,
                        "analysis": analysis,
                        "timestamp": time.time()
                    }
                    event_id = self.data_store.store_event(event_data)
                    
                    # Display analysis results
                    logger.info(f"LLM Analysis: {json.dumps(analysis, indent=2)}")
            except Exception as e:
                logger.error(f"Error in LLM processing: {e}")
    
    async def stop(self):
        """Stop the pipeline."""
        self.running = False
        
        # Stop audio capture
        if self.audio_pipeline:
            self.audio_pipeline.stop_capture()
        
        # Shutdown components
        if self.llm_analysis:
            self.llm_analysis.shutdown()
        
        if self.document_library:
            self.document_library.shutdown()
            
        if self.data_store:
            self.data_store.shutdown()
            
        if self.stt_engine:
            self.stt_engine.shutdown()
            
        if self.audio_pipeline:
            self.audio_pipeline.shutdown()
            
        logger.info("Pipeline stopped")
        return True

async def main():
    """Main function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Microphone to LLM pipeline")
    parser.add_argument("--engine", choices=["mock", "faster-whisper", "whisper"], default="faster-whisper",
                        help="STT engine type to use")
    parser.add_argument("--device", type=int, default=0, help="Microphone device ID to use")
    parser.add_argument("--list-microphones", action="store_true", help="List available microphones and exit")
    args = parser.parse_args()
    
    # Handle microphone listing
    if args.list_microphones:
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            print("\n===== Available Audio Devices =====")
            info = p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            for i in range(numdevices):
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    print(f"Device ID {i}: {device_info.get('name')}")
            
            p.terminate()
            return 0
        except ImportError:
            print("PyAudio not installed. Cannot list microphones.")
            return 1
    
    # Configure mock if requested
    if args.engine == "mock":
        os.environ["USE_MOCK_STT"] = "1"
    else:
        os.environ["USE_MOCK_STT"] = "0"
    
    # Create and run pipeline
    pipeline = MicrophoneToLLMPipeline(args.engine, args.device)
    initialized = await pipeline.initialize()
    
    if initialized:
        print("\n===== TCCC.ai Microphone to LLM Pipeline =====")
        print("Speak into your microphone to see the complete pipeline in action.")
        print("Press Ctrl+C to stop.\n")
        await pipeline.start()
    else:
        print("Failed to initialize pipeline")
        
    return 0

if __name__ == "__main__":
    asyncio.run(main())