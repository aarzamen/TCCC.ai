#!/usr/bin/env python3

import os
import sys
import yaml
import time
import json
import subprocess
from pathlib import Path

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Setup environment
os.environ["USE_MOCK_STT"] = "1"  # Use mock STT for reliability
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{project_dir}/src"

# Import key components
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine 
from tccc.llm_analysis import LLMAnalysis

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TCCC.Pipeline")

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configurations
audio_config = load_yaml_config(os.path.join(project_dir, 'config', 'audio_pipeline.yaml'))
stt_config = load_yaml_config(os.path.join(project_dir, 'config', 'stt_engine.yaml'))
llm_config = load_yaml_config(os.path.join(project_dir, 'config', 'llm_analysis.yaml'))

# Configure microphone (Razer Seiren V3 Mini)
if 'io' not in audio_config:
    audio_config['io'] = {}
if 'input_sources' not in audio_config['io']:
    audio_config['io']['input_sources'] = []

# Set up the Razer microphone source
razer_source_found = False
for source in audio_config['io']['input_sources']:
    if source.get('type') == 'microphone':
        source['device_id'] = 0  # Razer Seiren V3 Mini
        razer_source_found = True
        break

if not razer_source_found:
    audio_config['io']['input_sources'].append({
        'name': 'razer_mic',
        'type': 'microphone',
        'device_id': 0
    })

# Set default input to Razer microphone
audio_config['io']['default_input'] = 'razer_mic'

# Initialize components
print("\n===== Starting TCCC Microphone Pipeline =====")
print("Initializing components...\n")

# Initialize Audio Pipeline
audio_pipeline = AudioPipeline()
if not audio_pipeline.initialize(audio_config):
    print("Failed to initialize Audio Pipeline")
    sys.exit(1)

# Initialize STT Engine
stt_engine = create_stt_engine("mock", stt_config)
if not stt_engine.initialize(stt_config):
    print("Failed to initialize STT Engine")
    sys.exit(1)

# Initialize LLM Analysis
llm_analysis = LLMAnalysis()
if not llm_analysis.initialize(llm_config):
    print("Failed to initialize LLM Analysis")
    sys.exit(1)

# Start audio capture
print("\nStarting audio capture from Razer Seiren V3 Mini...")
mic_source = None
for source in audio_pipeline.get_available_sources():
    if source['type'] == 'microphone':
        mic_source = source['name']
        break

if not mic_source:
    print("Error: No microphone source found")
    sys.exit(1)

if not audio_pipeline.start_capture(mic_source):
    print("Failed to start audio capture")
    sys.exit(1)

print("\n===== TCCC.ai Microphone Pipeline Active =====")
print("Speak into your Razer Seiren V3 Mini to see the pipeline in action")
print("Press Ctrl+C to stop\n")

# Main processing loop
try:
    while True:
        # Get audio from pipeline
        audio_stream = audio_pipeline.get_audio_stream()
        if audio_stream:
            audio_data = audio_stream.read()
            if audio_data is not None and len(audio_data) > 0:
                # Transcribe the audio
                result = stt_engine.transcribe_segment(audio_data)
                
                if result and 'text' in result and result['text'].strip():
                    text = result['text']
                    print(f"\nTranscription: {text}")
                    
                    # Process with LLM
                    try:
                        analysis = llm_analysis.analyze_transcription(text)
                        if analysis:
                            print("\nLLM Analysis:")
                            for key, value in analysis.items():
                                if isinstance(value, dict) or isinstance(value, list):
                                    print(f"  {key}: {json.dumps(value, indent=2)}")
                                else:
                                    print(f"  {key}: {value}")
                            print("\n" + "-"*50)
                    except Exception as e:
                        print(f"Error in LLM processing: {e}")
        
        # Sleep to prevent high CPU usage
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopping pipeline...")
finally:
    # Clean up
    audio_pipeline.stop_capture()
    audio_pipeline.shutdown()
    stt_engine.shutdown()
    llm_analysis.shutdown()
    print("\nPipeline stopped. Exiting.")