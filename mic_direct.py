#!/usr/bin/env python3
"""
Direct microphone to STT to LLM pipeline.
Simplified for immediate execution without dependencies or configuration issues.
"""

import os
import sys
import time
import yaml

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Force mock mode for immediate execution
os.environ["USE_MOCK_STT"] = "1"
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{project_dir}/src"

# Import essential modules
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine 
from tccc.llm_analysis.llm_analysis import LLMAnalysis

# Load minimal configurations
with open(os.path.join(project_dir, 'config', 'audio_pipeline.yaml'), 'r') as f:
    audio_config = yaml.safe_load(f)

with open(os.path.join(project_dir, 'config', 'stt_engine.yaml'), 'r') as f:
    stt_config = yaml.safe_load(f)

with open(os.path.join(project_dir, 'config', 'llm_analysis.yaml'), 'r') as f:
    llm_config = yaml.safe_load(f)

# Set Razer microphone (device 0)
if 'io' not in audio_config:
    audio_config['io'] = {}
if 'input_sources' not in audio_config['io']:
    audio_config['io']['input_sources'] = []

mic_configured = False
for source in audio_config['io']['input_sources']:
    if source.get('type') == 'microphone':
        source['device_id'] = 0  # Razer Seiren
        mic_configured = True
        break

if not mic_configured:
    audio_config['io']['input_sources'].append({
        'name': 'razer_mic', 
        'type': 'microphone',
        'device_id': 0
    })
    audio_config['io']['default_input'] = 'razer_mic'

# Initialize components
print("\n===== TCCC.ai Pipeline Initialization =====")

# Initialize audio pipeline
audio_pipeline = AudioPipeline()
audio_pipeline.initialize(audio_config)

# Initialize STT engine
stt_engine = create_stt_engine("mock", stt_config)
stt_engine.initialize(stt_config)

# Initialize LLM analysis
llm = LLMAnalysis()
llm.initialize(llm_config)

# Get microphone source
mic_source = None
for source in audio_pipeline.get_available_sources():
    if source['type'] == 'microphone':
        mic_source = source['name']
        break

if not mic_source:
    print("No microphone source found. Using test file instead.")
    # Fall back to test file
    for source in audio_pipeline.get_available_sources():
        if source['type'] == 'file':
            mic_source = source['name']
            break

# Start capture
print(f"\nStarting audio capture from source: {mic_source}")
audio_pipeline.start_capture(mic_source)

print("\n===== Pipeline Active =====")
print("Speak into your microphone to see transcriptions")
print("Press Ctrl+C to stop\n")

# Process audio in a loop
try:
    while True:
        # Get audio
        audio_stream = audio_pipeline.get_audio_stream()
        if audio_stream:
            audio_data = audio_stream.read()
            if audio_data is not None and len(audio_data) > 0:
                # Transcribe
                result = stt_engine.transcribe_segment(audio_data)
                
                # Process if we have text
                if result and 'text' in result and result['text'].strip():
                    text = result['text']
                    print(f"\nTranscription: {text}")
                    
                    # Generate reports with LLM
                    try:
                        # Try different report types
                        print("\nGenerating medical report...")
                        report = llm.generate_zmist_report([{
                            "text": text, 
                            "timestamp": time.time()
                        }])
                        if report:
                            print("\n----- ZMIST Report from LLM -----")
                            print(report)
                            print("-"*40)
                    except Exception as e:
                        print(f"Error generating reports: {e}")
                        
        # Sleep to prevent CPU overuse
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Clean up
    audio_pipeline.stop_capture()
    audio_pipeline.shutdown()
    stt_engine.shutdown()
    print("\nPipeline stopped")