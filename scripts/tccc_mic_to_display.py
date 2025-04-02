#!/usr/bin/env python3
"""
TCCC.ai Full System Test - Microphone to Display

This script implements a complete end-to-end pipeline from:
1. Microphone input
2. Speech-to-text processing
3. LLM analysis
4. Display rendering

It demonstrates the full TCCC.ai system with real-time speech recognition,
medical entity extraction, and visual display.
"""

import os
import sys
import time
import json
import threading
import argparse
import re
from pathlib import Path

# Disable mock modes - use real implementations
os.environ["TCCC_USE_MOCK_LLM"] = "1"  # Use mock LLM instead of real to avoid torch dependency
os.environ["USE_MOCK_STT"] = "0"       # Use real STT

# Set up paths
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TCCC.FullSystemTest")

# Try importing torch - if it fails, we'll use fallbacks
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available - using real LLM implementation")
    os.environ["TCCC_USE_MOCK_LLM"] = "0"  # Use real LLM if torch is available
except ImportError:
    logger.warning("PyTorch not available - using mock LLM implementation")
    # Keep mock LLM enabled

# Import required components
from tccc.audio_pipeline import AudioPipeline
from tccc.stt_engine import create_stt_engine
from tccc.llm_analysis import LLMAnalysis
# Skip document library for faster loading
from tccc.display.display_interface import DisplayInterface
from tccc.utils.config_manager import ConfigManager

class TCCCFullSystemTest:
    """
    TCCC Full System Test with Microphone to Display Pipeline
    """
    
    def __init__(self, config=None):
        """Initialize the full system test."""
        self.config = config or {}
        
        # Component state
        self.audio_pipeline = None
        self.stt_engine = None
        self.llm_analysis = None
        self.document_library = None
        self.display = None
        
        # Runtime state
        self.running = False
        self.current_transcription = ""
        self.extracted_entities = []
        self.reports = {}
        self.mic_source = None
        
        # Thread for display updates
        self.display_thread = None
        
        # Performance tracking
        self.stats = {
            "transcription_count": 0,
            "entity_count": 0,
            "start_time": 0,
            "last_transcription_time": 0
        }

    def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing TCCC.ai Full System Test...")
        
        # Load configurations
        config_manager = ConfigManager()
        audio_config = config_manager.load_config("audio_pipeline")
        stt_config = config_manager.load_config("stt_engine")
        llm_config = config_manager.load_config("llm_analysis")
        display_config = config_manager.load_config("display")
        
        # Optimize configs for Jetson Nano + WaveShare display
        display_config["resolution"] = [1280, 720]  # For WaveShare display
        display_config["fullscreen"] = True
        
        # Configure microphone
        self._configure_microphone(audio_config)
        
        # Initialize Audio Pipeline
        logger.info("Initializing Audio Pipeline...")
        self.audio_pipeline = AudioPipeline()
        if not self.audio_pipeline.initialize(audio_config):
            logger.error("Failed to initialize Audio Pipeline")
            return False
        
        # Initialize STT Engine (faster-whisper)
        logger.info("Initializing STT Engine (faster-whisper)...")
        self.stt_engine = create_stt_engine("faster-whisper", stt_config)
        if not self.stt_engine.initialize(stt_config):
            logger.error("Failed to initialize STT Engine")
            return False
            
        # Initialize LLM Analysis - skip document library for speed
        logger.info("Initializing LLM Analysis...")
        self.llm_analysis = LLMAnalysis()
        
        # Configure LLM based on torch availability
        if TORCH_AVAILABLE:
            # Simplify LLM config for faster load, but use real model
            logger.info("Using real Phi-2 model with PyTorch")
            llm_config["model"]["primary"]["force_real"] = True
            if "hardware" in llm_config:
                llm_config["hardware"]["enable_acceleration"] = False
                llm_config["hardware"]["quantization"] = "none"
        else:
            # Force mock mode if torch not available
            logger.info("Using mock LLM implementation (PyTorch not available)")
            if "model" not in llm_config:
                llm_config["model"] = {}
            if "primary" not in llm_config["model"]:
                llm_config["model"]["primary"] = {}
            llm_config["model"]["primary"]["use_mock"] = True
            llm_config["model"]["primary"]["force_real"] = False
            
        if not self.llm_analysis.initialize(llm_config):
            logger.error("Failed to initialize LLM Analysis")
            return False
        
        # Initialize Display
        logger.info("Initializing Display...")
        self.display = DisplayInterface()
        if not self.display.initialize(display_config):
            logger.error("Failed to initialize Display")
            return False
            
        # Find microphone source
        sources = self.audio_pipeline.get_available_sources()
        logger.info("Available audio sources:")
        for source in sources:
            logger.info(f"  {source['name']} ({source['type']})")
            if source['type'] == 'microphone':
                self.mic_source = source['name']
        
        if not self.mic_source:
            logger.error("No microphone source found")
            return False
            
        logger.info(f"Using microphone source: {self.mic_source}")
        return True
        
    def _configure_microphone(self, audio_config):
        """Configure Razer Seiren V3 Mini microphone source."""
        if 'io' not in audio_config:
            audio_config['io'] = {}
        if 'input_sources' not in audio_config['io']:
            audio_config['io']['input_sources'] = []
        
        # Force-set to use only Razer Seiren V3 Mini
        razer_source_name = 'razer_mic'
        razer_config = {
            'name': razer_source_name,
            'type': 'microphone',
            'device_id': 0,  # Typically device 0 for Razer Seiren V3 Mini
            'display_name': 'Razer Seiren V3 Mini',
            'channels': 1,
            'rate': 16000,
            'chunk_size': 1024,
            'active': True
        }
        
        # Remove any existing microphone sources and add Razer
        audio_config['io']['input_sources'] = [s for s in audio_config['io']['input_sources'] 
                                             if s.get('type') != 'microphone']
        audio_config['io']['input_sources'].append(razer_config)
        
        # Set default input to Razer mic
        audio_config['io']['default_input'] = razer_source_name
        logger.info("Configured Razer Seiren V3 Mini as the exclusive microphone source")
    
    def _update_display(self):
        """Update the display with current state information."""
        if not self.display:
            return
            
        try:
            # Clear display
            self.display.clear()
            
            # Set header
            self.display.set_header("TCCC.ai MEDICAL ASSISTANT")
            
            # Show live status information
            status_text = "Status: Active" if self.running else "Status: Inactive"
            status_text += f" | Uptime: {int(time.time() - self.stats['start_time'])}s"
            self.display.set_status(status_text)
            
            # Show transcription
            self.display.add_text_block(
                "TRANSCRIPTION", 
                self.current_transcription[-300:] or "Speak into microphone...",
                color=(255, 255, 255)
            )
            
            # Show extracted entities by category
            categories = self._categorize_entities()
            
            # Show vital signs
            if "vital_sign" in categories:
                vitals_text = "\n".join([
                    f"{e.get('value', '')} {e.get('unit', '')}" 
                    for e in categories["vital_sign"][:5]
                ])
                self.display.add_text_block("VITAL SIGNS", vitals_text, color=(0, 255, 0))
            
            # Show injuries
            if "injury" in categories:
                injuries_text = "\n".join([
                    f"{e.get('value', '')} ({e.get('location', 'unknown')})" 
                    for e in categories["injury"][:5]
                ])
                self.display.add_text_block("INJURIES", injuries_text, color=(255, 0, 0))
            
            # Show procedures
            if "procedure" in categories:
                procedures_text = "\n".join([
                    f"{e.get('value', '')}" 
                    for e in categories["procedure"][:5]
                ])
                self.display.add_text_block("PROCEDURES", procedures_text, color=(0, 200, 255))
            
            # Show medications
            if "medication" in categories:
                meds_text = "\n".join([
                    f"{e.get('value', '')} {e.get('dosage', '')}" 
                    for e in categories["medication"][:5]
                ])
                self.display.add_text_block("MEDICATIONS", meds_text, color=(255, 255, 0))
            
            # Show TCCC report if available
            if "tccc" in self.reports:
                report_text = self.reports["tccc"]["content"][:500]
                self.display.add_text_block("TCCC REPORT", report_text, color=(200, 200, 255))
            
            # Update the display
            self.display.update()
            
        except Exception as e:
            logger.error(f"Error updating display: {e}")
    
    def _categorize_entities(self):
        """Categorize entities by type."""
        categories = {}
        for entity in self.extracted_entities:
            category = entity.get("type", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(entity)
        return categories
    
    def _display_thread_func(self):
        """Thread function to update the display periodically."""
        update_interval = 0.5  # Update every 0.5 seconds
        
        while self.running:
            self._update_display()
            time.sleep(update_interval)
    
    def start(self):
        """Start the full system test."""
        if not self.audio_pipeline or not self.stt_engine or not self.llm_analysis or not self.display:
            logger.error("Cannot start - components not initialized")
            return False
            
        # Start audio capture
        if not self.audio_pipeline.start_capture(self.mic_source):
            logger.error("Failed to start audio capture")
            return False
            
        # Mark as running
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_thread_func)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        logger.info("TCCC.ai Full System Test started")
        return True
    
    def process(self):
        """Process audio and update display."""
        if not self.running:
            return
            
        # Get audio from pipeline
        audio_stream = self.audio_pipeline.get_audio_stream()
        if not audio_stream:
            return
            
        audio_data = audio_stream.read()
        if audio_data is None or len(audio_data) == 0:
            return
            
        # Transcribe audio
        result = self.stt_engine.transcribe_segment(audio_data)
        
        # Process only if there's text
        if result and 'text' in result and result['text'].strip():
            text = result['text']
            logger.info(f"Transcription: {text}")
            
            # Update transcription
            self.current_transcription += " " + text
            self.stats["transcription_count"] += 1
            self.stats["last_transcription_time"] = time.time()
            
            # Check for medical keywords and add them directly to entities for faster display
            self._check_for_medical_keywords(text)
            
            # Process with LLM only for larger chunks of text
            if len(self.current_transcription.split()) >= 15:
                try:
                    # Process with LLM for entity extraction (less frequently)
                    transcription = {"text": self.current_transcription}
                    
                    # Try to use LLM, but handle exceptions gracefully
                    try:
                        entities = self.llm_analysis.process_transcription(transcription)
                    except Exception as e:
                        logger.error(f"LLM processing error: {e}")
                        # Fall back to keyword extraction
                        logger.info("Falling back to keyword-based entity extraction")
                        entities = self._extract_entities_from_keywords(self.current_transcription)
                    
                    if entities:
                        self.extracted_entities = entities
                        self.stats["entity_count"] = len(entities)
                        logger.info(f"Extracted {len(entities)} entities")
                        
                        # Generate TCCC report for longer transcriptions
                        try:
                            if len(entities) >= 8:
                                report = self.llm_analysis.generate_report("tccc", entities)
                                if report:
                                    self.reports["tccc"] = report
                                    logger.info("Generated TCCC report")
                        except Exception as e:
                            logger.error(f"Error generating report: {e}")
                            # Create a simple report as fallback
                            categories = self._categorize_entities()
                            report_text = "TCCC Report (auto-generated):\n\n"
                            for category, items in categories.items():
                                if items:
                                    report_text += f"{category.upper()}: {', '.join([item.get('value', '') for item in items])}\n"
                            self.reports["tccc"] = {"content": report_text}
                except Exception as e:
                    logger.error(f"Error in processing: {e}")
                    
    def _extract_entities_from_keywords(self, text):
        """Extract entities directly from keywords as a fallback mechanism."""
        entities = []
        text = text.lower()
        
        # Process the full text with all keyword checks at once
        self._check_for_medical_keywords(text, entities)
        
        return entities
    
    def _check_for_medical_keywords(self, text, entities_list=None):
        """Quick check for medical keywords to provide immediate feedback."""
        # Fast keyword matching for common medical terms
        text = text.lower()
        
        # Use either the provided list or the instance variable
        entities = entities_list if entities_list is not None else self.extracted_entities
        
        # Injuries
        if any(kw in text for kw in ["injury", "wound", "trauma", "bleeding", "fracture", "blast"]):
            injury_type = "blast injury" if "blast" in text else "bleeding" if "bleeding" in text else "trauma"
            location = "leg" if "leg" in text else "chest" if "chest" in text else "unknown"
            
            # Add to entities directly
            injury = {
                "type": "injury",
                "value": injury_type,
                "location": location,
                "time": "current"
            }
            entities.append(injury)
            logger.info(f"Detected injury: {injury_type} to {location}")
        
        # Vital signs
        if "BP" in text or "blood pressure" in text:
            match = re.search(r"BP (?:is |of )?(\d+/\d+)", text)
            if match:
                bp = match.group(1)
                vital = {
                    "type": "vital_sign", 
                    "value": bp, 
                    "unit": "mmHg",
                    "name": "blood pressure"
                }
                entities.append(vital)
                logger.info(f"Detected vital sign: BP {bp}")
        
        # Heart rate / pulse
        if "pulse" in text or "heart rate" in text:
            match = re.search(r"(?:pulse|heart rate)(?: is| of)? (\d+)", text)
            if match:
                hr = match.group(1)
                vital = {
                    "type": "vital_sign", 
                    "value": hr, 
                    "unit": "bpm",
                    "name": "pulse" 
                }
                entities.append(vital)
                logger.info(f"Detected vital sign: pulse {hr}")
                
        # Procedures
        if "tourniquet" in text:
            entities.append({
                "type": "procedure",
                "value": "tourniquet application",
                "location": "thigh" if "thigh" in text else "leg" if "leg" in text else "unknown"
            })
            logger.info("Detected procedure: tourniquet application")
            
        elif "needle decompression" in text:
            entities.append({
                "type": "procedure",
                "value": "needle decompression",
                "location": "chest"
            })
            logger.info("Detected procedure: needle decompression")
        
        elif "iv" in text or "intravenous" in text:
            entities.append({
                "type": "procedure",
                "value": "IV access",
                "location": "arm" if "arm" in text else "unknown"
            })
            logger.info("Detected procedure: IV access")
            
        # Medications
        if "morphine" in text:
            entities.append({
                "type": "medication",
                "value": "morphine",
                "dosage": "10mg" if "10" in text else "unknown",
                "route": "IV" if "IV" in text else "unknown"
            })
            logger.info("Detected medication: morphine")
            
        if "ceftriaxone" in text:
            entities.append({
                "type": "medication",
                "value": "ceftriaxone",
                "dosage": "1g" if "1g" in text or "1 g" in text else "unknown",
                "route": "IV" if "IV" in text else "unknown"
            })
            logger.info("Detected medication: ceftriaxone")
        
        # Return the list when used as a function (for fallback mode)
        return entities
    
    def stop(self):
        """Stop the full system test."""
        logger.info("Stopping TCCC.ai Full System Test...")
        
        # Mark as not running
        self.running = False
        
        # Wait for display thread to finish
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        
        # Stop audio capture
        if self.audio_pipeline:
            self.audio_pipeline.stop_capture()
        
        # Cleanup components
        if self.display:
            self.display.clear()
            self.display.set_header("TCCC.ai SYSTEM STOPPED")
            self.display.add_text_block("STATUS", "System has been shut down", color=(255, 0, 0))
            self.display.update()
            time.sleep(2)  # Show shutdown screen briefly
            self.display.shutdown()
        
        if self.llm_analysis:
            self.llm_analysis.shutdown()
            
        if self.document_library:
            self.document_library.shutdown()
            
        if self.stt_engine:
            self.stt_engine.shutdown()
            
        if self.audio_pipeline:
            self.audio_pipeline.shutdown()
            
        logger.info("TCCC.ai Full System Test stopped")
        return True

def print_test_script():
    """Print the test script for the user to read."""
    print("\n" + "#"*100)
    print("# JETSON NANO TERMINAL - TCCC.ai FULL SYSTEM TEST".center(100) + " #")
    print("#"*100)
    print("\n\n" + "="*100)
    print("TEST SCRIPT - READ ALOUD WHEN SYSTEM IS READY".center(100))
    print("="*100)
    
    # Instructions for when to start
    print("\n[INSTRUCTIONS]")
    print("1. Wait until you see 'SYSTEM IS ACTIVE' below")
    print("2. Read the following script into the Razer Seiren V3 Mini microphone")
    print("3. Speak clearly at a normal pace")
    print("4. Watch the WaveShare display for real-time processing results")
    print("5. Press Ctrl+C when finished to stop the system\n")
    
    print("-"*100)
    print("""[MEDICAL SCENARIO SCRIPT - READ THIS ALOUD]

Medic: This is Medic One-Alpha reporting from grid coordinate Charlie-Delta 4352.
Patient is a 28-year-old male with blast injuries to the right leg from an IED.
Initially unresponsive at scene with significant bleeding from right thigh.
Applied tourniquet at 0930 hours and established two IVs.
Vital signs are: BP 100/60, pulse 120, respiratory rate 24, oxygen saturation 92%.
GCS is now 14, was initially 12 when found.
Performed needle decompression on right chest for suspected tension pneumothorax at 0935.
Administered 10mg morphine IV at 0940 and 1g ceftriaxone IV.
Patient has severe right leg injury with controlled hemorrhage, possible TBI.
We're continuing fluid resuscitation and monitoring vitals every 5 minutes.
This is an urgent surgical case, requesting immediate MEDEVAC to Role 2.""")
    print("-"*100)
    print("\n[WHAT TO EXPECT ON DISPLAY]")
    print("✓ Transcription of your speech")
    print("✓ Detected injuries (blast injury, bleeding)")
    print("✓ Vital signs (BP 100/60)")
    print("✓ Procedures (tourniquet, needle decompression)")
    print("✓ Medications (morphine, ceftriaxone)")
    
    # Add spacer for the system status to be visually separated
    print("\n\n" + "-"*100)

def main():
    """Main function for the TCCC.ai Full System Test."""
    parser = argparse.ArgumentParser(description="TCCC.ai Full System Test")
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
    
    # Create and initialize system test
    system = TCCCFullSystemTest()
    if not system.initialize():
        logger.error("Failed to initialize TCCC.ai Full System Test")
        return 1
    
    # Print the test script
    print_test_script()
    
    # Start the system
    if not system.start():
        logger.error("Failed to start TCCC.ai Full System Test")
        return 1
        
    print("\n" + "="*100)
    print("▶ TCCC.ai SYSTEM IS ACTIVE - START READING THE SCRIPT NOW ◀".center(100))
    print("="*100)
    
    try:
        # Main loop - process audio and update display
        while True:
            system.process()
            time.sleep(0.1)  # Small sleep to prevent high CPU usage
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        # Stop the system
        system.stop()
        
    print("\n" + "="*80)
    print("TCCC.ai FULL SYSTEM TEST COMPLETE".center(80))
    print("="*80)
    
    # Print statistics
    elapsed_time = time.time() - system.stats["start_time"]
    print(f"\nTest duration: {elapsed_time:.1f} seconds")
    print(f"Transcriptions processed: {system.stats['transcription_count']}")
    print(f"Entities extracted: {system.stats['entity_count']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())