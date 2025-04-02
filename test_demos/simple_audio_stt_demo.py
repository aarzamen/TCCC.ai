#!/usr/bin/env python3
"""
TCCC.ai Simple Audio Pipeline to STT Demo

This script demonstrates the basic flow of capturing audio via the AudioPipeline,
processing it, and transcribing it using the STTEngine.

-----------------------------------------------------------------------
Metadata:
    Created By:     Cascade (via Windsurf)
    Date Created:   2025-04-01
    Project:        TCCC.ai
    Relevant Task:  Testing real AudioPipeline -> STTEngine integration
    Origin:         User requested a simpler alternative to the full 
                    verification_script_system_enhanced.py for testing 
                    microphone input to transcription output. Required 
                    clearer terminal feedback for status.
-----------------------------------------------------------------------

Usage:
  - Ensure config files (audio_pipeline.yaml, stt_engine.yaml) are present
    in ../config relative to the project root.
  - Run from the project's test_demos directory:
    python simple_audio_stt_demo.py
  - Press Ctrl+C to stop.
"""

import os
import sys
import asyncio
import logging
import signal
import time

# Add project root to path to allow importing TCCC modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go one level up from script dir to get project root
sys.path.insert(0, project_root)

try:
    from tccc.audio_pipeline import AudioPipeline
    from tccc.stt_engine import STTEngine
    from tccc.utils import ConfigManager
    from tccc.processing_core.processing_core import ModuleState # Ensure ModuleState is available
except ImportError as e:
    print(f"Error importing TCCC modules: {e}")
    print("Ensure the script is run from the project root or the TCCC library is installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioSTTDemo")

# --- Configuration ---
# Use ConfigManager to load settings from yaml files
CONFIG_DIR = os.path.join(project_root, 'config')
print(f"Attempting to load configuration from: {CONFIG_DIR}") # Added print
config_manager = ConfigManager(config_dir=CONFIG_DIR)
try:
    AUDIO_CONFIG = config_manager.load_config('audio_pipeline.yaml')
    STT_CONFIG = config_manager.load_config('stt_engine.yaml')
    # Ensure the 'io' section and 'default_input' exist in AUDIO_CONFIG
    if 'io' not in AUDIO_CONFIG or 'default_input' not in AUDIO_CONFIG['io']:
         raise KeyError("Configuration 'audio_pipeline.yaml' missing 'io.default_input'")
    INPUT_SOURCE_NAME = AUDIO_CONFIG['io']['default_input']
    print("Configuration loaded successfully.") # Added print

except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {e}. Ensure config files exist in {CONFIG_DIR}")
    sys.exit(1)
except KeyError as e:
     logger.error(f"Missing key in configuration: {e}")
     sys.exit(1)
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    sys.exit(1)


# --- Global Variables ---
stop_event = asyncio.Event()
is_listening = False # Flag to track listening state

# --- Signal Handler ---
def handle_signal(sig, frame):
    logger.info("Signal received, initiating shutdown...")
    stop_event.set()

# --- Main Demo Logic ---
async def run_demo():
    """Initializes components and runs the audio processing loop."""
    global is_listening # Allow modification of the global flag
    audio_pipeline = None
    stt_engine = None

    try:
        # --- Initialize Components ---
        print("\n" + "="*20 + " INITIALIZING " + "="*20) # Added header
        logger.info("Initializing Audio Pipeline...")
        audio_pipeline = AudioPipeline(config=AUDIO_CONFIG)
        if not audio_pipeline.initialize():
            logger.error("Failed to initialize Audio Pipeline.")
            return
        logger.info(">>> Audio Pipeline Initialized <<<") # Enhanced message

        logger.info("Initializing STT Engine...")
        stt_engine = STTEngine(config=STT_CONFIG)
        if not stt_engine.initialize():
            logger.error("Failed to initialize STT Engine.")
            return
        logger.info(">>> STT Engine Initialized <<<") # Enhanced message

        # --- Start Capture ---
        print("\n" + "="*20 + " STARTING CAPTURE " + "="*16) # Added header
        logger.info(f"Starting audio capture from source: {INPUT_SOURCE_NAME}...")
        # Assuming start_capture takes the source name, adjust if needed based on AudioPipeline API
        if not audio_pipeline.start_capture(source_name=INPUT_SOURCE_NAME):
            logger.error("Failed to start audio capture.")
            return
        logger.info(">>> Audio capture started. Speak into the microphone. Press Ctrl+C to stop. <<<")
        is_listening = True # Set flag
        print("\n" + "="*20 + " LISTENING... " + "="*20) # Added header


        # --- Processing Loop ---
        while not stop_event.is_set():
            # Optional: Add a visual indicator that it's actively looping/listening
            # print(".", end="", flush=True) # Uncomment for a simple dot indicator

            audio_chunk = await audio_pipeline.get_processed_audio_chunk() # Or appropriate method

            if audio_chunk:
                if is_listening:
                    print("\n" + "="*20 + " PROCESSING AUDIO " + "="*16) # Indicate processing start
                    is_listening = False # Reset flag once audio is detected

                try:
                    # Assume audio_chunk has data, timestamp, sample_rate etc. or is raw bytes
                    # Adjust the call based on what STTEngine expects
                    # If it expects raw bytes: stt_engine.transcribe_segment(audio_chunk.data, ...)
                    # If it expects a specific object type, ensure audio_pipeline provides it.
                    # This might require adjustment based on actual AudioPipeline output format.

                    # Placeholder: Using a generic call, assuming transcribe_segment handles the chunk object
                    # You might need to adapt this based on the actual API
                    logger.debug(f"Received audio chunk, size: {len(getattr(audio_chunk, 'data', audio_chunk))} bytes") # Log chunk size
                    transcription_result = await stt_engine.transcribe_segment(audio_chunk)

                    if transcription_result and transcription_result.get('text'):
                        text = transcription_result['text'].strip()
                        confidence = transcription_result.get('confidence', 'N/A')
                        if text: # Avoid printing empty transcriptions
                             # Clearer output format
                             print("-" * 60)
                             print(f" DETECTED SPEECH: '{text}'")
                             try:
                                 print(f"    Confidence: {float(confidence):.2f}")
                             except (ValueError, TypeError):
                                 print(f"    Confidence: {confidence}") # Print as is if not float
                             print("-" * 60)
                             # Reset flag to show listening again after transcription
                             if not is_listening:
                                 print("\n" + "="*20 + " LISTENING... " + "="*20)
                                 is_listening = True
                    elif transcription_result:
                        logger.warning(f"STT returned a result but no text: {transcription_result}")
                    else:
                        # Potentially normal if VAD is active and there's silence
                        logger.debug("No transcription result (silence or VAD active?)")

                except Exception as e:
                    logger.error(f"Error during transcription: {e}", exc_info=True)
                    # Optional: Add a small delay to prevent spamming errors
                    await asyncio.sleep(0.1)
            else:
                 # If no chunk and not already showing listening, show it
                 if not is_listening:
                    print("\n" + "="*20 + " LISTENING... " + "="*20)
                    is_listening = True
                 # No chunk available, wait briefly before checking again
                 await asyncio.sleep(0.05) # 50ms sleep

    except asyncio.CancelledError:
        logger.info("Task cancelled.")
    except Exception as e:
        logger.error(f"An error occurred during the demo: {e}", exc_info=True)
    finally:
        # --- Shutdown ---
        print("\n" + "="*20 + " SHUTTING DOWN " + "="*19) # Added header
        logger.info("Shutting down...")
        if audio_pipeline and audio_pipeline.get_status()['capturing']:
            logger.info("Stopping audio capture...")
            audio_pipeline.stop_capture()
        if audio_pipeline:
            logger.info("Shutting down Audio Pipeline...")
            audio_pipeline.shutdown()
        if stt_engine:
            logger.info("Shutting down STT Engine...")
            stt_engine.shutdown()
        logger.info("Demo finished.")
        print("=" * 55) # Footer


# --- Entry Point ---
if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("--- Starting Simple Audio -> STT Demo ---")
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping demo.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
