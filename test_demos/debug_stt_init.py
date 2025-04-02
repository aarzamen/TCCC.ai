
import sys
import os
import logging
import traceback
from datetime import datetime

# Add src directory to Python path to allow imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STTDebug")

# --- Attempt to import STTEngine ---
try:
    from tccc.stt_engine.stt_engine import STTEngine, ModelManager # Import ModelManager too for type hints if needed
    logger.info("Successfully imported STTEngine and ModelManager")
except ImportError as e:
    logger.error(f"Failed to import STTEngine or ModelManager: {e}")
    logger.error("Make sure 'src' directory is in PYTHONPATH or script is run from project root.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred during import: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# --- Define Default STT Config ---
DEFAULT_STT_CONFIG = {
    "model": {
        "type": "whisper",
        "size": "tiny", # Using the small model for testing
        "path": "models/stt",
        "language": "en",
        "beam_size": 5
    },
    "hardware": {
        "enable_acceleration": False, # Keep False for CPU testing
        "cuda_device": -1,
        "use_tensorrt": False,
        "quantization": "none"
    },
    "streaming": {
        "enabled": False, # Disable streaming for simple init test
        "max_context_length_sec": 30
    },
    "diarization": {
        "enabled": False # Disable diarization for simple init test
    },
    "vad_filter": True
}

# --- Main Debug Logic ---
if __name__ == "__main__":
    logger.info("Starting STT Engine Initialization Test...")
    stt_engine_instance = None
    initialization_successful = False

    try:
        logger.info(f"Creating STTEngine instance...")
        # Pass the config directly during instantiation or separately to initialize
        # Assuming STTEngine constructor doesn't require config, and initialize does
        stt_engine_instance = STTEngine()
        logger.info("STTEngine instance created.")

        logger.info(f"Initializing STTEngine with config: {DEFAULT_STT_CONFIG}")
        initialization_successful = stt_engine_instance.initialize(DEFAULT_STT_CONFIG)

        if initialization_successful:
            logger.info("STT Engine INITIALIZATION SUCCEEDED according to initialize() return value.")
            # Optionally, check status as well
            status = stt_engine_instance.get_status()
            logger.info(f"STT Engine status after init: {status}")
            if status.get('status') == 'READY' or status.get('status') == 'ACTIVE':
                 logger.info("STT Engine status is OPERATIONAL (READY/ACTIVE).")
            else:
                 logger.warning(f"STT Engine status is NOT OPERATIONAL: {status.get('status')}")

        else:
            logger.error("STT Engine INITIALIZATION FAILED according to initialize() return value.")
            # Attempt to get status even on failure, might provide clues
            try:
                status = stt_engine_instance.get_status()
                logger.warning(f"STT Engine status after failed init: {status}")
            except Exception as status_e:
                logger.error(f"Could not get status after failed init: {status_e}")


    except Exception as e:
        logger.error(f"An EXCEPTION occurred during STTEngine instantiation or initialization:")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {e}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        initialization_successful = False # Ensure this is marked as false on exception

    logger.info("--- STT Initialization Test Finished ---")
    if initialization_successful:
        logger.info("Overall Result: Success (Initialization returned True)")
    else:
        logger.error("Overall Result: Failure")

