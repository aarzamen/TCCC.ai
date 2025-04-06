import logging
import sys
import os
import time

# Ensure the src directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from tccc.audio_pipeline.audio_pipeline import AudioPipeline
    from tccc.processing_core.processing_core import ModuleState # Corrected import path
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Basic Logging Setup ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format, stream=sys.stdout)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

if __name__ == "__main__":
    logger.info("--- Starting Minimal AudioPipeline Instantiation Test ---")

    # Use an empty config for the most basic test
    test_config = {}
    logger.info(f"Using config: {test_config}")

    pipeline_instance = None
    try:
        logger.info("Attempting to instantiate AudioPipeline...")
        start_time = time.time()
        # Explicitly pass the empty config dictionary
        pipeline_instance = AudioPipeline(config=test_config)
        end_time = time.time()
        logger.info(f"AudioPipeline instantiation SUCCESSFUL! Time taken: {end_time - start_time:.4f} seconds")
        logger.info(f"Instance created: {pipeline_instance}")
        if hasattr(pipeline_instance, 'status'):
             logger.info(f"Instance status after init: {pipeline_instance.status}")
        else:
             logger.warning("Instance does not have a 'status' attribute after init.")

    except Exception as e:
        logger.exception(f"AudioPipeline instantiation FAILED with Python exception: {e}")
        # Explicitly log the traceback
        import traceback
        traceback.print_exc()

    finally:
        logger.info("--- Minimal AudioPipeline Instantiation Test Finished ---")

        # Check if instance was created before trying to access attributes
        if pipeline_instance is not None:
             logger.info(f"Final check: Instance exists: {pipeline_instance}")
        else:
             logger.info("Final check: Instance was not successfully created.")
