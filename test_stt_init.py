import sys
import os
import logging
import traceback

# --- Path Setup ---
# Ensure the 'src' directory is in the Python path
project_root = os.path.abspath(os.path.dirname(__file__)) 
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print(f"DEBUG: Project Root: {project_root}")
print(f"DEBUG: Added to sys.path: {src_path}")
print(f"DEBUG: Current sys.path: {sys.path}")

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_stt_init")
print("DEBUG: Basic logging configured.")

# --- Attempt STTEngine Import ---
try:
    print("DEBUG: Attempting to import STTEngine...")
    from tccc.stt_engine.stt_engine import STTEngine
    print("DEBUG: STTEngine imported successfully.")
except ImportError as ie:
    print(f"FATAL: Failed to import STTEngine: {ie}")
    print(traceback.format_exc())
    sys.exit(1)
except Exception as e:
    print(f"FATAL: An unexpected error occurred during STTEngine import: {e}")
    print(traceback.format_exc())
    sys.exit(1)

# --- Configuration Loading ---
try:
    print("DEBUG: Attempting to import config loader...")
    from tccc.utils.config import load_config 
    print("DEBUG: Config loader imported.")
    
    config_dir = os.path.join(project_root, 'config')
    config_file = 'jetson_mvp.yaml'
    print(f"DEBUG: Loading config '{config_file}' from '{config_dir}'")
    
    # Ensure the config directory exists
    if not os.path.isdir(config_dir):
        print(f"FATAL: Config directory not found: {config_dir}")
        sys.exit(1)
        
    # Load the full config first
    full_config_obj = load_config([config_file], config_dir=str(config_dir))

    # --- MODIFIED: Access config using .get() method ---
    # Correctly get the stt_engine section using the .get() method of the Config object
    stt_config = full_config_obj.get('stt_engine', None) # Returns dict or None
    # --- END MODIFICATION ---

    if stt_config is None:
         logger.warning("STT Engine config section not found in jetson_mvp.yaml using .get(). Proceeding with None.")
         # STTEngine.__init__ handles None config, so this is okay for testing init itself
    elif not isinstance(stt_config, dict):
         logger.error(f"Expected 'stt_engine' config section retrieved via .get() to be a dictionary, but got {type(stt_config)}. Cannot proceed.")
         sys.exit(1)
    else:
         logger.debug(f"Successfully retrieved stt_engine config section via .get(): {stt_config}")

except ImportError:
    print("FATAL: Could not import tccc.utils.config.load_config. Make sure utils are available.")
    sys.exit(1)
except FileNotFoundError:
    print(f"FATAL: Configuration file '{config_file}' not found in '{config_dir}'.")
    sys.exit(1)
except AttributeError:
    # Catch if .get() itself doesn't exist, though it's standard for many config libs
    print(f"FATAL: The loaded config object (type: {type(full_config_obj)}) does not have a .get() method.") 
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Error loading or accessing configuration: {e}")
    print(traceback.format_exc())
    sys.exit(1)


# --- STTEngine Instantiation Test ---
if stt_config: # Proceed only if config was loaded
    try:
        logger.info("Attempting to instantiate STTEngine...")
        print("--- BEFORE STTEngine INSTANTIATION ---")
        
        # THE ACTUAL INSTANTIATION CALL
        stt_instance = STTEngine(stt_config) 
        
        print("--- AFTER STTEngine INSTANTIATION ---")
        logger.info("STTEngine instantiated successfully!")
        
        # Optional: Try initializing it as well, if instantiation succeeds
        # logger.info("Attempting to initialize STTEngine...")
        # init_success = await stt_instance.initialize(stt_config) # Requires running in async context if initialize is async
        # if init_success:
        #     logger.info("STTEngine initialized successfully!")
        # else:
        #     logger.error("STTEngine initialization failed.")

    except Exception as e:
        logger.error(f"FAILED to instantiate STTEngine: {e}", exc_info=True)
        print(f"--- EXCEPTION DURING STTEngine INSTANTIATION --- ")
        print(traceback.format_exc()) # Print traceback explicitly
    finally:
        logger.info("Test script finished.")
else:
    logger.error("Skipping STTEngine instantiation test because config loading failed.")
