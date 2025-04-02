import argparse
import asyncio
import logging
import os
import sys
import signal # Added for graceful shutdown handling

# Ensure the src directory is in the Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("DEBUG: Attempting to import TCCCSystem...") # Add debug print
from tccc.system.system import TCCCSystem, SystemState
print("DEBUG: TCCCSystem imported successfully.") # Add debug print

from tccc.utils.logging import configure_logging, get_logger
from tccc.utils.config import load_config # Assuming config loading utility exists

# --- Globals ---
logger = None # Initialize logger globally
keep_running = True # Flag for main loop

# --- Signal Handler ---
def signal_handler(sig, frame):
    """Handles termination signals for graceful shutdown."""
    global keep_running
    print(f"Received signal {sig}. Initiating shutdown...")
    logger.info(f"Received signal {sig}. Initiating shutdown...")
    keep_running = False

# --- Main Execution ---
async def main():
    """Main entry point for the TCCC System."""
    global logger, keep_running
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="TCCC.ai System Main Entry Point")
    parser.add_argument(
        "--config", 
        type=str, 
        default="jetson_mvp.yaml", # Default config filename
        help="Configuration file name (e.g., jetson_mvp.yaml). Assumed to be in the directory specified by --config-dir."
    )
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="config", # Default config directory relative to project root
        help="Path to the configuration directory."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # --- Basic Setup ---
    # Go up four levels: __main__.py -> system -> tccc -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

    # --- Logging Configuration ---
    # Configure before creating the system instance
    # Use a dedicated logger for the main script
    try:
        # Assuming log directory is defined relative to project root or via config
        log_dir = os.path.join(project_root, "logs") 
        configure_logging(log_level=args.log_level.upper(), log_dir=log_dir)
        logger = get_logger("__main__") # Use __main__ logger
        print("Logging configured.") # Simple print for early feedback
        logger.info("Logging configured.")
    except Exception as log_err:
        print(f"FATAL: Could not configure logging: {log_err}")
        sys.exit(1)

    # --- Configuration Loading ---
    try:
        # Construct full path to config directory (project_root is defined above)
        config_dir_path = os.path.join(project_root, args.config_dir) # Correct path relative to project root
        config_filename = args.config # Use the filename directly
        config_file_list = [config_filename]

        # Check if config directory exists
        print(f"DEBUG: Checking existence of config directory: {config_dir_path}") # DEBUG
        if not os.path.exists(config_dir_path):
            logger.error(f"Configuration directory not found: {config_dir_path}")
            sys.exit(1)
            
        logger.info(f"Loading configuration file '{config_filename}' from directory '{config_dir_path}'")
        
        # Call load_config with directory and a list containing the filename
        # NOTE: Assumes tccc.utils.config.Config handles loading dependencies like base.yaml
        # If not, additional logic might be needed here to explicitly load base configs first.
        config = load_config(config_files=config_file_list, config_dir=config_dir_path)
        logger.info(f"Successfully loaded configuration.")
            
    except FileNotFoundError: # This check is now done above, but keep for safety
        logger.error(f"Configuration file not found: {config_dir_path}")
        sys.exit(1)
    except Exception as cfg_err:
        logger.error(f"Error loading configuration: {cfg_err}", exc_info=True) # Add traceback
        sys.exit(1)

    # --- System Initialization ---
    system = None
    try:
        logger.info("Creating TCCCSystem instance...")
        system = TCCCSystem()
        logger.info("TCCCSystem instance created.")
        
        logger.info("Initializing TCCC System...")
        # Pass the loaded config dictionary directly
        initialized = await system.initialize(config=config) 
        
        if initialized:
            logger.info("TCCC System initialized successfully.")
            
            # --- Start Audio Capture ---
            logger.info("__main__: Attempting to start audio capture...")
            capture_started = system.start_audio_capture()
            if capture_started:
                logger.info("Audio capture started successfully.")
            else:
                logger.warning("Audio capture failed to start. Check logs.")
                # Depending on requirements, we might want to exit here
                # keep_running = False
                
            # --- Keep Running Loop (Placeholder) ---
            # This loop keeps the main thread alive. 
            # In a real application, this might involve waiting for specific events,
            # handling user input, or integrating with an external event loop.
            logger.info("System running. Press Ctrl+C to exit.")
            while keep_running:
                # Check system status periodically or wait for events
                if system.get_state() == SystemState.ERROR:
                     logger.error("System entered ERROR state. Initiating shutdown.")
                     keep_running = False
                await asyncio.sleep(1) # Prevent busy-waiting
            logger.info("Main loop exited.")

        else:
            logger.error("TCCC System initialization failed. See previous logs for details.")
            # Optionally print errors gathered during init
            errors = system.get_errors()
            if errors:
                 logger.error("Initialization errors reported:")
                 for comp, err_list in errors.items():
                     for err in err_list:
                         logger.error(f"  - [{comp}]: {err}")

    except Exception as e:
        logger.critical(f"An unexpected error occurred during system startup or execution: {e}", exc_info=True)
    
    # --- Graceful Shutdown ---
    finally:
        if system:
            logger.info("Shutting down TCCC System...")
            system.shutdown() # Shutdown is synchronous
            logger.info("TCCC System shutdown complete.")
        else:
            logger.info("No system instance to shut down.")

# --- Script Execution ---
if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle kill command
    
    try:
         asyncio.run(main())
         logger.info("Main function finished.")
    except KeyboardInterrupt:
         # This might be caught by the signal handler first, but handle here too just in case.
         logger.info("KeyboardInterrupt received in main block. Exiting.")
    except Exception as main_err:
        # Log any exception that might occur outside the main async function itself
        # (though most should be caught within main's try/except)
        if logger:
             logger.critical(f"Critical error outside main async function: {main_err}", exc_info=True)
        else:
            # If logger failed, print to stderr as last resort
            print(f"CRITICAL ERROR (logging unavailable): {main_err}", file=sys.stderr)
        sys.exit(1) # Ensure non-zero exit code on error
    finally:
        # Final log message if logger is available
        if logger:
             logger.info("Script execution finished.")
        else:
             print("Script execution finished (logging unavailable).")
        sys.exit(0) # Ensure zero exit code on clean exit
