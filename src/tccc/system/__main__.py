import argparse
import asyncio
import logging
import os
import sys
import signal # Added for graceful shutdown handling
import traceback

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
shutdown_event = None # Event to signal shutdown

# --- Signal Handler ---
def signal_handler(sig, frame):
    """Handles termination signals for graceful shutdown."""
    global logger
    print(f"Received signal {sig}. Initiating shutdown...")
    logger.info(f"Received signal {sig}. Initiating shutdown...")

# --- Main Execution ---
async def main(args):
    """Main asynchronous function - Takes parsed args."""
    global logger
    
    # --- Basic Setup ---
    # Go up four levels: __main__.py -> system -> tccc -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

    # --- Configuration Loading ---
    try:
        # Directly use the directory and filename arguments
        config_dir_path = args.config_dir
        config_filename = args.config
        config_file_list = [config_filename] # Pass only the filename

        # Check if config directory exists
        print(f"DEBUG: Checking existence of config directory: {config_dir_path}") # DEBUG
        if not os.path.isdir(config_dir_path): # Use isdir for directory check
            logger.error(f"Configuration directory not found: {config_dir_path}")
            sys.exit(1)
            
        logger.info(f"Loading configuration file '{config_filename}' from directory '{config_dir_path}'")
        
        # Call load_config with the directory path and the list of filenames
        # The Config class inside load_config will handle joining paths correctly.
        config_instance = load_config(config_files=config_file_list, config_dir=config_dir_path)
        config = config_instance.config # Get the underlying dict
        logger.info(f"Successfully loaded configuration.")
            
    except FileNotFoundError: # Should be caught by Config class now
        logger.error(f"Configuration file not found: {config_dir_path}")
        sys.exit(1)
    except Exception as cfg_err:
        logger.error(f"Error loading configuration: {cfg_err}", exc_info=True) # Add traceback
        sys.exit(1)

    system = TCCCSystem() # Instantiate without config dict

    def signal_handler():
        logger.info("Shutdown signal received. Setting shutdown event.")
        shutdown_event.set()

    # Add signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler) # Handle Ctrl+C
        loop.add_signal_handler(signal.SIGTERM, signal_handler) # Handle termination signal
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        logger.warning("Signal handlers not fully supported on this platform.")
        # Consider alternative shutdown mechanisms for Windows if needed

    try:
        logger.info("Initializing TCCC System...")
        # Pass the loaded config dictionary here
        await system.initialize(config=config, mock_modules=args.mock_modules) 
        logger.info("TCCC System Initialized. Running...")
        logger.info("Press Ctrl+C to shut down.")
        
        # Keep running based on duration or until shutdown signal
        if args.duration is not None and args.duration > 0:
            logger.info(f"Running for specified duration: {args.duration} seconds...")
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=args.duration)
                logger.info("Shutdown event received before duration expired.")
            except asyncio.TimeoutError:
                logger.info(f"{args.duration} second runtime finished.")
        else:
            logger.info("Running indefinitely until shutdown signal (Ctrl+C)...")
            await shutdown_event.wait() # Wait here until signal_handler sets the event
            logger.info("Shutdown event received.")

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in main execution: {e}")
        logger.critical(traceback.format_exc())
    finally:
        logger.info("Initiating final system shutdown...")
        await system.shutdown() # Ensure shutdown is called
        logger.info("TCCC System shutdown process complete.")

# --- Script Execution ---
if __name__ == "__main__":
    # Go up four levels: __main__.py -> system -> tccc -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="TCCC.ai System Main Entry Point")
    parser.add_argument("-c", "--config", default="jetson_mvp.yaml", help="Configuration file name relative to the config directory (default: jetson_mvp.yaml)")
    parser.add_argument("--config-dir", default=os.path.join(project_root, 'config'), help="Path to the configuration directory")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level (default: INFO)")
    parser.add_argument("--log-file", default=None, help="Path to a file to write logs to (optional)")
    parser.add_argument("--mock-modules", nargs='*', help="List of modules to mock (e.g., 'stt_engine' 'llm_analysis')")
    parser.add_argument("--duration", type=int, default=None, help="Optional duration in seconds to run before automatic shutdown.") # Added duration argument
    args = parser.parse_args()

    # Setup logging here using parsed args before starting async
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        try:
            log_handlers.append(logging.FileHandler(args.log_file))
        except Exception as e:
            print(f"Error setting up log file handler: {e}")

    logging.basicConfig(level=log_level, format=log_format, handlers=log_handlers)
    logger = logging.getLogger(__name__) # Set the global logger
    logger.info("Logging configured.")

    # Define shutdown event globally for signal handlers
    shutdown_event = asyncio.Event()

    # --- Signal Handling Setup ---
    # Set up signal handlers before starting the event loop
    signal.signal(signal.SIGINT, signal_handler) # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle kill command

    try:
        asyncio.run(main(args)) # Pass parsed args to main
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
