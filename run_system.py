#!/usr/bin/env python3
"""
TCCC.ai System Runner Script - Runs the TCCC system with all components.
"""

import os
import sys
import time
import argparse
import logging
import signal
from pathlib import Path

# Set up proper path for imports
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_dir, 'logs', 'system.log'))
    ]
)
logger = logging.getLogger("TCCC.System")

try:
    from tccc.system.system import TCCCSystem, SystemState
    from tccc.utils.config import ConfigManager
    
    # Try to import display components if available
    try:
        from tccc.display.display_interface import DisplayInterface
        display_available = True
    except ImportError:
        logger.warning("Display interface not available")
        display_available = False
        
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class SystemRunner:
    """Runner class for the TCCC.ai system."""
    
    def __init__(self, with_display=False, use_microphone=False, test_mode=False):
        """Initialize the system runner."""
        self.system = None
        self.display = None
        self.config_manager = ConfigManager()
        self.with_display = with_display and display_available
        self.use_microphone = use_microphone
        self.test_mode = test_mode
        self.running = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.shutdown()
        
    def initialize(self):
        """Initialize the TCCC.ai system."""
        logger.info("Initializing TCCC.ai system...")
        
        # Load configurations
        self.config_manager.load_all_configs()
        
        # Create system with loaded configurations
        self.system = TCCCSystem()
        
        # Initialize the system
        logger.info("Starting system initialization...")
        init_success = self.system.initialize()
        
        if not init_success:
            logger.error("System initialization failed")
            return False
        
        # Initialize display if requested
        if self.with_display:
            try:
                logger.info("Initializing display interface...")
                self.display = DisplayInterface()
                self.display.initialize()
                self.system.register_display(self.display)
                logger.info("Display interface initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize display: {e}")
                self.with_display = False
        
        logger.info("System initialization complete")
        return True
    
    def run(self):
        """Run the TCCC.ai system."""
        if not self.system:
            logger.error("System not initialized")
            return False
        
        self.running = True
        logger.info("Starting TCCC.ai system...")
        
        try:
            # Start the system
            self.system.start()
            
            # If in test mode, run a short test and exit
            if self.test_mode:
                logger.info("Running in test mode")
                self.run_test()
                return True
            
            # Main loop
            while self.running:
                # Check system status
                status = self.system.get_status()
                
                if status['state'] == SystemState.ERROR.value:
                    logger.error(f"System error: {status['state_message']}")
                    self.running = False
                    break
                
                # Update display if available
                if self.with_display and self.display:
                    self.display.update()
                
                # Sleep to prevent high CPU usage
                time.sleep(0.1)
                
            return True
            
        except Exception as e:
            logger.error(f"Error running system: {e}")
            return False
        finally:
            self.shutdown()
    
    def run_test(self):
        """Run a short test with sample data."""
        logger.info("Running system test...")
        
        # Use sample audio file for testing
        sample_audio = os.path.join(project_dir, 'test_data', 'test_speech.wav')
        
        if not os.path.exists(sample_audio):
            logger.error(f"Sample audio file not found: {sample_audio}")
            return
        
        logger.info(f"Processing sample audio: {sample_audio}")
        
        # Process sample audio
        result = self.system.process_audio_file(sample_audio)
        
        logger.info(f"Test completed with result: {result}")
        
        # Keep the system running for a few seconds to show output
        time.sleep(10)
    
    def shutdown(self):
        """Shutdown the TCCC.ai system."""
        self.running = False
        
        if self.system:
            logger.info("Shutting down TCCC.ai system...")
            self.system.shutdown()
            
        if self.with_display and self.display:
            logger.info("Shutting down display...")
            self.display.shutdown()
            
        logger.info("System shutdown complete")

def main():
    """Main function to run the TCCC.ai system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TCCC.ai System Runner")
    parser.add_argument("--with-display", action="store_true", help="Enable display interface")
    parser.add_argument("--use-microphone", action="store_true", help="Use microphone for audio input")
    parser.add_argument("--test", action="store_true", help="Run in test mode with sample data")
    args = parser.parse_args()
    
    # Create system runner
    runner = SystemRunner(
        with_display=args.with_display,
        use_microphone=args.use_microphone,
        test_mode=args.test
    )
    
    # Initialize the system
    if not runner.initialize():
        logger.error("Failed to initialize system")
        sys.exit(1)
    
    # Run the system
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(project_dir, 'logs'), exist_ok=True)
    
    # Run the main function
    main()