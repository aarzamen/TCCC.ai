#!/usr/bin/env python3
"""
Test script for the TCCC LLM Analysis module.

This script tests the LLM analysis functionality by:
1. Loading the LLM engine
2. Sending sample transcriptions for analysis
3. Displaying the analysis results
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import logging
import asyncio

# Add project source to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMAnalysisTest")

# Import TCCC modules
from tccc.llm_analysis.llm_analysis import LLMAnalysis
from tccc.utils.config import Config

# Sample transcriptions for testing
SAMPLE_TRANSCRIPTIONS = [
    "Request immediate medical evacuation at grid coordinates Delta-7-9-2",
    "Enemy forces spotted moving northeast approximately 500 meters from our position",
    "We need resupply of ammunition and medical supplies as soon as possible",
    "The bridge at checkpoint Charlie has been damaged but is still passable for light vehicles",
    "Civilian population in sector Bravo-3 requires assistance, multiple injuries reported"
]

class LLMTester:
    def __init__(self, config_path, model=None):
        self.config_path = config_path
        self.model_override = model
        self.llm_analysis = None
        
    def load_config(self):
        """Load configuration from file"""
        try:
            config = Config.from_yaml(self.config_path)
            llm_config = config.get('llm_analysis', {})
            
            # Override model if specified
            if self.model_override:
                logger.info(f"Overriding model to: {self.model_override}")
                llm_config['model']['primary']['name'] = self.model_override
            
            logger.info(f"LLM configuration: {llm_config}")
            return llm_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    async def initialize(self):
        """Initialize the LLM analysis module"""
        try:
            llm_config = self.load_config()
            
            # Initialize LLM analysis
            logger.info("Initializing LLM Analysis module...")
            self.llm_analysis = LLMAnalysis()
            success = await self.llm_analysis.initialize(llm_config)
            
            if success:
                logger.info("LLM Analysis module initialized successfully")
                return True
            else:
                logger.error("Failed to initialize LLM Analysis module")
                return False
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    async def analyze_transcription(self, text):
        """Analyze a single transcription using the LLM engine"""
        try:
            logger.info(f"Analyzing transcription: '{text}'")
            
            # Create a simple event structure
            event = {
                "type": "transcription",
                "source": "test_script",
                "timestamp": time.time(),
                "data": {
                    "text": text,
                    "is_partial": False
                }
            }
            
            # Process the event
            analysis_start = time.time()
            analysis_result = await self.llm_analysis.process_event(event)
            analysis_time = time.time() - analysis_start
            
            if analysis_result:
                logger.info(f"Analysis complete in {analysis_time:.2f}s")
                return {
                    "input": text,
                    "result": analysis_result,
                    "processing_time": analysis_time
                }
            else:
                logger.warning("No analysis result returned")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing transcription: {e}")
            return None
    
    async def run_test_suite(self, custom_inputs=None):
        """Run analysis on a set of sample transcriptions"""
        results = []
        
        # Use custom inputs if provided, otherwise use samples
        test_inputs = custom_inputs if custom_inputs else SAMPLE_TRANSCRIPTIONS
        
        for idx, text in enumerate(test_inputs, 1):
            logger.info(f"Testing sample {idx}/{len(test_inputs)}")
            result = await self.analyze_transcription(text)
            if result:
                results.append(result)
            
            # Small delay between tests
            if idx < len(test_inputs):
                await asyncio.sleep(1)
        
        return results
    
    def format_results(self, results):
        """Format the results for display"""
        output = []
        
        for idx, result in enumerate(results, 1):
            output.append(f"==== Analysis Result {idx} ====")
            output.append(f"Input: {result['input']}")
            output.append(f"Processing time: {result['processing_time']:.2f}s")
            output.append("Analysis:")
            
            # Format the analysis result
            analysis = result.get('result', {})
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    if isinstance(value, dict):
                        output.append(f"  {key}:")
                        for subkey, subvalue in value.items():
                            output.append(f"    {subkey}: {subvalue}")
                    else:
                        output.append(f"  {key}: {value}")
            else:
                output.append(f"  {analysis}")
            
            output.append("")
        
        return "\n".join(output)
    
    async def shutdown(self):
        """Shutdown the LLM analysis module"""
        try:
            if self.llm_analysis:
                await self.llm_analysis.shutdown()
            logger.info("LLM Analysis module shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def run_test(args):
    """Run the LLM analysis test"""
    tester = LLMTester(args.config, args.model)
    
    try:
        # Initialize components
        if not await tester.initialize():
            logger.error("Initialization failed")
            return 1
        
        # Run the test suite
        custom_inputs = None
        if args.input_file:
            try:
                with open(args.input_file, 'r') as f:
                    custom_inputs = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(custom_inputs)} custom inputs from {args.input_file}")
            except Exception as e:
                logger.error(f"Error loading custom inputs: {e}")
                return 1
        
        logger.info("Starting analysis tests...")
        results = await tester.run_test_suite(custom_inputs)
        
        # Process and display results
        if results:
            logger.info(f"Completed {len(results)} analysis tests successfully")
            
            # Save results if requested
            if args.output_file:
                try:
                    with open(args.output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Results saved to {args.output_file}")
                except Exception as e:
                    logger.error(f"Error saving results: {e}")
            
            # Display formatted results
            print("\n" + tester.format_results(results))
            
            # Shut down
            await tester.shutdown()
            return 0
        else:
            logger.error("Test completed without any valid results")
            await tester.shutdown()
            return 1
    except Exception as e:
        logger.error(f"Test failed: {e}")
        await tester.shutdown()
        return 1

def main():
    """Main entry point for the LLM analysis test"""
    parser = argparse.ArgumentParser(description="Test TCCC LLM Analysis")
    parser.add_argument("--config", default=str(project_root / "config/jetson_mvp.yaml"), 
                      help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                      help="Override the model specified in config")
    parser.add_argument("--input-file", type=str, default=None,
                      help="File containing custom input texts (one per line)")
    parser.add_argument("--output-file", type=str, default=None,
                      help="File to save JSON results")
    args = parser.parse_args()
    
    # Create and run the event loop
    try:
        asyncio.run(run_test(args))
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
