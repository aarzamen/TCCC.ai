#!/usr/bin/env python3
"""
Integration test runner for TCCC.ai system.
Executes all integration tests and reports results.
"""

import os
import sys
import unittest
import argparse
import time
import asyncio
from datetime import datetime

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test modules
from tests.integration.test_system_integration import TestSystemInitialization, TestSystemOperation, TestDataFlow
from tests.integration.test_error_handling import TestErrorHandling, TestResourceConstraints
from tests.integration.test_performance import PerformanceBenchmarks


def run_tests(test_categories=None, verbose=False):
    """
    Run the selected test categories
    
    Args:
        test_categories (list): Categories of tests to run
        verbose (bool): Whether to show verbose output
    """
    # Define test suites
    test_suites = {
        'system': unittest.TestLoader().loadTestsFromTestCase(TestSystemInitialization),
        'operation': unittest.TestLoader().loadTestsFromTestCase(TestSystemOperation),
        'dataflow': unittest.TestLoader().loadTestsFromTestCase(TestDataFlow),
        'error': unittest.TestLoader().loadTestsFromTestCase(TestErrorHandling),
        'resources': unittest.TestLoader().loadTestsFromTestCase(TestResourceConstraints),
        'performance': unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmarks),
    }
    
    # Create master test suite
    master_suite = unittest.TestSuite()
    
    # Add selected test suites
    if not test_categories:
        # Run all tests if none specified
        test_categories = list(test_suites.keys())
    
    for category in test_categories:
        if category in test_suites:
            master_suite.addTest(test_suites[category])
        else:
            print(f"Warning: Unknown test category '{category}'")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    start_time = time.time()
    result = runner.run(master_suite)
    end_time = time.time()
    
    # Report results
    print("\n" + "="*80)
    print(f"TCCC.ai Integration Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Ran {result.testsRun} tests in {end_time - start_time:.2f} seconds")
    print(f"Success: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Show any failures or errors
    if result.failures or result.errors:
        print("\nDetails of failed tests:")
        
        for test, trace in result.failures:
            print(f"\n--- FAILURE in {test} ---")
            print(trace)
        
        for test, trace in result.errors:
            print(f"\n--- ERROR in {test} ---")
            print(trace)
    
    # Return success/failure
    return len(result.failures) + len(result.errors) == 0


def main():
    """Main entry point for the test runner"""
    parser = argparse.ArgumentParser(description='Run TCCC.ai integration tests')
    parser.add_argument('--categories', '-c', nargs='+', 
                        choices=['system', 'operation', 'dataflow', 'error', 
                                'resources', 'performance', 'all'],
                        help='Test categories to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose test output')
    args = parser.parse_args()
    
    # Process 'all' category
    categories = args.categories
    if categories and 'all' in categories:
        categories = None  # None means run all
    
    # Run tests
    success = run_tests(categories, args.verbose)
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())