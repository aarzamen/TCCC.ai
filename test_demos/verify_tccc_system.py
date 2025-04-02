#!/usr/bin/env python3
"""
TCCC System Verification Script.
This script provides a unified interface to verify all system components.
"""

import os
import sys
import argparse
import shutil
import subprocess
import time
from pathlib import Path

# Add project to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'src'))

def run_command(command, timeout=60):
    """Run a command and return its output."""
    try:
        # Use bash explicitly to ensure source command works
        result = subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after {} seconds".format(timeout)

def check_environment():
    """Check the environment for TCCC."""
    print("\n=== Checking Environment ===")
    
    # Check virtual environment
    activate_path = os.path.join(project_dir, 'venv', 'bin', 'activate')
    if os.path.exists(activate_path):
        print("✓ Virtual environment found")
    else:
        print("✗ Virtual environment not found at {}".format(activate_path))
        return False
    
    # Run the environment check script
    cmd = "cd {} && source venv/bin/activate && python verify_environment.py".format(project_dir)
    code, output, error = run_command(cmd)
    
    if code == 0:
        print("✓ Environment verification passed")
        return True
    else:
        print("✗ Environment verification failed")
        print("Error: {}".format(error))
        return False

def verify_audio_pipeline():
    """Verify the audio pipeline."""
    print("\n=== Verifying Audio Pipeline ===")
    
    cmd = "cd {} && source venv/bin/activate && python verification_script_audio_pipeline.py".format(project_dir)
    code, output, error = run_command(cmd)
    
    if "Verification complete!" in output:
        print("✓ Audio pipeline verification passed")
        return True
    else:
        print("✗ Audio pipeline verification failed")
        print("Error: {}".format(error))
        return False

def verify_stt_engine():
    """Verify the STT engine."""
    print("\n=== Verifying STT Engine ===")
    
    cmd = "cd {} && source venv/bin/activate && python verification_script_stt_engine.py".format(project_dir)
    code, output, error = run_command(cmd)
    
    if "Verification complete!" in output:
        print("✓ STT engine verification passed")
        return True
    else:
        print("✗ STT engine verification failed")
        print("Error: {}".format(error))
        return False

def verify_event_system():
    """Verify the event system."""
    print("\n=== Verifying Event System ===")
    
    cmd = "cd {} && source venv/bin/activate && python verification_script_event_schema_simple.py".format(project_dir)
    code, output, error = run_command(cmd)
    
    # Note that output can be in error when subprocess is too verbose
    combined_output = output + error
    
    if "Event system verification complete: All tests PASSED" in combined_output:
        print("✓ Event system verification passed")
        return True
    else:
        print("✗ Event system verification failed")
        if "Error running tests" in combined_output:
            print("Error: Exception occurred during testing")
        return False

def verify_display_components():
    """Verify the display components."""
    print("\n=== Verifying Display Components ===")
    
    # Use the simplified display verification script for MVP readiness check
    cmd = "cd {} && source venv/bin/activate && python verification_script_display_enhanced_simple.py".format(project_dir)
    code, output, error = run_command(cmd)
    
    # Check both output and error for the success message
    combined_output = output + error
    if "Display verification complete: All tests PASSED" in combined_output:
        print("✓ Display components verification passed")
        return True
    else:
        print("✗ Display components verification failed")
        return False

def verify_display_integration():
    """Verify the display integration with event system."""
    print("\n=== Verifying Display-Event Integration ===")
    
    # For MVP readiness, we'll use the same simplified display verification
    # which also checks basic integration capabilities
    cmd = "cd {} && source venv/bin/activate && python verification_script_display_enhanced_simple.py".format(project_dir)
    code, output, error = run_command(cmd)
    
    # Check both output and error for the success message
    combined_output = output + error
    
    # If the display verification worked and it mentions integration passing
    if "Display Integration: PASSED" in combined_output:
        print("✓ Display-event integration verification passed")
        return True
    else:
        print("✗ Display-event integration verification failed")
        return False

def verify_end_to_end(duration=10, use_mock=True, use_file=True):
    """Verify the end-to-end audio to STT pipeline."""
    print("\n=== Verifying End-to-End Audio to STT Pipeline ===")
    
    mock_flag = "--mock" if use_mock else ""
    file_flag = "--file" if use_file else ""
    
    cmd = "cd {} && source venv/bin/activate && python verify_audio_stt_e2e.py {} {} --duration {}".format(
        project_dir, mock_flag, file_flag, duration
    )
    code, output, error = run_command(cmd, timeout=duration+30)
    
    if "Verification PASSED" in output:
        print("✓ End-to-end verification passed")
        return True
    else:
        print("✗ End-to-end verification failed")
        return False

def optimize_audio_for_razer():
    """Optimize audio pipeline for Razer Seiren V3 Mini."""
    print("\n=== Optimizing Audio Pipeline for Razer Seiren V3 Mini ===")
    
    cmd = "cd {} && source venv/bin/activate && python optimize_audio_razer.py".format(project_dir)
    code, output, error = run_command(cmd, timeout=120)
    
    if "OPTIMIZATION COMPLETE" in output:
        print("✓ Audio pipeline optimization completed")
        return True
    else:
        print("✗ Audio pipeline optimization failed or was interrupted")
        return False

def verify_rag_system():
    """Verify the RAG system."""
    print("\n=== Verifying RAG System ===")
    
    cmd = "cd {} && source venv/bin/activate && python verification_script_rag.py".format(project_dir)
    code, output, error = run_command(cmd)
    
    if "RAG verification complete" in output:
        print("✓ RAG system verification passed")
        return True
    else:
        print("✗ RAG system verification failed")
        print("Error: {}".format(error))
        return False

def print_verification_summary(results):
    """Print a summary of verification results."""
    print("\n\n=== Verification Summary ===")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ Passed" if result else "✗ Failed"
        print(f"{name}: {status}")
    
    print("\nOverall: {}/{} verifications passed".format(passed, total))
    
    if passed == total:
        print("\n✅ All verifications passed - System is ready to use")
    else:
        print("\n⚠️ Some verifications failed - System may not be fully functional")
    
    # Write the results to a file for reference
    with open(os.path.join(project_dir, 'verification_status.txt'), 'w') as f:
        f.write("=== TCCC Verification Results ===\n")
        f.write("Timestamp: {}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        
        for name, result in results.items():
            status = "PASSED" if result else "FAILED"
            f.write(f"{name}: {status}\n")
        
        f.write("\nOverall: {}/{} verifications passed\n".format(passed, total))
        
        if passed == total:
            f.write("\nSystem is ready to use\n")
        else:
            f.write("\nSystem may not be fully functional\n")

def perform_mvp_readiness_check():
    """Perform a comprehensive MVP readiness check."""
    print("\n=== Performing MVP Readiness Check ===")
    print("This will run a series of checks to validate the core functionality of the TCCC system")
    
    results = {}
    
    # Step 1: Check environment and critical prerequisites
    print("\n-- Step 1: Checking Environment --")
    results["Environment"] = check_environment()
    
    if not results["Environment"]:
        print("\n❌ Environment check failed - cannot proceed with verification")
        print_verification_summary(results)
        return 1
    
    # Step 2: Check core components individually
    print("\n-- Step 2: Checking Core Components --")
    results["Audio Pipeline"] = verify_audio_pipeline()
    results["STT Engine"] = verify_stt_engine()
    results["Event System"] = verify_event_system()
    
    # Step 3: Check basic integration points
    print("\n-- Step 3: Checking Basic Integration --")
    results["Audio-STT (Mock/File)"] = verify_end_to_end(duration=5, use_mock=True, use_file=True)
    
    # Step 4: Check display components if previous steps passed
    step4_enabled = results["Audio Pipeline"] and results["STT Engine"] and results["Event System"]
    if step4_enabled:
        print("\n-- Step 4: Checking Display Components --")
        results["Display Components"] = verify_display_components()
        if results["Display Components"]:
            results["Display-Event Integration"] = verify_display_integration()
    else:
        print("\n-- Step 4: Skipping Display Components (Core components failed) --")
        results["Display Components"] = False
        results["Display-Event Integration"] = False
    
    # Step 5: Check RAG system if core functionality passed
    step5_enabled = results["Audio-STT (Mock/File)"]
    if step5_enabled:
        print("\n-- Step 5: Checking RAG System --")
        results["RAG System"] = verify_rag_system()
    else:
        print("\n-- Step 5: Skipping RAG System (Core integration failed) --")
        results["RAG System"] = False
    
    # Print summary
    print_verification_summary(results)
    
    # Define critical components for MVP
    critical_components = [
        "Environment",
        "Audio Pipeline", 
        "STT Engine", 
        "Event System", 
        "Audio-STT (Mock/File)"
    ]
    
    # Define enhanced components (good to have but not critical)
    enhanced_components = [
        "Display Components",
        "Display-Event Integration",
        "RAG System"
    ]
    
    # Check status of critical components
    critical_passed = all(results.get(component, False) for component in critical_components)
    enhanced_passed = all(results.get(component, False) for component in enhanced_components)
    
    # Print MVP status
    if critical_passed:
        if enhanced_passed:
            print("\n✅ MVP VERIFICATION COMPLETE: All components passed - System is ready for deployment")
        else:
            print("\n🟡 MVP MINIMUM REQUIREMENTS MET: Core functionality works but some enhanced features failed")
            print("   The system can be deployed with limited functionality")
    else:
        print("\n❌ MVP VERIFICATION FAILED: Core functionality is not working properly")
        print("   The system is not ready for deployment")
    
    # Save detailed results to a file
    result_file = os.path.join(project_dir, 'mvp_verification_results.txt')
    with open(result_file, 'w') as f:
        f.write("=== TCCC MVP Verification Results ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Critical Components:\n")
        for component in critical_components:
            status = "PASSED" if results.get(component, False) else "FAILED"
            f.write(f"  {component}: {status}\n")
        
        f.write("\nEnhanced Components:\n")
        for component in enhanced_components:
            status = "PASSED" if results.get(component, False) else "FAILED"
            f.write(f"  {component}: {status}\n")
        
        f.write(f"\nOverall MVP Status: {'PASSED' if critical_passed else 'FAILED'}\n")
        if critical_passed and not enhanced_passed:
            f.write("Note: Core functionality working but some enhanced features failed\n")
    
    print(f"\nDetailed results saved to: {result_file}")
    
    return 0 if critical_passed else 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="TCCC System Verification")
    parser.add_argument("--all", action="store_true", help="Run all verifications")
    parser.add_argument("--env", action="store_true", help="Verify environment")
    parser.add_argument("--audio", action="store_true", help="Verify audio pipeline")
    parser.add_argument("--stt", action="store_true", help="Verify STT engine")
    parser.add_argument("--e2e", action="store_true", help="Verify end-to-end pipeline")
    parser.add_argument("--display", action="store_true", help="Verify display components")
    parser.add_argument("--event", action="store_true", help="Verify event system")
    parser.add_argument("--rag", action="store_true", help="Verify RAG system")
    parser.add_argument("--optimize", action="store_true", help="Optimize audio for Razer microphone")
    parser.add_argument("--real", action="store_true", help="Use real STT engine instead of mock")
    parser.add_argument("--mic", action="store_true", help="Use microphone instead of file input")
    parser.add_argument("--duration", type=int, default=10, help="Duration for tests in seconds")
    parser.add_argument("--mvp", action="store_true", help="Perform comprehensive MVP readiness check")
    args = parser.parse_args()
    
    # Special case: MVP readiness check
    if args.mvp:
        return perform_mvp_readiness_check()
    
    # If no specific checks requested, run all
    if not (args.env or args.audio or args.stt or args.e2e or args.display or 
            args.event or args.rag or args.optimize):
        args.all = True
    
    # Results to track
    results = {}
    
    # Check environment
    if args.env or args.all:
        results["Environment"] = check_environment()
    
    # Verify audio pipeline
    if args.audio or args.all:
        results["Audio Pipeline"] = verify_audio_pipeline()
    
    # Verify STT engine
    if args.stt or args.all:
        results["STT Engine"] = verify_stt_engine()
    
    # Verify event system
    if args.event or args.all:
        results["Event System"] = verify_event_system()
    
    # Verify display components
    if args.display or args.all:
        results["Display Components"] = verify_display_components()
        results["Display-Event Integration"] = verify_display_integration()
    
    # Verify RAG system
    if args.rag or args.all:
        results["RAG System"] = verify_rag_system()
    
    # Verify end-to-end pipeline
    if args.e2e or args.all:
        # Process real/mock and mic/file flags
        use_mock = not args.real
        use_file = not args.mic
        
        type_str = "Real" if not use_mock else "Mock"
        input_str = "Microphone" if not use_file else "File"
        
        results[f"End-to-End ({type_str}/{input_str})"] = verify_end_to_end(
            duration=args.duration,
            use_mock=use_mock,
            use_file=use_file
        )
    
    # Optimize audio pipeline
    if args.optimize:
        results["Audio Optimization"] = optimize_audio_for_razer()
    
    # Print summary
    print_verification_summary(results)
    
    # Return success if all passed
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())