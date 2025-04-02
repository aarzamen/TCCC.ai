#!/usr/bin/env python3
"""
TCCC Medical Terminology Test.

This script tests the system's ability to recognize and process 
tactical combat casualty care terminology and protocols.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title} ".center(70, "="))
    print("=" * 70 + "\n")

def record_audio(duration=10.0, fs=16000, device=0):
    """
    Record audio from the microphone.
    
    Args:
        duration: Recording duration in seconds
        fs: Sample rate
        device: Audio device ID
        
    Returns:
        Recorded audio as numpy array
    """
    try:
        import sounddevice as sd
        
        print(f"Recording {duration} seconds of audio...")
        print("Starting in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Recording now! Please verbalize the tactical medical scenario...")
        
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, 
                          dtype='float32', device=device)
        sd.wait()
        
        print("Recording complete!")
        return recording.flatten()
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return np.zeros(int(duration * fs), dtype=np.float32)

def save_audio(audio, fs=16000, filename="tccc_test.wav"):
    """Save audio to file."""
    try:
        import soundfile as sf
        sf.write(filename, audio, fs)
        print(f"Audio saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def transcribe_audio(audio_file, enhance_audio=True):
    """
    Transcribe audio using the TCCC STT engine.
    
    Args:
        audio_file: Path to audio file
        enhance_audio: Whether to apply audio enhancement
        
    Returns:
        Dictionary with transcription results
    """
    print("Transcribing audio with TCCC STT engine...")
    
    try:
        # First try to load and configure the STT engine directly
        try:
            from tccc.stt_engine.stt_engine import STTEngine
            import soundfile as sf
            
            # Load audio
            audio, sr = sf.read(audio_file)
            
            # Initialize STT engine
            engine = STTEngine()
            engine.initialize({
                "model": {
                    "type": "whisper",
                    "size": "small",  # Use small model for better accuracy
                },
                "hardware": {
                    "enable_acceleration": True
                }
            })
            
            # Preprocess audio if enhancement is enabled
            if enhance_audio:
                audio = preprocess_audio(audio)
            
            # Transcribe
            result = engine.transcribe_segment(audio)
            
            # Shutdown engine
            engine.shutdown()
            
            return result
            
        except Exception as e:
            print(f"Error using direct STT engine: {e}")
            # Fall back to command-line approach
            raise RuntimeError("Direct STT failed")
            
    except Exception:
        # Fallback: Run the transcription script as a subprocess
        try:
            import subprocess
            
            # Determine the script to use
            if os.path.exists("microphone_to_text.py"):
                script = "microphone_to_text.py"
            elif os.path.exists("stt_with_file.py"):
                script = "stt_with_file.py"
            else:
                # Find any script that might handle STT
                candidates = []
                for root, _, files in os.walk("."):
                    for file in files:
                        if file.endswith(".py") and ("stt" in file.lower() or "transcribe" in file.lower()):
                            candidates.append(os.path.join(root, file))
                
                if candidates:
                    script = candidates[0]
                else:
                    raise FileNotFoundError("No STT script found")
            
            # Build command with appropriate flags
            cmd = [
                "python", script, 
                "--file", audio_file
            ]
            
            if enhance_audio:
                cmd.append("--enhance")
            
            # Run the command
            print(f"Running command: {' '.join(cmd)}")
            output = subprocess.check_output(cmd, universal_newlines=True)
            
            # Extract transcription from output
            transcription = ""
            for line in output.splitlines():
                if "Transcription:" in line:
                    transcription = line.split("Transcription:", 1)[1].strip()
                    break
            
            return {
                "text": transcription,
                "segments": [{"text": transcription}]
            }
            
        except Exception as e:
            print(f"Error running STT script: {e}")
            
            # Last resort: manually process the file with a simple function
            return {
                "text": "Failed to transcribe audio",
                "error": str(e)
            }

def preprocess_audio(audio, sample_rate=16000):
    """
    Apply audio preprocessing to enhance speech quality.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate
        
    Returns:
        Processed audio
    """
    try:
        from scipy import signal
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Apply high-pass filter to remove low-frequency noise
        b, a = signal.butter(4, 120 / (sample_rate / 2), 'highpass')
        audio = signal.filtfilt(b, a, audio)
        
        # Apply light compression
        threshold = 0.1
        ratio = 2.0
        
        # Simple compressor
        above_threshold = audio > threshold
        below_neg_threshold = audio < -threshold
        
        audio[above_threshold] = threshold + (audio[above_threshold] - threshold) / ratio
        audio[below_neg_threshold] = -threshold + (audio[below_neg_threshold] + threshold) / ratio
        
        # Normalize again after processing
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
            
        return audio
        
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return audio

def evaluate_transcription(transcription, test_type):
    """
    Evaluate transcription accuracy for TCCC terminology.
    
    Args:
        transcription: Transcription text
        test_type: Type of test conducted
        
    Returns:
        Dictionary with evaluation results
    """
    # Medical terminology by category
    medical_terms = {
        "vital_signs": [
            "blood pressure", "pulse", "respiration", "respiratory rate", 
            "heart rate", "temperature", "pulse ox", "oxygen saturation",
            "spo2", "bp", "hr", "rr"
        ],
        "injuries": [
            "gunshot wound", "gsw", "laceration", "puncture", "fracture", 
            "amputation", "blast injury", "traumatic brain injury", "tbi",
            "tension pneumothorax", "hemothorax", "penetrating trauma"
        ],
        "procedures": [
            "tourniquet", "chest seal", "needle decompression", "nasal airway",
            "nasopharyngeal airway", "npa", "hemostatic gauze", "pressure dressing",
            "combat gauze", "chest tube", "cricothyroidotomy", "cric", "surgical airway"
        ],
        "medications": [
            "tranexamic acid", "txa", "fentanyl", "ketamine", "antibiotics",
            "morphine", "epinephrine", "combat pill pack", "hextend", "moxifloxacin"
        ],
        "assessments": [
            "massive hemorrhage", "airway", "respiration", "circulation", "head",
            "marche", "avpu", "alert", "verbal", "painful", "unresponsive", 
            "gcs", "glasgow coma scale", "tics"
        ],
        "evacuation": [
            "medevac", "casevac", "evacuation category", "urgent", "priority", 
            "routine", "tactical evacuation", "strategic evacuation", "9-liner",
            "nine-liner", "dust-off", "extraction point"
        ]
    }
    
    # Convert transcription to lowercase for case-insensitive matching
    transcription_lower = transcription.lower()
    
    # Count terms found in each category
    found_terms = {}
    total_found = 0
    
    for category, terms in medical_terms.items():
        found_in_category = []
        
        for term in terms:
            if term in transcription_lower:
                found_in_category.append(term)
                total_found += 1
        
        found_terms[category] = found_in_category
    
    # Calculate timing precision if test_type is appropriate
    timing_precision = None
    if test_type in ["trauma_assessment", "mass_casualty", "evacuation"]:
        # Look for time markers in the transcription
        time_patterns = [
            "t plus", "t minus", "minutes ago", "seconds ago", 
            "eta", "estimated time", "at time", "hours ago"
        ]
        
        found_time_markers = [pattern for pattern in time_patterns 
                             if pattern in transcription_lower]
        
        timing_precision = len(found_time_markers) > 0
    
    # Overall assessment
    total_possible = sum(len(terms) for terms in medical_terms.values())
    accuracy = total_found / total_possible if total_possible > 0 else 0
    
    return {
        "found_terms": found_terms,
        "total_found": total_found,
        "total_possible": total_possible,
        "timing_precision": timing_precision,
        "accuracy": accuracy,
        "transcription": transcription
    }

def save_results(results, filename="tccc_terminology_results.json"):
    """Save evaluation results to file."""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def generate_report(test_type, transcription, evaluation):
    """Generate a human-readable report."""
    report = []
    
    report.append("# TCCC Terminology Recognition Test Report")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Test Type: {test_type}")
    
    report.append("\n## Transcription")
    report.append(f"```\n{transcription}\n```")
    
    report.append("\n## Evaluation")
    
    # Calculate percentage
    accuracy_pct = evaluation["accuracy"] * 100
    report.append(f"Overall Recognition Rate: {accuracy_pct:.1f}%")
    report.append(f"Terms Recognized: {evaluation['total_found']} of {evaluation['total_possible']}")
    
    if evaluation["timing_precision"] is not None:
        report.append(f"Timing Precision: {'✓ Detected' if evaluation['timing_precision'] else '✗ Not Detected'}")
    
    report.append("\n## Recognition by Category")
    
    for category, terms in evaluation["found_terms"].items():
        report.append(f"\n### {category.replace('_', ' ').title()}")
        if terms:
            for term in terms:
                report.append(f"- ✓ {term}")
        else:
            report.append("- *No terms recognized in this category*")
    
    report.append("\n## Recommendations")
    
    if accuracy_pct < 50:
        report.append("- Consider using enhanced noise reduction settings")
        report.append("- Position microphone closer to the speaker")
        report.append("- Speak more clearly and deliberately when using TCCC terminology")
        report.append("- Update the STT engine with a custom medical terminology model")
    elif accuracy_pct < 80:
        report.append("- Current setup is adequate but could be improved")
        report.append("- Consider adding custom medical vocabulary for better recognition")
        report.append("- Practice consistent pronunciation of critical terms")
    else:
        report.append("- Current setup is performing well for TCCC terminology")
        report.append("- Continue monitoring performance in different environments")
        report.append("- Consider expanding the medical vocabulary database")
    
    return "\n".join(report)

def run_medical_terminology_test(test_type):
    """
    Run a test for medical terminology recognition.
    
    Args:
        test_type: Type of test to run (options: trauma_assessment, 
                  mass_casualty, evacuation, field_care)
    """
    # Define test prompts
    prompts = {
        "trauma_assessment": 
            "Please verbalize a TCCC trauma assessment for a casualty with "
            "multiple gunshot wounds, including vital signs assessment and "
            "interventions performed.",
            
        "mass_casualty": 
            "Please verbalize a triage report for a mass casualty incident "
            "with multiple casualties of varying severity, including injury "
            "types and evacuation priorities.",
            
        "evacuation": 
            "Please verbalize a MEDEVAC request (9-liner) for a casualty "
            "with a traumatic amputation and signs of shock.",
            
        "field_care": 
            "Please verbalize the tactical field care procedures you would "
            "perform for a casualty with tension pneumothorax and "
            "controlled extremity hemorrhage."
    }
    
    # Get the appropriate prompt
    if test_type not in prompts:
        print(f"Error: Unknown test type '{test_type}'")
        print(f"Available test types: {', '.join(prompts.keys())}")
        return 1
    
    prompt = prompts[test_type]
    
    print_header(f"TCCC {test_type.replace('_', ' ').title()} Test")
    
    print("In this test, you will verbalize a tactical medicine scenario.")
    print(f"Scenario: {prompt}")
    print("Please speak clearly and use standard TCCC terminology.")
    print("The system will record and transcribe your response.")
    
    # Record the audio
    print_header("Recording Audio")
    audio = record_audio(duration=30.0)  # Longer duration for medical scenarios
    
    if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
        print("Error: Recorded audio is empty or extremely quiet")
        return 1
    
    # Save the audio
    audio_file = f"tccc_{test_type}_test.wav"
    save_audio(audio, filename=audio_file)
    
    # Transcribe the audio
    print_header("Transcribing Audio")
    result = transcribe_audio(audio_file, enhance_audio=True)
    
    if "error" in result:
        print(f"Error transcribing audio: {result['error']}")
        transcription = "Transcription failed"
    else:
        transcription = result["text"]
        print(f"Transcription:\n{transcription}")
    
    # Evaluate the transcription
    print_header("Evaluating Transcription")
    evaluation = evaluate_transcription(transcription, test_type)
    
    # Print summary
    accuracy_pct = evaluation["accuracy"] * 100
    print(f"Recognition Rate: {accuracy_pct:.1f}%")
    print(f"Terms Recognized: {evaluation['total_found']} of {evaluation['total_possible']}")
    
    # Save results
    results = {
        "test_type": test_type,
        "prompt": prompt,
        "transcription": transcription,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat()
    }
    
    save_results(results, f"tccc_{test_type}_results.json")
    
    # Generate report
    report = generate_report(test_type, transcription, evaluation)
    
    with open(f"tccc_{test_type}_report.md", 'w') as f:
        f.write(report)
    
    print(f"Report saved to tccc_{test_type}_report.md")
    
    print_header("Test Complete")
    
    if accuracy_pct < 50:
        print("⚠️ Poor recognition rate - adjustments recommended")
    elif accuracy_pct < 80:
        print("ℹ️ Adequate recognition rate - minor improvements possible")
    else:
        print("✅ Good recognition rate - system is performing well")
    
    return 0

def main():
    """Main function for TCCC terminology testing."""
    print_header("TCCC Medical Terminology Test")
    
    # Determine test type
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        # If no test type specified, prompt user
        print("Available test types:")
        print("1. trauma_assessment - TCCC trauma assessment")
        print("2. mass_casualty - Mass casualty triage report")
        print("3. evacuation - MEDEVAC request (9-liner)")
        print("4. field_care - Tactical field care procedures")
        
        try:
            choice = input("Select test type (1-4): ")
            test_types = ["trauma_assessment", "mass_casualty", "evacuation", "field_care"]
            test_type = test_types[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice, defaulting to trauma_assessment")
            test_type = "trauma_assessment"
    
    return run_medical_terminology_test(test_type)

if __name__ == "__main__":
    sys.exit(main())