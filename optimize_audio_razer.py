#!/usr/bin/env python3
"""
Audio Pipeline Optimizer for Razer Seiren V3 Mini microphone.
This script configures and tests the audio pipeline with optimized settings.
"""

import os
import sys
import time
import numpy as np
import threading
import yaml
from datetime import datetime
from pathlib import Path

# Add project to path
project_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_dir, 'src')
sys.path.insert(0, src_dir)

# Force real implementations
os.environ["USE_MOCK_STT"] = "0"

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioOptimizer")

# Import TCCC components
from tccc.audio_pipeline.audio_pipeline import AudioPipeline
from tccc.utils.vad_manager import VADManager, VADMode

class AudioPipelineOptimizer:
    """Optimizes the AudioPipeline for Razer Seiren V3 Mini microphone."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.audio_pipeline = None
        self.vad_manager = None
        self.config_path = os.path.join(project_dir, 'config', 'audio_pipeline.yaml')
        self.config = self.load_config(self.config_path)
        self.optimized_config = None
        self.optimized_config_path = os.path.join(project_dir, 'config', 'optimized_razer_audio.yaml')
        self.recording_path = os.path.join(project_dir, 'razer_test_recording.wav')
        self.vad_sensitivities = [1, 2, 3, 4]  # Different VAD sensitivity levels to test
        self.noise_strengths = [0.3, 0.5, 0.7]  # Different noise reduction strengths
        self.running = False
        self.meter_thread = None
        self.test_results = {}
    
    def load_config(self, config_path):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return None
    
    def save_config(self, config, config_path):
        """Save configuration to file."""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def configure_for_razer(self):
        """Configure audio pipeline specifically for Razer Seiren V3 Mini."""
        if not self.config:
            logger.error("No config loaded")
            return False
        
        # Create a copy of the config
        config = self.config.copy() if self.config else {}
        
        # Ensure required sections exist
        if 'io' not in config:
            config['io'] = {}
        if 'input_sources' not in config['io']:
            config['io']['input_sources'] = []
        
        # Configure Razer microphone
        razer_config = {
            'name': 'razer_mic',
            'type': 'microphone',
            'device_id': 0,  # Razer Seiren V3 Mini is typically device 0
            'display_name': 'Razer Seiren V3 Mini',
            'channels': 1,
            'rate': 16000,
            'chunk_size': 1024,
            'active': True
        }
        
        # Replace any existing microphone source
        config['io']['input_sources'] = [s for s in config['io']['input_sources'] 
                                      if s.get('type') != 'microphone']
        config['io']['input_sources'].append(razer_config)
        config['io']['default_input'] = 'razer_mic'
        
        # Configure VAD (Voice Activity Detection)
        if 'vad' not in config:
            config['vad'] = {}
        
        config['vad']['enabled'] = True
        config['vad']['mode'] = 'normal'  # We'll test different modes
        config['vad']['sensitivity'] = 3  # We'll test different sensitivities
        
        # Configure noise reduction
        if 'noise_reduction' not in config:
            config['noise_reduction'] = {}
        
        config['noise_reduction']['enabled'] = True
        config['noise_reduction']['strength'] = 0.5  # We'll test different strengths
        
        # Configure battlefield audio enhancement if available
        if 'battlefield_audio' not in config:
            config['battlefield_audio'] = {}
        
        config['battlefield_audio']['enabled'] = True
        
        # Store the base configuration
        self.optimized_config = config
        return True
    
    def initialize_components(self):
        """Initialize AudioPipeline and VADManager."""
        if not self.optimized_config:
            logger.error("No optimized config available")
            return False
        
        # Initialize VAD Manager
        self.vad_manager = VADManager()
        if not self.vad_manager:
            logger.error("Failed to create VAD Manager")
            return False
        
        # Initialize Audio Pipeline
        self.audio_pipeline = AudioPipeline()
        if not self.audio_pipeline:
            logger.error("Failed to create Audio Pipeline")
            return False
        
        # Initialize with config
        if not self.audio_pipeline.initialize(self.optimized_config):
            logger.error("Failed to initialize Audio Pipeline")
            return False
        
        logger.info("Components initialized successfully")
        return True
    
    def _audio_meter_thread(self):
        """Thread for displaying audio levels."""
        if not self.audio_pipeline:
            return
        
        last_speech = False
        speech_count = 0
        silence_count = 0
        
        while self.running:
            try:
                # Get audio stream
                audio_stream = self.audio_pipeline.get_audio_stream()
                if not audio_stream:
                    time.sleep(0.1)
                    continue
                
                # Read audio data
                audio_data = audio_stream.read()
                if audio_data is None or len(audio_data) == 0:
                    time.sleep(0.1)
                    continue
                
                # Calculate audio level
                level = np.max(np.abs(audio_data)) * 100 if isinstance(audio_data, np.ndarray) else 0
                
                # Check if speech detected
                is_speech = self.audio_pipeline.enhanced_speech_detection(audio_data) if hasattr(self.audio_pipeline, 'enhanced_speech_detection') else False
                
                if is_speech:
                    speech_count += 1
                    speech_indicator = "🟢 SPEECH"
                else:
                    silence_count += 1
                    speech_indicator = "⚪ silence"
                
                # Visual meter
                bars = int(level / 5)
                meter = "█" * min(bars, 20) + " " * (20 - min(bars, 20))
                
                # Print meter
                print(f"\rLevel: {level:5.1f}% |{meter}| {speech_indicator} | Speech ratio: {speech_count/(speech_count+silence_count+0.001):.2f}", end="")
                
                # Short sleep
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in audio meter: {e}")
                time.sleep(0.1)
    
    def test_vad_sensitivity(self, sensitivity):
        """Test a specific VAD sensitivity level."""
        if not self.audio_pipeline:
            logger.error("Audio pipeline not initialized")
            return None
        
        # Update VAD sensitivity
        config = self.optimized_config.copy()
        config['vad']['sensitivity'] = sensitivity
        
        # Reinitialize with new config
        self.audio_pipeline.stop_capture()
        if not self.audio_pipeline.initialize(config):
            logger.error(f"Failed to initialize with sensitivity {sensitivity}")
            return None
        
        # Start capture
        if not self.audio_pipeline.start_capture('razer_mic'):
            logger.error(f"Failed to start capture with sensitivity {sensitivity}")
            return None
        
        # Initialize metrics
        speech_count = 0
        total_count = 0
        start_time = time.time()
        
        # Start audio meter thread
        self.running = True
        self.meter_thread = threading.Thread(target=self._audio_meter_thread)
        self.meter_thread.daemon = True
        self.meter_thread.start()
        
        print(f"\n\n{'='*50}")
        print(f"Testing VAD sensitivity: {sensitivity} (5 seconds)")
        print(f"{'='*50}")
        print("Speak into the microphone to test speech detection")
        
        # Run test for 5 seconds
        try:
            for i in range(50):  # 5 seconds (10 samples per second)
                # Get audio stream
                audio_stream = self.audio_pipeline.get_audio_stream()
                if not audio_stream:
                    time.sleep(0.1)
                    continue
                
                # Read audio data
                audio_data = audio_stream.read()
                if audio_data is None or len(audio_data) == 0:
                    time.sleep(0.1)
                    continue
                
                # Check if speech detected
                is_speech = self.audio_pipeline.enhanced_speech_detection(audio_data) if hasattr(self.audio_pipeline, 'enhanced_speech_detection') else False
                
                if is_speech:
                    speech_count += 1
                total_count += 1
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            # Stop meter thread
            self.running = False
            if self.meter_thread:
                self.meter_thread.join(timeout=1.0)
            
            # Stop capture
            self.audio_pipeline.stop_capture()
        
        elapsed_time = time.time() - start_time
        speech_ratio = speech_count / total_count if total_count > 0 else 0
        
        result = {
            'sensitivity': sensitivity,
            'speech_count': speech_count,
            'total_count': total_count,
            'speech_ratio': speech_ratio,
            'elapsed_time': elapsed_time
        }
        
        print(f"\nSensitivity {sensitivity} results:")
        print(f"  Speech count: {speech_count}/{total_count} ({speech_ratio:.2f})")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        
        return result
    
    def optimize_vad(self):
        """Test different VAD sensitivity levels to find the optimal one."""
        if not self.audio_pipeline:
            logger.error("Audio pipeline not initialized")
            return False
        
        print("\n\n" + "="*50)
        print("VAD SENSITIVITY OPTIMIZATION".center(50))
        print("="*50)
        print("\nThis test will optimize VAD sensitivity for speech detection.")
        print("Please speak during each test to see how well speech is detected.")
        print("Press Ctrl+C to skip a test and move to the next sensitivity level.\n")
        
        vad_results = {}
        
        # Test each sensitivity level
        for sensitivity in self.vad_sensitivities:
            result = self.test_vad_sensitivity(sensitivity)
            if result:
                vad_results[sensitivity] = result
        
        # Find optimal sensitivity (highest speech ratio)
        optimal_sensitivity = max(vad_results, key=lambda k: vad_results[k]['speech_ratio']) if vad_results else 3
        
        print("\n\nVAD sensitivity results:")
        for sensitivity, result in vad_results.items():
            print(f"  Sensitivity {sensitivity}: Speech ratio {result['speech_ratio']:.2f}")
        
        print(f"\nOptimal VAD sensitivity: {optimal_sensitivity}")
        
        # Update config with optimal sensitivity
        self.optimized_config['vad']['sensitivity'] = optimal_sensitivity
        
        # Store results
        self.test_results['vad'] = {
            'tested_sensitivities': list(vad_results.keys()),
            'optimal_sensitivity': optimal_sensitivity,
            'results': vad_results
        }
        
        return True
    
    def optimize_noise_reduction(self):
        """Test different noise reduction strengths to find the optimal one."""
        if not self.audio_pipeline:
            logger.error("Audio pipeline not initialized")
            return False
        
        print("\n\n" + "="*50)
        print("NOISE REDUCTION OPTIMIZATION".center(50))
        print("="*50)
        print("\nThis test will optimize noise reduction strength.")
        print("Please speak during each test to see how well noise is reduced.")
        print("Press Ctrl+C to skip a test and move to the next strength level.\n")
        
        noise_results = {}
        
        # Use optimal VAD sensitivity from previous test
        optimal_sensitivity = self.optimized_config['vad']['sensitivity']
        
        # Test each noise reduction strength
        for strength in self.noise_strengths:
            # Update audio pipeline with new noise reduction strength
            if not self.audio_pipeline.set_quality_parameters({'noise_reduction': {'strength': strength}}):
                logger.error(f"Failed to set noise reduction strength {strength}")
                continue
            
            print(f"\n\n{'='*50}")
            print(f"Testing noise reduction strength: {strength} (5 seconds)")
            print(f"{'='*50}")
            print("Speak into the microphone to test noise reduction")
            
            # Start capture
            if not self.audio_pipeline.start_capture('razer_mic'):
                logger.error(f"Failed to start capture with strength {strength}")
                continue
            
            # Initialize metrics
            snr_values = []
            start_time = time.time()
            
            # Start audio meter thread
            self.running = True
            self.meter_thread = threading.Thread(target=self._audio_meter_thread)
            self.meter_thread.daemon = True
            self.meter_thread.start()
            
            # Run test for 5 seconds
            try:
                for i in range(50):  # 5 seconds (10 samples per second)
                    # Get audio stream
                    audio_stream = self.audio_pipeline.get_audio_stream()
                    if not audio_stream:
                        time.sleep(0.1)
                        continue
                    
                    # Read audio data
                    audio_data = audio_stream.read()
                    if audio_data is None or len(audio_data) == 0:
                        time.sleep(0.1)
                        continue
                    
                    # Estimate SNR (very basic)
                    if isinstance(audio_data, np.ndarray) and len(audio_data) > 0:
                        signal = np.mean(np.abs(audio_data))
                        noise = np.std(audio_data)
                        snr = signal / (noise + 1e-10)
                        snr_values.append(snr)
                    
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\nTest interrupted by user")
            finally:
                # Stop meter thread
                self.running = False
                if self.meter_thread:
                    self.meter_thread.join(timeout=1.0)
                
                # Stop capture
                self.audio_pipeline.stop_capture()
            
            elapsed_time = time.time() - start_time
            avg_snr = np.mean(snr_values) if snr_values else 0
            
            result = {
                'strength': strength,
                'snr_values': snr_values,
                'avg_snr': avg_snr,
                'elapsed_time': elapsed_time
            }
            
            print(f"\nStrength {strength} results:")
            print(f"  Average SNR: {avg_snr:.4f}")
            print(f"  Elapsed time: {elapsed_time:.2f}s")
            
            noise_results[strength] = result
        
        # Find optimal strength (highest average SNR)
        optimal_strength = max(noise_results, key=lambda k: noise_results[k]['avg_snr']) if noise_results else 0.5
        
        print("\n\nNoise reduction strength results:")
        for strength, result in noise_results.items():
            print(f"  Strength {strength}: Average SNR {result['avg_snr']:.4f}")
        
        print(f"\nOptimal noise reduction strength: {optimal_strength}")
        
        # Update config with optimal strength
        self.optimized_config['noise_reduction']['strength'] = optimal_strength
        
        # Store results
        self.test_results['noise_reduction'] = {
            'tested_strengths': list(noise_results.keys()),
            'optimal_strength': optimal_strength,
            'results': noise_results
        }
        
        return True
    
    def record_test_audio(self, duration=10):
        """Record a test audio sample with optimized settings."""
        if not self.audio_pipeline:
            logger.error("Audio pipeline not initialized")
            return False
        
        # Initialize with optimized config
        if not self.audio_pipeline.initialize(self.optimized_config):
            logger.error("Failed to initialize with optimized config")
            return False
        
        # Start capture
        if not self.audio_pipeline.start_capture('razer_mic'):
            logger.error("Failed to start capture")
            return False
        
        # Start audio meter thread
        self.running = True
        self.meter_thread = threading.Thread(target=self._audio_meter_thread)
        self.meter_thread.daemon = True
        self.meter_thread.start()
        
        # Create array to store audio data
        recorded_audio = []
        
        print("\n\n" + "="*50)
        print("RECORDING TEST AUDIO".center(50))
        print("="*50)
        print(f"\nRecording {duration} seconds of audio with optimized settings...")
        print("Please speak into the microphone to test the optimized settings.")
        print("This recording will be saved for verification.\n")
        
        # Record audio for duration
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                # Get audio stream
                audio_stream = self.audio_pipeline.get_audio_stream()
                if not audio_stream:
                    time.sleep(0.1)
                    continue
                
                # Read audio data
                audio_data = audio_stream.read()
                if audio_data is None or len(audio_data) == 0:
                    time.sleep(0.1)
                    continue
                
                # Store audio data
                if isinstance(audio_data, np.ndarray):
                    recorded_audio.append(audio_data)
                
                # Show remaining time
                remaining = duration - (time.time() - start_time)
                print(f"\rRecording: {remaining:.1f}s remaining   ", end="")
                
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            # Stop meter thread
            self.running = False
            if self.meter_thread:
                self.meter_thread.join(timeout=1.0)
            
            # Stop capture
            self.audio_pipeline.stop_capture()
        
        # Combine audio data
        if not recorded_audio:
            logger.error("No audio data recorded")
            return False
        
        combined_audio = np.concatenate(recorded_audio)
        
        # Save audio to file
        try:
            import soundfile as sf
            sf.write(self.recording_path, combined_audio, 16000)
            print(f"\nRecorded audio saved to {self.recording_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
    
    def run_optimization(self):
        """Run full optimization process."""
        print("\n" + "="*50)
        print("RAZER MICROPHONE AUDIO PIPELINE OPTIMIZATION".center(50))
        print("="*50)
        
        # Configure for Razer
        if not self.configure_for_razer():
            logger.error("Failed to configure for Razer")
            return False
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        try:
            # Optimize VAD sensitivity
            if not self.optimize_vad():
                logger.warning("VAD optimization failed or was skipped")
            
            # Optimize noise reduction
            if not self.optimize_noise_reduction():
                logger.warning("Noise reduction optimization failed or was skipped")
            
            # Record test audio
            if not self.record_test_audio(10):
                logger.warning("Test audio recording failed or was skipped")
            
            # Save optimized config
            if not self.save_config(self.optimized_config, self.optimized_config_path):
                logger.error("Failed to save optimized config")
                return False
            
            print("\n\n" + "="*50)
            print("OPTIMIZATION COMPLETE".center(50))
            print("="*50)
            print(f"\nOptimized configuration saved to {self.optimized_config_path}")
            print(f"Test recording saved to {self.recording_path}")
            print("\nOptimized settings:")
            print(f"  VAD sensitivity: {self.optimized_config['vad']['sensitivity']}")
            print(f"  Noise reduction strength: {self.optimized_config['noise_reduction']['strength']}")
            print("\nTo use these settings in your application:")
            print(f"  1. Copy {os.path.basename(self.optimized_config_path)} to {os.path.basename(self.config_path)}")
            print("  2. Or update your code to load the optimized config directly")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\nOptimization interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return False
        finally:
            # Clean up
            if self.audio_pipeline:
                self.audio_pipeline.stop_capture()
                if hasattr(self.audio_pipeline, 'shutdown'):
                    self.audio_pipeline.shutdown()

def main():
    """Main function."""
    optimizer = AudioPipelineOptimizer()
    success = optimizer.run_optimization()
    
    if success:
        print("\nOptimization completed successfully!")
        return 0
    else:
        print("\nOptimization failed or was interrupted.")
        return 1

if __name__ == "__main__":
    sys.exit(main())