#!/usr/bin/env python3
"""
Battlefield Audio Enhancer for the TCCC System

This module provides practical audio enhancements for outdoor battlefield environments:
1. Realistic noise reduction tailored for outdoor settings
2. Adaptive gain control for varying speaker distances
3. Robust voice activity detection in high-noise conditions
"""

import os
import sys
import time
import wave
import numpy as np
import logging
import threading
import collections
import argparse
import soundfile as sf
from scipy import signal
from typing import Dict, List, Tuple, Optional, Union, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import TCCC components as needed
from tccc.utils.logging import get_logger
from tccc.utils.config_manager import ConfigManager
from tccc.audio_pipeline.audio_pipeline import AudioProcessor

# Configure logging
logger = get_logger("battlefield_audio_enhancer")

class BattlefieldAudioEnhancer:
    """
    Enhanced audio processing specifically optimized for battlefield scenarios.
    Designed to run efficiently on Jetson hardware with practical noise reduction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the battlefield audio enhancer.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Load default config if none provided
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.load_config("audio_pipeline")
            
            # Add battlefield-specific enhancements
            if 'battlefield_filtering' not in config:
                config['battlefield_filtering'] = {
                    'enabled': True,
                    'outdoor_mode': True,
                    'adaptive_processing': True,
                    'distance_compensation': True
                }
            
            # Add voice isolation if not present
            if 'voice_isolation' not in config:
                config['voice_isolation'] = {
                    'enabled': True,
                    'strength': 0.8,
                    'focus_width': 200,  # Hz
                    'voice_boost_db': 6   # dB
                }
        
        self.config = config
        
        # Initialize audio processor with battlefield optimizations
        self.audio_processor = AudioProcessor(config)
        
        # Voice frequency range (Hz) - optimized for tactical commands
        self.voice_freq_range = (85, 3800)  # Wider range for outdoor conditions
        
        # Adaptive settings
        self.adaptive_settings = {
            'noise_floor': 0.005,           # Initial noise floor estimate
            'speech_threshold': 0.01,       # Initial speech threshold
            'distance_factor': 1.0,         # Initial distance factor
            'environment_type': 'outdoor',  # Default environment type
            'noise_estimate': np.zeros(1024 // 2 + 1, dtype=np.float32),  # Initial noise estimate
            'gain_adaptation_speed': 0.05,  # How quickly to adapt to level changes
            'signal_history': collections.deque(maxlen=20),  # Signal level history
            'noise_history': collections.deque(maxlen=50),   # Noise level history
            'last_update_time': time.time(),
            'calibration_count': 0
        }
        
        # Frequency bands for processing (Hz)
        self.freq_bands = {
            'sub_bass': (20, 80),       # Very low rumble
            'bass': (80, 250),          # Low end
            'low_mid': (250, 800),      # Lower voice components
            'mid': (800, 2500),         # Main voice area
            'high_mid': (2500, 5000),   # Upper voice/consonants
            'high': (5000, 8000)        # High frequency detail
        }
        
        # Band-specific noise reduction strengths
        # More aggressive in non-voice bands, gentler in voice bands
        self.band_strengths = {
            'sub_bass': 0.90,   # Very aggressive for low rumble
            'bass': 0.85,       # Strong for low-frequency noise
            'low_mid': 0.60,    # Gentler in lower voice range
            'mid': 0.50,        # Gentlest in main voice range
            'high_mid': 0.65,   # Moderate in upper voice range
            'high': 0.80        # Strong in high range where noise dominates
        }
        
        # Outdoor-specific settings (can be toggled)
        self.outdoor_settings = {
            'wind_reduction_strength': 0.85,    # Wind noise reduction
            'transient_protection': True,       # Protect against sudden loud sounds
            'distance_compensation': True,      # Enhance for varying distances
            'environmental_adaptation': True    # Adapt to environment over time
        }
        
        # Runtime performance metrics
        self.performance_metrics = {
            'processing_times': collections.deque(maxlen=100),
            'speech_segments_detected': 0,
            'noise_segments_detected': 0,
            'average_speech_level': 0.0,
            'average_noise_level': 0.0
        }
        
        # Calibrate initial settings
        self._calibrate_initial_settings()
        
        logger.info("Battlefield Audio Enhancer initialized with practical optimizations")
    
    def _calibrate_initial_settings(self):
        """
        Calibrate initial settings based on environment.
        This is a lightweight process that runs at startup.
        """
        # Pre-compute frequency bin indexes for each band
        # This improves performance by avoiding repeated calculations
        self.band_bins = {}
        chunk_size = self.config.get('audio', {}).get('chunk_size', 1024)
        sample_rate = self.config.get('audio', {}).get('sample_rate', 16000)
        
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Convert frequencies to FFT bin indexes
            low_bin = int(low_freq * chunk_size / sample_rate)
            high_bin = int(high_freq * chunk_size / sample_rate)
            
            # Store bin ranges for efficient processing
            self.band_bins[band_name] = (low_bin, high_bin)
            
        # Also precompute voice frequency bins
        self.voice_bins = (
            int(self.voice_freq_range[0] * chunk_size / sample_rate),
            int(self.voice_freq_range[1] * chunk_size / sample_rate)
        )
        
        logger.info("Initial audio processing settings calibrated")
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, bool]:
        """
        Process audio with battlefield-optimized enhancements.
        
        Args:
            audio_data: Input audio data (numpy array)
            sample_rate: Sample rate of audio in Hz
            
        Returns:
            Tuple of (processed audio data, is_speech flag)
        """
        start_time = time.time()
        
        # Ensure proper format (float32 in range [-1, 1])
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32767.0
        
        # Track signal levels for adaptive processing
        current_level = np.sqrt(np.mean(audio_data ** 2))
        self.adaptive_settings['signal_history'].append(current_level)
        
        # Apply noise reduction (optimized for outdoor environments)
        if self.config.get('noise_reduction', {}).get('enabled', True):
            audio_data = self._apply_enhanced_noise_reduction(audio_data)
        
        # Apply wind noise reduction if in outdoor mode
        if self.outdoor_settings['wind_reduction_strength'] > 0 and self.outdoor_settings['environmental_adaptation']:
            audio_data = self._apply_wind_noise_reduction(audio_data)
        
        # Apply adaptive distance compensation
        if self.outdoor_settings['distance_compensation']:
            audio_data = self._apply_distance_compensation(audio_data)
        
        # Apply adaptive gain control
        audio_data = self._apply_adaptive_gain(audio_data)
        
        # Apply voice isolation if enabled
        if self.config.get('voice_isolation', {}).get('enabled', True):
            audio_data = self._apply_voice_isolation(audio_data)
        
        # Detect speech with enhanced VAD optimized for battlefield
        is_speech = self._enhanced_vad(audio_data)
        
        # Apply transient protection if enabled
        if self.outdoor_settings['transient_protection']:
            audio_data = self._protect_from_transients(audio_data)
        
        # Log performance
        processing_time = (time.time() - start_time) * 1000  # ms
        self.performance_metrics['processing_times'].append(processing_time)
        
        # Update speech/noise statistics
        if is_speech:
            self.performance_metrics['speech_segments_detected'] += 1
            self.performance_metrics['average_speech_level'] = (
                0.95 * self.performance_metrics['average_speech_level'] + 
                0.05 * current_level
            )
        else:
            self.performance_metrics['noise_segments_detected'] += 1
            # Update noise estimate more slowly
            self.adaptive_settings['noise_history'].append(current_level)
            # Only update noise floor during silence
            if len(self.adaptive_settings['noise_history']) > 10:
                self.adaptive_settings['noise_floor'] = (
                    0.9 * self.adaptive_settings['noise_floor'] + 
                    0.1 * np.percentile(list(self.adaptive_settings['noise_history']), 20)
                )
        
        # Adaptive threshold updates
        self._update_adaptive_thresholds()
        
        # Periodically adapt to environmental conditions
        current_time = time.time()
        if current_time - self.adaptive_settings['last_update_time'] > 5:  # Every 5 seconds
            self._adapt_to_environment()
            self.adaptive_settings['last_update_time'] = current_time
        
        # Return processed audio and speech detection flag
        return audio_data, is_speech
    
    def _apply_enhanced_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply enhanced noise reduction optimized for battlefield conditions.
        Uses multi-band processing for more precise control.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Noise-reduced audio data
        """
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Process each frequency band separately for more precise control
        for band_name, (low_bin, high_bin) in self.band_bins.items():
            if high_bin > low_bin:
                # Get band-specific reduction strength
                strength = self.band_strengths[band_name]
                
                # Current band magnitude
                band_magnitude = magnitude[low_bin:high_bin]
                
                # Current noise estimate for this band
                noise_estimate = self.adaptive_settings['noise_estimate'][low_bin:high_bin]
                
                # Safety check for array lengths
                if len(band_magnitude) == len(noise_estimate) and len(band_magnitude) > 0:
                    # Calculate band power
                    band_power = band_magnitude ** 2
                    
                    # Update noise estimate for this band during low-energy frames
                    frame_energy = np.mean(band_power)
                    is_low_energy = frame_energy < (np.mean(noise_estimate ** 2) * 1.5)
                    
                    if is_low_energy or self.adaptive_settings['calibration_count'] < 20:
                        # Slower noise update rate for stability
                        learn_rate = 0.1 if self.adaptive_settings['calibration_count'] < 20 else 0.02
                        noise_estimate = noise_estimate * (1 - learn_rate) + band_magnitude * learn_rate
                        self.adaptive_settings['noise_estimate'][low_bin:high_bin] = noise_estimate
                    
                    # Compute spectral subtraction gain with oversubtraction
                    # More aggressive in non-voice bands, gentler in voice bands
                    gain = np.maximum(
                        1.0 - strength * (noise_estimate / (band_magnitude + 1e-6)),
                        0.05  # Minimum gain to avoid complete suppression
                    )
                    
                    # Apply gain
                    magnitude[low_bin:high_bin] = band_magnitude * gain
                    
                    # Increment calibration counter if needed
                    if self.adaptive_settings['calibration_count'] < 50:
                        self.adaptive_settings['calibration_count'] += 1
        
        # Reconstruct signal
        fft_reduced = magnitude * np.exp(1j * phase)
        reduced_audio = np.fft.irfft(fft_reduced)
        
        # Ensure output length matches input
        reduced_audio = reduced_audio[:len(audio_data)]
        
        return reduced_audio
    
    def _apply_wind_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply focused wind noise reduction for outdoor environments.
        Wind noise is characterized by low-frequency rumble with particular
        temporal and spectral patterns.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Audio with reduced wind noise
        """
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Get low-frequency bins (wind is predominantly below 200Hz)
        low_bin, _ = self.band_bins['sub_bass']
        mid_bin, _ = self.band_bins['bass']
        high_bin = self.band_bins['low_mid'][0]
        
        # Only process if we have sufficient bins
        if high_bin > low_bin and high_bin <= len(magnitude):
            # Check for wind-like spectral characteristics (smooth, concentrated energy in low freqs)
            low_energy = np.mean(magnitude[low_bin:mid_bin])
            mid_energy = np.mean(magnitude[mid_bin:high_bin])
            
            # Wind typically has strong low band and rapidly falling energy
            wind_characteristic = (low_energy > 2 * mid_energy) and (low_energy > 0.01)
            
            # If wind-like characteristics detected, apply focused reduction
            if wind_characteristic:
                # Create gradually increasing reduction (more at lower frequencies)
                reduction_curve = np.linspace(
                    self.outdoor_settings['wind_reduction_strength'], 
                    0.3, 
                    high_bin - low_bin
                )
                
                # Ensure compatible shapes for multiplication
                if len(reduction_curve) == high_bin - low_bin:
                    # Apply frequency-dependent reduction
                    magnitude[low_bin:high_bin] = magnitude[low_bin:high_bin] * (1.0 - reduction_curve)
        
        # Reconstruct signal
        fft_reduced = magnitude * np.exp(1j * phase)
        reduced_audio = np.fft.irfft(fft_reduced)
        
        # Ensure output length matches input
        reduced_audio = reduced_audio[:len(audio_data)]
        
        return reduced_audio
    
    def _apply_distance_compensation(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply distance compensation to enhance speech at varying distances.
        This boosts certain frequencies based on estimated speaker distance.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Distance-compensated audio
        """
        # Estimate distance based on signal characteristics
        distance_factor = self._estimate_speaker_distance(audio_data)
        
        # Store for other functions to use
        self.adaptive_settings['distance_factor'] = distance_factor
        
        # Only apply compensation if significant distance detected
        if distance_factor > 1.2:
            # Convert to frequency domain
            fft_data = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            
            # Get speech clarity bands (consonants lose power with distance)
            _, high_mid_low = self.band_bins['mid']
            high_mid_low, high_mid_high = self.band_bins['high_mid']
            
            # Compensate mid-high frequencies that carry speech intelligibility
            # These attenuate with distance due to air absorption
            if high_mid_high > high_mid_low and high_mid_low < len(magnitude):
                # Distance-based compensation (more compensation with greater distance)
                compensation = np.linspace(
                    1.0,  # No boost at the low end
                    1.0 + min(0.8, (distance_factor - 1.0) * 0.5),  # Max boost at high end
                    high_mid_high - high_mid_low
                )
                
                # Apply frequency-dependent boost
                magnitude[high_mid_low:high_mid_high] = magnitude[high_mid_low:high_mid_high] * compensation
            
            # Reconstruct signal
            fft_compensated = magnitude * np.exp(1j * phase)
            compensated_audio = np.fft.irfft(fft_compensated)
            
            # Ensure output length matches input
            compensated_audio = compensated_audio[:len(audio_data)]
            
            return compensated_audio
        
        # No significant compensation needed
        return audio_data
    
    def _apply_adaptive_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply intelligent adaptive gain control based on signal characteristics.
        Automatically adjusts to speaker distance and environment.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Gain-controlled audio
        """
        # Calculate current RMS level
        current_rms = np.sqrt(np.mean(audio_data ** 2)) + 1e-6
        current_db = 20 * np.log10(current_rms)
        
        # Get target level based on environment and distance
        base_target_db = self.config.get('enhancement', {}).get('target_level_db', -16)
        
        # Adjust target based on distance factor
        distance_adjustment = min(6, max(0, (self.adaptive_settings['distance_factor'] - 1.0) * 4))
        
        # Adjust target level based on environment
        environment_adjustment = 2 if self.adaptive_settings['environment_type'] == 'outdoor' else 0
        
        # Calculate final target level
        target_db = base_target_db + distance_adjustment + environment_adjustment
        
        # Calculate required gain
        gain_db = target_db - current_db
        
        # Apply soft limiting to prevent excessive gain for very quiet signals
        if gain_db > 15:
            # Compress gains above 15dB
            gain_db = 15 + (gain_db - 15) * 0.5
        
        # Apply maximum gain limit for safety
        gain_db = min(gain_db, 24)
        
        # Convert to linear gain
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        gained_audio = audio_data * gain_linear
        
        # Soft limiting to prevent clipping
        gained_audio = np.tanh(gained_audio)
        
        return gained_audio
    
    def _apply_voice_isolation(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply targeted voice isolation to enhance speech intelligibility.
        Focuses on preserving frequencies crucial for command understanding.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Voice-enhanced audio
        """
        # Get voice isolation parameters
        strength = self.config.get('voice_isolation', {}).get('strength', 0.8)
        focus_width = self.config.get('voice_isolation', {}).get('focus_width', 200)
        voice_boost_db = self.config.get('voice_isolation', {}).get('voice_boost_db', 6)
        
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        # Create voice emphasis filter
        emphasis = np.ones_like(magnitude)
        
        # Get voice bins
        voice_low_bin, voice_high_bin = self.voice_bins
        
        # Define key intelligibility center frequency (where consonants are most important)
        # For tactical commands, clarity of consonants (s, t, k, p) is crucial
        speech_clarity_bin = int(2000 * len(emphasis) / self.config.get('audio', {}).get('sample_rate', 16000))
        
        # Create bell curve for emphasis, focusing on speech clarity range
        if voice_high_bin > voice_low_bin and voice_high_bin <= len(emphasis):
            # Create emphasis curve with peak at clarity frequency
            for i in range(voice_low_bin, voice_high_bin):
                # Calculate distance from clarity center
                distance = abs(i - speech_clarity_bin) / (focus_width * len(emphasis) / self.config.get('audio', {}).get('sample_rate', 16000))
                
                # Apply emphasis with Gaussian falloff
                if distance < 2.0:  # Within 2 standard deviations
                    # Convert boost from dB to linear and apply with distance-based falloff
                    boost_linear = 10 ** ((voice_boost_db * np.exp(-(distance ** 2))) / 20)
                    emphasis[i] = boost_linear
        
        # Apply emphasis filter
        magnitude = magnitude * emphasis
        
        # Reduce extreme low and high frequencies for cleaner result
        sub_bass_bin = self.band_bins['sub_bass'][1]  # End of sub-bass
        magnitude[:sub_bass_bin] *= 0.3  # Significant reduction of rumble
        
        # Attenuate above voice range for cleaner output
        if voice_high_bin < len(magnitude):
            high_attenuation = np.linspace(1.0, 0.2, len(magnitude) - voice_high_bin)
            magnitude[voice_high_bin:] *= high_attenuation
        
        # Reconstruct audio
        fft_isolated = magnitude * np.exp(1j * phase)
        isolated_audio = np.fft.irfft(fft_isolated)
        
        # Ensure output length matches input
        isolated_audio = isolated_audio[:len(audio_data)]
        
        return isolated_audio
    
    def _protect_from_transients(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Protect against sudden loud transients like gunshots or explosions.
        Uses intelligent envelope detection to avoid clipping while preserving dynamics.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Transient-protected audio
        """
        # Calculate short-term energy using overlapping windows
        frame_size = 64  # Short windows to catch fast transients
        hop_size = 32
        
        # Safety check for sufficient data
        if len(audio_data) < frame_size:
            return audio_data
        
        # Calculate energy in overlapping frames
        energies = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i+frame_size]
            energies.append(np.mean(frame ** 2))
        
        # Convert to numpy array for vectorized operations
        frame_energies = np.array(energies)
        
        # Detect sudden energy spikes (transients)
        has_transients = False
        if len(frame_energies) > 1:
            # Calculate frame-to-frame energy ratios
            energy_ratios = frame_energies[1:] / (frame_energies[:-1] + 1e-6)
            
            # Detect significant jumps
            transient_threshold = 4.0  # 6dB sudden increase
            has_transients = np.any(energy_ratios > transient_threshold)
        
        # If transients detected, apply protection
        if has_transients:
            # Apply soft limiting with adaptive threshold
            # Find a reasonable limit based on signal characteristics
            abs_max = np.max(np.abs(audio_data))
            
            # For extreme transients, use more aggressive limiting
            if abs_max > 0.8:
                # Hyperbolic tangent provides smooth limiting
                # Scale input to limit only the peaks
                scaling = 0.85 / abs_max
                limited_audio = np.tanh(audio_data * scaling) / scaling
                return limited_audio
        
        # No significant transients, return original
        return audio_data
    
    def _enhanced_vad(self, audio_data: np.ndarray) -> bool:
        """
        Enhanced Voice Activity Detection optimized for battlefield conditions.
        Uses multiple features to robustly detect speech even in noise.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            True if speech detected, False otherwise
        """
        # Multi-feature VAD for robust detection
        features = {}
        
        # 1. Energy-based detection
        current_energy = np.mean(audio_data ** 2)
        features['energy'] = current_energy > self.adaptive_settings['noise_floor'] * 3
        
        # 2. Spectral balance (speech has specific spectral distribution)
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        
        # Calculate energy in different bands
        band_energies = {}
        for band_name, (low_bin, high_bin) in self.band_bins.items():
            if high_bin > low_bin and high_bin <= len(magnitude):
                band_energies[band_name] = np.mean(magnitude[low_bin:high_bin] ** 2)
        
        # Speech typically has energy concentrated in mid bands
        if 'mid' in band_energies and 'sub_bass' in band_energies and band_energies['sub_bass'] > 0:
            speech_balance = band_energies['mid'] / band_energies['sub_bass']
            features['spectral_balance'] = speech_balance > 2.0
        else:
            features['spectral_balance'] = False
        
        # 3. Zero-crossing rate (higher for speech with consonants)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data)))) / len(audio_data)
        features['zero_crossings'] = zero_crossings > 0.02  # Typical speech threshold
        
        # 4. Check for speech formants
        voice_low_bin, voice_high_bin = self.voice_bins
        if voice_high_bin > voice_low_bin and voice_high_bin <= len(magnitude):
            voice_spectrum = magnitude[voice_low_bin:voice_high_bin]
            
            # Simple formant detection - look for spectral peaks
            if len(voice_spectrum) > 10:
                # Smooth spectrum for more reliable peak detection
                smooth_spectrum = np.convolve(voice_spectrum, np.ones(5)/5, mode='same')
                
                # Find local maxima (peaks)
                peaks = []
                for i in range(2, len(smooth_spectrum)-2):
                    if (smooth_spectrum[i] > smooth_spectrum[i-1] and
                        smooth_spectrum[i] > smooth_spectrum[i-2] and
                        smooth_spectrum[i] > smooth_spectrum[i+1] and
                        smooth_spectrum[i] > smooth_spectrum[i+2]):
                        peaks.append((i, smooth_spectrum[i]))
                
                # Calculate average peak prominence (height relative to surroundings)
                if peaks:
                    prominences = []
                    for idx, peak_val in peaks:
                        # Simple prominence calculation
                        left_min = min(smooth_spectrum[max(0, idx-10):idx])
                        right_min = min(smooth_spectrum[idx+1:min(len(smooth_spectrum), idx+11)])
                        prominence = peak_val - max(left_min, right_min)
                        prominences.append(prominence)
                    
                    avg_prominence = np.mean(prominences) if prominences else 0
                    features['formants'] = avg_prominence > 0.01 and len(peaks) >= 2
                else:
                    features['formants'] = False
            else:
                features['formants'] = False
        else:
            features['formants'] = False
            
        # 5. Temporal continuity (speech has time structure)
        # Use static context in this implementation for simplicity
        
        # Combine features for final decision
        # Weighting: energy (30%), spectral balance (30%), zero-crossings (10%), formants (30%)
        speech_probability = (
            0.3 * features['energy'] +
            0.3 * features['spectral_balance'] +
            0.1 * features['zero_crossings'] +
            0.3 * features['formants']
        )
        
        # Final decision with adaptive threshold
        return speech_probability > 0.5
    
    def _estimate_speaker_distance(self, audio_data: np.ndarray) -> float:
        """
        Estimate relative speaker distance based on audio characteristics.
        Uses spectral cues and signal level to estimate distance factor.
        
        Args:
            audio_data: Audio data in float format [-1, 1]
            
        Returns:
            Distance factor (1.0 = normal, >1.0 = farther)
        """
        # Multiple features for distance estimation
        
        # 1. Overall level (distant speakers are quieter)
        current_level = np.sqrt(np.mean(audio_data ** 2))
        
        # Calculate average level from history
        avg_level = np.mean(list(self.adaptive_settings['signal_history'])) if self.adaptive_settings['signal_history'] else current_level
        
        # Level-based distance (inverse relationship)
        if avg_level > 0:
            level_distance = min(3.0, 0.01 / (avg_level + 1e-6))
        else:
            level_distance = 1.0
        
        # 2. Spectral tilt (high frequencies attenuate with distance)
        # Convert to frequency domain
        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        
        # Calculate high-to-mid frequency ratio
        high_mid_low, high_mid_high = self.band_bins['high_mid']
        mid_low, mid_high = self.band_bins['mid']
        
        if high_mid_high > high_mid_low and mid_high > mid_low:
            high_energy = np.mean(magnitude[high_mid_low:high_mid_high])
            mid_energy = np.mean(magnitude[mid_low:mid_high])
            
            # Higher ratio = closer speaker (more high frequency content)
            if mid_energy > 0:
                spectral_distance = min(3.0, 2.0 / (high_energy / mid_energy + 0.5))
            else:
                spectral_distance = 1.5
        else:
            spectral_distance = 1.0
        
        # Combine factors (with more weight on spectral characteristics)
        combined_distance = 0.4 * level_distance + 0.6 * spectral_distance
        
        # Apply smoothing for stability
        current_distance = self.adaptive_settings['distance_factor']
        smoothed_distance = current_distance * 0.8 + combined_distance * 0.2
        
        # Limit to reasonable range
        return max(1.0, min(3.0, smoothed_distance))
    
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on observed audio characteristics."""
        # Get signal statistics
        if len(self.adaptive_settings['signal_history']) > 5:
            signal_levels = np.array(list(self.adaptive_settings['signal_history']))
            
            # Update speech threshold based on observed levels
            # Use a percentile approach for robustness against outliers
            if len(signal_levels) > 10:
                p75 = np.percentile(signal_levels, 75)
                p25 = np.percentile(signal_levels, 25)
                
                # Set threshold between noise floor and typical speech level
                self.adaptive_settings['speech_threshold'] = (
                    self.adaptive_settings['speech_threshold'] * 0.9 +
                    0.1 * (self.adaptive_settings['noise_floor'] * 2 + p25) / 3
                )
    
    def _adapt_to_environment(self):
        """
        Periodically adapt processing parameters to environmental conditions.
        This provides gradual adaptation to changing battlefield conditions.
        """
        # Analyze recent audio characteristics to determine environment type
        if len(self.adaptive_settings['signal_history']) > 10:
            # Calculate signal variability
            signal_levels = np.array(list(self.adaptive_settings['signal_history']))
            level_variability = np.std(signal_levels) / (np.mean(signal_levels) + 1e-6)
            
            # High variability suggests outdoor/battlefield conditions
            if level_variability > 0.5:
                self.adaptive_settings['environment_type'] = 'outdoor'
                # Increase wind reduction for outdoor
                self.outdoor_settings['wind_reduction_strength'] = min(0.9, self.outdoor_settings['wind_reduction_strength'] + 0.05)
            else:
                self.adaptive_settings['environment_type'] = 'indoor'
                # Decrease wind reduction for indoor
                self.outdoor_settings['wind_reduction_strength'] = max(0.2, self.outdoor_settings['wind_reduction_strength'] - 0.05)
    
    def process_file(self, input_file: str, output_file: str) -> bool:
        """
        Process an audio file with battlefield audio enhancements.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            
        Returns:
            Success status
        """
        try:
            # Load audio file
            logger.info(f"Loading audio file: {input_file}")
            audio_data, sample_rate = sf.read(input_file)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Process audio in chunks for memory efficiency
            chunk_size = self.config.get('audio', {}).get('chunk_size', 1024)
            processed_chunks = []
            
            logger.info(f"Processing audio file with sample rate {sample_rate}Hz")
            
            # Process in overlapping chunks for better results
            overlap = chunk_size // 4
            pos = 0
            
            # Show progress
            total_chunks = len(audio_data) // (chunk_size - overlap)
            progress_interval = max(1, total_chunks // 20)  # Show progress ~20 times
            
            chunk_counter = 0
            speech_chunks = 0
            
            while pos < len(audio_data):
                # Get chunk (with overlap for smooth processing)
                end_pos = min(pos + chunk_size, len(audio_data))
                chunk = audio_data[pos:end_pos]
                
                # Process chunk
                processed_chunk, is_speech = self.process_audio(chunk, sample_rate)
                
                # Add to processed data (discarding overlap except for the first chunk)
                if pos == 0:
                    processed_chunks.append(processed_chunk)
                else:
                    processed_chunks.append(processed_chunk[overlap:])
                
                # Update counters
                chunk_counter += 1
                if is_speech:
                    speech_chunks += 1
                
                # Show progress
                if chunk_counter % progress_interval == 0:
                    logger.info(f"Progress: {pos/len(audio_data)*100:.1f}% - " +
                               f"Speech detected in {speech_chunks}/{chunk_counter} chunks")
                
                # Move position with overlap
                pos = end_pos - overlap
                if pos >= len(audio_data):
                    break
            
            # Combine processed chunks
            processed_audio = np.concatenate(processed_chunks)
            
            # Convert to int16 for output
            processed_audio_int16 = np.clip(processed_audio * 32767, -32768, 32767).astype(np.int16)
            
            # Save to output file
            sf.write(output_file, processed_audio_int16, sample_rate)
            
            # Get processing stats
            avg_processing_time = np.mean(self.performance_metrics['processing_times'])
            
            logger.info(f"Audio processing complete: {output_file}")
            logger.info(f"Processed {chunk_counter} chunks, " +
                       f"detected speech in {speech_chunks} chunks " +
                       f"({speech_chunks/chunk_counter*100:.1f}%)")
            logger.info(f"Average processing time: {avg_processing_time:.2f}ms per chunk")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the audio enhancer.
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            'average_processing_time_ms': np.mean(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0,
            'speech_segments_detected': self.performance_metrics['speech_segments_detected'],
            'noise_segments_detected': self.performance_metrics['noise_segments_detected'],
            'speech_to_noise_ratio': (self.performance_metrics['speech_segments_detected'] / 
                                    max(1, self.performance_metrics['speech_segments_detected'] + 
                                       self.performance_metrics['noise_segments_detected'])),
            'average_speech_level': self.performance_metrics['average_speech_level'],
            'average_noise_level': self.performance_metrics['average_noise_level'],
            'estimated_snr_db': 20 * np.log10((self.performance_metrics['average_speech_level'] + 1e-6) / 
                                           (self.performance_metrics['average_noise_level'] + 1e-6)),
            'adaptive_settings': {
                'noise_floor': self.adaptive_settings['noise_floor'],
                'speech_threshold': self.adaptive_settings['speech_threshold'],
                'distance_factor': self.adaptive_settings['distance_factor'],
                'environment_type': self.adaptive_settings['environment_type']
            }
        }
        
        return stats

class MicrophoneProcessor:
    """
    Real-time processor for microphone input with battlefield enhancements.
    Handles continuous audio capture, processing, and optional output.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize microphone processor.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Load default config if none provided
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.load_config("audio_pipeline")
        
        self.config = config
        
        # Initialize audio settings
        audio_config = config.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.channels = audio_config.get('channels', 1)
        self.chunk_size = audio_config.get('chunk_size', 1024)
        self.device_id = audio_config.get('device_id', 0)
        
        # Initialize battlefield enhancer
        self.enhancer = BattlefieldAudioEnhancer(config)
        
        # Runtime state
        self.is_running = False
        self.stream = None
        self.audio = None
        self.capture_thread = None
        
        # Processing stats
        self.stats = {
            'chunks_processed': 0,
            'speech_chunks': 0,
            'start_time': 0,
            'average_level': 0.0,
            'speech_detected': False
        }
        
        # Buffer for audio output
        self.output_buffer = collections.deque(maxlen=100)
        
        # Import PyAudio here to avoid dependency when it's not needed
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.audio = pyaudio.PyAudio()
            logger.info("PyAudio initialized for microphone capture")
        except ImportError:
            logger.error("PyAudio not installed. Microphone capture will not work.")
            self.pyaudio = None
    
    def start(self, save_output: bool = False, output_file: str = None, 
              show_visual: bool = False) -> bool:
        """
        Start capturing and processing audio from microphone.
        
        Args:
            save_output: Whether to save processed audio to file
            output_file: Path to output WAV file (if saving)
            show_visual: Whether to show real-time visual feedback
            
        Returns:
            Success status
        """
        if self.is_running:
            logger.warning("Microphone processor already running")
            return False
        
        if not self.audio:
            logger.error("PyAudio not initialized")
            return False
        
        try:
            # Initialize audio output if needed
            self.save_output = save_output
            self.output_file = output_file
            self.show_visual = show_visual
            self.output_data = []
            
            # Open audio stream
            self.stream = self.audio.open(
                format=self.pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk_size
            )
            
            # Mark as running
            self.is_running = True
            self.stats['start_time'] = time.time()
            
            # Start capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                name="MicrophoneCapture",
                daemon=True
            )
            self.capture_thread.start()
            
            logger.info(f"Started microphone capture with battlefield audio enhancements")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start microphone capture: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop capturing and processing audio.
        
        Returns:
            Success status
        """
        if not self.is_running:
            logger.warning("Microphone processor not running")
            return False
        
        try:
            # Mark as stopped
            self.is_running = False
            
            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            
            # Close audio stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Save output if requested
            if self.save_output and self.output_data:
                self._save_output()
            
            # Log stats
            duration = time.time() - self.stats['start_time']
            logger.info(f"Stopped microphone capture after {duration:.1f}s")
            logger.info(f"Processed {self.stats['chunks_processed']} chunks, " +
                      f"speech detected in {self.stats['speech_chunks']} chunks " +
                      f"({self.stats['speech_chunks']/max(1, self.stats['chunks_processed'])*100:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping microphone capture: {e}")
            return False
    
    def _capture_loop(self):
        """Main audio capture and processing loop."""
        logger.info("Capture loop started")
        
        try:
            while self.is_running:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Normalize to float [-1, 1]
                float_audio = audio_data.astype(np.float32) / 32767.0
                
                # Process with battlefield enhancer
                processed_audio, is_speech = self.enhancer.process_audio(float_audio, self.sample_rate)
                
                # Update stats
                self.stats['chunks_processed'] += 1
                current_level = np.sqrt(np.mean(processed_audio ** 2))
                self.stats['average_level'] = 0.9 * self.stats['average_level'] + 0.1 * current_level
                self.stats['speech_detected'] = is_speech
                
                if is_speech:
                    self.stats['speech_chunks'] += 1
                
                # Convert back to int16 for output
                processed_int16 = np.clip(processed_audio * 32767, -32768, 32767).astype(np.int16)
                
                # Store in output buffer (always, for get_audio)
                self.output_buffer.append(processed_int16)
                
                # Save if requested
                if self.save_output:
                    self.output_data.append(processed_int16)
                
                # Show visual feedback if requested
                if self.show_visual:
                    self._show_visual(processed_audio, is_speech, current_level)
                
        except Exception as e:
            if self.is_running:  # Only log if we're supposed to be running
                logger.error(f"Error in capture loop: {e}")
        
        logger.info("Capture loop ended")
    
    def _show_visual(self, audio_data: np.ndarray, is_speech: bool, level: float):
        """
        Show real-time visual feedback of audio processing.
        
        Args:
            audio_data: Processed audio data
            is_speech: Whether speech was detected
            level: Current audio level
        """
        try:
            # Display level meter
            meter_width = 50
            level_scaled = min(1.0, level * 10)  # Scale for better visibility
            meter = '█' * int(level_scaled * meter_width) + '░' * (meter_width - int(level_scaled * meter_width))
            
            # Format with colors based on speech detection
            if is_speech:
                status = "\033[1;92mSPEAKING\033[0m"  # Green
            else:
                status = "\033[1;90mSILENCE\033[0m"   # Gray
            
            # Calculate level in dB
            db_level = 20 * np.log10(level + 1e-6)
            
            # Clear line and print status
            sys.stdout.write(f"\r\033[K|{meter}| {db_level:.1f} dB - {status}")
            sys.stdout.flush()
            
        except Exception as e:
            # Quietly handle visualization errors
            pass
    
    def _save_output(self):
        """Save processed audio to output file."""
        if not self.output_data:
            logger.warning("No output data to save")
            return
        
        try:
            # Concatenate all chunks
            output_audio = np.concatenate(self.output_data)
            
            # Save to WAV file
            output_file = self.output_file or "battlefield_enhanced_output.wav"
            
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(output_audio.tobytes())
            
            logger.info(f"Saved processed audio to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving output audio: {e}")
    
    def get_audio(self) -> np.ndarray:
        """
        Get the latest processed audio chunk.
        
        Returns:
            Latest processed audio chunk or empty array if none available
        """
        if not self.output_buffer:
            return np.array([], dtype=np.int16)
        
        return self.output_buffer[-1]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary of processing statistics
        """
        # Get base stats
        stats = dict(self.stats)
        
        # Add duration
        stats['duration'] = time.time() - self.stats['start_time'] if self.is_running else 0
        
        # Add enhancer performance stats
        stats.update(self.enhancer.get_performance_stats())
        
        return stats
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        
        Returns:
            List of audio device information dictionaries
        """
        if not self.audio:
            logger.error("PyAudio not initialized")
            return []
        
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:  # Only input devices
                devices.append({
                    'index': i,
                    'name': info.get('name', ''),
                    'channels': info.get('maxInputChannels', 0),
                    'sample_rate': info.get('defaultSampleRate', 0)
                })
        
        return devices

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Battlefield Audio Enhancer")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--file", "-f", action="store_true",
                         help="Process an audio file")
    mode_group.add_argument("--microphone", "-m", action="store_true",
                         help="Process microphone input in real-time")
    mode_group.add_argument("--list-devices", "-l", action="store_true",
                         help="List available audio input devices")
    
    # File processing options
    parser.add_argument("--input", "-i", type=str,
                      help="Input audio file (for file mode)")
    parser.add_argument("--output", "-o", type=str,
                      help="Output audio file")
    
    # Microphone options
    parser.add_argument("--device", "-d", type=int, default=0,
                      help="Audio input device ID (for microphone mode)")
    parser.add_argument("--duration", "-t", type=int, default=10,
                      help="Recording duration in seconds (for microphone mode)")
    parser.add_argument("--save", "-s", action="store_true",
                      help="Save processed microphone audio to file")
    
    # Configuration options
    parser.add_argument("--config", "-c", type=str,
                      help="Path to custom configuration file")
    
    # Display options
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Show detailed processing information")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = None
    if args.config:
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config_from_file(args.config)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return 1
    
    # List audio devices
    if args.list_devices:
        try:
            processor = MicrophoneProcessor(config)
            devices = processor.list_devices()
            
            print("\nAvailable audio input devices:")
            print("-" * 60)
            for device in devices:
                print(f"ID: {device['index']}, Name: {device['name']}")
                print(f"  Channels: {device['channels']}, Sample Rate: {device['sample_rate']}Hz")
            print("-" * 60)
            return 0
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return 1
    
    # Process audio file
    if args.file:
        if not args.input:
            logger.error("Input file required for file mode")
            return 1
        
        output_file = args.output or f"{os.path.splitext(args.input)[0]}_enhanced.wav"
        
        try:
            enhancer = BattlefieldAudioEnhancer(config)
            logger.info(f"Processing file: {args.input}")
            success = enhancer.process_file(args.input, output_file)
            
            if success:
                logger.info(f"Processing complete: {output_file}")
                # Print performance stats
                stats = enhancer.get_performance_stats()
                logger.info(f"Average processing time: {stats['average_processing_time_ms']:.2f}ms per chunk")
                logger.info(f"Speech/noise segments: {stats['speech_segments_detected']}/{stats['noise_segments_detected']}")
                logger.info(f"Estimated SNR: {stats['estimated_snr_db']:.1f}dB")
                return 0
            else:
                logger.error("Processing failed")
                return 1
                
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return 1
    
    # Process microphone input
    if args.microphone:
        try:
            # Prepare microphone processor
            processor = MicrophoneProcessor(config)
            
            # Update device ID if specified
            if args.device is not None:
                processor.device_id = args.device
            
            # Determine output file if saving
            output_file = args.output
            if args.save and not output_file:
                output_file = f"microphone_enhanced_{int(time.time())}.wav"
            
            print("\nStarting real-time battlefield audio enhancement...")
            print(f"Recording from device {processor.device_id} for {args.duration} seconds")
            if args.save:
                print(f"Saving processed audio to: {output_file}")
            print("\nAudio level meter (press Ctrl+C to stop):")
            print("-" * 60)
            
            # Start processing
            processor.start(save_output=args.save, output_file=output_file, show_visual=True)
            
            # Run for specified duration or until interrupted
            try:
                time.sleep(args.duration)
            except KeyboardInterrupt:
                print("\nStopped by user")
            
            # Stop processing
            processor.stop()
            
            # Print final stats
            stats = processor.get_stats()
            print("\n" + "-" * 60)
            print(f"Processed {stats['chunks_processed']} chunks in {stats['duration']:.1f}s")
            print(f"Speech detected in {stats['speech_chunks']} chunks ({stats['speech_chunks']/max(1, stats['chunks_processed'])*100:.1f}%)")
            print(f"Average processing time: {stats['average_processing_time_ms']:.2f}ms per chunk")
            print(f"Estimated SNR: {stats['estimated_snr_db']:.1f}dB")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error processing microphone input: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())