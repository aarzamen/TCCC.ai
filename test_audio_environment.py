#!/usr/bin/env python3
"""
TCCC Audio Environment Test.

This script analyzes the current audio environment for noise levels,
SNR estimation, and frequency analysis to optimize audio processing.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

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
        print("Please speak some medical terminology during this recording")
        print("Starting in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Recording now! Please speak...")
        
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, 
                          dtype='float32', device=device)
        sd.wait()
        
        print("Recording complete!")
        return recording.flatten()
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return np.zeros(int(duration * fs), dtype=np.float32)

def save_audio(audio, fs=16000, filename="environment_test.wav"):
    """Save audio to file."""
    try:
        import soundfile as sf
        sf.write(filename, audio, fs)
        print(f"Audio saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def get_silence_segments(audio, fs=16000, threshold=0.01, min_duration=0.5):
    """
    Identify silent segments in the audio.
    
    Args:
        audio: Audio data
        fs: Sample rate
        threshold: Amplitude threshold for silence
        min_duration: Minimum duration for a silence segment in seconds
        
    Returns:
        List of (start, end) tuples for silence segments
    """
    # Convert amplitude threshold to power threshold
    power_threshold = threshold ** 2
    
    # Calculate short-term energy
    frame_length = int(0.02 * fs)  # 20ms frames
    hop_length = int(0.01 * fs)    # 10ms hop
    
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    energy = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        energy[i] = np.mean(frame ** 2)
    
    # Find silent frames
    silent_frames = energy < power_threshold
    
    # Group contiguous silent frames
    silent_segments = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = i
        elif not is_silent and in_silence:
            # End of silence
            in_silence = False
            silence_duration = (i - silence_start) * hop_length / fs
            if silence_duration >= min_duration:
                silent_segments.append((
                    silence_start * hop_length / fs,
                    i * hop_length / fs
                ))
    
    # Handle case where audio ends in silence
    if in_silence:
        silence_duration = (len(silent_frames) - silence_start) * hop_length / fs
        if silence_duration >= min_duration:
            silent_segments.append((
                silence_start * hop_length / fs,
                len(audio) / fs
            ))
    
    return silent_segments

def calculate_noise_profile(audio, fs=16000, silence_segments=None):
    """
    Calculate noise profile from audio.
    
    Args:
        audio: Audio data
        fs: Sample rate
        silence_segments: Optional list of (start, end) tuples for silence segments
        
    Returns:
        Dictionary with noise statistics
    """
    if silence_segments is None or len(silence_segments) == 0:
        # Use the whole audio as noise profile
        noise = audio
        print("No clear silence segments found, using entire audio for noise analysis")
    else:
        # Extract and concatenate silence segments
        print(f"Using {len(silence_segments)} silence segments for noise analysis")
        noise_segments = []
        for start_sec, end_sec in silence_segments:
            start = int(start_sec * fs)
            end = int(end_sec * fs)
            noise_segments.append(audio[start:end])
        
        if noise_segments:
            noise = np.concatenate(noise_segments)
        else:
            noise = audio
    
    # Calculate noise statistics
    noise_rms = np.sqrt(np.mean(noise ** 2))
    noise_peak = np.max(np.abs(noise))
    noise_std = np.std(noise)
    
    # Calculate spectral characteristics
    try:
        from scipy import signal
        
        # Calculate power spectral density
        f, psd = signal.welch(noise, fs, nperseg=1024)
        
        # Frequency bands
        bands = [
            (0, 100),      # Sub-bass
            (100, 300),    # Bass
            (300, 1000),   # Low-mid
            (1000, 3000),  # Mid
            (3000, 7000),  # High-mid
            (7000, 8000)   # High (limited by 16kHz sample rate)
        ]
        
        band_powers = {}
        for band_name, (low, high) in zip(
            ["sub_bass", "bass", "low_mid", "mid", "high_mid", "high"],
            bands
        ):
            # Find indices for the band
            indices = np.logical_and(f >= low, f <= high)
            band_powers[band_name] = np.mean(psd[indices])
        
        # Find dominant frequency band
        dominant_band = max(band_powers.items(), key=lambda x: x[1])[0]
        
    except Exception as e:
        print(f"Error calculating spectral characteristics: {e}")
        f, psd = None, None
        band_powers = {}
        dominant_band = "unknown"
    
    return {
        "rms": float(noise_rms),
        "peak": float(noise_peak),
        "std": float(noise_std),
        "freq_bands": band_powers,
        "dominant_band": dominant_band,
        "freq": f.tolist() if f is not None else None,
        "psd": psd.tolist() if psd is not None else None
    }

def analyze_speech_segments(audio, fs=16000, silence_segments=None):
    """
    Analyze speech segments in the audio.
    
    Args:
        audio: Audio data
        fs: Sample rate
        silence_segments: Optional list of (start, end) tuples for silence segments
        
    Returns:
        Dictionary with speech statistics
    """
    if silence_segments is None or len(silence_segments) == 0:
        # Use the whole audio
        speech = audio
        print("No clear silence segments found, using entire audio for speech analysis")
    else:
        # Extract speech segments (inverse of silence segments)
        speech_segments = []
        
        # Handle first segment if it's not silence
        if silence_segments[0][0] > 0:
            speech_segments.append((0, silence_segments[0][0]))
        
        # Handle segments between silence
        for i in range(len(silence_segments) - 1):
            speech_segments.append((silence_segments[i][1], silence_segments[i+1][0]))
        
        # Handle last segment if it's not silence
        if silence_segments[-1][1] < len(audio) / fs:
            speech_segments.append((silence_segments[-1][1], len(audio) / fs))
        
        # Extract and concatenate speech segments
        print(f"Analyzing {len(speech_segments)} speech segments")
        speech_chunks = []
        for start_sec, end_sec in speech_segments:
            start = int(start_sec * fs)
            end = int(end_sec * fs)
            speech_chunks.append(audio[start:end])
        
        if speech_chunks:
            speech = np.concatenate(speech_chunks)
        else:
            speech = audio
    
    # Calculate speech statistics
    speech_rms = np.sqrt(np.mean(speech ** 2))
    speech_peak = np.max(np.abs(speech))
    speech_std = np.std(speech)
    
    # Calculate spectral characteristics
    try:
        from scipy import signal
        
        # Calculate power spectral density
        f, psd = signal.welch(speech, fs, nperseg=1024)
        
        # Calculate spectral centroid
        if len(f) == len(psd) and len(f) > 0:
            spectral_centroid = np.sum(f * psd) / np.sum(psd)
        else:
            spectral_centroid = 0
        
    except Exception as e:
        print(f"Error calculating speech spectral characteristics: {e}")
        spectral_centroid = 0
    
    return {
        "rms": float(speech_rms),
        "peak": float(speech_peak),
        "std": float(speech_std),
        "spectral_centroid": float(spectral_centroid)
    }

def estimate_snr(speech_stats, noise_stats):
    """
    Estimate signal-to-noise ratio.
    
    Args:
        speech_stats: Speech statistics dictionary
        noise_stats: Noise statistics dictionary
        
    Returns:
        SNR in dB
    """
    speech_power = speech_stats["rms"] ** 2
    noise_power = noise_stats["rms"] ** 2
    
    if noise_power > 0:
        snr_linear = speech_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
    else:
        snr_db = float('inf')
    
    return snr_db

def recommend_audio_settings(noise_profile, snr_db):
    """
    Generate recommendations for audio processing settings.
    
    Args:
        noise_profile: Noise profile dictionary
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Dictionary with recommended settings
    """
    recommendations = {}
    
    # Noise gate threshold
    noise_threshold = min(0.05, max(0.005, noise_profile["peak"] * 1.5))
    recommendations["noise_gate_threshold"] = noise_threshold
    
    # Highpass filter to remove low-frequency noise
    dominant_band = noise_profile["dominant_band"]
    if dominant_band in ["sub_bass", "bass"]:
        recommendations["highpass_cutoff"] = 120  # Hz
    else:
        recommendations["highpass_cutoff"] = 80   # Hz
    
    # Compression settings based on SNR
    if snr_db < 10:
        # Poor SNR, use stronger compression
        recommendations["compression_ratio"] = 4.0
        recommendations["compression_threshold"] = -18  # dB
    elif snr_db < 20:
        # Moderate SNR
        recommendations["compression_ratio"] = 3.0
        recommendations["compression_threshold"] = -24  # dB
    else:
        # Good SNR
        recommendations["compression_ratio"] = 2.0
        recommendations["compression_threshold"] = -30  # dB
    
    # Noise reduction amount based on SNR
    if snr_db < 15:
        recommendations["noise_reduction_amount"] = "high"
    elif snr_db < 25:
        recommendations["noise_reduction_amount"] = "medium"
    else:
        recommendations["noise_reduction_amount"] = "low"
    
    return recommendations

def generate_report(audio_duration, noise_profile, speech_stats, snr_db, recommendations):
    """Generate an analysis report."""
    report = []
    
    report.append("# TCCC Audio Environment Analysis Report")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Audio Duration: {audio_duration:.1f} seconds\n")
    
    report.append("## Noise Analysis")
    report.append(f"RMS Level: {noise_profile['rms']:.6f}")
    report.append(f"Peak Level: {noise_profile['peak']:.6f}")
    
    report.append("\nFrequency Band Analysis:")
    for band, power in noise_profile["freq_bands"].items():
        report.append(f"- {band}: {power:.8f}")
    report.append(f"Dominant Band: {noise_profile['dominant_band']}")
    
    report.append("\n## Speech Analysis")
    report.append(f"RMS Level: {speech_stats['rms']:.6f}")
    report.append(f"Peak Level: {speech_stats['peak']:.6f}")
    report.append(f"Spectral Centroid: {speech_stats['spectral_centroid']:.1f} Hz")
    
    report.append(f"\n## Signal-to-Noise Ratio")
    report.append(f"Estimated SNR: {snr_db:.1f} dB")
    
    report.append("\n## Recommended Audio Settings")
    report.append(f"Noise Gate Threshold: {recommendations['noise_gate_threshold']:.6f}")
    report.append(f"Highpass Filter Cutoff: {recommendations['highpass_cutoff']} Hz")
    report.append(f"Compression Ratio: {recommendations['compression_ratio']}")
    report.append(f"Compression Threshold: {recommendations['compression_threshold']} dB")
    report.append(f"Noise Reduction Amount: {recommendations['noise_reduction_amount']}")
    
    return "\n".join(report)

def plot_audio_analysis(audio, fs, noise_profile, silence_segments=None, filename="audio_analysis.png"):
    """
    Generate analysis plots.
    
    Args:
        audio: Audio data
        fs: Sample rate
        noise_profile: Noise profile dictionary
        silence_segments: Optional list of (start, end) tuples for silence segments
        filename: Output filename
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Time domain plot
        time = np.arange(len(audio)) / fs
        axes[0].plot(time, audio)
        axes[0].set_title("Waveform")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        
        # Highlight silence segments
        if silence_segments:
            for start, end in silence_segments:
                axes[0].axvspan(start, end, color='red', alpha=0.3)
        
        # Spectrogram
        axes[1].specgram(audio, NFFT=1024, Fs=fs, noverlap=512)
        axes[1].set_title("Spectrogram")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Frequency (Hz)")
        
        # PSD plot
        if noise_profile["freq"] is not None and noise_profile["psd"] is not None:
            f = np.array(noise_profile["freq"])
            psd = np.array(noise_profile["psd"])
            
            axes[2].semilogy(f, psd)
            axes[2].set_title("Power Spectral Density of Noise")
            axes[2].set_xlabel("Frequency (Hz)")
            axes[2].set_ylabel("PSD (V^2/Hz)")
            
            # Highlight frequency bands
            bands = [
                ("Sub-bass", 0, 100),
                ("Bass", 100, 300),
                ("Low-mid", 300, 1000),
                ("Mid", 1000, 3000),
                ("High-mid", 3000, 7000),
                ("High", 7000, 8000)
            ]
            
            colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']
            
            for (name, low, high), color in zip(bands, colors):
                band_indices = np.logical_and(f >= low, f <= high)
                if np.any(band_indices):
                    axes[2].fill_between(
                        f[band_indices], 
                        psd[band_indices], 
                        alpha=0.3, 
                        color=color, 
                        label=name
                    )
            
            axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Audio analysis plot saved to {filename}")
        
    except Exception as e:
        print(f"Error generating plots: {e}")

def save_audio_config(recommendations, filename="audio_config.yaml"):
    """Save recommended audio configuration to file."""
    try:
        config = [
            "# TCCC Audio Processing Configuration",
            f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "audio_processing:",
            f"  noise_gate_threshold: {recommendations['noise_gate_threshold']:.6f}",
            f"  highpass_cutoff: {recommendations['highpass_cutoff']}",
            f"  compression_ratio: {recommendations['compression_ratio']}",
            f"  compression_threshold: {recommendations['compression_threshold']}",
            f"  noise_reduction_amount: \"{recommendations['noise_reduction_amount']}\"",
            "",
            "microphone:",
            "  device: 0  # Razer Seiren V3 Mini",
            "  sample_rate: 16000",
            "  channels: 1",
            "  format: float32",
            "",
            "vad:",
            "  enabled: true",
            "  mode: 3  # 0-3, higher is more aggressive",
            "  frame_size: 480  # 30ms at 16kHz",
            "",
            "preprocessing:",
            "  remove_dc_offset: true",
            "  preemphasis: 0.97",
            "  normalization: true"
        ]
        
        with open(filename, 'w') as f:
            f.write('\n'.join(config))
            
        print(f"Audio configuration saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving audio configuration: {e}")
        return False

def main():
    """Main function for audio environment analysis."""
    print_header("TCCC Audio Environment Test")
    
    # Load device configuration
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "config", "device_config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            device = config['audio']['input']['device']
            print(f"Using configured audio input device: {device}")
        else:
            device = 0
            print(f"Using default audio input device: {device}")
    except Exception:
        device = 0
        print(f"Using default audio input device: {device}")
    
    # Record audio
    print_header("Recording Audio")
    audio = record_audio(duration=15.0, device=device)
    
    if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
        print("Error: Recorded audio is empty or extremely quiet")
        return 1
    
    # Save audio
    save_audio(audio, filename="environment_test.wav")
    
    # Analyze audio
    print_header("Analyzing Audio")
    
    # Find silence segments
    print("Identifying silence segments...")
    silence_segments = get_silence_segments(audio)
    print(f"Found {len(silence_segments)} silence segments")
    
    # Calculate noise profile
    print("Calculating noise profile...")
    noise_profile = calculate_noise_profile(audio, silence_segments=silence_segments)
    print(f"Noise RMS: {noise_profile['rms']:.6f}")
    print(f"Dominant noise frequency band: {noise_profile['dominant_band']}")
    
    # Analyze speech segments
    print("Analyzing speech segments...")
    speech_stats = analyze_speech_segments(audio, silence_segments=silence_segments)
    print(f"Speech RMS: {speech_stats['rms']:.6f}")
    
    # Estimate SNR
    snr_db = estimate_snr(speech_stats, noise_profile)
    print(f"Estimated SNR: {snr_db:.1f} dB")
    
    # Generate recommendations
    print_header("Generating Recommendations")
    recommendations = recommend_audio_settings(noise_profile, snr_db)
    
    for key, value in recommendations.items():
        print(f"{key}: {value}")
    
    # Generate report
    print_header("Generating Report")
    report = generate_report(len(audio) / 16000, noise_profile, speech_stats, snr_db, recommendations)
    
    with open("audio_environment_report.md", 'w') as f:
        f.write(report)
    
    print("Report saved to audio_environment_report.md")
    
    # Generate plots
    print("Generating analysis plots...")
    plot_audio_analysis(audio, 16000, noise_profile, silence_segments)
    
    # Save recommended config
    save_audio_config(recommendations, "config/audio_processing.yaml")
    
    print_header("Test Complete")
    print(f"Estimated SNR: {snr_db:.1f} dB")
    
    if snr_db < 10:
        print("⚠️ Low SNR detected - noise reduction recommended")
    elif snr_db < 20:
        print("ℹ️ Moderate SNR - some noise reduction may help")
    else:
        print("✅ Good SNR - minimal noise reduction needed")
    
    print("Generated recommendations for optimal audio processing")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())