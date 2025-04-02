# TCCC Audio Showcase Demo Guide

## Overview

The TCCC Audio Showcase Demo is a comprehensive demonstration of the audio pipeline capabilities in the TCCC system. It provides a visual and interactive way to experience the following features:

- High-quality audio capture with configurable settings
- Real-time audio processing with multiple enhancement options
- Voice activity detection with visual feedback
- Real-time transcription with performance metrics
- Visual audio level monitoring and spectral display
- Performance benchmarking and comparison between enhancement modes

This guide will help you get the most out of the demo application.

## Installation

The demo is pre-installed with the TCCC project. You can launch it using:

1. The desktop shortcut: `TCCC_Audio_Showcase.desktop`
2. From the command line: `python3 tccc_audio_showcase_demo.py`

## Interface

The demo provides a comprehensive terminal-based interface with the following elements:

- ASCII logo and system information header
- Status display with current enhancement mode and performance metrics
- Real-time audio level meter with voice activity detection
- Recent transcriptions display
- Keyboard command menu

## Enhancement Modes

The demo supports multiple audio enhancement modes:

1. **None**: Raw audio with no enhancement (baseline)
2. **Basic**: Simple spectral subtraction and normalization
3. **Battlefield**: Specialized enhancement for tactical environments
   - Adaptive gain control
   - Outdoor noise filtering
   - Transient protection (for gunshots, explosions)
   - Distance compensation
   - Wind noise reduction
4. **FullSubNet**: Deep learning-based speech enhancement
   - Neural network-based noise suppression
   - GPU-accelerated processing (when available)
   - Advanced spectral mapping
5. **Combined**: Uses both Battlefield and FullSubNet enhancers in sequence

## Controls

The demo is controlled via keyboard:

- **Space**: Start/pause recording
- **1-5**: Switch between enhancement modes:
  - 1: None
  - 2: Basic
  - 3: Battlefield
  - 4: FullSubNet
  - 5: Combined
- **V**: Toggle visualization (requires matplotlib)
- **B**: Run benchmark comparison of all enhancement modes
- **Q**: Quit the application

## Visualization

When enabled (press 'V'), the visualization display shows:

1. **Waveform**: Real-time display of the audio waveform
2. **Spectrum**: Frequency spectrum of the audio
3. **Level**: Audio level history with voice activity detection

## Benchmark Mode

Pressing 'B' runs a comprehensive benchmark comparing all available enhancement modes. The benchmark:

1. Captures multiple audio samples
2. Processes each sample through all enhancement modes
3. Measures processing time, real-time factor (RTF), and latency
4. Saves detailed results to a file
5. Displays a summary in the interface

## Output Files

The demo saves several files to the `audio_showcase_output` directory:

- `raw_audio.wav`: Original unprocessed audio
- `enhanced_audio.wav`: Enhanced audio using the selected mode
- `transcription.txt`: Transcription of the spoken audio
- `benchmark_results.txt`: Detailed benchmark results (if run)

## Tips for Best Results

1. **Microphone Setup**: Use the Razer Seiren V3 Mini microphone when available
2. **Noise Calibration**: Remain silent during the initial calibration
3. **Enhancement Selection**:
   - Use "Basic" for quiet environments
   - Use "Battlefield" for noisy outdoor environments
   - Use "FullSubNet" for general noise reduction
   - Use "Combined" for maximum noise reduction (higher latency)
4. **Hardware Acceleration**: The demo will automatically use GPU acceleration if available

## Troubleshooting

1. **Audio Input Issues**:
   - Check microphone connection
   - Run `arecord -l` to list available audio devices
   - Specify device with `--device ID` parameter
2. **Visual Display Issues**:
   - Ensure matplotlib is installed
   - Try running in a larger terminal window
3. **Performance Issues**:
   - Switch to a simpler enhancement mode
   - Check CPU/GPU usage with monitoring tools
4. **Battlefield/FullSubNet Missing**:
   - These are optional components
   - Check for error messages during startup

## Technical Details

The demo integrates several TCCC components:

- **Audio Pipeline**: Captures and processes audio in real-time
- **STT Engine**: Transcribes enhanced audio using Faster Whisper
- **VAD Manager**: Detects when someone is speaking
- **Enhancement Modules**: Process audio for improved quality

## Extending the Demo

The demo code is designed to be extensible. You can:

1. Add new enhancement algorithms by extending the enhancers dictionary
2. Add new visualization types by extending the matplotlib plots
3. Modify benchmark parameters for more detailed analysis
4. Add support for different audio formats and sample rates

## Performance Metrics

The demo tracks and displays several performance metrics:

- **RTF (Real-Time Factor)**: How fast the processing is compared to real-time
  - RTF < 1.0 means faster than real-time
  - RTF > 1.0 means slower than real-time
- **Latency**: End-to-end processing time in milliseconds
- **Level**: Audio level in decibels (dB)
- **CPU/GPU Usage**: Resource utilization during processing

This data helps evaluate the trade-offs between enhancement quality and performance.