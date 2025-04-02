# TCCC Audio-to-Text System: Comprehensive Usage Guide

## Introduction

The TCCC Audio-to-Text System provides a high-performance speech recognition solution optimized for tactical environments. This guide explains how to use the system in various configurations.

## Quick Start

For immediate use, run:

```bash
./run_optimized_audio_stt.sh
```

This launches the system in standard mode with microphone input.

## Core Commands

### Basic Usage

```bash
./run_optimized_audio_stt.sh                    # Standard mode with microphone
./run_optimized_audio_stt.sh --battlefield      # Battlefield noise reduction mode
./run_optimized_audio_stt.sh --file --input-file test_data/test_speech.wav  # Process audio file
./run_optimized_audio_stt.sh --display          # Show transcription on display
```

### Model Options

```bash
./run_optimized_audio_stt.sh --model tiny.en    # Fast, lightweight model (default)
./run_optimized_audio_stt.sh --model base       # Medium accuracy/performance balance
./run_optimized_audio_stt.sh --model small      # Higher accuracy, more resources
```

### Performance Optimization

```bash
python preload_stt_models.py                    # Preload models at system startup
python audio_status_monitor.py                  # Monitor system performance metrics
```

## Advanced Options

### Testing and Verification

```bash
python verify_audio_stt_e2e.py --file           # Test with file input
python verify_audio_stt_e2e.py --mock           # Test with mock STT engine
python verify_audio_stt_e2e.py --cache          # Test model caching system
python verify_audio_stt_e2e.py --cache --file --model tiny.en  # Comprehensive test
```

### Development and Debugging

```bash
python test_model_cache.py                      # Test caching system in isolation
python test_stt_with_mock.py                    # Test STT engine with mock implementation
python test_basic_cache.py                      # Basic cache functionality test
```

## System Modes

### Standard Mode

Default operation with balanced performance and accuracy. Suitable for most environments with moderate background noise.

```bash
./run_optimized_audio_stt.sh
```

### Battlefield Mode

Enhanced audio processing for noisy environments. Applies specialized filtering algorithms to improve speech recognition in challenging acoustic environments.

```bash
./run_optimized_audio_stt.sh --battlefield
```

Features:
- Noise suppression
- Signal enhancement
- Adaptive VAD (Voice Activity Detection)
- Specialized audio preprocessing

### File Processing Mode

Process pre-recorded audio files instead of microphone input.

```bash
./run_optimized_audio_stt.sh --file --input-file [path_to_file]
```

Supported formats:
- WAV files (16-bit PCM)
- MP3 files (automatically converted)
- Pre-segmented battlefield recordings

## Hardware Considerations

### Microphone Configuration

The system is optimized for the Razer Seiren V3 Mini microphone (device ID 0). To use a different microphone:

```bash
./run_optimized_audio_stt.sh --device-id [device_number]
```

### Jetson Optimization

When running on Jetson hardware, the system automatically applies specialized optimizations:

- Model quantization to reduce memory usage
- CUDA acceleration for TensorRT compatibility
- Reduced model size selection for resource constraints
- Memory-efficient caching strategies

## Troubleshooting

### Common Issues

1. **"Module not initialized" error**
   - Ensure virtual environment is activated: `source venv/bin/activate`
   - Verify model files are present in models directory

2. **High latency or slow processing**
   - Try smaller model: `--model tiny.en`
   - Ensure GPU acceleration is enabled if available
   - Run `python preload_stt_models.py` before starting the system

3. **Poor transcription accuracy in noisy conditions**
   - Use battlefield mode: `--battlefield`
   - Position microphone closer to speaker
   - Run audio preprocessing calibration: `python configure_audio_enhancement.py`

4. **"StreamBuffer error" or audio capture issues**
   - Check microphone connection
   - Verify correct device ID with `python direct_mic_test.py --list-devices`
   - Restart the system if persistent

## Performance Tips

1. **For fastest startup:**
   - Run `python preload_stt_models.py` during system initialization
   - Use the default tiny.en model for quickest response

2. **For best accuracy:**
   - Use the small model: `--model small`
   - Enable battlefield mode even for moderately noisy environments
   - Set the system to use higher quality audio capture: add `--high-quality` flag

3. **For resource-constrained devices:**
   - Use the tiny.en model with minimal quality settings
   - Disable display output to save resources
   - Run in optimized mode: add `--optimize-memory` flag

## Additional Resources

- **AUDIO_STT_BENCHMARKS.md**: Detailed performance metrics
- **AUDIO_STT_INTEGRATION_COMPLETE.md**: Implementation details and architecture
- **AUDIO_STT_MANAGER_REPORT.md**: Executive summary and key benefits

## Command Reference

| Command | Purpose | Example |
|---------|---------|---------|
| run_optimized_audio_stt.sh | Main launcher script | ./run_optimized_audio_stt.sh --battlefield |
| verify_audio_stt_e2e.py | End-to-end testing | python verify_audio_stt_e2e.py --cache --file |
| preload_stt_models.py | Background model loading | python preload_stt_models.py |
| audio_status_monitor.py | System performance tracking | python audio_status_monitor.py |