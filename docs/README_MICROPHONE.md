# TCCC Microphone System

This document describes the TCCC (Tactical Combat Casualty Care) microphone speech recognition system implemented on the Jetson Nano with a Razer Seiren V3 Mini microphone and WaveShare display.

## System Components

- **Razer Seiren V3 Mini Microphone** (USB, device ID 0)
- **Jetson Nano** (processing platform)
- **WaveShare Display** (1280x800 resolution)
- **Faster-Whisper STT Engine** (speech-to-text)
- **Phi-2 LLM Model** (for analysis)

## Quick Start

1. Connect the Razer Seiren V3 Mini microphone to any USB port on the Jetson Nano
2. Connect the WaveShare display via HDMI
3. Double-click the "TCCC Microphone" desktop icon or run `./tccc_mic_launcher.sh` from terminal

## Functionality

The TCCC Microphone System provides:

1. **High-Quality Audio Capture**
   - 44.1kHz sampling rate for superior audio quality
   - Noise reduction and normalization
   - Battlefield audio enhancement for outdoor use
   - Support for FullSubNet speech enhancement on supported hardware
   - Advanced audio chunk management for optimal processing

2. **Real-time Speech Transcription**
   - Uses actual Faster-Whisper STT model (not mock)
   - Medical term recognition for TCCC context
   - Real-time visual feedback during speech
   - Terminal-based graphical interface

3. **Optimized for Jetson Nano**
   - GPU acceleration where available
   - Memory-optimized processing
   - Terminal-based graphical interface for display
   - Automatic launch in visible window

## File Outputs

When you run the system, it produces:

- `improved_transcription.txt` - The transcribed speech with timestamps
- `improved_audio.wav` - The processed audio used for transcription (16kHz)
- `highquality_audio.wav` - The original high-quality audio (44.1kHz)

## Advanced Usage

The microphone system can be run with additional options:

```bash
# Run with specific audio enhancement mode
./tccc_mic_launcher.sh --enhancement battlefield

# Available enhancement modes:
# - auto (default - uses best available enhancer)
# - battlefield (optimized for outdoor environments)
# - fullsubnet (deep learning based enhancement)
# - both (applies both enhancers in sequence)
# - none (no enhancement, basic noise reduction only)
```

## Troubleshooting

### ALSA Warnings

You may see ALSA warnings like:
```
ALSA lib pcm_dmix.c:1032:(snd_pcm_dmix_open) unable to open slave
```

These are common and generally do not affect functionality. They are related to ALSA trying to access devices it doesn't need for our application.

### No Microphone Detected

If no microphone is detected:
1. Check USB connections
2. Verify the microphone is recognized by the system:
   ```bash
   arecord -l
   ```
3. Try a different USB port or microphone

### Display Issues
- Verify the WaveShare display is connected via HDMI
- Check if `echo $DISPLAY` returns `:0`
- Ensure X server is running with `ps aux | grep X`

### Low Audio Levels
1. Check microphone physical volume controls if present
2. Adjust system input volume:
   ```bash
   alsamixer
   ```
3. Position the microphone closer to the audio source

### Performance Issues
- Close other applications to free memory
- Reduce enhancement level with `--enhancement none`
- Verify GPU acceleration is enabled

## Integration with TCCC System

The microphone system is integrated with the full TCCC pipeline:

1. Audio capture from Razer microphone
2. Audio enhancement and noise reduction
3. Speech-to-text conversion using Faster-Whisper
4. Medical term recognition
5. LLM analysis using Phi-2 model
6. RAG-based document lookup
7. Display on WaveShare screen

## Testing Tools

Two diagnostic scripts are also available for testing:

### 1. Direct PyAudio Test

This script bypasses the TCCC modules and directly tests microphone functionality:

```bash
python direct_mic_test.py [device_id]
```

### 2. TCCC Audio Pipeline Test

This script tests just the TCCC AudioPipeline with microphone input:

```bash
python test_microphone_tccc.py
```

## For Developers

The main components are:

- `microphone_to_text.py` - Main speech capture and processing script
- `tccc_mic_launcher.sh` - Launcher script for easy execution
- `TCCC_Microphone.desktop` - Desktop shortcut for system integration

The full TCCC system uses the MicrophoneSource class in `audio_pipeline.py` to capture audio, which:

1. Initializes the audio capture device
2. Processes audio in real-time (noise reduction, VAD)
3. Provides processed audio to downstream components (STT, analysis)

The implementation has been optimized for battlefield conditions with enhanced noise filtering and voice activity detection.