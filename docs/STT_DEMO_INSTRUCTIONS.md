# Speech-to-Text Demo - Available Commands

## Basic Usage

```bash
# List available microphones
python demo_stt_microphone.py --list-microphones

# Run with mock STT engine (fastest)
python demo_stt_microphone.py --engine mock --device 0

# Run with Whisper (needs CUDA/GPU)
python demo_stt_microphone.py --engine whisper --device 0

# Run with Faster Whisper (optimized for Jetson)
python demo_stt_microphone.py --engine faster-whisper --device 0
```

## Tips
- Select the appropriate microphone ID using --device
- For best performance on Jetson hardware, use the faster-whisper engine
- Use Ctrl+C to stop the demo at any time

## Troubleshooting

### VAD Energy Threshold Error

If you encounter this error:
```
Error processing audio: 'AudioProcessor' object has no attribute 'vad_energy_threshold'
```

The issue occurs when the WebRTC Voice Activity Detection (VAD) module is not available, and the fallback mode doesn't properly initialize all required variables. This has been fixed in the latest version.

### Other Audio Errors

- **ALSA errors**: These are common system messages and usually don't affect functionality
- **Tensor optimization warnings**: These are related to the performance optimization for Jetson hardware, but don't affect the core functionality

## Testing Voice Recognition

The system is optimized to recognize medical terminology. Try speaking phrases like:
- "Tension pneumothorax"
- "Apply direct pressure to the wound"
- "Administer tourniquets"
- "Hemorrhage control"
- "Intraosseous access"