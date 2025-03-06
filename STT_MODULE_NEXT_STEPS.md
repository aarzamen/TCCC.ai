# STT Module Improvement Plan

## Current Status
- Basic mock STT implementation is in place
- Need to enhance STT engine for battlefield conditions
- Faster Whisper integration needs optimization for Jetson

## Next Implementation Steps

### 1. Audio Quality Enhancement
- Implement battlefield noise cancellation
- Add adaptive gain control for varying distances
- Create voice activity detection to reduce false transcriptions

### 2. Medical Vocabulary Integration
- Integrate TCCC medical terminology
- Add specialized military/tactical terminology
- Implement context-aware terminology correction

### 3. Performance Optimization
- Quantize model for Jetson hardware
- Implement streaming transcription for real-time feedback
- Add dynamic quality settings based on available resources

### 4. Testing Framework
- Create automated testing with sample battlefield audio
- Develop WER (Word Error Rate) measurement for medical terms
- Implement continuous quality monitoring

## Technical Debt To Address
- `mock_stt.py` needs to match interface with real implementation
- Current model loading has hardcoded paths
- Missing error handling for audio capture failures
- Need proper resource cleanup on shutdown

## Dependencies
- Faster Whisper library
- PyAudio for audio capture
- TensorRT for model optimization on Jetson

## References
- See `BATTLEFIELD_AUDIO_IMPROVEMENTS.md` for more details
- Existing STT verification script shows testing approach