# FullSubNet Integration Plan

## Overview
This document outlines the steps to integrate the FullSubNet speech enhancement model into the TCCC project's audio processing pipeline.

## Implementation Steps

1. **Clone/Download FullSubNet Repository**
   - Source: https://github.com/Audio-WestlakeU/FullSubNet
   - Target directory: `/home/ama/tccc-project/fullsubnet_integration`

2. **Install Dependencies**
   - PyTorch and torchaudio
   - SoundFile library
   - Additional FullSubNet requirements

3. **Download Pre-trained Model**
   - Identify suitable pre-trained model from FullSubNet
   - Download model weights

4. **Create Integration Module**
   - Create `fullsubnet_enhancer.py` wrapper class
   - Implement preprocessing and postprocessing functions
   - Ensure GPU acceleration via CUDA

5. **Pipeline Integration**
   - Insert FullSubNet processing before sending audio to Whisper
   - Modify `microphone_to_text.py` to use FullSubNet
   - Add configuration options

6. **Testing and Benchmarking**
   - Create comparison scripts for before/after audio quality
   - Measure transcription accuracy improvements
   - Monitor GPU/CPU usage

## Folder Structure
```
/fullsubnet_integration/
├── fullsubnet/               # Cloned repository
├── models/                   # Pre-trained models
├── fullsubnet_enhancer.py    # Integration wrapper
├── test_fullsubnet.py        # Testing script
└── README.md                 # Documentation
```

## Implementation Details

### FullSubNet Wrapper
The wrapper will:
1. Handle audio format conversions
2. Process audio chunks in real-time
3. Leverage GPU acceleration
4. Provide fallback options

### Pipeline Integration
The integration will:
1. Add FullSubNet as an optional preprocessing step
2. Allow toggling between battlefield enhancer and FullSubNet
3. Allow cascading both enhancers if desired
