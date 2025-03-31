# FullSubNet Speech Enhancement for TCCC

This integration adds the FullSubNet speech enhancement model to the TCCC project's microphone processing pipeline.

## Overview

FullSubNet is a state-of-the-art speech enhancement model developed by Westlake University. It uses deep learning to remove noise from speech, significantly improving speech quality and intelligibility in challenging conditions. This implementation is specifically optimized for Nvidia Jetson hardware.

## Installation

To install FullSubNet and its dependencies:

```bash
cd /home/ama/tccc-project/fullsubnet_integration
./fullsubnet_setup.sh
```

This script will:
1. Clone the FullSubNet repository
2. Install necessary dependencies
3. Download a pre-trained model
4. Configure the integration

## Usage

FullSubNet is integrated with the microphone_to_text.py script and can be used in three ways:

### 1. Automatic mode (default)
```bash
python microphone_to_text.py
```
This will automatically use FullSubNet if available.

### 2. Specific enhancer mode
```bash
python microphone_to_text.py --enhancement fullsubnet
python microphone_to_text.py --enhancement battlefield
python microphone_to_text.py --enhancement both
python microphone_to_text.py --enhancement none
```

### 3. For testing and comparison
```bash
cd /home/ama/tccc-project/fullsubnet_integration
python test_fullsubnet.py --benchmark --input /path/to/audio.wav
```

## Performance

On the Jetson hardware:
- Processing time: ~50-100ms per chunk (depends on model size and GPU)
- Real-time factor: ~0.2-0.5x (can process audio faster than real-time)
- GPU memory usage: ~500MB-1GB (depends on model size)
- Estimated SNR improvement: 5-10dB (depends on noise conditions)

## Components

- `fullsubnet_enhancer.py`: Main integration class that wraps the FullSubNet model
- `test_fullsubnet.py`: Testing and benchmarking script
- `fullsubnet_setup.sh`: Installation script
- `fullsubnet_config.yaml`: Configuration file

## Implementation Notes

- The enhancer uses CUDA acceleration when available
- Mixed precision is used for faster inference
- Automatic fallback to CPU if GPU is not available
- Seamless integration with existing Whisper STT pipeline
