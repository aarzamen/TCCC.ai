# faster-whisper-small Model

This directory contains the faster-whisper-small model for speech recognition in the TCCC.ai system.

## Model Description

faster-whisper-small is Nexa AI's optimized implementation of OpenAI's Whisper small-sized model, designed for efficient speech-to-text transcription on edge devices. This model provides a good balance between accuracy and performance, making it well-suited for deployment on the Jetson Orin Nano platform.

## Technical Specifications

- **Architecture**: Encoder-decoder transformer with cross-attention
- **Base Model**: Whisper small
- **Parameters**: ~244 million
- **Embedding Dimension**: 768
- **Encoder Layers**: 12
- **Decoder Layers**: 12
- **Attention Heads**: 12
- **Model Size**: ~500MB
- **Memory Usage**: ~1GB (FP16), ~500MB (INT8)
- **Input**: Log-mel spectrogram (80 mel bands)
- **Output**: Text tokens with timestamps
- **Supported Languages**: Primarily English
- **Backend**: CTranslate2 (optimized inference)

## Performance

- **Real-time Factor (RTF)**: 0.3-0.4x (2.5-3x faster than real-time)
- **Accuracy**: Comparable to original Whisper small model
- **Word Error Rate (WER)**: ~11% on general English, ~15-20% on medical terminology
- **Latency**: ~200-400ms for typical utterances
- **Throughput**: 2.5-3x real-time on Jetson Orin Nano GPU

## Integration with TCCC.ai

The model is used by the STT Engine module for transcribing medical audio recordings. It supports:

1. **Batch Processing**: Transcribing pre-recorded audio
2. **Streaming**: Real-time transcription with partial results
3. **Word Timestamps**: Precise timing for each word
4. **Speaker Diarization**: When combined with speaker diarization module
5. **Medical Vocabulary**: Enhanced with domain-specific terminology

## Optimization for Jetson

For optimal performance on Jetson Orin Nano:

1. Use FP16 precision (`compute_type: "float16"`)
2. Enable CUDA acceleration (`enable_acceleration: true`)
3. Set appropriate thread count (`cpu_threads: 6`)
4. Apply VAD filtering for natural speech (`vad_filter: true`)
5. Configure batch size based on available memory

## Usage

The model will be automatically downloaded when first used. The files are cached in this directory for future use.

The model is loaded by the STT Engine's FasterWhisperSTT class:

```python
# This happens automatically in the STT Engine initialization
from faster_whisper import WhisperModel

model = WhisperModel(
    model_size_or_path="models/faster-whisper-small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16",
    cpu_threads=6
)
```

## Source and License

- **Source**: https://github.com/guillaumekln/faster-whisper
- **Base Model**: https://github.com/openai/whisper
- **License**: MIT License
- **Citation**: 
  ```
  @article{radford2022whisper,
    title={Robust Speech Recognition via Large-Scale Weak Supervision},
    author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
    journal={arXiv preprint arXiv:2212.04356},
    year={2022}
  }
  ```

## Alternatives

If this model doesn't meet your requirements, consider these alternatives:

- **faster-whisper-tiny**: Smallest model, fastest inference, reduced accuracy
- **faster-whisper-medium**: Higher accuracy, requires more memory and compute
- **faster-whisper-large-v2**: Highest accuracy, significantly more resources required
- **Standard Whisper ONNX**: Alternative implementation if CTranslate2 isn't available