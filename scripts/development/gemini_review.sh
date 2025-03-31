#!/bin/bash
# gemini_review.sh - Comprehensive Code Review with Gemini 2.5 Pro (Focused Version)

# Ensure we're in the project root directory
cd "$(dirname "$0")/../.."

# Define the log file for analysis output
ANALYSIS_LOG="gemini_code_review.txt"

# Get the most recent verification status
if [ -f "audio_stt_integration_verified.txt" ]; then
  VERIFICATION_STATUS=$(cat audio_stt_integration_verified.txt)
else
  VERIFICATION_STATUS="No verification status found"
fi

echo "Preparing focused code review on critical components..."

# Create a comprehensive prompt for Gemini
PROMPT="Conduct a detailed code review focusing on the integration between AudioPipeline and STTEngine components in the TCCC project. The verification status shows: '$VERIFICATION_STATUS'. We have recently fixed issues with StreamBuffer API mismatches, ONNX conversion fallbacks, torch.compiler patching for speaker diarization, and proper resource cleanup in the STTEngine. Identify any remaining issues, technical debt, or potential optimizations. Structure your analysis by subsystem and provide specific code examples where applicable."

# Use Gemini to analyze the key files in STT Engine and Audio Pipeline directly
echo "Analyzing core integration files..."
FILES="src/tccc/stt_engine/stt_engine.py src/tccc/stt_engine/model_cache_manager.py src/tccc/audio_pipeline/audio_pipeline.py src/tccc/audio_pipeline/stream_buffer.py src/tccc/utils/event_bus.py"
./scripts/development/gemini_analyze.sh -o $ANALYSIS_LOG files "$PROMPT" $FILES

echo "Code review completed! Results saved to $ANALYSIS_LOG"
echo "You can review the findings using: cat $ANALYSIS_LOG"