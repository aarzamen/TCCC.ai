# TCCC.ai Test Modules

This directory contains a set of test scripts designed to verify the functionality of the TCCC.ai system's key components and their integration. These test scripts are specifically created to focus on the Audio→STT→LLM pipeline, bypassing the DataStore functionality as noted in the project requirements.

## Test Scripts Overview

1. **01_test_audio_pipeline.py** - Tests the audio capture functionality
   - Validates microphone configuration and input levels
   - Displays real-time audio diagnostics
   - Includes visualization option for audio levels

2. **02_test_stt_engine.py** - Tests the Speech-to-Text engine
   - Captures audio and processes it through the STT engine
   - Displays transcription results with timing information
   - Validates the sequential processing of audio chunks

3. **03_test_llm_analysis.py** - Tests the LLM analysis module
   - Processes sample transcriptions through the LLM engine
   - Displays analysis results and processing times
   - Supports custom input texts and result saving

4. **04_test_module_interfaces.py** - Tests integration between modules
   - Validates the Audio→STT→LLM pipeline connections
   - Ensures proper event propagation between components
   - Generates comprehensive interface test reports

5. **05_test_mvp_flow.py** - Tests the complete MVP flow
   - Runs all components together in an end-to-end test
   - Bypasses DataStore functionality as required
   - Captures and saves all test results for analysis

## Usage Instructions

### Prerequisites
- Ensure your audio device is properly configured in `config/jetson_mvp.yaml`
- The current configuration uses device index 0 (default system microphone)

### Running the Tests

Start with basic component tests before testing the full pipeline:

```bash
# 1. Test audio pipeline to verify microphone is working
python test_modules_claude37think/01_test_audio_pipeline.py --duration 20 --visualize

# 2. Test STT engine functionality
python test_modules_claude37think/02_test_stt_engine.py --duration 30

# 3. Test LLM analysis with sample inputs
python test_modules_claude37think/03_test_llm_analysis.py

# 4. Test module interfaces
python test_modules_claude37think/04_test_module_interfaces.py --duration 45

# 5. Run complete MVP flow test
python test_modules_claude37think/05_test_mvp_flow.py --duration 60
```

### Common Options
- `--config`: Specify an alternative config file
- `--device`: Override the audio device index
- `--duration`: Set the test duration in seconds
- `--output-file/--output-dir`: Save results to file/directory

## Troubleshooting

- If no audio is detected, verify your microphone is working and properly configured
- For STT issues, check that the audio levels are sufficient (use `--visualize` with the audio test)
- If the LLM isn't generating analyses, verify the transcription pipeline is working first
- Check the enhanced debug logs which now include audio level monitoring

## Notes

These tests integrate the memory that the DataStore functionality has been commented out in `system.py` for MVP testing. The tests are designed to work with this temporary setup, focusing on the core Audio→STT→LLM pipeline functionality.
