# TCCC Project Commands

## Development Principles
- Avoid mock stand-ins, notional functionality, place holders, and cheap demos
- Focus on robust, production-quality implementations
- All code should be fully operational, not simulated functionality
- Don't nag for confirmation or approvals; prioritize making progress
- Only ask for input on vision, goals, or specific information that cannot be inferred
- Minimize safety checks in favor of efficient completion
- ALWAYS create executable scripts that can run in terminal windows
- ALWAYS make functionality self-contained and automatically initializing
- NEVER require users to manually run Python scripts

## Hardware Configuration
- ALWAYS use the Razer Seiren V3 Mini microphone (device ID 0)
- ALWAYS use the actual functional LLM model (Phi-2), not mock implementations
- ALWAYS create desktop shortcuts for any functionality
- ALWAYS make scripts executable with `chmod +x script_name.sh`

## Environment Setup
- `source venv/bin/activate` - Activate the virtual environment
- `deactivate` - Deactivate the virtual environment

## Verification Commands
- `./run_all_verifications.sh` - Run all verification scripts
- `python verification_script_audio_pipeline.py` - Run audio pipeline verification
- `python verification_script_stt_engine.py` - Run STT engine verification
- `python verification_script_llm_analysis.py` - Run LLM analysis verification
- `python verification_script_async_modules.py` - Verify async/sync interface compatibility
- `./verify_audio_stt_e2e.py --file --cache` - Test end-to-end audio-to-STT pipeline with model caching

## Audio-STT Module Integration
- See `AUDIO_STT_INTEGRATION_PLAN.md` for detailed integration tasks and status
- Model caching system implemented with reference counting for faster performance
- Model preloading at system startup added via preload_stt_models.py
- Optimized audio-to-STT pipeline available via ./run_optimized_audio_stt.sh
- Desktop shortcuts added for different modes (standard, battlefield)
- End-to-end verification with verify_audio_stt_e2e.py --cache option

## Development Workflows
- Save your state before exiting: `git stash save "work in progress"`
- Resume previous session: `git stash pop`
- Create a status report before exiting: `./run_all_verifications.sh > verification_status.log`

## Session Management
- Create a session marker: `echo "Last session: $(date)" > .last_session`
- Record current tasks: `echo "Current tasks: task1, task2" >> .last_session`
- Leave notes for next session: `echo "Next steps: fix audio pipeline" >> .last_session`

## Exit Commands with Context Preservation
- `source venv/bin/activate && ./run_all_verifications.sh > last_status.log && echo "Session ended: $(date)" >> .last_session`
- `python -c "import json; open('.session_state.json', 'w').write(json.dumps({'last_component': 'audio_pipeline', 'status': 'needs_fixing'}))"` 

## Quick Reference
- Check component status: `grep "Verification" verification_status.log`
- List modified files: `git status -s`
- Save current branch work: `git add . && git commit -m "WIP: Save current progress" && git push origin HEAD`
- Hardware note: Display resolution 1560x720 via HDMI

## Working with Large Files
- Use `grep -n "pattern" filename` to find specific lines in large files
- Use `tail -n 100 filename` or `head -n 100 filename` to view beginning or end of large files
- Use Python chunk processing: `python -c "with open('file.py') as f: print(''.join(f.readlines()[500:600]))"`
- Use file splitting: `split -l 1000 large_file.py split_file_`
- For large Python files, use GrepTool pattern matching for specific functions: `"def\s+function_name"`
- When examining large Python modules, first search for class definitions: `"class\s+ClassName"`

## Gemini 2.5 Pro Integration
- Use Gemini 2.5 Pro's 1M token context window for large-scale code analysis
- Configuration stored in: `scripts/development/gemini_config.json`
- API key already configured: Gemini API key set up in config file
- Commands (log file approach for SSH sessions):
  - Analyze a module: `./scripts/development/gemini_analyze.sh -o /home/ama/tccc-project/gemini_analysis_log.txt module "query" module_name`
  - Analyze specific files: `./scripts/development/gemini_analyze.sh -o /home/ama/tccc-project/gemini_analysis_log.txt files "query" file1 file2`
  - Search with glob pattern: `./scripts/development/gemini_analyze.sh -o /home/ama/tccc-project/gemini_analysis_log.txt glob "query" "src/**/*.py"`
- Log file accessed at: `/home/ama/tccc-project/gemini_analysis_log.txt`
- For large analyses, allow 30-60 seconds for the log file to populate
- Documentation: `scripts/development/GEMINI_TOOL_README.md`

## Efficient Code Search
- Use the efficient search utility: `./scripts/development/efficient_search.sh "search pattern"`
- Search specific file types: `./scripts/development/efficient_search.sh -i "*.py" "AudioPipeline"`
- Search specific directories: `./scripts/development/efficient_search.sh -p "src/tccc" "initialize"`
- List only matching files: `./scripts/development/efficient_search.sh -l "def process_audio"`
- This utility automatically excludes:
  - Large model files (*.gguf, *.pth, *.faiss)
  - Virtual environment directories
  - Audio recordings (*.wav)
  - Log files and transcripts
  - Cache directories

## Transcription Errors
- If a filename looks like a transcription error, it's likely meant to be an existing file
- Common examples: "CLAUDE.md" might be transcribed as "Quad code.md", "Cloud code.md", etc.
- Always check for similar existing files before creating new ones
- When in doubt, assume CLAUDE.md is the intended reference

## Completed - Audio Pipeline to STT Integration
- ✅ Integration complete! See `AUDIO_STT_INTEGRATION_COMPLETE.md` for details
- ✅ Fixed critical integration issues. See `audio_stt_integration_verified.txt` for summary.
- Run optimized pipeline: `./run_optimized_audio_stt.sh`
- Run with battlefield mode: `./run_optimized_audio_stt.sh --battlefield`
- Run with file input: `./run_optimized_audio_stt.sh --file --input-file test_data/test_speech.wav`
- Test model caching: `python verify_audio_stt_e2e.py --cache --file`
- Run model preloading: `python preload_stt_models.py`
- Status and testing: `python audio_status_monitor.py`

## Critical Fixes Implemented
- ✅ Fixed StreamBuffer API compatibility between different implementations
- ✅ Improved ONNX conversion with proper fallback to PyTorch
- ✅ Added patching for torch.compiler to prevent speaker diarization errors
- ✅ Implemented proper shutdown() method in STTEngine with resource cleanup
- ✅ Added enhanced error handling and recovery throughout the pipeline

## Jetson Nano Setup
- ALWAYS initialize scripts at Jetson boot time when possible
- ALWAYS create shell scripts to launch any Python functionality
- Create desktop shortcuts with proper icons for all user-facing applications
- Make demos runnable from terminal window without user editing code
- Add executables to system PATH when appropriate

## Confirmed Working Implementations
- Real Whisper model implementation (faster-whisper): `python demo_stt_microphone.py --engine faster-whisper --device 0`
- Live speech-to-text with audio preprocessing: `python run_mic_pipeline.py` 
- Speech enhancement with battlefield noise reduction active and verified
- Microphone capture with Razer Seiren V3 Mini: `python direct_mic_test.py --device 0`
- Display output through standard HDMI: `python test_display_enhanced.py`
- Complete STT pipeline with display: `python microphone_to_text.py --use-display`
- Optimized audio-to-STT pipeline with model caching: `./run_optimized_audio_stt.sh`
- Model preloading for faster startup: `python preload_stt_models.py`
- Battlefield audio enhancement mode: `./run_optimized_audio_stt.sh --battlefield`
- Audio status monitoring: `python audio_status_monitor.py`
- End-to-end testing with model caching: `python verify_audio_stt_e2e.py --cache --file`