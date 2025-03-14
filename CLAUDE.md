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
- ALWAYS use the WaveShare display with 1280x800 resolution
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

## Current Focus - System Integration
- Run system verification: `python verification_script_system_enhanced.py`
- Test data flow: `python test_system_data_flow.py`
- Debug event processing: `python -c "from tccc.system.system import TCCCSystem; s = TCCCSystem(); s.initialize({}); s.process_event({'type': 'test', 'data': 'test'}); print(s.get_status())"`

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
- Display output on WaveShare screen (1280x800): `python test_display_enhanced.py`
- Complete STT pipeline with display: `python microphone_to_text.py --use-display`

## Multi-Agent Collaboration
- Split tasks between multiple Claude instances for parallel work
- Each agent focuses on specific components:
  - Agent 1: Audio pipeline and STT engine
  - Agent 2: LLM analysis and document library
  - Agent 3: Processing core and system integration
- Use shared documentation for coordination:
  - Create task files with naming pattern: `AGENT<N>_TASK.md`
  - Update shared status in `AGENT_STATUS.md`
  - Leave code comments with `<!-- AGENT<N>: comment -->` format
- Before merging changes, run `./check_agent_conflicts.sh` to verify no conflicts
- Use workflow: `python create_multi_agent_workspace.py --agents 3 --task "System Integration"`