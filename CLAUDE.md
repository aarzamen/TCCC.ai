# TCCC Project Commands

## Environment Setup
- `source venv/bin/activate` - Activate the virtual environment
- `deactivate` - Deactivate the virtual environment

## Verification Commands
- `./run_all_verifications.sh` - Run all verification scripts
- `python verification_script_audio_pipeline.py` - Run audio pipeline verification
- `python verification_script_stt_engine.py` - Run STT engine verification
- `python verification_script_llm_analysis.py` - Run LLM analysis verification

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

## Current Focus - STT Engine Improvements
- Reference document: `STT_MODULE_NEXT_STEPS.md`
- Run STT verification: `python verification_script_stt_engine.py`
- Test battlefield audio: `python test_battlefield_audio.py`
- Test mock implementation: `python -c "from tccc.stt_engine.mock_stt import MockSTTEngine; m = MockSTTEngine(); m.initialize({}); print(m.transcribe_segment(None))"`