# TCCC.ai Version Control Checklist

This checklist is designed to help properly version all project components and resolve the current inconsistencies in tracking.

## Untracked Files That Need Immediate Tracking

These files are currently untracked but appear to be integral to the project. Review and commit each:

### Documentation
- [ ] BATTLEFIELD_AUDIO_IMPROVEMENTS.md
- [ ] DISPLAY_SETUP_GUIDE.md
- [ ] TCCC_DEPLOYMENT_GUIDE.md
- [ ] TCCC_INTEGRATION_REFERENCE.md
- [ ] rag_system_improvements.md

### Configuration
- [ ] config/jetson_optimizer.yaml

### Scripts
- [ ] configure_razor_mini3.sh
- [ ] deployment_script.sh
- [ ] download_models.py
- [ ] make_deployment_guide.sh
- [ ] run_system.py
- [ ] setup_jetson_mvp.sh

### Code
- [ ] src/tccc/display/ (entire directory)
- [ ] src/tccc/document_library/medical_vocabulary.py
- [ ] src/tccc/document_library/query_engine.py
- [ ] src/tccc/document_library/vector_store.py
- [ ] src/tccc/system/ (entire directory)
- [ ] src/tccc/utils/jetson_integration.py
- [ ] src/tccc/utils/jetson_optimizer.py

### Tests
- [ ] test_battlefield_audio.py
- [ ] test_config_loading.py
- [ ] test_display_basic.py
- [ ] test_jetson_integration.py
- [ ] test_tccc_casualty_card.py
- [ ] test_waveshare_display.py
- [ ] tests/integration/test_display_integration.py
- [ ] tests/mocks/ (entire directory)
- [ ] verification_script_display.py
- [ ] verification_script_jetson_optimizer.py
- [ ] verification_script_processing_core.py
- [ ] verification_script_rag_mock.py

### Data (requires special handling)
- [ ] Determine if these should be tracked with Git LFS:
  - [ ] data/document_index/chunks.json
  - [ ] data/document_index/documents.json

## Commit Process

For each file:

1. Review content for completeness and correctness
2. Ensure no sensitive information (keys, credentials) is included
3. Complete any placeholder/incomplete implementations
4. Add tests if missing
5. Commit with semantic message: `type(scope): description`

## Modified Files Needing Review

These files are modified but not committed yet. Review changes:

- [ ] README.md
- [ ] config/audio_pipeline.yaml
- [ ] requirements.txt
- [ ] src/tccc/audio_pipeline/audio_pipeline.py
- [ ] verification_script_rag.py

## Next Steps After Tracking

1. Create comprehensive test suite for all components
2. Set up GitHub Actions for CI/CD pipeline
3. Implement Git hooks for pre-commit checks
4. Conduct code review of all newly tracked components