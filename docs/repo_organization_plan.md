# TCCC.ai Repository Organization Plan

## New Repository Structure

```
tccc-ai/
├── .github/                    # GitHub-specific files
│   ├── CODEOWNERS              # Defines ownership of code sections
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md       # Template for bug reports
│   │   ├── feature_request.md  # Template for feature requests
│   │   └── documentation.md    # Template for documentation issues
│   ├── PULL_REQUEST_TEMPLATE.md # Template for pull requests
│   └── workflows/              # GitHub Actions workflows
│       ├── ci.yml              # Main CI workflow
│       ├── docs.yml            # Documentation generation workflow
│       └── releases.yml        # Release workflow
├── .gitignore                  # Updated Git ignore file
├── docs/                       # Documentation
│   ├── architecture/           # System architecture docs
│   │   ├── images/             # Architecture diagrams
│   │   └── system_architecture.md
│   ├── development/            # Development guides
│   │   ├── contribution_guide.md
│   │   ├── development_guide.md
│   │   └── testing_guide.md
│   ├── guides/                 # User guides 
│   │   ├── deployment.md
│   │   ├── display_setup.md
│   │   └── microphone_setup.md
│   ├── modules/                # Module-specific documentation
│   │   ├── audio_pipeline.md
│   │   ├── document_library.md
│   │   ├── llm_analysis.md
│   │   ├── processing_core.md
│   │   └── stt_engine.md
│   ├── index.md                # Main documentation page
│   └── mkdocs.yml              # MkDocs configuration
├── scripts/                    # Utility scripts
│   ├── ci/                     # CI/CD scripts
│   │   ├── run_tests.sh
│   │   └── verify_dependencies.sh
│   ├── deployment/             # Deployment scripts
│   │   ├── create_deployment_package.sh
│   │   └── deployment_script.sh
│   ├── development/            # Development scripts
│   │   ├── claude_code/        # Claude Code helpers
│   │   │   ├── extract_cloudcode_section.sh
│   │   │   ├── prepare_claude_session.sh
│   │   │   └── use_snippet.sh
│   │   ├── setup_git_hooks.sh
│   │   └── setup_workspace.sh
│   ├── display/                # Display setup scripts
│   │   ├── configure_waveshare_display.sh
│   │   ├── fix_waveshare_display.sh
│   │   └── setup_waveshare_display.sh
│   ├── hardware/               # Hardware configuration scripts
│   │   ├── configure_razor_mini3.sh
│   │   └── install_torch_jetson.sh
│   └── models/                 # Model download scripts
│       ├── download_models.py
│       ├── download_phi2_gguf.sh
│       ├── download_silero_vad.py
│       └── download_stt_model.py
├── src/                        # Source code
│   ├── tccc/                   # Main package
│   │   ├── __init__.py
│   │   ├── audio_pipeline/     # Audio processing
│   │   │   ├── __init__.py
│   │   │   ├── audio_pipeline.py
│   │   │   └── stream_buffer.py
│   │   ├── cli/                # Command-line interface
│   │   │   ├── __init__.py
│   │   │   └── commands.py
│   │   ├── data_store/         # Data storage
│   │   │   ├── __init__.py
│   │   │   └── data_store.py
│   │   ├── display/            # Display interface
│   │   │   ├── __init__.py
│   │   │   └── display_interface.py
│   │   ├── document_library/   # Document management
│   │   │   ├── __init__.py
│   │   │   ├── cache_manager.py
│   │   │   ├── document_library.py
│   │   │   ├── document_processor.py
│   │   │   ├── medical_vocabulary.py
│   │   │   ├── query_engine.py
│   │   │   ├── response_generator.py
│   │   │   └── vector_store.py
│   │   ├── form_generator/     # Form generation
│   │   │   ├── __init__.py
│   │   │   ├── field_extractor.py
│   │   │   └── form_generator.py
│   │   ├── llm_analysis/       # LLM analysis
│   │   │   ├── __init__.py
│   │   │   ├── llm_analysis.py
│   │   │   ├── phi_gguf_model.py
│   │   │   └── phi_model.py
│   │   ├── processing_core/    # Core processing
│   │   │   ├── __init__.py
│   │   │   ├── entity_extractor.py
│   │   │   ├── intent_classifier.py
│   │   │   ├── plugin_manager.py
│   │   │   ├── processing_core.py
│   │   │   ├── resource_monitor.py
│   │   │   ├── sentiment_analyzer.py
│   │   │   └── state_manager.py
│   │   ├── stt_engine/         # Speech-to-text
│   │   │   ├── __init__.py
│   │   │   ├── faster_whisper_stt.py
│   │   │   └── stt_engine.py
│   │   ├── system/             # System integration
│   │   │   ├── __init__.py
│   │   │   ├── display_integration.py
│   │   │   └── system.py
│   │   └── utils/              # Utilities
│   │       ├── __init__.py
│   │       ├── audio_chunk_manager.py
│   │       ├── audio_data_converter.py
│   │       ├── config.py
│   │       ├── config_manager.py
│   │       ├── event_schema.py
│   │       ├── jetson_integration.py
│   │       ├── jetson_optimizer.py
│   │       ├── logging.py
│   │       ├── module_adapter.py
│   │       ├── tensor_optimization.py
│   │       └── vad_manager.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Test configuration
│   ├── integration/            # Integration tests
│   │   ├── __init__.py
│   │   ├── test_audio_pipeline_integration.py
│   │   ├── test_display_integration.py
│   │   ├── test_system_integration.py
│   │   └── test_vad_integration.py
│   ├── resources/              # Test resources
│   │   ├── audio/              # Test audio files
│   │   │   ├── test_mic.wav
│   │   │   └── test_speech.wav
│   │   ├── documents/          # Test documents
│   │   │   ├── test_json.json
│   │   │   ├── test_markdown.md
│   │   │   └── test_text.txt
│   │   └── models/             # Small test models
│   │       └── README.md
│   └── unit/                   # Unit tests
│       ├── __init__.py
│       ├── audio_pipeline/
│       │   ├── __init__.py
│       │   └── test_audio_pipeline.py
│       ├── data_store/
│       │   ├── __init__.py
│       │   └── test_data_store.py
│       ├── document_library/
│       │   ├── __init__.py
│       │   └── test_document_library.py
│       ├── llm_analysis/
│       │   ├── __init__.py
│       │   └── test_llm_analysis.py
│       ├── processing_core/
│       │   ├── __init__.py
│       │   ├── test_module_registration.py
│       │   ├── test_processing_core.py
│       │   ├── test_resource_allocation.py
│       │   └── test_state_manager.py
│       ├── stt_engine/
│       │   ├── __init__.py
│       │   └── test_stt_engine.py
│       └── utils/
│           ├── __init__.py
│           └── test_logging.py
├── examples/                   # Usage examples
│   ├── audio_processing/       # Audio processing examples
│   │   ├── battlefield_audio_enhancer.py
│   │   └── run_enhanced_audio.sh
│   ├── full_system/            # Complete system examples
│   │   ├── run_full_tccc_demo.sh
│   │   └── tccc_mic_to_display.py
│   ├── microphone/             # Microphone examples
│   │   ├── direct_mic_test.py
│   │   └── microphone_to_text.py
│   ├── rag/                    # RAG examples
│   │   ├── jetson_rag_explorer.py
│   │   └── rag_explorer.py
│   └── stt/                    # STT examples
│       ├── demo_stt_microphone.py
│       └── simple_stt_demo.py
├── tools/                      # Development tools
│   ├── cloud_code/             # Claude Code guidance
│   │   ├── CLOUDCODE.md
│   │   ├── CLOUDCODE_README.md
│   │   └── CLOUDCODE_SNIPPETS.txt
│   └── verification/           # Verification scripts
│       ├── run_all_verifications.sh
│       ├── verification_script_audio_pipeline.py
│       ├── verification_script_system.py
│       └── verify_phi2.py
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── LICENSE                     # Project license (MIT)
├── pyproject.toml              # Modern Python project configuration
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution guidelines
├── CODE_OF_CONDUCT.md          # Code of conduct
├── SECURITY.md                 # Security policy
└── setup.py                    # Setup script (for backward compatibility)
```

## Implementation Notes

1. **Key Changes:**
   - Organized all loose scripts into appropriate directories under `scripts/`, `examples/`, and `tools/`
   - Maintained the core source code structure in `src/tccc/`
   - Consolidated test code into the `tests/` directory with a clear separation between unit and integration tests
   - Created a structured documentation system under `docs/`
   - Added proper GitHub configuration files in `.github/`
   - Preserved critical files like README.md, CONTRIBUTING.md, and SECURITY.md at the repository root for public visibility

2. **Guiding Principles:**
   - Maintained backward compatibility where possible
   - Followed Python project best practices
   - Ensured clear module boundaries for code review
   - Properly structured documentation for public consumption
   - Organized utility scripts by purpose and function
   - Created a clear separation between examples and core code

3. **Next Steps:**
   - Create GitHub Actions workflows for CI/CD
   - Update import paths in all Python files
   - Create a proper documentation system with MkDocs
   - Configure code quality tools