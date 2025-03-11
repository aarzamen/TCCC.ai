#!/bin/bash

# TCCC.ai Enhanced Workspace Setup Script
# This script sets up a multi-terminal workspace with Claude instances for TCCC.ai model integration

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"
BROWSER="firefox"  # Change to your preferred browser (firefox, google-chrome, etc.)
CLAUDE_URL="https://claude.ai/chat"

# Check dependencies
if ! command -v wmctrl &> /dev/null; then
    echo "Installing wmctrl..."
    sudo apt-get update && sudo apt-get install -y wmctrl
fi

if ! command -v xdpyinfo &> /dev/null; then
    echo "Installing x11-utils for xdpyinfo..."
    sudo apt-get update && sudo apt-get install -y x11-utils
fi

# Set up model-integration branch
echo "Setting up model-integration branch..."
git checkout -b model-integration 2>/dev/null || git checkout model-integration

# Create model directories if they don't exist
mkdir -p "$PROJECT_DIR/models/stt" "$PROJECT_DIR/models/llm" "$PROJECT_DIR/models/embeddings"

# Function to open a terminal with specific profile and commands
open_terminal() {
    local profile="$1"
    local title="$2"
    local commands="$3"
    local geometry="$4"
    
    gnome-terminal --window-with-profile="$profile" \
                   --title="$title" \
                   --geometry="$geometry" \
                   -- bash -c "cd $PROJECT_DIR && source $VENV_ACTIVATE && echo -e '\e]0;$title\a' && $commands; exec bash"
}

# Function to open Claude in a browser with specific instructions
open_claude_with_prompt() {
    local title="$1"
    local position="$2"
    local module="$3"
    local prompt_file="/tmp/claude_prompt_${module}.txt"
    
    # Create prompt file based on module
    case "$module" in
        "stt")
            cat > "$prompt_file" << EOF
I'll be working exclusively on the STT Engine module to implement a real Whisper model instead of the mock implementation. I will:

1. First review the current STT Engine module structure, interfaces, and mock implementations
2. Download an appropriate Whisper model variant for the Jetson Orin Nano
3. Update the module to use the real model instead of mocks
4. Implement proper interfaces for audio transcription
5. Run verification tests to ensure functionality
6. Document all changes made

I'll respect module boundaries and only modify STT Engine files. Let's start by examining the current STT Engine code structure.
EOF
            ;;
        "llm")
            cat > "$prompt_file" << EOF
I'll be working exclusively on the LLM Analysis module to implement a real language model instead of the mock implementation. I will:

1. First review the current LLM Analysis module structure, interfaces, and mock implementations
2. Select and download an appropriate LLM model for the Jetson Orin Nano (Llama-2, Phi-2, etc.)
3. Update the module to use the real model instead of mocks
4. Implement proper interfaces for text processing and medical entity extraction
5. Run verification tests to ensure functionality
6. Document all changes made

I'll respect module boundaries and only modify LLM Analysis files. Let's start by examining the current LLM Analysis code structure.
EOF
            ;;
        "doc")
            cat > "$prompt_file" << EOF
I'll be working exclusively on the Document Library (RAG) module to implement real embedding models and vector database instead of the mock implementation. I will:

1. First review the current Document Library module structure, interfaces, and mock implementations
2. Select and download an appropriate embedding model for the Jetson Orin Nano
3. Set up a vector database (FAISS, Chroma, etc.) with sample medical documents
4. Update the module to use the real components instead of mocks
5. Run verification tests to ensure functionality
6. Document all changes made

I'll respect module boundaries and only modify Document Library files. Let's start by examining the current Document Library code structure.
EOF
            ;;
        "integration")
            cat > "$prompt_file" << EOF
As the Integration Coordinator, my role is to monitor and synchronize the work of the other three terminals. I will:

1. Track the progress of each module's implementation
2. Identify and resolve cross-module dependencies or conflicts
3. Monitor system resources to prevent overload
4. Update the central configuration to reflect model changes
5. Prepare for integration testing once individual modules are complete
6. Help debug any issues that span module boundaries
7. Alert if system resources approach critical levels

Let's start by examining the overall project structure and dependencies between modules.
EOF
            ;;
    esac
    
    # Open browser with Claude
    $BROWSER "$CLAUDE_URL" &
    sleep 3
    
    # Get window ID of the most recently opened browser window
    local window_id=$(wmctrl -l | grep "$BROWSER" | tail -1 | awk '{print $1}')
    
    # Set window name for easier identification
    wmctrl -i -r "$window_id" -T "$title"
    
    # Position the window based on provided coordinates
    wmctrl -i -r "$window_id" -e "$position"
    
    # Display instructions to copy/paste the prompt
    echo -e "\n>>> CLAUDE PROMPT FOR $module MODULE <<<"
    echo -e "Copy and paste this prompt into the Claude window titled \"$title\":"
    echo -e "------------------------------------------------------------"
    cat "$prompt_file"
    echo -e "------------------------------------------------------------\n"
}

# Function to create a module-specific README file
create_module_readme() {
    local module="$1"
    local readme_file="$PROJECT_DIR/${module}_IMPLEMENTATION_NOTES.md"
    
    case "$module" in
        "STT")
            cat > "$readme_file" << EOF
# STT Engine Implementation Notes

## Current Status
- [ ] Code structure review complete
- [ ] Model selection complete
- [ ] Model download script created
- [ ] Jetson hardware acceleration implemented
- [ ] Audio chunking optimized
- [ ] Verification testing passed

## Key Files
- \`src/tccc/stt_engine/faster_whisper_stt.py\`
- \`src/tccc/stt_engine/stt_engine.py\`
- \`verification_script_stt_engine.py\`

## Implementation Steps
1. Review current mockups and interfaces
2. Download appropriate Whisper model
3. Implement model initialization
4. Add Jetson-specific optimizations
5. Update verification script
6. Test and validate

## Resources
- Model storage: \`models/stt/\`
- Memory budget: 1-2GB
- Expected inference time: <500ms per 5s audio chunk

## Integration Notes
- Input from Audio Pipeline
- Output to LLM Analysis
- Status reporting to Integration Coordinator
EOF
            ;;
        "LLM")
            cat > "$readme_file" << EOF
# LLM Analysis Implementation Notes

## Current Status
- [ ] Code structure review complete
- [ ] Model selection complete
- [ ] Model download script created
- [ ] Inference optimization implemented
- [ ] Medical entity extraction working
- [ ] Verification testing passed

## Key Files
- \`src/tccc/llm_analysis/phi_model.py\`
- \`src/tccc/llm_analysis/llm_analysis.py\`
- \`verification_script_llm_analysis.py\`

## Implementation Steps
1. Review current mockups and interfaces
2. Download appropriate LLM (Phi-2 or Llama-2)
3. Implement model loading and inference
4. Optimize for Jetson hardware
5. Create medical entity extraction pipeline
6. Test and validate

## Resources
- Model storage: \`models/llm/\`
- Memory budget: 2-3GB
- Expected inference time: <2s per request

## Integration Notes
- Input from STT Engine
- Output to Document Library and Form Generator
- Status reporting to Integration Coordinator
EOF
            ;;
        "DOC")
            cat > "$readme_file" << EOF
# Document Library Implementation Notes

## Current Status
- [ ] Code structure review complete
- [ ] Embedding model selection complete
- [ ] Vector database setup complete
- [ ] Query interface implemented
- [ ] Response generation optimized
- [ ] Verification testing passed

## Key Files
- \`src/tccc/document_library/vector_store.py\`
- \`src/tccc/document_library/document_library.py\`
- \`src/tccc/document_library/query_engine.py\`
- \`verification_script_document_library.py\`

## Implementation Steps
1. Review current mockups and interfaces
2. Download appropriate embedding model
3. Set up FAISS vector database
4. Process sample medical documents
5. Implement query and retrieval
6. Test and validate

## Resources
- Model storage: \`models/embeddings/\`
- Document storage: \`data/documents/\`
- Memory budget: 1GB
- Expected query time: <200ms

## Integration Notes
- Input from LLM Analysis
- Output to LLM Analysis and Processing Core
- Status reporting to Integration Coordinator
EOF
            ;;
        "INTEGRATION")
            cat > "$readme_file" << EOF
# Integration Coordinator Notes

## Current Status
- [ ] Module interface analysis complete
- [ ] Configuration files updated
- [ ] Resource monitoring implemented
- [ ] Integration tests created
- [ ] End-to-end verification passing

## Key Files
- \`src/tccc/system/system.py\`
- \`verification_script_system.py\`
- \`run_system.py\`
- \`config/*.yaml\`

## Integration Checklist
1. STT Engine ↔ Audio Pipeline
2. STT Engine ↔ LLM Analysis
3. LLM Analysis ↔ Document Library
4. Document Library ↔ Processing Core
5. All modules ↔ System Integration

## Resource Allocation
- STT Engine: 1-2GB RAM
- LLM Analysis: 2-3GB RAM
- Document Library: 1GB RAM
- System Overhead: 2-3GB RAM

## System Verification Tests
- \`verification_script_system.py\`
- \`test_battlefield_audio.py\`
- \`run_all_verifications.sh\`
EOF
            ;;
    esac
    
    echo "Created implementation notes for $module module: $readme_file"
}

# Create READMEs for each module
create_module_readme "STT"
create_module_readme "LLM"
create_module_readme "DOC"
create_module_readme "INTEGRATION"

# Get screen dimensions
SCREEN_WIDTH=$(xdpyinfo | awk '/dimensions/{print $2}' | cut -d 'x' -f1)
SCREEN_HEIGHT=$(xdpyinfo | awk '/dimensions/{print $2}' | cut -d 'x' -f2)

# Calculate window dimensions
HALF_WIDTH=$((SCREEN_WIDTH / 2 - 20))
QUARTER_HEIGHT=$((SCREEN_HEIGHT / 2 - 60))
DASHBOARD_HEIGHT=180
DASHBOARD_Y=$((SCREEN_HEIGHT - DASHBOARD_HEIGHT - 60))

echo "Setting up TCCC.ai workspace with Claude instances for model integration..."

# Terminal 1: STT Engine Module (Top-left)
echo "Setting up STT Engine terminal and Claude instance..."
open_terminal "CC agent 1" "TCCC: STT Engine" "echo -e '\033[1;33m===== STT ENGINE MODULE =====\033[0m\n' && cat STT_IMPLEMENTATION_NOTES.md && echo -e '\nPreparing STT Engine environment...' && find src/tccc/stt_engine -type f | xargs ls -la && python -c \"from tccc.stt_engine.stt_engine import STTEngine; print('\\nSTT Engine module initialized successfully')\"" "80x24+0+0"
open_claude_with_prompt "Claude: STT Engine" "0,0,0,$HALF_WIDTH,$QUARTER_HEIGHT" "stt"

# Terminal 2: LLM Analysis Module (Top-right)
echo "Setting up LLM Analysis terminal and Claude instance..."
open_terminal "CC agent 2" "TCCC: LLM Analysis" "echo -e '\033[1;36m===== LLM ANALYSIS MODULE =====\033[0m\n' && cat LLM_IMPLEMENTATION_NOTES.md && echo -e '\nPreparing LLM Analysis environment...' && find src/tccc/llm_analysis -type f | xargs ls -la && python -c \"from tccc.llm_analysis.llm_analysis import LLMAnalysis; print('\\nLLM Analysis module initialized successfully')\"" "80x24+0+0"
open_claude_with_prompt "Claude: LLM Analysis" "0,$HALF_WIDTH,0,$HALF_WIDTH,$QUARTER_HEIGHT" "llm"

# Terminal 3: Document Library Module (Bottom-left)
echo "Setting up Document Library terminal and Claude instance..."
open_terminal "CC agent 3" "TCCC: Document Library" "echo -e '\033[1;32m===== DOCUMENT LIBRARY MODULE =====\033[0m\n' && cat DOC_IMPLEMENTATION_NOTES.md && echo -e '\nPreparing Document Library environment...' && find src/tccc/document_library -type f | xargs ls -la && python -c \"from tccc.document_library.document_library import DocumentLibrary; print('\\nDocument Library module initialized successfully')\"" "80x24+0+0"
open_claude_with_prompt "Claude: Document Library" "0,0,$QUARTER_HEIGHT,$HALF_WIDTH,$QUARTER_HEIGHT" "doc"

# Terminal 4: Integration Coordinator (Bottom-right)
echo "Setting up Integration Coordinator terminal and Claude instance..."
open_terminal "CC agent 4" "TCCC: Integration" "echo -e '\033[1;35m===== INTEGRATION COORDINATOR =====\033[0m\n' && cat INTEGRATION_IMPLEMENTATION_NOTES.md && echo -e '\nPreparing Integration environment...' && ./run_all_verifications.sh" "80x24+0+0"
open_claude_with_prompt "Claude: Integration" "0,$HALF_WIDTH,$QUARTER_HEIGHT,$HALF_WIDTH,$QUARTER_HEIGHT" "integration"

# Dashboard at the bottom
echo "Setting up Dashboard terminal..."
open_terminal "CC agent 1" "TCCC: Dashboard" "echo -e '\033[1;37m===== TCCC SYSTEM DASHBOARD =====\033[0m\n' && echo -e 'System resources:\n' && free -h && echo -e '\nStarting dashboard...' && sleep 2 && python dashboard.py" "160x12+0+0"
wmctrl -r "TCCC: Dashboard" -e "0,0,$DASHBOARD_Y,$SCREEN_WIDTH,$DASHBOARD_HEIGHT"

# Instructions and summary
cat << EOF

===============================
TCCC.ai Model Integration Workspace
===============================

Workspace Layout:
- Top-left: STT Engine Terminal + Claude instance
- Top-right: LLM Analysis Terminal + Claude instance
- Bottom-left: Document Library Terminal + Claude instance
- Bottom-right: Integration Coordinator Terminal + Claude instance
- Bottom: Dashboard spanning full width

Created Implementation Notes:
- STT_IMPLEMENTATION_NOTES.md
- LLM_IMPLEMENTATION_NOTES.md
- DOC_IMPLEMENTATION_NOTES.md
- INTEGRATION_IMPLEMENTATION_NOTES.md

Architecture Overview:
- TCCC_IMPLEMENTATION_ARCHITECTURE.md

Next Steps:
1. Copy the displayed prompts into each Claude window
2. Review the module implementation notes
3. Start working on each module according to the plan
4. Check integration points regularly

EOF

echo "TCCC.ai model integration workspace setup complete!"