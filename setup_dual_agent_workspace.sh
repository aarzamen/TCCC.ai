#!/bin/bash

# TCCC.ai Dual-Agent Workspace Setup Script
# This script sets up a workspace with two Claude instances for TCCC.ai model integration

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
    local agent_type="$3"
    local prompt_file="/tmp/claude_prompt_${agent_type}.txt"
    
    # Create prompt file based on agent type
    case "$agent_type" in
        "agent1")
            cat > "$prompt_file" << EOF
I'll be working on implementing real models for the TCCC.ai project, focusing on the STT Engine and Document Library modules:

For the STT Engine module:
1. First review the current STT Engine module structure and interfaces
2. Download an appropriate Whisper model variant for the Jetson Orin Nano
3. Update the STT module to use the real model instead of mocks
4. Implement proper interfaces for audio transcription
5. Run verification tests to ensure functionality

For the Document Library module:
1. Implement real embedding models and vector database
2. Select and download an appropriate embedding model 
3. Set up a vector database (FAISS) with sample medical documents
4. Update the module to use real components instead of mocks
5. Ensure query interface is functioning properly

Let's start by examining the current STT Engine code structure, which is our first priority.
EOF
            ;;
        "agent2")
            cat > "$prompt_file" << EOF
I'll be working on implementing real models for the TCCC.ai project, focusing on the LLM Analysis module and System Integration:

For the LLM Analysis module:
1. First review the current LLM Analysis module structure and interfaces
2. Select and download an appropriate LLM model for the Jetson Orin Nano (Phi-2)
3. Update the module to use the real model instead of mocks
4. Implement proper interfaces for text processing and medical entity extraction
5. Run verification tests to ensure functionality

For System Integration:
1. Monitor integration between modules
2. Update the central configuration to reflect model changes
3. Prepare integration testing once individual modules are complete
4. Help debug any issues that span module boundaries
5. Ensure resource monitoring is properly implemented

Let's start by examining the current LLM Analysis code structure, which is our first priority.
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
    echo -e "\n>>> CLAUDE PROMPT FOR $agent_type <<<"
    echo -e "Copy and paste this prompt into the Claude window titled \"$title\":"
    echo -e "------------------------------------------------------------"
    cat "$prompt_file"
    echo -e "------------------------------------------------------------\n"
}

# Create the implementation architecture document
cat > "$PROJECT_DIR/TCCC_DUAL_AGENT_PLAN.md" << EOF
# TCCC.ai Dual-Agent Implementation Plan

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TCCC.ai System                                  │
│                                                                         │
│  ┌──────────────┐    ┌───────────────┐    ┌────────────────────┐        │
│  │              │    │               │    │                    │        │
│  │  STT Engine  │───►│  LLM Analysis │───►│  Document Library  │        │
│  │              │    │               │    │                    │        │
│  └──────────────┘    └───────────────┘    └────────────────────┘        │
│        ▲                    ▲                       ▲                   │
│        │                    │                       │                   │
│        └────────────────────┼───────────────────────┘                   │
│                             │                                           │
│                      ┌──────────────┐                                   │
│                      │              │                                   │
│                      │ Integration  │                                   │
│                      │ Coordinator  │                                   │
│                      │              │                                   │
│                      └──────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Dual-Agent Implementation Strategy

### Agent 1: STT Engine + Document Library
- Focus on audio processing and information retrieval
- Responsible for model selection, download, and integration
- Implement hardware acceleration and optimizations
- Create interfaces for integration with other modules

### Agent 2: LLM Analysis + System Integration
- Focus on language processing and system coordination
- Implement real LLM model and medical entity extraction
- Create integration points between modules
- Monitor resource usage and system performance

### Claude Code as Orchestrator
- Coordinate between both agents
- Resolve cross-module issues
- Track implementation progress
- Ensure consistent interfaces
- Manage resource allocation

## Implementation Timeline

### Phase 1: Preparation (Day 1)
- Environment setup and dependency installation
- Module interface analysis
- Model selection and resource planning

### Phase 2: Model Implementation (Days 2-4)
- Agent 1: Implement STT Engine with Whisper model
- Agent 2: Implement LLM Analysis with Phi-2
- Agent 1: Implement Document Library with embedding model
- Agent 2: Prepare system integration framework

### Phase 3: Integration (Day 5)
- Connect STT Engine to LLM Analysis
- Connect LLM Analysis to Document Library
- Implement resource monitoring
- Create end-to-end verification tests

### Phase 4: Verification & Documentation (Day 6)
- Run comprehensive test suite
- Optimize performance bottlenecks
- Document implementation details
- Create deployment package

## Module-Specific Implementation Details

### STT Engine (Agent 1)
- Model: Whisper Tiny or Base (INT8 quantized)
- Memory budget: 1-2GB
- Key files:
  - \`src/tccc/stt_engine/faster_whisper_stt.py\`
  - \`src/tccc/stt_engine/stt_engine.py\`
  - \`verification_script_stt_engine.py\`

### Document Library (Agent 1)
- Embedding model: MiniLM or BERT-tiny
- Vector DB: FAISS with flat index
- Memory budget: 1GB
- Key files:
  - \`src/tccc/document_library/vector_store.py\`
  - \`src/tccc/document_library/document_library.py\`
  - \`src/tccc/document_library/query_engine.py\`

### LLM Analysis (Agent 2)
- Model: Phi-2 (INT8 quantized) or Llama-2 7B (4-bit)
- Memory budget: 2-3GB
- Key files:
  - \`src/tccc/llm_analysis/phi_model.py\`
  - \`src/tccc/llm_analysis/llm_analysis.py\`
  - \`verification_script_llm_analysis.py\`

### System Integration (Agent 2)
- Resource monitoring and management
- Configuration management
- Key files:
  - \`src/tccc/system/system.py\`
  - \`run_system.py\`
  - \`verification_script_system.py\`

## Resource Management

### Memory Allocation Strategy
- STT Engine: 1-2GB RAM
- LLM Analysis: 2-3GB RAM
- Document Library: 1GB RAM
- System Overhead: 2-3GB RAM

### Resource Monitoring
- Memory Warning Threshold: 75%
- Memory Critical Threshold: 90%
- CPU Throttling: 85% sustained usage
- Staged model loading to optimize memory usage

## Integration Points

### STT Engine → LLM Analysis
- Interface: \`analyze_transcript(text, context) → structured_data\`

### LLM Analysis → Document Library
- Interface: \`retrieve_documents(query) → ranked_results\`

### All Modules → System Integration
- Status reporting
- Resource monitoring
- Error handling

## Success Criteria
1. All verification scripts pass
2. End-to-end pipeline operates without errors
3. System stays within resource budgets
4. Latency meets acceptable thresholds
5. Documentation is complete

## Next Steps
1. Begin implementation with STT Engine
2. Proceed to LLM Analysis
3. Integrate Document Library
4. Complete System Integration
5. Run comprehensive verification
EOF

echo "Created implementation plan: TCCC_DUAL_AGENT_PLAN.md"

# Create agent-specific implementation notes
cat > "$PROJECT_DIR/AGENT1_IMPLEMENTATION_NOTES.md" << EOF
# Agent 1: STT Engine & Document Library Implementation Notes

## STT Engine Module

### Current Status
- [ ] Code structure review complete
- [ ] Model selection complete
- [ ] Model download script created
- [ ] Jetson hardware acceleration implemented
- [ ] Audio chunking optimized
- [ ] Verification testing passed

### Key Files
- \`src/tccc/stt_engine/faster_whisper_stt.py\`
- \`src/tccc/stt_engine/stt_engine.py\`
- \`verification_script_stt_engine.py\`

### Implementation Steps
1. Review current mockups and interfaces
2. Download appropriate Whisper model
3. Implement model initialization
4. Add Jetson-specific optimizations
5. Update verification script
6. Test and validate

### Resources
- Model storage: \`models/stt/\`
- Memory budget: 1-2GB
- Expected inference time: <500ms per 5s audio chunk

## Document Library Module

### Current Status
- [ ] Code structure review complete
- [ ] Embedding model selection complete
- [ ] Vector database setup complete
- [ ] Query interface implemented
- [ ] Response generation optimized
- [ ] Verification testing passed

### Key Files
- \`src/tccc/document_library/vector_store.py\`
- \`src/tccc/document_library/document_library.py\`
- \`src/tccc/document_library/query_engine.py\`
- \`verification_script_document_library.py\`

### Implementation Steps
1. Review current mockups and interfaces
2. Download appropriate embedding model
3. Set up FAISS vector database
4. Process sample medical documents
5. Implement query and retrieval
6. Test and validate

### Resources
- Model storage: \`models/embeddings/\`
- Document storage: \`data/documents/\`
- Memory budget: 1GB
- Expected query time: <200ms
EOF

cat > "$PROJECT_DIR/AGENT2_IMPLEMENTATION_NOTES.md" << EOF
# Agent 2: LLM Analysis & System Integration Implementation Notes

## LLM Analysis Module

### Current Status
- [ ] Code structure review complete
- [ ] Model selection complete
- [ ] Model download script created
- [ ] Inference optimization implemented
- [ ] Medical entity extraction working
- [ ] Verification testing passed

### Key Files
- \`src/tccc/llm_analysis/phi_model.py\`
- \`src/tccc/llm_analysis/llm_analysis.py\`
- \`verification_script_llm_analysis.py\`

### Implementation Steps
1. Review current mockups and interfaces
2. Download appropriate LLM (Phi-2 or Llama-2)
3. Implement model loading and inference
4. Optimize for Jetson hardware
5. Create medical entity extraction pipeline
6. Test and validate

### Resources
- Model storage: \`models/llm/\`
- Memory budget: 2-3GB
- Expected inference time: <2s per request

## System Integration Module

### Current Status
- [ ] Module interface analysis complete
- [ ] Configuration files updated
- [ ] Resource monitoring implemented
- [ ] Integration tests created
- [ ] End-to-end verification passing

### Key Files
- \`src/tccc/system/system.py\`
- \`verification_script_system.py\`
- \`run_system.py\`
- \`config/*.yaml\`

### Implementation Steps
1. Review module interfaces
2. Update configuration files
3. Implement resource monitoring
4. Create integration tests
5. Run end-to-end verification

### Integration Checklist
1. STT Engine ↔ Audio Pipeline
2. STT Engine ↔ LLM Analysis
3. LLM Analysis ↔ Document Library
4. Document Library ↔ Processing Core
5. All modules ↔ System Integration
EOF

echo "Created implementation notes for both agents"

# Get screen dimensions
SCREEN_WIDTH=$(xdpyinfo | awk '/dimensions/{print $2}' | cut -d 'x' -f1)
SCREEN_HEIGHT=$(xdpyinfo | awk '/dimensions/{print $2}' | cut -d 'x' -f2)

# Calculate window dimensions
HALF_WIDTH=$((SCREEN_WIDTH / 2 - 20))
WINDOW_HEIGHT=$((SCREEN_HEIGHT / 2 - 80))
DASHBOARD_HEIGHT=150
DASHBOARD_Y=$((SCREEN_HEIGHT - DASHBOARD_HEIGHT - 60))

echo "Setting up TCCC.ai dual-agent workspace..."

# Terminal 1: Agent 1 (STT Engine + Document Library)
echo "Setting up Agent 1 terminal and Claude instance..."
open_terminal "CC agent 1" "TCCC: Agent 1 (STT + Doc)" "echo -e '\033[1;33m===== AGENT 1: STT ENGINE + DOCUMENT LIBRARY =====\033[0m\n' && cat AGENT1_IMPLEMENTATION_NOTES.md && echo -e '\nPreparing environment...' && echo -e '\nSTT Engine files:' && find src/tccc/stt_engine -type f | sort && echo -e '\nDocument Library files:' && find src/tccc/document_library -type f | sort" "100x30+0+0"
open_claude_with_prompt "Claude: Agent 1" "0,0,0,$HALF_WIDTH,$WINDOW_HEIGHT" "agent1"

# Terminal 2: Agent 2 (LLM Analysis + System Integration)
echo "Setting up Agent 2 terminal and Claude instance..."
open_terminal "CC agent 2" "TCCC: Agent 2 (LLM + Integration)" "echo -e '\033[1;36m===== AGENT 2: LLM ANALYSIS + SYSTEM INTEGRATION =====\033[0m\n' && cat AGENT2_IMPLEMENTATION_NOTES.md && echo -e '\nPreparing environment...' && echo -e '\nLLM Analysis files:' && find src/tccc/llm_analysis -type f | sort && echo -e '\nSystem files:' && find src/tccc/system -type f 2>/dev/null | sort" "100x30+0+0"
open_claude_with_prompt "Claude: Agent 2" "0,$HALF_WIDTH,0,$HALF_WIDTH,$WINDOW_HEIGHT" "agent2"

# Terminal 3: Orchestrator Terminal
echo "Setting up Orchestrator terminal..."
open_terminal "CC agent 4" "TCCC: Orchestrator" "echo -e '\033[1;35m===== ORCHESTRATOR TERMINAL =====\033[0m\n' && cat TCCC_DUAL_AGENT_PLAN.md && echo -e '\nRunning initial verification...' && ./run_all_verifications.sh" "100x30+0+0"
wmctrl -r "TCCC: Orchestrator" -e "0,0,$WINDOW_HEIGHT,$SCREEN_WIDTH,$WINDOW_HEIGHT"

# Dashboard at the bottom
echo "Setting up Dashboard terminal..."
open_terminal "CC agent 3" "TCCC: Dashboard" "echo -e '\033[1;37m===== TCCC SYSTEM DASHBOARD =====\033[0m\n' && echo -e 'System resources:\n' && free -h && echo -e '\nCurrent git branch: ' && git branch --show-current && echo -e '\nStarting dashboard...' && sleep 2 && python dashboard.py" "160x10+0+0"
wmctrl -r "TCCC: Dashboard" -e "0,0,$DASHBOARD_Y,$SCREEN_WIDTH,$DASHBOARD_HEIGHT"

# Instructions and summary
cat << EOF

===============================
TCCC.ai Dual-Agent Workspace
===============================

Workspace Layout:
- Top-left: Agent 1 Terminal (STT Engine + Document Library)
- Top-right: Agent 2 Terminal (LLM Analysis + System Integration)
- Middle: Orchestrator Terminal (coordination and monitoring)
- Bottom: Dashboard

Created Documentation:
- TCCC_DUAL_AGENT_PLAN.md - Overall implementation strategy
- AGENT1_IMPLEMENTATION_NOTES.md - Detailed notes for Agent 1
- AGENT2_IMPLEMENTATION_NOTES.md - Detailed notes for Agent 2

Next Steps:
1. Copy the displayed prompts into each Claude window
2. Start implementation according to the plan
3. Use this terminal (Orchestrator) to monitor progress and coordinate
4. Check integration points regularly

EOF

echo "TCCC.ai dual-agent workspace setup complete!"