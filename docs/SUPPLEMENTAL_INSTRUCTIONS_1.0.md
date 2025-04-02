# TCCC.ai Project Guide: Claude Code Addendum

This addendum provides specific guidance for using Claude Code CLI effectively with the TCCC.ai project on Jetson hardware, focusing on resolving current integration issues and moving from mock implementations to actual hardware testing with the Razor Mini 3 USB-C microphone and USB-C headset for audio I/O.

## 1. Resolving Current Integration Issues

### 1.1 Fix Missing Verification Scripts

The system is missing verification scripts for some modules. Let's create these first:

```bash
# Create missing verification script for Processing Core
claude "Create a verification script for the Processing Core module that:
1. Tests module registration functionality
2. Validates resource allocation methods
3. Verifies state transitions between all operational modes
4. Tests system status reporting
5. Confirms proper error handling

Save this as 'verification_script_processing_core.py' in the project root."
```

### 1.2 Fix Document Library Configuration

The Document Library verification is failing due to a missing 'embedding' key in configuration:

```bash
# Update Document Library verification script
claude "Fix the Document Library verification script that's failing with KeyError: 'embedding'.

The error occurs in verification_script_document_library.py at line 182:
config[\"embedding\"][\"cache_dir\"] = os.path.join(data_dir, \"models/embeddings\")

Either update the script to use the correct configuration key structure or modify the default configuration to include the required embedding section."
```

### 1.3 Fix System Integration Module

The System Integration verification is failing with a module import error:

```bash
# Fix System Integration module structure
claude "Fix the System Integration module that's failing with ModuleNotFoundError: No module named 'src.tccc.system'.

1. Create the missing module structure at src/tccc/system/
2. Implement a basic TCCCSystem class with SystemState enum
3. Ensure proper imports in verification_script_system_enhanced.py
4. Update any related module dependencies"
```

## 2. Installing Actual Dependencies on Jetson

### 2.1 Nexa AI SDK Installation

```bash
# Install Nexa AI SDK
claude "Create an installation script for the Nexa AI SDK on Jetson Orin Nano.

The script should:
1. Install all necessary prerequisites
2. Configure CUDA and TensorRT integration
3. Install the Nexa AI SDK with proper Jetson optimizations
4. Download and configure the preferred models:
   - faster-whisper-tiny.en for STT
   - LLaMA-3-8B-Instruct for LLM analysis
   - all-MiniLM-L12-v2 for document embedding
5. Verify installation with basic tests for each model
6. Set up proper configuration for the TCCC modules

Save this as 'install_nexa_dependencies.sh' and ensure it's executable."
```

### 2.2 Audio Processing Dependencies

```bash
# Install audio processing dependencies
claude "Create an installation script for audio processing dependencies on Jetson Orin Nano with the Razor Mini 3 USB-C microphone.

The script should:
1. Install required system packages for audio (ALSA, PortAudio)
2. Install PyAudio with proper Jetson compatibility
3. Install Silero VAD for voice activity detection
4. Configure the Razor Mini 3 USB-C microphone specifically
   - Set appropriate sample rates and bit depth
   - Configure proper gain levels
   - Set as default recording device
5. Configure the USB-C headset as output device (for future use)
6. Verify microphone access and recording quality
7. Test integration with the Audio Pipeline module
8. Include diagnostics for troubleshooting USB audio issues on Jetson

Save this as 'install_audio_dependencies.sh' and ensure it's executable."
```

#### 2.2.1 Razor Mini 3 Microphone Configuration

```bash
# Create Razor Mini 3 configuration script
claude "Create a dedicated configuration script for the Razor Mini 3 USB-C microphone on Jetson Orin Nano.

The script should:
1. Detect the Razor Mini 3 device automatically
2. Set optimal audio parameters (48kHz, 24-bit)
3. Configure noise cancellation features if available
4. Test recording quality with spectrum analysis
5. Create an ALSA configuration profile specific to this device
6. Provide troubleshooting for common USB audio issues on Jetson
7. Include documentation on microphone specifications and capabilities

Save this as 'configure_razor_mini3.sh' and ensure it's executable."
```

### 2.3 Database and Storage Setup

```bash
# Set up database and storage
claude "Create a setup script for the SQLite database and storage requirements.

The script should:
1. Configure SQLite with WAL mode and proper optimization for Jetson
2. Create the necessary database schema for events, reports, and sessions
3. Set up proper file permissions and storage locations
4. Configure backup mechanisms
5. Test database performance on the Jetson hardware
6. Verify integration with the Data Store module

Save this as 'setup_database.sh' and ensure it's executable."
```

## 3. Testing with Claude Code on Hardware

### 3.1 Hardware Verification Framework

```bash
# Create hardware verification framework
claude "Create a comprehensive hardware verification framework for the TCCC.ai system on Jetson Orin Nano.

The framework should:
1. Test audio capture with actual microphone input
2. Verify STT transcription performance on real audio
3. Measure LLM inference times for medical text analysis
4. Test end-to-end processing pipeline with actual hardware
5. Monitor and report system resource usage (CPU, GPU, memory)
6. Verify thermal management during extended operation
7. Include clear pass/fail criteria for each component

Save this as 'hardware_verification.py' in the project root."
```

### 3.2 Guided Testing Session

Here's a step-by-step guided testing session to run with Claude Code:

```bash
# 1. Verify system dependencies
claude "Run a system check to verify all dependencies are properly installed on this Jetson Orin Nano.

Check for:
- CUDA and TensorRT availability
- Nexa AI SDK installation
- PyAudio and audio device access
- SQLite with proper configuration
- All required Python packages

Generate a detailed report of findings and identify any missing components."

# 2. Run module tests on hardware
claude "Create a test script that verifies each module's performance on actual Jetson hardware.

The script should:
1. Test Audio Pipeline with live microphone input
2. Test STT Engine with recorded audio samples
3. Test Processing Core with actual resource monitoring
4. Test LLM Analysis with medical transcription samples
5. Test Document Library with vector search benchmarks
6. Test Data Store with concurrent write operations
7. Report performance metrics for each module

Save this as 'hardware_module_tests.py' in the project root."

# 3. End-to-end pipeline test
claude "Create an end-to-end pipeline test that simulates a realistic medical scenario.

The test should:
1. Record or play sample medical audio (like tourniquet application procedure)
2. Process through the complete pipeline from audio to reports
3. Generate MEDEVAC and ZMIST reports from the extracted events
4. Query relevant protocols from the Document Library
5. Measure end-to-end latency and resource usage
6. Compare results with expected outputs
7. Generate a comprehensive performance report

Save this as 'end_to_end_hardware_test.py' in the project root."
```

## 4. Optimizing Module Configuration

### 4.1 Nexa Model Configuration

```bash
# Create optimized configuration for Nexa models
claude "Create optimized configuration files for the Nexa AI models on Jetson Orin Nano.

Generate configuration files for:
1. faster-whisper-tiny.en with optimal batch size and threading
2. LLaMA-3-8B-Instruct with proper quantization and context size
3. all-MiniLM-L12-v2 with efficient vector dimensions and indexing

Each configuration should:
- Respect the 8GB memory limitation
- Optimize for Jetson's 67 TOPS capability
- Include power management settings
- Support the operational states defined in the architecture

Save these in the config/ directory with appropriate naming."
```

### 4.2 Resource Allocation Strategy

```bash
# Create resource allocation strategy
claude "Develop a resource allocation strategy for the TCCC.ai system on Jetson Orin Nano.

The strategy should:
1. Define CPU, GPU, and memory allocation for each module
2. Implement dynamic resource scaling based on operational state
3. Prioritize critical modules during high-demand scenarios
4. Include thermal management considerations
5. Optimize power usage for different scenarios
6. Define resource monitoring and reallocation triggers

Save this as 'resource_allocation_strategy.md' in the project documentation."
```

## 5. Best Practices for Claude Code CLI on Jetson

### 5.1 Claude Code CLI Configuration

For optimal performance when working with the TCCC.ai project on Jetson hardware, configure Claude Code CLI as follows:

```bash
# Configure Claude Code for Jetson development
claude config set --global theme dark
claude config set --global verbose true
claude config set --global preferredNotifChannel terminal_bell

# Configure allowed tools for development
claude config add allowedTools "Bash(pip install*)"
claude config add allowedTools "Bash(python*)"
claude config add allowedTools "Bash(sudo systemctl*)"
claude config add allowedTools "Bash(nvidia-smi)"
claude config add allowedTools "Bash(jtop)"

# Configure paths to ignore
claude config add ignorePatterns "models/*.onnx"
claude config add ignorePatterns "models/*.gguf"
claude config add ignorePatterns "models/*.bin"
claude config add ignorePatterns "data/**"
```

### 5.2 Effective Claude Code Workflows

For effective development using Claude Code CLI with the TCCC.ai project:

1. **Use specific task requests**: Rather than generic requests, ask Claude to perform specific diagnostic or development tasks
   ```
   claude "Diagnose the ALSA audio initialization failures in the Audio Pipeline module and suggest corrections"
   ```

2. **Share error logs directly**: When debugging issues, share complete error logs
   ```
   claude "Analyze this error log from the LLM module and identify the root cause:
   
   <paste actual error log>"
   ```

3. **Request specific hardware tests**: Have Claude create scripts that test specific hardware features
   ```
   claude "Create a script to benchmark inference time for the Nexa LLaMA-3-8B model on the Jetson Orin Nano"
   ```

4. **Use compact mode regularly**: Keep context efficient with the /compact command
   ```
   /compact
   ```

5. **Break complex tasks into stages**: For hardware integration, break tasks into manageable pieces
   ```
   # Stage 1: Verify hardware access
   claude "Create a script to verify microphone access and recording capability"
   
   # Stage 2: Test audio processing
   claude "Now create a script to process the recorded audio with noise reduction and VAD"
   
   # Stage 3: Test integration with STT
   claude "Now extend the script to pass processed audio to the STT engine and verify transcription"
   ```

## 6. Hardware-Specific Verification

### 6.1 Jetson Hardware Verification

```bash
# Create Jetson-specific verification tools
claude "Create a Jetson Orin Nano verification toolkit for the TCCC.ai project.

The toolkit should include:
1. GPU utilization monitoring during inference operations
2. Memory usage tracking across all modules
3. Thermal monitoring during extended operation
4. Power consumption measurement in different operational states
5. I/O performance testing for database and file operations
6. Network latency measurement for potential cloud fallback

Save this as a Python module in src/tccc/utils/jetson_verification/"
```

### 6.2 Audio Pipeline Hardware Verification

```bash
# Create Audio Pipeline hardware verification
claude "Create an Audio Pipeline hardware verification script for the TCCC.ai project.

The script should:
1. Test microphone initialization and configuration
2. Verify audio capture with different sample rates and formats
3. Measure voice activity detection accuracy with real audio
4. Test noise reduction effectiveness in different environments
5. Measure audio processing latency on the Jetson hardware
6. Verify proper threading and buffer management
7. Test integration with the STT Engine

Save this as 'verify_audio_hardware.py' in the project root."
```

## 7. Continuous Integration Strategy

### 7.1 Local Testing Pipeline

```bash
# Create local testing pipeline
claude "Create a local testing pipeline for the TCCC.ai project on Jetson Orin Nano.

The pipeline should:
1. Run unit tests for each module
2. Verify hardware integration for each component
3. Measure performance metrics and compare to baselines
4. Test end-to-end functionality with actual hardware
5. Generate comprehensive test reports
6. Support different operational scenarios (field, training, debugging)

Save this as 'local_test_pipeline.sh' and ensure it's executable."
```

### 7.2 Module Integration Schedule

Based on the current verification results, prioritize module integration as follows:

1. Fix Processing Core verification (**FAILED**)
2. Fix Document Library verification (**FAILED**)  
3. Fix System Integration (**FAILED**)
4. Verify Audio Pipeline on actual hardware (**PASSED** on mock)
5. Verify STT Engine with Nexa model (**PASSED** on mock)
6. Verify LLM Analysis with Nexa model (**PASSED** on mock)
7. Verify Data Store with actual database (**PASSED** on mock)
8. End-to-end integration testing

## 8. Log Analysis and Project Status Review

### 8.1 Log File Analysis Instructions

Use the following commands with Claude Code to analyze log files and determine next steps:

```bash
# Analyze verification logs to determine next steps
claude "Analyze the latest verification log files in the logs/ directory and:
1. Identify which modules are passing and failing verification
2. Determine the root causes of any failures
3. Suggest specific fixes for each issue found
4. Prioritize the issues based on dependency order
5. Recommend what to work on next

Use the project architectural dependencies to make intelligent recommendations."
```

```bash
# Generate progress report and next steps
claude "Review the current project state by examining:
1. All verification logs in logs/
2. Git commit history to see what's been implemented
3. Current configuration files in config/
4. Module implementation status in src/

Then:
1. Summarize the current state of the TCCC.ai project
2. Identify which components are complete vs. incomplete
3. List any blocking issues preventing progress
4. Suggest the next 3 most impactful tasks to focus on
5. Provide specific commands to execute those tasks"
```

### 8.2 Feature Suggestion Mode

When the project seems to be in good shape, use Claude Code to suggest improvements:

```bash
# Request improvement suggestions
claude "The TCCC.ai system is currently working for basic functionality. Suggest:
1. Three performance optimizations for the Jetson Orin Nano
2. Two new features that would enhance medical documentation
3. One integration with external systems that would be valuable
4. Testing scenarios to validate system robustness
5. Ways to improve the accuracy of the Nexa models for medical terminology

For each suggestion, provide a brief implementation plan."
```

## 9. Basic Environment and Claude Code Setup

Follow these exact steps to set up your environment and run Claude Code:

### 9.1 Idiot-Proof Environment Setup

```bash
# Step 1: Open a terminal window

# Step 2: Navigate to your home directory
cd ~

# Step 3: Go to the tccc-project directory
cd tccc-project

# Step 4: Activate the virtual environment
source venv/bin/activate
# You should see (venv) at the beginning of your command prompt

# Step 5: Verify you're in the right place
pwd
# Should show: /home/yourusername/tccc-project

# Step 6: Make sure the virtual environment is active
python -c "import sys; print(sys.prefix)"
# Should show the path to your virtual environment

# Step 7: Run Claude Code
claude

# If you want to start with a specific query:
claude "What should I work on next?"
```

### 9.2 Helpful Claude Code Commands

Once Claude Code is running:

- Type `/help` to see available commands
- Type `/clear` to reset the conversation
- Type `/compact` to save context space
- Type `/bug` to report issues
- Press CTRL+C to exit Claude Code

## 10. Cross-Model Communication Prompt Template

Use this prompt template when communicating with other advanced LLMs about the TCCC.ai project:

```
# TCCC.ai Project Status Report for Advanced LLM Analysis

## Project Overview
TCCC.ai is a frontier, state-of-the-art combat medical documentation system running on the NVIDIA Jetson Orin Nano platform. It captures audio during medical procedures, transcribes speech to text, processes information through language models, and creates structured medical documentation.

## Current Hardware Configuration
- NVIDIA Jetson Orin Nano 8GB with 67 TOPS processing power
- Razor Mini 3 USB-C microphone for audio capture
- USB-C headset for potential audio output
- Samsung 960 EVO 1TB NVMe storage

## Project Architecture
The system consists of 6 core modules with the following verification status:
- Audio Pipeline: PASSED (mock verification)
- STT Engine: PASSED (mock verification)
- Processing Core: FAILED verification
- LLM Analysis: PASSED (mock verification)
- Document Library: FAILED verification
- Data Store: PASSED (mock verification)
- System Integration: FAILED verification

## AI Model Selection
- Speech-to-Text: Nexa AI's faster-whisper-tiny.en
- Language Model: Nexa AI's LLaMA-3-8B-Instruct
- Document Embeddings: Nexa AI's all-MiniLM-L12-v2

## Current Development Status
[INSERT PROJECT STATUS DETAILS HERE - Briefly describe the current state, recent progress, and known issues]

## Requested Assistance
[INSERT SPECIFIC REQUEST HERE - What you need help with]

## Claude Code Context
This project uses Claude Code CLI (beta) for development assistance. Claude Code works in the terminal, understands the codebase through direct filesystem access, and can take actions like editing files and running commands. 

When providing advice or generating solutions, please include specific Claude Code commands that can be executed directly in the terminal, like:

```bash
claude "Create a script that fixes the Document Library configuration issue"
```

Prefer step-by-step instructions that can be executed through Claude Code, and provide context on what each step accomplishes. Remember Claude Code has access to the project structure and can read/edit files directly, so it's helpful to reference actual project paths and files.

Note: Claude Code responds well to specific, focused requests for individual module improvements rather than broad, vague instructions.
```

By following this addendum alongside the comprehensive guide, you'll be able to transition from mock implementations to actual hardware testing on the Jetson Orin Nano platform with the Razor Mini 3 microphone, successfully integrating all TCCC.ai modules for real-world deployment.