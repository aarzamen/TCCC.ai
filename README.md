# TCCC.ai

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="200" height="200" style="display: block; margin: 0 auto;">
  <!-- Define the layering order -->
  <defs>
    <clipPath id="clipMicrophone">
      <circle cx="250" cy="250" r="40"/>
    </clipPath>
  </defs>
  <!-- Main circle background - military olive green -->
  <circle cx="250" cy="250" r="240" fill="#5D6C41" stroke="#333333" stroke-width="10"/>
  
  <!-- Clipboard base -->
  <rect x="125" y="100" width="250" height="320" rx="10" ry="10" fill="#E6E6C8" stroke="#333333" stroke-width="8"/>
  
  <!-- Document lines -->
  <line x1="150" y1="160" x2="350" y2="160" stroke="#8A8F61" stroke-width="6"/>
  <line x1="150" y1="200" x2="350" y2="200" stroke="#8A8F61" stroke-width="6"/>
  <line x1="150" y1="240" x2="350" y2="240" stroke="#8A8F61" stroke-width="6"/>
  <line x1="150" y1="280" x2="350" y2="280" stroke="#8A8F61" stroke-width="6"/>
  <line x1="150" y1="320" x2="350" y2="320" stroke="#8A8F61" stroke-width="6"/>
  <line x1="150" y1="360" x2="350" y2="360" stroke="#8A8F61" stroke-width="6"/>
  
  <!-- Medical cross - red -->
  <rect x="170" y="60" width="40" height="80" fill="#E62020"/>
  <rect x="150" y="80" width="80" height="40" fill="#E62020"/>
  
  <!-- EKG heartbeat line - more prominent and extends through the microphone -->
  <path d="M120,230 L160,230 L175,160 L200,300 L225,180 L250,260 L275,200 L300,150 L325,230 L380,230" 
        fill="none" stroke="#E62020" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>
        
  <!-- EKG highlight for extra visibility -->
  <path d="M120,230 L160,230 L175,160 L200,300 L225,180 L250,260 L275,200 L300,150 L325,230 L380,230" 
        fill="none" stroke="#FF4444" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  
  <!-- Microphone/audio circle - moved to appear above the EKG line -->
  <circle cx="250" cy="250" r="40" fill="#5D6C41" stroke="#333333" stroke-width="4"/>
  
  <!-- Microphone icon -->
  <rect x="240" y="230" width="20" height="40" rx="10" ry="10" fill="#000000"/>
  <path d="M230,255 L230,265 C230,280 240,290 250,290 C260,290 270,280 270,265 L270,255" 
        fill="none" stroke="#000000" stroke-width="6" stroke-linecap="round"/>
  <line x1="250" y1="290" x2="250" y2="300" stroke="#000000" stroke-width="6" stroke-linecap="round"/>
  <line x1="235" y1="300" x2="265" y2="300" stroke="#000000" stroke-width="6" stroke-linecap="round"/>
</svg>

## Project Overview

TCCC.ai (Tactical Combat Casualty Care AI) is an edge-deployed AI system designed to function as a "black box recorder" for combat medicine. Operating completely offline on the NVIDIA Jetson Orin Nano platform, it captures and processes audio during medical procedures to create structured documentation and timeline records without requiring network connectivity.

This project modernizes the traditional field documentation process, replacing the paper-based DD Form 1380s that are typically hastily completed with Sharpie markers and taped or zip-tied to patients' arms. These forms are often difficult to read, can be damaged by blood or environmental conditions, and risk being blown away during helicopter evacuations due to rotor wash. Unlike conventional medical scribes or EMR systems, TCCC.ai is purpose-built for the practical challenges of combat medicine documentation, providing a more durable and legible alternative that preserves critical patient information throughout the evacuation chain.

## Current Status

The project has reached a significant milestone with approximately 22,000 lines of original Python code developed across six core functional modules. Remarkably, this entire codebase was created using Anthropic's Claude Code CLI (beta) tool, which allowed me to bring my field medical background into software development without traditional programming experience. The core system architecture and module interfaces have been defined and implemented, with most individual components working in isolation but requiring further integration testing.

### Component Status

| Module | Status |
|--------|--------|
| Audio Pipeline | Passes verification but needs integration work |
| STT Engine | Fixed and working properly |
| Processing Core | Passes verification but needs integration |
| LLM Analysis | Fixed with tensor optimization implemented |
| Document Library | Working properly in isolation |
| Data Store | Working with no issues identified |
| System Integration | Currently failing at the verification stage |

![Videoshop_2025-03-19_13-03-35-540](https://github.com/user-attachments/assets/ef56819d-0927-4e58-a5be-4a9a190c0f01)


**Primary Development Focus:** Fixing system integration and event flow between components.

## Technical Architecture

The system follows a modular architecture with clean interfaces between components:

1. **Audio Pipeline:** Captures and processes speech input using PyAudio with Silero VAD and multi-stage battlefield audio enhancement, including:
   - Battlefield-specific noise filtering for gunshots, explosions, vehicle noise, and wind noise
   - FullSubNet neural speech enhancement for challenging acoustic conditions
   - Adaptive distance compensation and voice isolation techniques
   - Advanced audio chunk management system for variable-sized processing
   - Real-time processing optimized for Jetson hardware
2. **Speech-to-Text:** Implements a fine-tuned Whisper model (faster-whisper-tiny.en) with battlefield-optimized VAD parameters and enhanced accuracy through audio pre-processing
3. **LLM Analysis:** Utilizes Microsoft's Phi-2 model (quantized for edge deployment) to extract medical events and procedures
4. **RAG System:** Employs all-MiniLM-L12-v2 embedding model with a local vector database (FAISS/Chroma) for document retrieval
5. **Timeline Database:** SQLite database with WAL mode optimized for the Jetson's NVMe storage
6. **Report Generation:** Jinja2-based templating system for standardized medical documentation output

All models are heavily quantized (INT8/INT4) and optimized using TensorRT for the Jetson's ARM architecture. The system uses ZeroMQ for lightweight inter-module communication and implements custom resource management to handle the 8GB memory constraint.

### Memory Allocation Strategy

```
┌─────────────────────────────────────────────┐
│ Total System Memory (8GB on Jetson Orin)    │
├─────────────┬───────────────┬───────────────┤
│ STT Engine  │ LLM Analysis  │  Doc Library  │
│ (1-2GB)     │ (2-3GB)       │  (1GB)        │
├─────────────┴───────────────┴───────────────┤
│ System Overhead + Processing Core (2-3GB)   │
└─────────────────────────────────────────────┘
```

## Hardware Implementation

The current implementation runs on custom hardware consisting of:

- NVIDIA Jetson Orin Nano 8GB (67 TOPS processing power)
- Custom 3D-printed enclosures in a clamshell/pocket PC form factor
- DIY 20V Li-ion Molicel 21700 battery packs in rifle magazine form factor (6-8 hour operation)
- Waveshare 6.25" display with HDMI interface
- Removable wireless components for EMCON compliance

The device is specifically designed to fit inside the center chest admin pouch commonly used in military flak jackets, with battery packs that slot into standard magazine pouches for easy swapping in the field.

## Why This Matters

Combat medicine presents unique documentation challenges that standard systems aren't designed to address. In battlefield conditions, providers face:

- **Environmental Factors:** Documentation materials exposed to weather, bodily fluids, and physical damage
- **Cognitive Demands:** Managing multiple casualties while maintaining situational awareness
- **Time Constraints:** Critical interventions like tourniquets demand immediate attention, often at the expense of documentation
- **Protocol Complexity:** TCCC guidelines require specific procedures and timing that are difficult to track manually
- **Continuity of Care:** Information must follow patients through complex evacuation chains
- **Accountability:** Clear records protect providers from retrospective criticism of decisions made under duress

Current methods fall short in these environments. Paper documentation is vulnerable to destruction, electronic systems require connectivity and time that isn't available, and retrospective documentation results in significant information gaps and timeline inaccuracies.

## Development Approach

This project was developed using Claude Code CLI (beta) - an AI-assisted development approach that bridged the gap between field medicine knowledge and software implementation. This approach represents a shift where domain understanding can directly translate into functional systems without requiring extensive programming background.

The development process followed these principles:

- Creating clean module interfaces first
- Implementing basic functionality module by module
- Optimizing for edge deployment on constrained hardware
- Testing each component in isolation

## Installation & Setup

### Hardware Requirements

- NVIDIA Jetson Orin Nano (4GB+ RAM)
- JetPack 5.1.1 or later
- 15GB+ storage (SSD recommended)
- Optional: USB microphone for audio input (Razer Seiren V3 Mini supported)

### Basic Setup

```bash
# Clone repository
git clone https://github.com/aarzamen/TCCC.ai.git
cd TCCC.ai

# Environment setup
python -m venv venv
source venv/bin/activate

# Installation
pip install -e .

# Run individual module verification
python verification_script_audio_pipeline.py
```

## Current Capabilities

The system now has several enhanced capabilities:

1. **Battlefield Audio Processing:** Multi-stage noise reduction pipeline with neural enhancement
   - Specialized filters for battlefield acoustic conditions (gunfire, explosions, vehicles)
   - Adaptive processing based on environmental conditions
   - Distance compensation for varying speaker positions
   - Wind and background noise suppression optimized for outdoor environments
2. **Robust Speech Recognition:** STT engine with enhanced accuracy through audio pre-processing
   - Battlefield-optimized voice activity detection parameters
   - Integration with FullSubNet neural speech enhancement
   - Automatic adjustment to varying audio conditions
3. **Real-time Operation:** Complete pipeline from audio capture to transcription and analysis
   - Efficient processing optimized for Jetson hardware
   - CUDA acceleration for neural speech enhancement
   - Automatic resource management for constrained hardware

While individual components now function effectively, several integration challenges remain in creating a system that maintains functionality under suboptimal conditions rather than failing completely.

## Next Steps

The immediate focus areas include:

1. Resolving the core system event loop architecture
2. Implementing a reliable module initialization sequence
3. Streamlining event passing between components
4. Enhancing system-wide logging for better diagnostics
5. Conducting integrated testing on physical hardware
6. Further optimizing models for the Jetson platform constraints

The goal is to achieve a system stable enough for field testing within two months.

## Personal Perspective

This project stems from my experiences as a General Medical Officer with Marine Corps infantry units over the past decade. Working in field conditions provided firsthand insight into documentation challenges: casualty cards damaged or lost during care, critical information missing during handoffs, and the gap between enterprise medical systems and frontline needs.

With military service gradually winding down, this project serves dual purposes – addressing a practical need I've encountered repeatedly and developing a skill set that combines field medicine understanding with technology development. 

What distinguishes TCCC.ai from many healthcare AI projects is its design philosophy: instead of requiring ideal conditions, it embraces constraints like limited power, absence of connectivity, and harsh environments as fundamental design parameters. The result prioritizes reliability and practicality in conditions where conventional systems fail.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Technical Features

_[Claude thinking... pondering the silicon depths... consulting my neural pathways...]_

Claude suggested the following items be included for completeness (and to justify where all those extra API calls went):

1. **Display Interface Implementation**: The project includes a comprehensive UI for the WaveShare 6.25" display with dual-mode interface: a "live view" showing real-time transcription with event timeline, and a "TCCC card view" with digital representation of the DD Form 1380 including an interactive anatomical diagram. Touch input supports mode switching and interactive data entry.

2. **Advanced Tensor Optimization Framework**: A custom tensor optimization layer manages constrained hardware operation through:
   - Mixed precision operations (FP16/INT8)
   - Memory-efficient tensor chunking
   - TensorRT acceleration with cached engines
   - Adaptive precision selection based on available resources

3. **Jetson-Specific Adaptations**: Hardware-aware features including:
   - Dynamic model size selection based on platform capabilities
   - Precision adjustments optimized for Ampere architecture
   - Memory usage monitoring during transcription
   - Automatic warmup procedures for consistent performance

4. **Performance Profile System**: Three operational modes with distinct resource priorities:
   - Emergency Mode: Maximum performance for critical situations
   - Field Mode: Balanced performance and power efficiency
   - Training Mode: Extended battery life with higher precision

5. **Touch Interface Technology**: Full support for the WaveShare 6.25" capacitive touchscreen:
   - Auto-calibration for different orientation modes
   - Multi-touch gesture recognition for zooming anatomical diagrams
   - Customizable touch regions for quick actions

_[Claude has completed his rumination and declared these extra technical details with algorithmic confidence]_
