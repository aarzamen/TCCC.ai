# TCCC.ai Technical Integration Reference

This technical reference provides information for integrating and customizing the TCCC.ai system.

## System Architecture

The TCCC.ai system consists of seven main modules:

1. **Audio Pipeline**: Handles audio capture and preprocessing
2. **STT Engine**: Transcribes speech to text using Whisper models
3. **Processing Core**: Coordinates system components and processing
4. **Document Library**: Provides RAG capabilities for medical information
5. **LLM Analysis**: Extracts medical events and generates reports
6. **Data Store**: Persists events, reports, and system state
7. **Display Interface**: Provides visual interface on connected displays

## Configuration

The system's behavior can be customized through configuration files in the `config/` directory:

- `audio_pipeline.yaml`: Audio capture and processing settings
- `data_store.yaml`: Data persistence and backup configuration
- `document_library.yaml`: RAG system and embedding model settings
- `llm_analysis.yaml`: LLM model and inference settings
- `processing_core.yaml`: Core system settings and plugins
- `stt_engine.yaml`: Speech recognition models and parameters
- `jetson_optimizer.yaml`: Jetson-specific optimizations

## Hardware Optimization

### Jetson-Specific Optimizations

For Jetson devices, the system uses several optimizations:

1. **Model Quantization**: INT8/FP16 precision for models
2. **TensorRT Acceleration**: GPU-accelerated inference
3. **Memory Management**: Configurable limits to prevent OOM errors
4. **Power Profiles**: Dynamic adjustment based on workload

Example configuration in `jetson_optimizer.yaml`:

```yaml
jetson:
  enable_optimization: true
  power_mode: 15W
  memory_limit:
    stt_engine: 1.5G
    llm_analysis: 1G
  use_tensorrt: true
  quantization:
    whisper: int8
    phi2: int8_float16
```

## Display Integration

The display system is designed to work with various displays, prioritizing the WaveShare 6.25" display.

### Display Modes

The system supports two main display modes:

1. **Live View**: Three-column layout showing transcription, events, and card preview
2. **TCCC Card View**: Detailed view of the TCCC casualty card with anatomical diagram

### Environment Variables

Display configuration can be controlled via environment variables:

- `TCCC_ENABLE_DISPLAY`: Enable/disable display (0 or 1)
- `TCCC_DISPLAY_RESOLUTION`: Set resolution (e.g., "1560x720")
- `TCCC_DISPLAY_TYPE`: Display type identifier (e.g., "waveshare_6_25")
- `TCCC_DISPLAY_ORIENTATION`: Set to "landscape" or "portrait"

## API Reference

For custom integrations, TCCC.ai provides several Python APIs:

### System API

```python
from tccc.system.system import TCCCSystem
import asyncio

# Initialize system
system = TCCCSystem()

# Initialize with configuration (async method)
async def init_system():
    # Empty config will use defaults
    await system.initialize({})
    
    # For testing, you can use mock modules
    # mock_modules = ["audio_pipeline", "stt_engine"]
    # await system.initialize({}, mock_modules=mock_modules)

asyncio.run(init_system())

# Process events
event = {
    "type": "external_text",
    "text": "Patient shows signs of tension pneumothorax",
    "timestamp": 1709682904.5,
    "metadata": {"source": "manual_entry"}
}
event_id = system.process_event(event)

# Start audio capture
system.start_audio_capture()

# Generate reports
reports = system.generate_reports(["zmist", "medevac"])

# Query documents
results = system.query_documents("tension pneumothorax treatment", n_results=3)

# Get system status
status = system.get_status()

# Shutdown the system
system.shutdown()
```

### Display API

```python
from tccc.display.display_interface import DisplayInterface

# Initialize display
display = DisplayInterface(width=1560, height=720)
display.initialize()

# Update with new data
display.update_transcription("New transcription text...")
display.add_significant_event({"time": "14:30", "description": "Event description"})
display.update_card_data(card_data_dict)

# Control display mode
display.set_mode("live")  # or "card"
```

## Extending the System

### Custom Plugins

The Processing Core supports custom plugins:

1. Create a Python module in `src/tccc/processing_core/plugins/`
2. Implement the plugin interface
3. Register the plugin in `config/processing_core.yaml`

Example plugin:

```python
class CustomExtractor:
    def __init__(self):
        self.name = "custom_extractor"
    
    def initialize(self, core):
        # Setup code
        return True
    
    def process(self, text):
        # Processing logic
        return {"entities": [...]}
    
    def shutdown(self):
        # Cleanup code
        pass
```

### Custom Models

To use custom models:

1. Place model files in `models/` directory
2. Update configuration in corresponding YAML file
3. Implement model interface if needed

Example for custom STT model:

```yaml
# config/stt_engine.yaml
model:
  provider: custom
  name: my-custom-model
  path: /path/to/model
  params:
    beam_size: 5
    temperature: 0.0
```

## Monitoring and Debugging

For system monitoring:

1. **Logs**: Check `logs/` directory for detailed component logs
2. **Metrics**: Use `resource_monitor` for system resource tracking
3. **Verification**: Run verification scripts to validate components

```bash
# Run all verifications
./run_all_verifications.sh

# Run system integration verification
python verification_script_system_enhanced.py

# Run system integration verification with mock modules
python verification_script_system_enhanced.py --mock all

# Run basic system integration test
python test_system_integration.py

# Monitor resource usage
python -c "from tccc.utils.monitoring import print_resource_usage; print_resource_usage()"
```

### System Integration Verification

The enhanced system verification script runs several test stages:

1. **Initialization**: Verifies system and component initialization
2. **Module Verification**: Tests individual module functionality
3. **Integration Verification**: Tests module interactions and dependencies
4. **Data Flow**: Verifies end-to-end data processing
5. **Error Handling**: Tests system resilience to errors
6. **Performance**: Measures throughput and resource usage
7. **Security**: Validates system security features

When developing new components, ensure they pass all verification stages before integration.
