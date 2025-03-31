# Gemini Tool for TCCC Project

This tool provides an integration with Google's Gemini 2.5 Pro model to leverage its 1 million token context window for analyzing large portions of the TCCC codebase.

## Setup

1. Set up your Gemini API key in one of the following ways:
   - Set an environment variable: `export GEMINI_API_KEY=your_api_key_here`
   - Create a config file at `~/.gemini_config` with JSON content: `{"api_key": "your_api_key_here"}`
   - Create a config file at `scripts/development/gemini_config.json` with the same JSON format

2. Install required Python package (done automatically by the script if needed):
   ```
   pip install google-generativeai
   ```

## Usage

The user-friendly wrapper script provides several ways to use Gemini:

```bash
# Direct query with no additional context
./gemini_analyze.sh query "How does the TCCC system work?"

# Analyze specific files
./gemini_analyze.sh files "Find integration issues" src/tccc/audio_pipeline/*.py src/tccc/stt_engine/*.py

# Analyze a directory
./gemini_analyze.sh dir "List all classes" src/tccc/document_library

# Use glob patterns to find files
./gemini_analyze.sh glob "Find event handlers" "src/tccc/**/*.py"

# Analyze a specific module
./gemini_analyze.sh module "Explain the pipeline" audio_pipeline

# Analyze a component across multiple modules
./gemini_analyze.sh component "How are events used?" event_bus

# Save output to a file
./gemini_analyze.sh -o analysis.txt module "Explain architecture" system
```

## Working with Claude Code

This tool is designed to complement Claude Code by:

1. Analyzing large portions of the codebase that exceed Claude's context window
2. Generating analyses that can be shared with Claude for further action
3. Exploring complex integrations across multiple modules

### Remote SSH Workflow with Claude Code

For working with Claude Code remotely via SSH, use the log file approach:

1. Claude Code suggests a gemini_analyze command with output redirection:
   ```bash
   ./scripts/development/gemini_analyze.sh -o gemini_analysis_log.txt module "Analyze audio pipeline" audio_pipeline
   ```

2. You run this command in your SSH terminal

3. After completion, inform Claude Code that the command is done:
   ```
   Command complete. Gemini's analysis is in gemini_analysis_log.txt.
   ```

4. Claude Code reads the file and incorporates the analysis:
   ```
   I'll read the analysis from the log file and integrate the findings...
   ```

This approach eliminates manual copying of large text blocks and works effectively over SSH connections.

## Advanced Usage

For more advanced usage, you can use the Python script directly:

```bash
./scripts/development/gemini_tool.py --query "Your query" --glob "**/*.py" --recursive
```

See `./scripts/development/gemini_tool.py --help` for all available options.