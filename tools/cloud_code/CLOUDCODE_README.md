# CloudCode: Claude Code CLI Framework for TCCC.ai

This guide explains how to use the CloudCode framework to efficiently work with Claude Code CLI for TCCC.ai development.

## 🚀 Overview

CloudCode is a framework that optimizes your interaction with Claude Code CLI by providing:

1. **Structured Documentation**: Modular, task-specific content for Claude's context window
2. **Helper Scripts**: Tools to extract relevant sections and prepare sessions
3. **Prompt Templates**: Ready-to-use snippets for common development tasks
4. **Best Practices**: Guidelines for efficient AI-assisted development

## 📋 Components

The CloudCode framework consists of:

- **CLOUDCODE.md**: The core reference document with modular sections
- **CLOUDCODE_SNIPPETS.txt**: Ready-to-use prompt templates
- **CLOUDCODE_README.md**: This guide document
- **Helper Scripts**:
  - `extract_cloudcode_section.sh`: Extract specific sections from CLOUDCODE.md
  - `prepare_claude_session.sh`: Prepare complete Claude sessions
  - `use_snippet.sh`: Extract and prepare snippets for use

## 📝 Core Documentation Structure

The CLOUDCODE.md file is organized into modular sections with specific tags:

- **[CORE]**: Essential project context that should always be included
- **[MODULE]**: Module-specific documentation for each system component
- **[WORKFLOW]**: Guidelines for specific development processes
- **[REFERENCE]**: Technical information about specific technologies
- **[META]**: Information about the CloudCode framework itself

## 🔧 Using the Helper Scripts

### Extract CloudCode Sections

Use `extract_cloudcode_section.sh` to pull specific sections from CLOUDCODE.md:

```bash
# List available sections
./extract_cloudcode_section.sh -l

# Extract a specific module section
./extract_cloudcode_section.sh MODULE "Audio Pipeline"

# Extract all CORE sections
./extract_cloudcode_section.sh -a CORE

# Extract CORE sections with shortcut
./extract_cloudcode_section.sh -c

# Save output to file
./extract_cloudcode_section.sh MODULE "STT Engine" -o stt_context.txt
```

### Prepare Complete Claude Sessions

Use `prepare_claude_session.sh` to create ready-to-use Claude sessions:

```bash
# Prepare session for specific module with all CORE sections
./prepare_claude_session.sh "Audio Pipeline"

# Use minimal context (only Project Context)
./prepare_claude_session.sh -m "Document Library"

# Include a workflow section
./prepare_claude_session.sh -w "Testing" "STT Engine"

# Include a reference section
./prepare_claude_session.sh -r "Optimization Techniques" "LLM Analysis"

# Save to specific file
./prepare_claude_session.sh -o session.txt "Processing Core"
```

### Use Snippet Templates

Use `use_snippet.sh` to extract and customize snippet templates:

```bash
# List available snippets
./use_snippet.sh -l

# Extract specific snippet
./use_snippet.sh 1

# Preview snippet without saving
./use_snippet.sh -p 3

# Extract snippet with CORE context
./use_snippet.sh -c 5

# Extract snippet with module-specific context
./use_snippet.sh 2 "Audio Pipeline"

# Save to specific file without opening editor
./use_snippet.sh -n -o my_prompt.txt 4
```

## 👨‍💻 Workflow Example

Here's a complete workflow example for implementing a new feature:

```bash
# 1. Prepare a session for your module
./prepare_claude_session.sh "STT Engine"

# 2. Start Claude with the prepared session
cat /tmp/claude_code/session_*.txt | claude

# 3. For a specific implementation task, use a snippet
./use_snippet.sh -c 1 "STT Engine"

# 4. Edit the snippet in your editor to add details
# (Editor opens automatically)

# 5. Send the customized snippet to Claude
cat /tmp/claude_code/snippet_*.txt | claude

# 6. For debugging, use the debug snippet
./use_snippet.sh 2
```

## 🌟 Best Practices

### Context Optimization

1. **Include Only What's Needed**: Only include sections relevant to your current task
2. **Use Minimal Context**: For simple tasks, use the `-m` flag with `prepare_claude_session.sh`
3. **Use Progressive Detail**: Start with high-level prompts, then add details progressively
4. **Use /compact Command**: When conversations get long, use `/compact` in Claude Code CLI

### Effective Prompting

1. **Be Specific**: Replace all placeholders in snippets with detailed information
2. **Reference Existing Code**: Include examples of similar functionality from the codebase
3. **Specify Constraints**: Always mention performance and resource constraints
4. **Include Interface Details**: Provide expected interfaces and contract requirements
5. **Use Modular Prompting**: Break complex tasks into smaller, focused prompts

### Session Management

1. **Save Sessions**: Save important sessions using `-o` flag with descriptive filenames
2. **Document Tasks**: Create brief task descriptions at the start of each session
3. **Summarize Progress**: At the end of sessions, ask Claude to summarize progress
4. **Track File Changes**: Keep note of which files were modified in each session

## 🔄 Framework Maintenance

As the project evolves:

1. **Update Module Documentation**: Keep module interfaces and implementations up-to-date
2. **Refine Snippets**: Adjust snippets based on common usage patterns
3. **Add New Sections**: Add new modules and reference sections as needed
4. **Organize by Use Case**: Group related information for common development scenarios

## 🆘 Troubleshooting

If you encounter issues:

1. **Context Limits**: If Claude misses information, you're likely exceeding the context window
   - Use `/compact` command
   - Use fewer sections with `-m` flag
   - Split complex tasks into multiple sessions

2. **Script Errors**: Ensure scripts have execution permissions
   ```bash
   chmod +x extract_cloudcode_section.sh prepare_claude_session.sh use_snippet.sh
   ```

3. **Unclear Responses**: Ensure your prompts include:
   - Specific requirements and acceptance criteria
   - Interface details and expected behavior
   - Error handling expectations
   - Performance considerations

## 🌐 Resources

- **Claude Documentation**: https://docs.anthropic.com/claude/
- **Claude Code CLI**: https://github.com/anthropics/claude-code/
- **Project Repository**: [TCCC.ai Repository URL]