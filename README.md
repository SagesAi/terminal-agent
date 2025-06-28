# Terminal Agent

A powerful, intelligent agent for the terminal that supports natural language interaction to perform system diagnostics, command translation, and automated software installation.

<p align="center">
  <img src="./images/terminal_agent_logo.png" alt="Terminal Agent Logo" width="200"/>
</p>

<p align="center">
  <img src="./images/demo.gif" alt="Terminal Agent Demo" width="700"/>
</p>

## Features

### 1. System Diagnostics
- Analyze system issues described in natural language
- Automatically generate and execute diagnostic commands
- Provide comprehensive diagnostic reports and solution recommendations

### 2. Command Translation
- Convert natural language instructions into specific terminal commands
- Support for common operations like checking memory usage, CPU load, disk space, etc.
- Display and execute translated commands, returning results to the terminal

### 3. Automated Software Installation
- Process natural language software installation requests
- Automatically identify the appropriate installation method (apt, yum, snap, etc.)
- Execute the installation process, handling potential issues, and providing success/failure feedback

### 4. Multi-LLM Provider Support
- Support for multiple LLM providers:
  - OpenAI (GPT-3.5, GPT-4)
  - DeepSeek
  - Google Gemini
  - Anthropic Claude
  - Ollama (local open-source models)
  - VLLM (high-performance inference server)
- Flexible provider selection based on availability and preference
- Consistent interface across all providers

### 5. Privacy-Focused Local Models
- Run entirely offline with local models via Ollama and VLLM
- No data sent to external APIs when using local models
- Support for various open-source models (Llama, Mistral, etc.)
- Easy switching between cloud and local providers

### 6. Remote Execution Support
- Execute commands on remote servers via SSH
- Seamless integration between local and remote operations
- Support for key-based and password authentication
- Configurable sudo permissions for remote operations


## Architecture

Terminal Agent is built with a modular architecture:

- **Core Components**:
  - `LLMClient`: Flexible client supporting multiple LLM providers
  - `CommandExecutor`: Safe execution of system commands
  - `CommandAnalyzer`: Analysis and validation of generated commands
  - `CommandOrchestrator`: Coordination of command sequences

- **Modules**:
  - `DiagnosticsModule`: System problem diagnosis
  - `CommandTranslatorModule`: Natural language to command translation
  - `SoftwareInstallerModule`: Automated software installation
  - `ReActModule`: ReAct-based reasoning for complex tasks

- **LLM Providers**:
  - Modular provider system with consistent interfaces
  - Easy extension to support additional providers

## Installation

### Install from Source

1. Clone the repository
```bash
git clone https://github.com/SagesAi/terminal-agent.git
cd terminal-agent
```

2. (Optional) Create a Python 3.12 environment using conda
```bash
conda create -n terminal-agent python=3.12
conda activate terminal-agent
```

3. Install the package and dependencies
```bash
pip install -r requirements.txt
pip install -e .
```


## Configuration

Terminal Agent can be configured through environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `DEEPSEEK_API_KEY` | DeepSeek API key | None |
| `GOOGLE_API_KEY` | Google API key for Gemini | None |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | None |
| `JINA_API_KEY` | Jina AI API key for web crawling | None |
| `TERMINAL_AGENT_PROVIDER` | Default LLM provider | `openai` |
| `TERMINAL_AGENT_MODEL` | Default model for the selected provider | Provider-specific |
| `TERMINAL_AGENT_API_BASE` | Custom API base URL | Provider-specific |
| `OLLAMA_API_BASE` | Ollama API base URL | `http://localhost:11434` |
| `VLLM_API_BASE` | VLLM API base URL | `http://localhost:8000` |
| `VLLM_API_KEY` | VLLM API key (if required) | None |
| `REMOTE_EXECUTION_ENABLED` | Enable remote execution via SSH | `false` |
| `REMOTE_HOST` | Remote host address | None |
| `REMOTE_USER` | Remote username | None |
| `REMOTE_PORT` | Remote SSH port | `22` |
| `REMOTE_AUTH_TYPE` | Authentication type (`key` or `password`) | `key` |
| `REMOTE_KEY_PATH` | Path to SSH private key | `~/.ssh/id_rsa` |
| `REMOTE_PASSWORD` | SSH password (if using password auth) | None |
| `REMOTE_SUDO_ENABLED` | Allow sudo commands on remote host | `false` |


1. Set up your API key(s)
```bash
# Copy the example .env file to the config directory
mkdir -p ~/.terminal_agent
cp .env.example ~/.terminal_agent/.env

# Edit the .env file to add your API keys
# For example:
# OPENAI_API_KEY=your_api_key_here
# DEEPSEEK_API_KEY=your_api_key_here
# GOOGLE_API_KEY=your_api_key_here
# ANTHROPIC_API_KEY=your_api_key_here
# JINA_API_KEY=your_api_key_here  # For web crawling functionality
```
2. env config for deepseek

```bash
DEEPSEEK_API_KEY=sk-xxx

TERMINAL_AGENT_PROVIDER=deepseek

#only support deepseek-chat
TERMINAL_AGENT_MODEL=deepseek-chat
```
3. env config for anthropic

```bash
ANTHROPIC_API_KEY=xxx

TERMINAL_AGENT_PROVIDER=anthropic

TERMINAL_AGENT_MODEL=claude-3-7-sonnet-20250219
```
4. env config for vllm

```bash
VLLM_API_BASE=http://localhost:8000

TERMINAL_AGENT_PROVIDER=vllm

#--served-model-name $modelname
TERMINAL_AGENT_MODEL=modelname
```

## Key Features

### User and AI Mode Switching

Terminal Agent supports seamless switching between AI assistant mode and direct user command mode:

```
# AI Mode (default)
[Terminal Agent] > Find large log files and compress them

# Switch to User Mode
[Terminal Agent] > @user

# Direct command execution in User Mode
[Terminal Agent] > ls -la

# Switch back to AI Mode
[Terminal Agent] > @ai
```

### Code LSP Support

Terminal Agent includes Language Server Protocol (LSP) integration for intelligent code analysis:

- Syntax error detection
- Code completion suggestions
- Type checking
- Reference finding
- Semantic analysis

For detailed installation and configuration of code analysis tools, see:
- [Code Analysis Tools (English)](docs/code_analysis_en.md)
- [代码分析工具 (中文)](docs/code_analysis.md)

### Memory System

The built-in memory system enables Terminal Agent to:

- Remember context across multiple interactions
- Intelligently compress and prioritize important information
- Recall previous commands and their outcomes
- Maintain awareness of user preferences and environment details

## Usage

### Basic Usage

1. Run Terminal Agent
```bash
# If installed from source:
terminal-agent
```

2. Example Usage Scenarios

- System Diagnostics
```
[Terminal Agent] > My system is running slow, can you help diagnose the issue?
```

- Command Translation
```
[Terminal Agent] > Find all files larger than 100MB in my home directory
```

- Software Installation
```
[Terminal Agent] > Install Docker on my system
```

- Code Repository Analysis
```
[Terminal Agent] > Analyze this Python repository and summarize its structure
```

- Unit Test Generation
```
[Terminal Agent] > Generate unit tests for the functions in utils/helpers.py
```

- Code Refactoring
```
[Terminal Agent] > Refactor this function to improve performance and readability
```


### Using Local Models

Terminal Agent supports local LLM deployments through Ollama and VLLM, providing privacy and flexibility.

#### Ollama Integration

[Ollama](https://ollama.com) allows you to run various open-source models locally on your machine.

1. **Install Ollama**:
   - Download and install from [ollama.com](https://ollama.com)
   - Start the Ollama service: `ollama serve`

2. **Pull a model**:
   ```bash
   # Pull a model (examples)
   ollama pull llama3
   ollama pull mistral
   ollama pull llama2
   ```

3. **Use with Terminal Agent**:
   ```bash
   # Use Ollama with Terminal Agent
   TERMINAL_AGENT_PROVIDER=ollama terminal-agent
   
   # Specify a particular model
   TERMINAL_AGENT_PROVIDER=ollama TERMINAL_AGENT_MODEL=mistral terminal-agent
   
   # Or configure in .env file
   echo "TERMINAL_AGENT_PROVIDER=ollama" >> ~/.terminal_agent/.env
   echo "TERMINAL_AGENT_MODEL=llama3" >> ~/.terminal_agent/.env
   ```

#### VLLM Integration

[VLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference engine for LLMs.

1. **Install and start VLLM**:
   ```bash
   # Install VLLM
   pip install vllm
   
   # Start VLLM OpenAI-compatible server with a model
   python -m vllm.entrypoints.openai.api_server --model <your-model-name>
   ```

2. **Use with Terminal Agent**:
   ```bash
   # Use VLLM with Terminal Agent
   TERMINAL_AGENT_PROVIDER=vllm terminal-agent
   
   # Specify custom API base if not using default
   TERMINAL_AGENT_PROVIDER=vllm VLLM_API_BASE=http://localhost:8000 terminal-agent
   
   # Or configure in .env file
   echo "TERMINAL_AGENT_PROVIDER=vllm" >> ~/.terminal_agent/.env
   echo "VLLM_API_BASE=http://localhost:8000" >> ~/.terminal_agent/.env
   ```

### Security Considerations

- Terminal Agent will request user confirmation before executing any command
- Additional warnings are displayed for potentially dangerous commands
- It is not recommended to run this agent as a root user
- API keys are managed through environment variables for security

## Contributing

We welcome contributions to Terminal Agent! Here's how you can help:

1. **Report Issues**: Submit bugs or suggest features through GitHub issues
2. **Contribute Code**: 
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/amazing-feature`)
   - Commit your changes (`git commit -m 'Add amazing feature'`)
   - Push to the branch (`git push origin feature/amazing-feature`)
   - Open a Pull Request

3. **Add LLM Providers**: Extend support for additional LLM providers by implementing the `BaseLLMProvider` interface

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses various LLM APIs to provide intelligent functionality
- Special thanks to all contributors and the open-source community

---

<p align="center">
  Made with ❤️ for terminal lovers everywhere
</p>
