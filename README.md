# Terminal Agent

A powerful, intelligent agent for the terminal that supports natural language interaction to perform system diagnostics, command translation, and automated software installation.

<p align="center">
  <img src="./docs/images/terminal_agent_logo.png" alt="Terminal Agent Logo" width="200"/>
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

## Demo

<p align="center">
  <video width="720" controls>
    <source src="./docs/images/terminal_agent.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

<p align="center">
  <a href="./docs/images/terminal_agent.mp4">
    <img src="./docs/images/terminal_agent_logo.png" alt="点击查看演示视频" width="400"/>
    <br>
    <em>点击查看演示视频</em>
  </a>
</p>



## Installation

### Install from Source

1. Clone the repository
```bash
git clone https://github.com/SagesAi/terminal-agent.git
cd terminal-agent
```

2. Install the package and dependencies
```bash
pip install -e .
```

3. Set up your API key(s)
```bash
# Create a .env file in the current directory
echo "OPENAI_API_KEY=your_api_key_here" > .env
# Optional: Add keys for other providers
echo "DEEPSEEK_API_KEY=your_api_key_here" >> .env
echo "GOOGLE_API_KEY=your_api_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_api_key_here" >> .env
```

## Usage

1. Run Terminal Agent
```bash
# If installed from source:
python -m terminal_agent
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

- Specifying LLM Provider
```bash
# Use a specific provider
TERMINAL_AGENT_PROVIDER=gemini terminal-agent

# Use local models with Ollama
TERMINAL_AGENT_PROVIDER=ollama terminal-agent

# Use VLLM server
TERMINAL_AGENT_PROVIDER=vllm VLLM_API_BASE=http://localhost:8000 terminal-agent

# Or set in your .env file
echo "TERMINAL_AGENT_PROVIDER=claude" >> ~/.terminal_agent/.env
```

## Using Local Models

Terminal Agent supports local LLM deployments through Ollama and VLLM, providing privacy and flexibility.

### Ollama Integration

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

### VLLM Integration

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

## Security Considerations

- Terminal Agent will request user confirmation before executing any command
- Additional warnings are displayed for potentially dangerous commands
- It is not recommended to run this agent as a root user
- API keys are managed through environment variables for security

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

## Configuration

Terminal Agent can be configured through environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `DEEPSEEK_API_KEY` | DeepSeek API key | None |
| `GOOGLE_API_KEY` | Google API key for Gemini | None |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | None |
| `TERMINAL_AGENT_PROVIDER` | Default LLM provider | `openai` |
| `TERMINAL_AGENT_MODEL` | Default model for the selected provider | Provider-specific |
| `TERMINAL_AGENT_API_BASE` | Custom API base URL | Provider-specific |
| `OLLAMA_API_BASE` | Ollama API base URL | `http://localhost:11434` |
| `VLLM_API_BASE` | VLLM API base URL | `http://localhost:8000` |
| `VLLM_API_KEY` | VLLM API key (if required) | None |

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
