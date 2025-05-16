---
title: "Getting Started with Terminal Agent"
description: "This guide will introduce how to install and use Terminal Agent, helping you quickly get started with this powerful terminal DevOps assistant tool."
pubDate: "2025-05-09"
heroImage: "/images/getting-started.png"
author: "Terminal Agent Team"
readingTime: "8 minutes"
---

# Getting Started with Terminal Agent

Terminal Agent is a powerful terminal assistant tool that helps you interact with the terminal using natural language, performing system diagnostics, command translation, and automatic software installation. This guide will introduce how to install and use Terminal Agent, helping you quickly get started with this tool.

## Installing Terminal Agent

### Installation from Source Code

1. Clone the repository
```bash
git clone https://github.com/SagesAi/terminal-agent.git
cd terminal-agent
```

2. Install packages and dependencies
```bash
pip install -e .
```

3. Set API Keys
```bash
# Copy the example .env file to your config directory
mkdir -p ~/.terminal_agent
cp .env.example ~/.terminal_agent/.env

# Edit the .env file to add your API keys
# For example:
# OPENAI_API_KEY=your_api_key_here
# DEEPSEEK_API_KEY=your_api_key_here
# GOOGLE_API_KEY=your_api_key_here
# ANTHROPIC_API_KEY=your_api_key_here
```

## Basic Usage

### Running Terminal Agent

```bash
# If installed from source:
terminal-agent
```

### Example Use Cases

#### System Diagnostics
```
[Terminal Agent] > My system is running slow, can you help me diagnose the issue?
```

#### Command Translation
```
[Terminal Agent] > Find all files larger than 100MB in my home directory
```

#### Software Installation
```
[Terminal Agent] > Install Docker on my system
```

#### Specifying LLM Provider
```bash
# Using a specific provider
TERMINAL_AGENT_PROVIDER=gemini terminal-agent

# Using local model with Ollama
TERMINAL_AGENT_PROVIDER=ollama terminal-agent

# Using VLLM server
TERMINAL_AGENT_PROVIDER=vllm VLLM_API_BASE=http://localhost:8000 terminal-agent

# Or set in .env file
echo "TERMINAL_AGENT_PROVIDER=claude" >> ~/.terminal_agent/.env
```

## Using Local Models

Terminal Agent supports local LLM deployment via Ollama and VLLM, providing privacy and flexibility.

### Ollama Integration

[Ollama](https://ollama.com) allows you to run various open-source models on your local machine.

1. **Install Ollama:**
   - Download and install from [ollama.com](https://ollama.com)
   - Start Ollama service: `ollama serve`

2. **Pull models:**
   ```bash
   # Pull models (examples)
   ollama pull llama3
   ollama pull mistral
   ollama pull llama2
   ```

3. **Use with Terminal Agent:**
   ```bash
   # Use Ollama with Terminal Agent
   TERMINAL_AGENT_PROVIDER=ollama terminal-agent

   # Specify a specific model
   TERMINAL_AGENT_PROVIDER=ollama TERMINAL_AGENT_MODEL=mistral terminal-agent

   # Or configure in .env file
   echo "TERMINAL_AGENT_PROVIDER=ollama" >> ~/.terminal_agent/.env
   echo "TERMINAL_AGENT_MODEL=llama3" >> ~/.terminal_agent/.env
   ```

## Configuration Options

Terminal Agent can be configured via environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `DEEPSEEK_API_KEY` | DeepSeek API key | None |
| `GOOGLE_API_KEY` | Google Gemini API key | None |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | None |
| `TERMINAL_AGENT_PROVIDER` | Default LLM provider | `openai` |
| `TERMINAL_AGENT_MODEL` | Default model for provider | Provider-specific |
| `TERMINAL_AGENT_API_BASE` | Custom API base URL | Provider-specific |
| `OLLAMA_API_BASE` | Ollama API base URL | `http://localhost:11434` |
| `VLLM_API_BASE` | VLLM API base URL | `http://localhost:8000` |
| `VLLM_API_KEY` | VLLM API key (if needed) | None |

## Advanced Usage

Terminal Agent's modular architecture allows you to extend its functionality or customize its behavior. Check the `terminal_agent` directory in the source code for more information about its architecture and extension points.

We hope this guide helps you get started with Terminal Agent. If you have any questions or suggestions, please visit our [GitHub repository](https://github.com/SagesAi/terminal-agent).
