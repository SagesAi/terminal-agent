---
title: "Terminal Agent 入门指南"
description: "本文将介绍如何安装和使用 Terminal Agent，帮助您快速上手这个强大的终端devops助手工具。"
pubDate: "2025-05-09"
heroImage: "/images/getting-started.png"
author: "Terminal Agent 团队"
readingTime: "8 分钟"
---

# Terminal Agent 入门指南

Terminal Agent 是一个强大的终端助手工具，它可以帮助您通过自然语言与终端进行交互，执行系统诊断、命令翻译和自动软件安装等任务。本文将介绍如何安装和使用 Terminal Agent，帮助您快速上手这个工具。

## 安装 Terminal Agent

### 从源代码安装

1. 克隆仓库
```bash
git clone https://github.com/SagesAi/terminal-agent.git
cd terminal-agent
```

2. 安装包和依赖
```bash
pip install -e .
```

3. 设置 API 密钥
```bash
# 将示例 .env 文件复制到配置目录
mkdir -p ~/.terminal_agent
cp .env.example ~/.terminal_agent/.env

# 编辑 .env 文件添加您的 API 密钥
# 例如：
# OPENAI_API_KEY=your_api_key_here
# DEEPSEEK_API_KEY=your_api_key_here
# GOOGLE_API_KEY=your_api_key_here
# ANTHROPIC_API_KEY=your_api_key_here
```

## 基本使用

### 运行 Terminal Agent

```bash
# 如果从源代码安装：
terminal-agent
```

### 示例使用场景

#### 系统诊断
```
[Terminal Agent] > My system is running slow, can you help me diagnose the issue?
```

#### 命令翻译
```
[Terminal Agent] > Find all files larger than 100MB in my home directory
```

#### 软件安装
```
[Terminal Agent] > Install Docker on my system
```

#### 指定 LLM 提供商
```bash
# 使用特定提供商
TERMINAL_AGENT_PROVIDER=gemini terminal-agent

# 使用 Ollama 的本地模型
TERMINAL_AGENT_PROVIDER=ollama terminal-agent

# 使用 VLLM 服务器
TERMINAL_AGENT_PROVIDER=vllm VLLM_API_BASE=http://localhost:8000 terminal-agent

# 或在 .env 文件中设置
echo "TERMINAL_AGENT_PROVIDER=claude" >> ~/.terminal_agent/.env
```

## 使用本地模型

Terminal Agent 通过 Ollama 和 VLLM 支持本地 LLM 部署，提供隐私和灵活性。

### Ollama 集成

[Ollama](https://ollama.com) 允许您在本地机器上运行各种开源模型。

1. **安装 Ollama**:
   - 从 [ollama.com](https://ollama.com) 下载并安装
   - 启动 Ollama 服务: `ollama serve`

2. **拉取模型**:
   ```bash
   # 拉取模型（示例）
   ollama pull llama3
   ollama pull mistral
   ollama pull llama2
   ```

3. **与 Terminal Agent 一起使用**:
   ```bash
   # 使用 Ollama 和 Terminal Agent
   TERMINAL_AGENT_PROVIDER=ollama terminal-agent
   
   # 指定特定模型
   TERMINAL_AGENT_PROVIDER=ollama TERMINAL_AGENT_MODEL=mistral terminal-agent
   
   # 或在 .env 文件中配置
   echo "TERMINAL_AGENT_PROVIDER=ollama" >> ~/.terminal_agent/.env
   echo "TERMINAL_AGENT_MODEL=llama3" >> ~/.terminal_agent/.env
   ```

## 配置选项

Terminal Agent 可以通过环境变量或 `.env` 文件进行配置：

| 变量 | 描述 | 默认值 |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | 无 |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 无 |
| `GOOGLE_API_KEY` | Google Gemini API 密钥 | 无 |
| `ANTHROPIC_API_KEY` | Anthropic Claude API 密钥 | 无 |
| `TERMINAL_AGENT_PROVIDER` | 默认 LLM 提供商 | `openai` |
| `TERMINAL_AGENT_MODEL` | 所选提供商的默认模型 | 提供商特定 |
| `TERMINAL_AGENT_API_BASE` | 自定义 API 基础 URL | 提供商特定 |
| `OLLAMA_API_BASE` | Ollama API 基础 URL | `http://localhost:11434` |
| `VLLM_API_BASE` | VLLM API 基础 URL | `http://localhost:8000` |
| `VLLM_API_KEY` | VLLM API 密钥（如果需要） | 无 |

## 高级用法

Terminal Agent 的模块化架构允许您扩展其功能或自定义其行为。查看源代码中的 `terminal_agent` 目录，了解更多关于其架构和可能的扩展点的信息。

希望本指南能帮助您开始使用 Terminal Agent。如果您有任何问题或建议，请访问我们的 [GitHub 仓库](https://github.com/SagesAi/terminal-agent)。
