# Linux 终端智能Agent

一款基于Linux终端的智能Agent，支持自然语言交互，实现系统问题排查、命令翻译和软件自动安装。

## 功能特点

### 1. 问题排查功能
- 支持自然语言描述系统问题，如内存使用率高
- 自动分析问题描述，生成诊断命令并执行，返回诊断结果
- 提供问题诊断报告及建议解决方案

### 2. 命令翻译功能
- 将自然语言指令翻译为具体的Linux命令
- 支持常用命令翻译，如查看内存使用、CPU负载、磁盘空间等
- 显示并执行翻译后的命令，返回结果到终端

### 3. 软件自动安装功能
- 接受用户自然语言描述的软件安装需求
- 自动识别所需软件的安装方式（apt, yum, snap等）
- 自动执行安装过程，处理安装中可能遇到的问题，返回成功或失败提示

## 安装说明

### 前提条件
- Python 3.8+
- OpenAI API密钥

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/terminal_agent.git
cd terminal_agent
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 设置OpenAI API密钥
```bash
# 创建.env文件
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 使用方法

1. 运行终端Agent
```bash
python terminal_agent.py
```

2. 示例使用场景

- 系统诊断
```
[Terminal Agent] > 我的内存占用很高，帮我看一下原因
```

- 命令翻译
```
[Terminal Agent] > 查找home目录下所有大于100MB的文件
```

- 软件安装
```
[Terminal Agent] > 帮我安装Docker
```

## 安全注意事项

- Agent会在执行任何命令前请求用户确认
- 对于潜在危险的命令，会显示额外的警告提示
- 不建议以root用户运行此Agent

## 技术架构

- **编程语言**：Python
- **自然语言处理**：OpenAI GPT-4 API
- **终端交互**：Rich, Prompt Toolkit
- **系统信息收集**：psutil

## 贡献指南

欢迎提交问题报告和功能请求。如果您想贡献代码，请先开issue讨论您想要更改的内容。

## 许可证

[MIT](LICENSE)
