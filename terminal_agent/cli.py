#!/usr/bin/env python3
"""
Command-line interface for Terminal Agent
"""

import os
import sys
import pyfiglet
import logging
from rich.console import Console
from dotenv import load_dotenv, find_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
import typer
from datetime import datetime

from terminal_agent.core.agent import TerminalAgent
from terminal_agent.utils.command_executor import terminate_current_command, reset_stop_flag
from terminal_agent.utils.logging_config import configure_logging

# Initialize Rich console
console = Console()

# Define styles for prompt toolkit
style = Style.from_dict({
    'prompt': 'ansicyan bold',
})

app = typer.Typer()

@app.command()
def main():
    """Main entry point for the Terminal Agent CLI"""
    # Display welcome banner
    banner = pyfiglet.figlet_format("Terminal Agent", font="slant")
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print("[bold green]Your intelligent Linux terminal assistant[/bold green]")
    console.print("Type 'help' for usage information or 'exit' to quit\n")
    console.print("[bold yellow]Type 'stop' to terminate the currently running command and all subsequent operations[/bold yellow]\n")
    
    # Load environment variables from .env file
    # Try to find .env file in several locations
    home_config_dir = os.path.join(os.path.expanduser("~"), ".terminal_agent")
    home_env = os.path.join(home_config_dir, ".env")
    
    # 首先检查 ~/.terminal_agent/.env 文件
    if os.path.exists(home_env):
        dotenv_path = home_env
        console.print(f"[bold green]Loaded environment from: {dotenv_path}[/bold green]")
    else:
        # 如果在用户目录下没找到，再检查当前目录
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            console.print(f"[bold green]Loaded environment from: {dotenv_path}[/bold green]")
        else:
            console.print("[yellow]No .env file found. Looking for API keys in environment variables.[/yellow]")
            console.print("[yellow]You can create a .env file with your API keys for easier configuration:[/yellow]")
            
            # 打印 .env 示例
            console.print("\n[bold cyan]Example .env file content:[/bold cyan]")
            
            # 使用 Rich 的 Panel 和 Syntax 功能美化 .env 示例
            from rich.panel import Panel
            from rich.syntax import Syntax
            
            env_example = """# LLM Provider settings (uncomment the provider you want to use)
# OpenAI settings
OPENAI_API_KEY=your_openai_key_here
TERMINAL_AGENT_PROVIDER=openai
TERMINAL_AGENT_MODEL=gpt-4

# DeepSeek settings (alternative)
# DEEPSEEK_API_KEY=your_deepseek_key_here
# TERMINAL_AGENT_PROVIDER=deepseek
# TERMINAL_AGENT_MODEL=deepseek-chat

# Google Gemini settings (alternative)
# GOOGLE_API_KEY=your_google_key_here
# TERMINAL_AGENT_PROVIDER=gemini
# TERMINAL_AGENT_MODEL=gemini-pro

# Anthropic Claude settings (alternative)
# ANTHROPIC_API_KEY=your_anthropic_key_here
# TERMINAL_AGENT_PROVIDER=anthropic
# TERMINAL_AGENT_MODEL=claude-3-sonnet

# Ollama settings (for local models)
# TERMINAL_AGENT_PROVIDER=ollama
# TERMINAL_AGENT_MODEL=llama3  # or any model you have pulled in Ollama
# OLLAMA_API_BASE=http://localhost:11434  # Optional, default is http://localhost:11434

# VLLM settings (for local inference server)
# TERMINAL_AGENT_PROVIDER=vllm
# TERMINAL_AGENT_MODEL=llama-2-7b  # or any model loaded in your VLLM server
# VLLM_API_BASE=http://localhost:8000  # Optional, default is http://localhost:8000
# VLLM_API_KEY=your_vllm_key_here  # Optional, only if your VLLM server requires authentication

# Logging settings (optional)
# LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL"""
            
            # 创建语法高亮的内容
            syntax = Syntax(env_example, "ini", theme="monokai", line_numbers=True, word_wrap=True)
            
            # 创建面板
            panel = Panel(
                syntax,
                title="[bold green].env Example[/bold green]",
                border_style="cyan",
                padding=(1, 2),
                expand=False
            )
            
            # 显示面板
            console.print(panel)
            
            console.print("\n[yellow]You can create this file in one of these locations:[/yellow]")
            console.print(f"[bold cyan]1. Create ~/.terminal_agent/.env (recommended)[/bold cyan]")
            console.print("   mkdir -p ~/.terminal_agent && touch ~/.terminal_agent/.env")
            console.print("[cyan]2. Or create .env in your current directory[/cyan]")
            console.print("   touch .env")
            console.print("\n[dim]Then copy the example content above into your .env file and update with your API keys.[/dim]")
            
    # 加载找到的环境变量文件
    if dotenv_path:
        load_dotenv(dotenv_path)
    
    # 配置日志系统
    log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
    log_file = configure_logging(log_level_str=log_level_str, enable_file_logging=True)
    
    if log_level_str == "DEBUG":
        console.print(f"[bold green]日志级别设置为: {log_level_str}[/bold green]")
        console.print(f"[bold green]日志文件保存在: {log_file}[/bold green]")
    
    # Check for API keys and determine provider
    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    vllm_api_key = os.getenv("VLLM_API_KEY")  # Optional for VLLM

    # Check for API base URLs for local providers
    ollama_api_base = os.getenv("OLLAMA_API_BASE")
    vllm_api_base = os.getenv("VLLM_API_BASE")

    provider = os.getenv("TERMINAL_AGENT_PROVIDER", "openai").lower()
    model = os.getenv("TERMINAL_AGENT_MODEL", None)

    # 如果没有指定模型，使用提供商的默认模型
    if not model:
        # 导入相应的提供商类
        if provider == "openai":
            from terminal_agent.utils.llm_providers.openai import OpenAIProvider
            model = OpenAIProvider.get_default_model()
        elif provider == "deepseek":
            from terminal_agent.utils.llm_providers.deepseek import DeepSeekProvider
            model = DeepSeekProvider.get_default_model()
        elif provider == "gemini":
            from terminal_agent.utils.llm_providers.gemini import GeminiProvider
            model = GeminiProvider.get_default_model()
        elif provider == "anthropic":
            from terminal_agent.utils.llm_providers.anthropic import AnthropicProvider
            model = AnthropicProvider.get_default_model()
        elif provider == "ollama":
            from terminal_agent.utils.llm_providers.ollama import OllamaProvider
            model = OllamaProvider.get_default_model()
        elif provider == "vllm":
            from terminal_agent.utils.llm_providers.vllm import VLLMProvider
            model = VLLMProvider.get_default_model()

    # 设置 API 基础 URL
    api_base = None
    if provider == "openai" or provider == "deepseek" or provider == "gemini" or provider == "anthropic":
        api_base = os.getenv("TERMINAL_AGENT_API_BASE")
    elif provider == "ollama":
        api_base = ollama_api_base or "http://localhost:11434"
    elif provider == "vllm":
        api_base = vllm_api_base or "http://localhost:8000"

    # Validate API key based on provider
    api_key = None
    if provider == "openai":
        api_key = openai_api_key
        if not api_key:
            console.print("[bold red]Error: OPENAI_API_KEY not found in environment variables or .env file[/bold red]")
            console.print("[bold yellow]Please set your OpenAI API key using one of these methods:[/bold yellow]")
            console.print("1. Create a .env file in the current directory with: OPENAI_API_KEY=your_key_here")
            console.print("2. Create a .terminal_agent.env file in your home directory")
            console.print("3. Set the OPENAI_API_KEY environment variable before running the application")
            console.print("\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "deepseek":
        api_key = deepseek_api_key
        if not api_key:
            console.print("[bold red]Error: DEEPSEEK_API_KEY not found in environment variables or .env file[/bold red]")
            console.print("[bold yellow]Please set your DeepSeek API key using one of these methods:[/bold yellow]")
            console.print("1. Create a .env file in the current directory with: DEEPSEEK_API_KEY=your_key_here")
            console.print("2. Create a .terminal_agent.env file in your home directory")
            console.print("3. Set the DEEPSEEK_API_KEY environment variable before running the application")
            console.print("\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "gemini":
        api_key = google_api_key
        if not api_key:
            console.print("[bold red]Error: GOOGLE_API_KEY not found in environment variables or .env file[/bold red]")
            console.print("[bold yellow]Please set your Google API key using one of these methods:[/bold yellow]")
            console.print("1. Create a .env file in the current directory with: GOOGLE_API_KEY=your_key_here")
            console.print("2. Create a .terminal_agent.env file in your home directory")
            console.print("3. Set the GOOGLE_API_KEY environment variable before running the application")
            console.print("\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "anthropic":
        api_key = anthropic_api_key
        if not api_key:
            console.print("[bold red]Error: ANTHROPIC_API_KEY not found in environment variables or .env file[/bold red]")
            console.print("[bold yellow]Please set your Anthropic API key using one of these methods:[/bold yellow]")
            console.print("1. Create a .env file in the current directory with: ANTHROPIC_API_KEY=your_key_here")
            console.print("2. Create a .terminal_agent.env file in your home directory")
            console.print("3. Set the ANTHROPIC_API_KEY environment variable before running the application")
            console.print("\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "ollama":
        # 使用 OllamaProvider 检查服务可用性
        from terminal_agent.utils.llm_providers.ollama import OllamaProvider
        
        # 初始化 OllamaProvider 以检查服务可用性
        ollama_provider = OllamaProvider(api_key=None, model=model, api_base=api_base)
        is_available, available_models, error_message = ollama_provider.check_service_availability()
        
        if not is_available:
            console.print(f"[bold red]Error: {error_message}[/bold red]")
            console.print("[bold yellow]Please make sure Ollama is installed and running:[/bold yellow]")
            for instruction in ollama_provider.get_installation_instructions():
                console.print(instruction)
            sys.exit(1)
        
        if error_message:  # 服务可用但模型不存在
            console.print(f"[bold red]Warning: {error_message}[/bold red]")
            console.print(f"[bold yellow]You can pull the requested model with:[/bold yellow]")
            console.print(f"ollama pull {model}")
            
            # 如果有可用模型，使用第一个
            if available_models:
                model = available_models[0]
                console.print(f"[bold green]Using available model: {model}[/bold green]")
    elif provider == "vllm":
        # VLLM API 密钥是可选的
        api_key = vllm_api_key
        
        # 使用 VLLMProvider 检查服务可用性
        from terminal_agent.utils.llm_providers.vllm import VLLMProvider
        
        # 初始化 VLLMProvider 以检查服务可用性
        vllm_provider = VLLMProvider(api_key=api_key, model=model, api_base=api_base)
        is_available, available_models, error_message = vllm_provider.check_service_availability()
        
        if not is_available:
            console.print(f"[bold red]Error: {error_message}[/bold red]")
            console.print("[bold yellow]Please make sure VLLM server is running and accessible:[/bold yellow]")
            for instruction in vllm_provider.get_installation_instructions():
                console.print(instruction)
            sys.exit(1)
        
        if error_message and available_models:  # 服务可用但模型不存在
            console.print(f"[bold red]Warning: {error_message}[/bold red]")
            
            # 如果有可用模型，使用第一个
            model = available_models[0]
            console.print(f"[bold green]Using available model: {model}[/bold green]")
    else:
        console.print(f"[bold red]Error: Unsupported LLM provider: {provider}[/bold red]")
        console.print("[bold yellow]Please set TERMINAL_AGENT_PROVIDER to one of: 'openai', 'deepseek', 'gemini', 'anthropic', 'ollama', or 'vllm'[/bold yellow]")
        sys.exit(1)
    
    console.print(f"[bold green]Using {provider.upper()} API with model: {model}[/bold green]")
    
    # Initialize agent
    agent = TerminalAgent(
        api_key=api_key,
        provider=provider,
        model=model,
        api_base=api_base
    )
    
    # Initialize command history
    history_file = os.path.expanduser("~/.terminal_agent_history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory()
    )
    
    # 标记是否正在处理命令输出
    processing_output = False
    
    # Main interaction loop
    while True:
        try:
            # 重置处理标记，确保每次循环都能获取用户输入
            processing_output = False
                
            # Get user input with prompt toolkit for better UX
            user_input = session.prompt("\n[Terminal Agent] > ", style=style)
            
            # Skip empty inputs
            if not user_input.strip():
                continue
                
            # 处理特殊命令
            if user_input.lower() == 'stop':
                # 终止当前正在运行的命令和所有后续操作
                if terminate_current_command():
                    console.print("[bold yellow]已终止当前命令和所有后续操作[/bold yellow]")
                else:
                    console.print("[yellow]当前没有正在运行的命令[/yellow]")
                continue
                
            # 在每次新的用户输入前重置停止标志
            reset_stop_flag()
                
            # 去除用户输入最前面的空格
            user_input = user_input.lstrip()
                
            # Process the input
            agent.process_user_input(user_input)
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
            # 终止当前命令
            terminate_current_command()
            continue
        except EOFError:
            break
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
    
    console.print("[bold green]Exiting Terminal Agent. Goodbye![/bold green]")


if __name__ == "__main__":
    app()
