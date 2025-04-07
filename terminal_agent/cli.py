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
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        # If not found in current directory, check user's home directory
        home_env = os.path.join(os.path.expanduser("~"), ".terminal_agent.env")
        if os.path.exists(home_env):
            dotenv_path = home_env
    
    if dotenv_path:
        console.print(f"[bold green]Loaded environment from: {dotenv_path}[/bold green]")
        load_dotenv(dotenv_path)
    else:
        console.print("[yellow]No .env file found. Looking for API keys in environment variables.[/yellow]")
        console.print("[yellow]You can create a .env file with your API keys for easier configuration.[/yellow]")
    
    # 配置日志系统
    log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
    log_file = configure_logging(log_level_str=log_level_str, enable_file_logging=True)
    
    if log_level_str == "DEBUG":
        console.print(f"[bold green]日志级别设置为: {log_level_str}[/bold green]")
        console.print(f"[bold green]日志文件保存在: {log_file}[/bold green]")
    
    # Check for API keys and determine provider
    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-4" if provider == "openai" else "deepseek-chat")
    api_base = os.getenv("LLM_API_BASE")
    
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
            console.print("\nOr switch to DeepSeek by setting LLM_PROVIDER=deepseek and providing a DEEPSEEK_API_KEY")
            sys.exit(1)
    elif provider == "deepseek":
        api_key = deepseek_api_key
        if not api_key:
            console.print("[bold red]Error: DEEPSEEK_API_KEY not found in environment variables or .env file[/bold red]")
            console.print("[bold yellow]Please set your DeepSeek API key using one of these methods:[/bold yellow]")
            console.print("1. Create a .env file in the current directory with: DEEPSEEK_API_KEY=your_key_here")
            console.print("2. Create a .terminal_agent.env file in your home directory")
            console.print("3. Set the DEEPSEEK_API_KEY environment variable before running the application")
            console.print("\nOr switch to OpenAI by setting LLM_PROVIDER=openai and providing an OPENAI_API_KEY")
            sys.exit(1)
    else:
        console.print(f"[bold red]Error: Unsupported LLM provider: {provider}[/bold red]")
        console.print("[bold yellow]Please set LLM_PROVIDER to either 'openai' or 'deepseek'[/bold yellow]")
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
