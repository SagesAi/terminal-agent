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
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
import typer
from datetime import datetime

from terminal_agent.core.agent import TerminalAgent
from terminal_agent.utils.command_executor import terminate_current_command, reset_stop_flag
from terminal_agent.utils.logging_config import configure_logging
from terminal_agent.utils.command_forwarder import forwarder

# Initialize Rich console
console = Console()

# Define styles for prompt toolkit
style = Style.from_dict({
    'prompt': 'ansicyan bold',
    'multiline': 'ansiyellow',
    'continuation': 'ansiblue',
})

# Create file auto-completer


class FileCompleter(Completer):
    """File auto-completer that supports file name completion triggered by @file"""

    def __init__(self):
        self.path_completer = PathCompleter()
        self.max_depth = 3  # Maximum depth for recursive search

    def get_all_files(self, directory, max_depth=3, current_depth=0):
        """Recursively get all files and directories"""
        files = []
        if current_depth > max_depth:
            return files

        try:
            for item in os.listdir(directory):
                if item in ['.', '..', '.git', '__pycache__',
                            'venv', '.env', 'node_modules']:
                    continue

                full_path = os.path.join(directory, item)
                rel_path = os.path.relpath(full_path, os.getcwd())

                # If in current directory, don't show path
                if rel_path == item:
                    display_path = item
                else:
                    display_path = rel_path

                files.append((display_path, full_path))

                # Recursively process subdirectories
                if os.path.isdir(full_path):
                    files.extend(
                        self.get_all_files(
                            full_path,
                            max_depth,
                            current_depth + 1))
        except (PermissionError, FileNotFoundError):
            pass

        return files

    def get_completions(self, document, complete_event):
        text = document.text

        # Check if completion is triggered by @file
        if text.strip().lower().startswith("@file"):
            input_text = text.strip().lower()

            # Handle keyword-based completion (@filekeyword)
            if len(input_text) > 5 and ' ' not in input_text:
                keyword = input_text[5:]
                try:
                    current_dir = os.getcwd()
                    all_files = self.get_all_files(current_dir, self.max_depth)

                    # Smart matching with multiple strategies
                    filtered_files = []
                    for display, full in all_files:
                        lower_display = display.lower()
                        lower_keyword = keyword.lower()

                        # Scoring criteria (higher is better)
                        score = 0

                        # 1. Exact extension match (e.g., '.py')
                        if lower_keyword.startswith(
                                '.') and lower_display.endswith(lower_keyword):
                            score += 100

                        # 2. Starts with keyword
                        elif lower_display.startswith(lower_keyword):
                            score += 50

                        # 3. Contains keyword
                        elif lower_keyword in lower_display:
                            score += 10

                        # 4. Common pattern matches (e.g., 'test_')
                        if lower_keyword == 'test' and lower_display.startswith(
                                'test_'):
                            score += 30

                        if score > 0:
                            filtered_files.append((display, full, score))

                    # Sort by score (descending) then alphabetically
                    filtered_files.sort(key=lambda x: (-x[2], x[0]))
                    filtered_files = [(d, f) for d, f, _ in filtered_files]

                    # Provide filtered completion
                    for display_path, full_path in filtered_files:
                        is_dir = os.path.isdir(full_path)
                        display = display_path + "/" if is_dir else display_path
                        yield Completion(
                            text=display_path,
                            start_position=-len(input_text),
                            display=display,
                            style='fg:green' if is_dir else 'fg:cyan'
                        )
                    return
                except Exception as e:
                    pass

            # Original @file completion (no keyword)
            if input_text == "@file":
                # Recursively get all files in current directory
                try:
                    current_dir = os.getcwd()
                    all_files = self.get_all_files(current_dir, self.max_depth)
                    all_files.sort(key=lambda x: x[0])  # Sort by relative path

                    # Provide file completion
                    for display_path, full_path in all_files:
                        # Calculate display style: add / for directories, not
                        # for files
                        is_dir = os.path.isdir(full_path)
                        display = display_path + "/" if is_dir else display_path

                        # Return completion item, format as @filename
                        yield Completion(
                            text=display_path,  # Actual text to insert
                            start_position=-4,  # Replace from @, keep the @
                            display=display,  # Text to display
                            style='fg:green' if is_dir else 'fg:cyan'  # Directories in green, files in cyan
                        )
                except Exception as e:
                    console.print(
                        f"[bold red]Error listing files: {
                            str(e)}[/bold red]")
            # If starts with @file:, complete the path
            elif text.strip().lower().startswith("@file:"):
                # Extract path part
                path = text.strip()[6:]

                # Create a new document object containing only the path part
                path_document = Document(path)

                # Use PathCompleter to complete the path
                for completion in self.path_completer.get_completions(
                        path_document, complete_event):
                    # Adjust the start position of completion item
                    yield Completion(
                        text=completion.text,
                        start_position=completion.start_position,
                        display=completion.display,
                        style=completion.style
                    )


app = typer.Typer()


@app.command()
def main():
    """Main entry point for the Terminal Agent CLI"""
    # Display welcome banner
    banner = pyfiglet.figlet_format("Terminal Agent", font="slant")
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print(
        "[bold green]Your intelligent Linux terminal assistant[/bold green]")
    console.print("Type 'help' for usage information or 'exit' to quit\n")
    console.print(
        "[bold yellow]Type 'stop' to terminate the currently running command and all subsequent operations[/bold yellow]")
    console.print(
        "[bold magenta]Multiline input shortcuts: Esc then Enter to insert newline, Ctrl+J also inserts newline, Enter to submit[/bold magenta]\n")

    # Load environment variables from .env file
    # Try to find .env file in several locations
    home_config_dir = os.path.join(os.path.expanduser("~"), ".terminal_agent")
    home_env = os.path.join(home_config_dir, ".env")

    # First check ~/.terminal_agent/.env file
    if os.path.exists(home_env):
        dotenv_path = home_env
        console.print(f"[bold green]Loaded environment from: {
                      dotenv_path}[/bold green]")
    else:
        # If not found in user directory, check current directory
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            console.print(f"[bold green]Loaded environment from: {
                          dotenv_path}[/bold green]")
        else:
            console.print(
                "[yellow]No .env file found. Looking for API keys in environment variables.[/yellow]")
            console.print(
                "[yellow]You can create a .env file with your API keys for easier configuration:[/yellow]")

            # Print .env example
            console.print(
                "\n[bold cyan]Example .env file content:[/bold cyan]")

            # Use Rich's Panel and Syntax features to beautify .env example
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

# Kimi settings (alternative)
# KIMI_API_KEY=your_kimi_key_here
# TERMINAL_AGENT_PROVIDER=kimi
# TERMINAL_AGENT_MODEL=kimi-pro

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
            syntax = Syntax(
                env_example,
                "ini",
                theme="monokai",
                line_numbers=True,
                word_wrap=True)

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

            console.print(
                "\n[yellow]You can create this file in one of these locations:[/yellow]")
            console.print(
                f"[bold cyan]1. Create ~/.terminal_agent/.env (recommended)[/bold cyan]")
            console.print(
                "   mkdir -p ~/.terminal_agent && touch ~/.terminal_agent/.env")
            console.print(
                "[cyan]2. Or create .env in your current directory[/cyan]")
            console.print("   touch .env")
            console.print(
                "\n[dim]Then copy the example content above into your .env file and update with your API keys.[/dim]")

    # 加载找到的环境变量文件
    if dotenv_path:
        load_dotenv(dotenv_path)

    # 配置日志系统
    log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
    log_file = configure_logging(
        log_level_str=log_level_str,
        enable_file_logging=True)

    if log_level_str == "DEBUG":
        console.print(f"[bold green]日志级别设置为: {log_level_str}[/bold green]")
        console.print(f"[bold green]日志文件保存在: {log_file}[/bold green]")

    # Check for API keys and determine provider
    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    vllm_api_key = os.getenv("VLLM_API_KEY")  # Optional for VLLM
    kimi_api_key = os.getenv("KIMI_API_KEY")

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
        elif provider == "kimi":
            from terminal_agent.utils.llm_providers.kimi import KimiProvider
            model = KimiProvider.get_default_model()

    # 设置 API 基础 URL
    api_base = None
    if provider == "openai" or provider == "deepseek" or provider == "gemini" or provider == "anthropic" or provider == "kimi":
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
            console.print(
                "[bold red]Error: OPENAI_API_KEY not found in environment variables or .env file[/bold red]")
            console.print(
                "[bold yellow]Please set your OpenAI API key using one of these methods:[/bold yellow]")
            console.print(
                "1. Create a .env file in the current directory with: OPENAI_API_KEY=your_key_here")
            console.print(
                "2. Create a .terminal_agent.env file in your home directory")
            console.print(
                "3. Set the OPENAI_API_KEY environment variable before running the application")
            console.print(
                "\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "deepseek":
        api_key = deepseek_api_key
        if not api_key:
            console.print(
                "[bold red]Error: DEEPSEEK_API_KEY not found in environment variables or .env file[/bold red]")
            console.print(
                "[bold yellow]Please set your DeepSeek API key using one of these methods:[/bold yellow]")
            console.print(
                "1. Create a .env file in the current directory with: DEEPSEEK_API_KEY=your_key_here")
            console.print(
                "2. Create a .terminal_agent.env file in your home directory")
            console.print(
                "3. Set the DEEPSEEK_API_KEY environment variable before running the application")
            console.print(
                "\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "kimi":
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            console.print(
                "[bold red]Error: KIMI_API_KEY not found in environment variables or .env file[/bold red]")
            console.print(
                "[bold yellow]Please set your Kimi API key using one of these methods:[/bold yellow]")
            console.print(
                "1. Create a .env file in the current directory with: KIMI_API_KEY=your_key_here")
            console.print(
                "2. Create a .terminal_agent.env file in your home directory")
            console.print(
                "3. Set the KIMI_API_KEY environment variable before running the application")
            console.print(
                "\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "gemini":
        api_key = google_api_key
        if not api_key:
            console.print(
                "[bold red]Error: GOOGLE_API_KEY not found in environment variables or .env file[/bold red]")
            console.print(
                "[bold yellow]Please set your Google API key using one of these methods:[/bold yellow]")
            console.print(
                "1. Create a .env file in the current directory with: GOOGLE_API_KEY=your_key_here")
            console.print(
                "2. Create a .terminal_agent.env file in your home directory")
            console.print(
                "3. Set the GOOGLE_API_KEY environment variable before running the application")
            console.print(
                "\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "anthropic":
        api_key = anthropic_api_key
        if not api_key:
            console.print(
                "[bold red]Error: ANTHROPIC_API_KEY not found in environment variables or .env file[/bold red]")
            console.print(
                "[bold yellow]Please set your Anthropic API key using one of these methods:[/bold yellow]")
            console.print(
                "1. Create a .env file in the current directory with: ANTHROPIC_API_KEY=your_key_here")
            console.print(
                "2. Create a .terminal_agent.env file in your home directory")
            console.print(
                "3. Set the ANTHROPIC_API_KEY environment variable before running the application")
            console.print(
                "\nOr switch to another provider by setting TERMINAL_AGENT_PROVIDER and providing the corresponding API key")
            sys.exit(1)
    elif provider == "ollama":
        # 使用 OllamaProvider 检查服务可用性
        from terminal_agent.utils.llm_providers.ollama import OllamaProvider

        # 初始化 OllamaProvider 以检查服务可用性
        ollama_provider = OllamaProvider(
            api_key=None, model=model, api_base=api_base)
        is_available, available_models, error_message = ollama_provider.check_service_availability()

        if not is_available:
            console.print(f"[bold red]Error: {error_message}[/bold red]")
            console.print(
                "[bold yellow]Please make sure Ollama is installed and running:[/bold yellow]")
            for instruction in ollama_provider.get_installation_instructions():
                console.print(instruction)
            sys.exit(1)

        if error_message:  # 服务可用但模型不存在
            console.print(f"[bold red]Warning: {error_message}[/bold red]")
            console.print(
                f"[bold yellow]You can pull the requested model with: [/bold yellow]")
            console.print(f"ollama pull {model}")

            # 如果有可用模型，使用第一个
            if available_models:
                model = available_models[0]
                console.print(
                    f"[bold green]Using available model: {model}[/bold green]")
    elif provider == "vllm":
        # VLLM API 密钥是可选的
        api_key = vllm_api_key

        # 使用 VLLMProvider 检查服务可用性
        from terminal_agent.utils.llm_providers.vllm import VLLMProvider

        # 初始化 VLLMProvider 以检查服务可用性
        vllm_provider = VLLMProvider(
            api_key=api_key, model=model, api_base=api_base)
        is_available, available_models, error_message = vllm_provider.check_service_availability()

        if not is_available:
            console.print(f"[bold red]Error: {error_message}[/bold red]")
            console.print(
                "[bold yellow]Please make sure VLLM server is running and accessible:[/bold yellow]")
            for instruction in vllm_provider.get_installation_instructions():
                console.print(instruction)
            sys.exit(1)

        if error_message and available_models:  # 服务可用但模型不存在
            console.print(f"[bold red]Warning: {error_message}[/bold red]")

            # 如果有可用模型，使用第一个
            model = available_models[0]
            console.print(
                f"[bold green]Using available model: {model}[/bold green]")
    else:
        console.print(f"[bold red]Error: Unsupported LLM provider: {
                      provider}[/bold red]")
        console.print(
            "[bold yellow]Please set TERMINAL_AGENT_PROVIDER to one of: 'openai', 'deepseek', 'gemini', 'anthropic', 'ollama', or 'vllm'[/bold yellow]")
        sys.exit(1)

    console.print(
        f"[bold green]Using {
            provider.upper()} API with model: {model}[/bold green]")

    # Display remote execution status
    if forwarder.remote_enabled:
        console.print(f"[bold green]Remote execution enabled - Connected to: {
                      forwarder.host}@{forwarder.user}[/bold green]")

    # Initialize agent
    agent = TerminalAgent(
        api_key=api_key,
        provider=provider,
        model=model,
        api_base=api_base
    )

    # Initialize command history
    history_file = os.path.expanduser("~/.terminal_agent_history")

    # 创建键绑定
    kb = KeyBindings()

    # Multiline input state
    multiline_input = [False]

    @kb.add('escape', 'enter')
    def _(event):
        """Esc followed by Enter inserts a newline"""
        event.current_buffer.insert_text('\n')
        # Mark current input as multiline
        multiline_input[0] = True

    @kb.add('c-j')
    def _(event):
        """Ctrl+J inserts a newline (more compatible approach)"""
        event.current_buffer.insert_text('\n')
        # Mark current input as multiline
        multiline_input[0] = True

    @kb.add('enter')
    def _(event):
        """Enter key submits input"""
        # Check if file completion is in progress
        buffer = event.current_buffer
        completing = buffer.complete_state is not None

        # If in completion mode, Enter key only accepts completion result,
        # doesn't submit input
        if completing:
            buffer.complete_state = None
        else:
            # Not in completion state, submit input normally
            buffer.validate_and_handle()

    # Create prompt message function
    def get_prompt_tokens():
        if multiline_input[0]:
            return HTML(
                '<ansicyan><b>[Terminal Agent]</b></ansicyan> <ansiyellow><b>[Multiline]</b></ansiyellow> > ')
        else:
            return HTML('<ansicyan><b>[Terminal Agent]</b></ansicyan> > ')

    # 初始化文件补全器
    file_completer = FileCompleter()

    # Initialize PromptSession
    session = PromptSession(
        message=get_prompt_tokens,
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=False,  # Default to single-line mode, implement multiline via key bindings
        key_bindings=kb,
        enable_open_in_editor=True,  # Allow editing in external editor
        input_processors=[],  # Can add custom input processors
        complete_in_thread=True,  # Execute auto-completion in thread to avoid blocking
        completer=file_completer  # 使用自定义的文件补全器
    )

    # Flag to track if currently processing command output
    processing_output = False

    # Main interaction loop
    while True:
        try:
            # Reset processing flag to ensure user input can be captured in
            # each loop
            processing_output = False

            # Get user input with prompt toolkit for better UX
            user_input = session.prompt(style=style)

            # Reset multiline state
            multiline_input[0] = False

            # Skip empty inputs
            if not user_input.strip():
                continue

            # Process special commands
            if user_input.lower() == 'stop':
                # Terminate current running command and all subsequent
                # operations
                if terminate_current_command():
                    console.print(
                        "[bold yellow]Terminated current command and all subsequent operations[/bold yellow]")
                else:
                    console.print(
                        "[yellow]No command currently running[/yellow]")
                continue

            # Reset stop flag before each new user input
            reset_stop_flag()

            # Remove leading whitespace from user input
            user_input = user_input.lstrip()

            # Process the input
            agent.process_user_input(user_input)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
            # Terminate current command
            terminate_current_command()
            continue
        except EOFError:
            break
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")

    console.print("[bold green]Exiting Terminal Agent. Goodbye![/bold green]")


if __name__ == "__main__":
    app()
