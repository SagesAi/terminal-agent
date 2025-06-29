#!/usr/bin/env python3
"""
Core agent module for Terminal Agent
"""

import os
import platform
import subprocess
import re
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.system_info import get_system_info
from terminal_agent.utils.command_forwarder import forwarder
from terminal_agent.modules.react_module import ReActModule

# Initialize Rich console
console = Console()


class TerminalAgent:
    """Core Terminal Agent class"""
    
    def __init__(self, api_key: str, provider: str = "openai", 
                model: str = "gpt-4", api_base: Optional[str] = None):
        """
        Initialize Terminal Agent
        
        Args:
            api_key: API key for LLM provider
            provider: LLM provider (openai or deepseek)
            model: Model name to use
            api_base: Optional API base URL
        """
        # Initialize LLM client
        self.llm_client = LLMClient(
            api_key=api_key,
            provider=provider,
            model=model,
            api_base=api_base
        )
        
        # Get system information
        self.system_info = get_system_info()
        
        # Add remote execution information to system info if enabled
        if forwarder.remote_enabled:
            self.system_info["remote_execution"] = True
            self.system_info["remote_host"] = forwarder.host
            self.system_info["remote_user"] = forwarder.user
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize ReAct Agent
        self.react_agent = ReActModule(self.llm_client, self.system_info)
        
        # whether user mode is enabled
        self.user_mode = False
    
    def process_user_input(self, user_input: str) -> None:
        """
        Process user input using ReAct Agent as the default approach
        
        Args:
            user_input: Natural language input from user
        """
        # Update system info with current working directory
        # Check if remote execution is enabled
        if hasattr(forwarder, 'remote_enabled') and forwarder.remote_enabled:
            # Get current working directory from remote system
            exit_code, stdout, stderr = forwarder.forward_command("pwd")
            if exit_code == 0:
                self.system_info["current_working_directory"] = stdout.strip()
            else:
                console.print(f"[bold red]Failed to get remote working directory: {stderr}[/bold red]")
                self.system_info["current_working_directory"] = "<unknown remote directory>"
        else:
            # Get local working directory
            self.system_info["current_working_directory"] = os.getcwd()
        
        # Handle exit command
        if user_input.lower() in ["exit", "quit"]:
            exit(0)
            
        if user_input.lower() == "help":
            self._show_help()
            return
            
        # Handle mode switching commands
        command_handled = False
        ai_command = None  # Initialize ai_command variable
        # Handle @file syntax, directly use original input without generating additional ai_command
        # Handle @file or @file: commands (without path)
        if user_input.strip().lower() == "@file:" or user_input.strip().lower() == "@file":
            # List files in current directory for selection
            selected_file = self._list_and_select_files()
            if selected_file:
                # Use the selected file path directly
                self.user_mode = False  # Switch to AI mode
                console.print(f"[bold green]Processing file: [bold cyan]{selected_file}[/bold cyan][/bold green]")
                # Replace user input with the selected file path
                user_input = f"@{selected_file}"
                command_handled = False  # Continue processing
            else:
                console.print("[yellow]No file selected, operation cancelled[/yellow]")
                command_handled = True
                return
        # Handle @file:/path/to/file.py or @file/path/to/file.py syntax
        elif user_input.strip().lower().startswith("@file:") or user_input.strip().lower().startswith("@file/"):
            # Extract file path and possible additional content
            original_input = user_input.strip()
            prefix_end = 6  # Length of @file: or @file/
            remaining = original_input[prefix_end:].strip()
            file_path = ""
            extra_content = ""
            
            # Handle file paths wrapped in quotes
            if remaining.startswith('"') or remaining.startswith("'"):
                quote_char = remaining[0]  # Get quote type
                # Find matching end quote
                quote_end_pos = remaining[1:].find(quote_char)
                
                if quote_end_pos != -1:
                    # Extract file path inside quotes
                    file_path = remaining[1:quote_end_pos+1]
                    # Extract additional content after end quote
                    if len(remaining) > quote_end_pos + 2:
                        extra_content = remaining[quote_end_pos+2:].strip()
                else:
                    # If no end quote found, use all content as file path
                    file_path = remaining[1:]
            else:
                # No quotes, use space as separator
                space_pos = remaining.find(' ')
                
                if space_pos == -1:  # No additional content
                    file_path = remaining
                else:  # Has additional content
                    file_path = remaining[:space_pos].strip()
                    extra_content = remaining[space_pos:].strip()
            
            if os.path.exists(file_path):
                # Use the specified file path directly and keep additional content
                self.user_mode = False  # Switch to AI mode
                console.print(f"[bold green]Processing file: [bold cyan]{file_path}[/bold cyan][/bold green]")
                # Replace user input with file path and keep additional content
                user_input = f"@{file_path}{' ' + extra_content if extra_content else ''}"
                command_handled = False  # Continue processing
            else:
                console.print(f"[bold red]File {file_path} does not exist[/bold red]")
                command_handled = True
                return
        # Handle simple @ syntax (e.g. @path/to/file.py)
        elif user_input.strip().startswith("@") and not user_input.strip().lower().startswith("@user") and not user_input.strip().lower().startswith("@ai"):
            # Extract file path and possible additional content
            original_input = user_input.strip()
            remaining = original_input[1:].strip()  # Remove @ and any leading spaces
            file_path = ""
            extra_content = ""
            
            # Handle file paths wrapped in quotes
            if remaining.startswith('"') or remaining.startswith("'"):
                quote_char = remaining[0]  # Get quote type
                # Find matching end quote
                quote_end_pos = remaining[1:].find(quote_char)
                
                if quote_end_pos != -1:
                    # Extract file path inside quotes
                    file_path = remaining[1:quote_end_pos+1]
                    # Extract additional content after end quote
                    if len(remaining) > quote_end_pos + 2:
                        extra_content = remaining[quote_end_pos+2:].strip()
                else:
                    # If no end quote found, use all content as file path
                    file_path = remaining[1:]
            else:
                # No quotes, use space as separator
                space_pos = remaining.find(' ')
                
                if space_pos == -1:  # No additional content
                    file_path = remaining
                else:  # Has additional content
                    file_path = remaining[:space_pos].strip()
                    extra_content = remaining[space_pos:].strip()
            
            if os.path.exists(file_path):
                # Use the specified file path directly and keep additional content
                self.user_mode = False  # Switch to AI mode
                console.print(f"[bold green]Processing file: [bold cyan]{file_path}[/bold cyan][/bold green]")
                # Replace user input with file path and keep additional content
                user_input = f"@{file_path}{' ' + extra_content if extra_content else ''}"
                command_handled = False  # Continue processing
            else:
                console.print(f"[bold red]File {file_path} does not exist[/bold red]")
                command_handled = True
                return
                
        elif user_input.strip().lower() == "@user":
            # Switch to User Mode
            self.user_mode = True
            console.print("[bold green]Switched to User Mode. Type commands directly.[/bold green]")
            console.print("[dim]To switch back to AI Mode, type '@ai'[/dim]")
            command_handled = True
        elif user_input.strip().lower() == "@ai":
            # Switch to AI Mode
            self.user_mode = False
            console.print("[bold green]Switched to AI Mode. Ask me anything.[/bold green]")
            command_handled = True
        elif user_input.strip().lower().startswith("@ai "):
            # Switch to AI Mode and extract the command
            self.user_mode = False
            console.print("[bold green]Switched to AI Mode[/bold green]")
            
            # Extract the command after '@ai '
            ai_command = user_input[4:].strip()
            # This will be processed in the AI mode section if not empty
            if ai_command:
                command_handled = False  # Allow processing in AI mode
            else:
                command_handled = True   # Empty command, just switch mode
            
        # If a mode switching command was handled, skip further processing
        if command_handled:
            return
        
        # Process input based on current mode
        if self.user_mode:
            # User Mode: directly execute the command
            from terminal_agent.utils.command_executor import execute_command
            console.print(f"[dim]$ {user_input}[/dim]")
            # 设置 show_output=True 让 execute_command 直接显示输出，避免重复显示
            return_code, output, _ = execute_command(user_input, show_output=True, need_confirmation=False)
            
            # 不再重复显示输出，因为 execute_command 已经显示过了
            
            # If command execution failed, display error code
            if return_code != 0 and not output:
                console.print(f"[bold red]Command execution failed with code: {return_code}[/bold red]")
        else:
            # AI Mode: process with ReAct Agent
            # Determine which command to process
            command_to_process = ai_command if ai_command else user_input
            
            # Add user input to conversation history
            self.conversation_history.append({"role": "user", "content": command_to_process})
            
            # Process the input using ReAct Agent
            # Update system info with current working directory before processing
            # Check if remote execution is enabled
            if hasattr(forwarder, 'remote_enabled') and forwarder.remote_enabled:
                # Get current working directory from remote system
                exit_code, stdout, stderr = forwarder.forward_command("pwd")
                if exit_code == 0:
                    self.system_info["current_working_directory"] = stdout.strip()
                else:
                    logger.warning(f"Failed to get remote working directory: {stderr}")
                    self.system_info["current_working_directory"] = "<unknown remote directory>"
            else:
                # Get local working directory
                self.system_info["current_working_directory"] = os.getcwd()
            response = self.react_agent.process_query(command_to_process, self.conversation_history)
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Limit conversation history to last 20 exchanges to prevent context overflow
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
    
    def _show_help(self) -> None:
        """Display help information"""
        help_text = """
        # Terminal Agent Help
        
        Terminal Agent is an intelligent assistant that helps you with Linux terminal commands.
        
        ## Available Commands:
        
        - `help` - Show this help message
        - `exit` or `quit` - Exit Terminal Agent
        - `stop` - Terminate the currently running command and all subsequent operations
        
        ## Available Modes:
        
        - **AI Mode** - AI-driven mode where the agent automatically executes commands when needed (default)
        - **User Mode** - You directly type and execute commands like in a regular terminal
        
        ## Mode Switching:
        
        - `@user` - Switch to User Mode
        - `@ai` - Switch to AI Mode
        - `@ai <command>` - Switch to AI Mode and execute the specified command
        - `@file:` - List files in current directory and select one for AI to analyze
        - `@file:/path/to/file.py` - Directly specify a file for AI to analyze
        
        ## Example Queries:
        
        - "Check my system's disk usage"
        - "Install Docker on my system"
        - "How do I find all PDF files in my home directory?"
        - "Show me the process using the most CPU"
        - "Edit the file /etc/hosts"
        - "Analyze why my application is crashing"
        
        Terminal Agent uses a ReAct (Reasoning and Acting) approach to solve your tasks.
        It will think step-by-step, use shell commands when needed, and provide detailed explanations.
        """
        
        console.print(help_text)
        
    def _list_and_select_files(self) -> Optional[str]:
        """
        List files in the current directory and allow user to select one
        
        Returns:
            Optional[str]: Selected file path or None if cancelled
        """
        try:
            # Determine current directory
            current_dir = self.system_info.get("current_working_directory", os.getcwd())
            
            # Check if remote execution is enabled
            if hasattr(forwarder, 'remote_enabled') and forwarder.remote_enabled:
                # Get file list from remote system
                exit_code, stdout, stderr = forwarder.forward_command("ls -la")
                if exit_code != 0:
                    console.print(f"[bold red]Failed to list remote directory: {stderr}[/bold red]")
                    return None
                    
                # Parse ls output
                files = []
                for line in stdout.strip().split('\n'):
                    if line.startswith('total '):
                        continue
                    parts = line.split()
                    if len(parts) >= 9:
                        # Extract filename (could be multiple parts if it has spaces)
                        filename = ' '.join(parts[8:])
                        if filename not in ['.', '..']:
                            files.append(filename)
            else:
                # Get file list from local system
                files = [f for f in os.listdir(current_dir) if f not in ['.', '..']]  
            
            # Sort files alphabetically
            files.sort()
            
            # Create a table to display files
            table = Table(title=f"Files in {current_dir}")
            table.add_column("#", style="cyan")
            table.add_column("File Name", style="green")
            table.add_column("Type", style="yellow")
            
            # Add files to table
            for i, file in enumerate(files, 1):
                file_path = os.path.join(current_dir, file)
                file_type = "Directory" if os.path.isdir(file_path) else "File"
                table.add_row(str(i), file, file_type)
            
            # Display table
            console.print(table)
            
            # Prompt user to select a file
            console.print("[bold cyan]Please enter file number to select, or 'q' to cancel:[/bold cyan]")
            choice = Prompt.ask(">", default="q")
            
            # Handle user choice
            if choice.lower() == 'q':
                return None
                
            try:
                index = int(choice) - 1
                if 0 <= index < len(files):
                    selected_file = os.path.join(current_dir, files[index])
                    return selected_file
                else:
                    console.print("[bold red]Invalid selection[/bold red]")
                    return None
            except ValueError:
                console.print("[bold red]Invalid input[/bold red]")
                return None
                
        except Exception as e:
            console.print(f"[bold red]Error listing files: {str(e)}[/bold red]")
            return None
