#!/usr/bin/env python3
"""
Core agent module for Terminal Agent
"""

import os
import platform
import subprocess
from typing import Dict, List, Any, Optional
from rich.console import Console

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
                logger.warning(f"Failed to get remote working directory: {stderr}")
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
        ai_command = None
        
        if user_input.strip().lower() == "@user":
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
