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
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize ReAct Agent
        self.react_agent = ReActModule(self.llm_client, self.system_info)
    
    def process_user_input(self, user_input: str) -> None:
        """
        Process user input using ReAct Agent as the default approach
        
        Args:
            user_input: Natural language input from user
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Handle exit command
        if user_input.lower() in ["exit", "quit"]:
            exit(0)
        
        # Handle help command
        if user_input.lower() == "help":
            self._show_help()
            return
        
        # Process the input using ReAct Agent
        response = self.react_agent.process_query(user_input, self.conversation_history)
        
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
        
        - **ReAct Agent** - Uses reasoning and acting to solve complex tasks through shell commands
        
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
