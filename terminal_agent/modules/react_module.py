#!/usr/bin/env python3
"""
ReAct module for Terminal Agent

This module provides an interface to use the ReAct Agent within the Terminal Agent framework.
"""

import logging
import os
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.react.agent import create_react_agent, ReActAgent, ToolName
from terminal_agent.react.function_call_agent import create_function_call_agent, FunctionCallAgent
from terminal_agent.utils.logging_config import get_logger

# Initialize Rich console
console = Console()

# Get logger
logger = get_logger(__name__)


class ReActModule:
    """Module for using ReAct Agent in Terminal Agent"""
    
    def __init__(self, 
                llm_client: LLMClient, 
                system_info: Dict[str, Any],
                memory_enabled: bool = True,
                user_id: str = None):
        """
        Initialize ReAct module
        
        Args:
            llm_client: LLM client for API interactions
            system_info: Dictionary containing system information
            memory_enabled: Whether to enable the memory system
            user_id: User ID for the memory system (defaults to username)
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.command_analyzer = CommandAnalyzer(llm_client, system_info)
        self.memory_enabled = memory_enabled
        
        # Set user ID (default to system username if not provided)
        if user_id is None:
            import getpass
            user_id = getpass.getuser()
        self.user_id = user_id
        
        # Initialize memory database if enabled
        self.memory_db = None
        if memory_enabled:
            try:
                from terminal_agent.memory.memory_database import MemoryDatabase
                self.memory_db = MemoryDatabase()
                logger.info(f"Memory system enabled for user {self.user_id}")
            except ImportError as e:
                logger.warning(f"Failed to import memory modules: {e}. Memory system disabled.")
                self.memory_enabled = False
        
        # Create the Function Call ReAct agent
        self.agent = create_function_call_agent(
            llm_client, 
            system_info, 
            self.command_analyzer,
            memory_enabled=self.memory_enabled,
            memory_db=self.memory_db,
            user_id=self.user_id
        )
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Process a user query using the ReAct agent
        
        Args:
            query: Natural language query from user
            conversation_history: Optional conversation history for context
            
        Returns:
            Response text with results
        """
        # Display a friendly message indicating that we're processing the query
        #console.print(Panel(
        #    "[bold]Processing your request...[/bold]"
        #    "I'll complete your task right away.",
        #   title="[bold blue]Terminal Agent[/bold blue]",
        #   expand=False
        #))
        
        # Execute the query using the ReAct agent
        result = self.agent.execute(query)
        
        # Return the result
        return result
