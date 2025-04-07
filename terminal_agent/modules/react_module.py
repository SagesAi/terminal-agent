#!/usr/bin/env python3
"""
ReAct module for Terminal Agent

This module provides an interface to use the ReAct Agent within the Terminal Agent framework.
"""

import logging
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.react.agent import create_react_agent, ReActAgent, ToolName
from terminal_agent.utils.logging_config import get_logger

# Initialize Rich console
console = Console()

# 获取日志记录器
logger = get_logger(__name__)


class ReActModule:
    """Module for using ReAct Agent in Terminal Agent"""
    
    def __init__(self, llm_client: LLMClient, system_info: Dict[str, Any]):
        """
        Initialize ReAct module
        
        Args:
            llm_client: LLM client for API interactions
            system_info: Dictionary containing system information
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.command_analyzer = CommandAnalyzer(llm_client, system_info)
        
        # Create the ReAct agent
        self.agent = create_react_agent(llm_client, system_info, self.command_analyzer)
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Process a user query using the ReAct agent
        
        Args:
            query: Natural language query from user
            conversation_history: Optional conversation history for context
            
        Returns:
            Response text with results
        """
        # Display a message indicating that ReAct agent is processing
        console.print(Panel(
            "[bold]Processing your query using ReAct agent...[/bold]\n"
            "This agent will reason step-by-step and use tools to answer your query.",
            title="[bold blue]ReAct Agent[/bold blue]",
            expand=False
        ))
        
        # Execute the query using the ReAct agent
        result = self.agent.execute(query)
        
        # Return the result
        return result
