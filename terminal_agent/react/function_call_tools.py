#!/usr/bin/env python3
"""
Function Call tools for ReAct Agent

This module provides function call definitions for ReAct agent
using OpenAI's function calling format.
"""

from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
import json
import logging


from terminal_agent.react.tools.read_file import read_file_tool_handler, TOOL_SCHEMA as READ_FILE_SCHEMA
from terminal_agent.react.tools.write_file import write_file_tool_handler, TOOL_SCHEMA as WRITE_FILE_SCHEMA
from terminal_agent.react.tools.edit_file import edit_file_tool_handler, TOOL_SCHEMA as EDIT_FILE_SCHEMA
from terminal_agent.react.tools.web_search_tool import web_search_tool, TOOL_SCHEMA as WEB_SEARCH_SCHEMA
from terminal_agent.react.tools.web_page import web_page_tool, TOOL_SCHEMA as WEB_PAGE_SCHEMA
from terminal_agent.react.tools.buildin_tool import shell_function, SHELL_TOOL_SCHEMA
from terminal_agent.react.tools.buildin_tool import message_function, MESSAGE_TOOL_SCHEMA


logger = logging.getLogger(__name__)


class OpenAIToolRegistry:
    """Registry for managing tools in OpenAI function calling format."""
    
    def __init__(self):
        self._tools: List[Dict[str, Any]] = []
        self._handlers: Dict[str, Callable] = {}
    
    def register_tool(self, tool_definition: Dict[str, Any], handler: Callable) -> None:
        """Register a tool with its OpenAI-style definition and handler."""
        if "type" not in tool_definition or tool_definition["type"] != "function":
            raise ValueError("Tool definition must have type='function'")
        
        if "name" not in tool_definition.get("function", {}):
            raise ValueError("Tool definition must have a function.name")
        
        name = tool_definition["function"]["name"]
        self._tools.append(tool_definition)
        self._handlers[name] = handler
        logger.info(f"Registered OpenAI tool: {name}")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tools in OpenAI format."""
        return self._tools.copy()
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool with given arguments."""
        if name not in self._handlers:
            raise ValueError(f"Unknown tool: {name}")
        
        handler = self._handlers[name]
        return handler(arguments)
    
    def execute_function_call(self, function_call: Dict[str, Any]) -> str:
        """Execute a function call from OpenAI response."""
        name = function_call.get("name")
        arguments = function_call.get("arguments", {})
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return f"Error parsing arguments: {e}"
        
        return self.execute_tool(name, arguments)


openai_tool_registry = OpenAIToolRegistry()

def register_default_tools():
    """Register all default tools"""

    openai_tool_registry.register_tool(READ_FILE_SCHEMA, read_file_tool_handler)
    openai_tool_registry.register_tool(WRITE_FILE_SCHEMA, write_file_tool_handler)
    openai_tool_registry.register_tool(EDIT_FILE_SCHEMA, edit_file_tool_handler)
    openai_tool_registry.register_tool(WEB_SEARCH_SCHEMA, web_search_tool)
    openai_tool_registry.register_tool(WEB_PAGE_SCHEMA, web_page_tool)
    openai_tool_registry.register_tool(SHELL_TOOL_SCHEMA, shell_function)
    openai_tool_registry.register_tool(MESSAGE_TOOL_SCHEMA, message_function)
    
    

# Initialize default tools on import
register_default_tools()