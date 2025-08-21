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

logger = logging.getLogger(__name__)


class ToolDescriptionWrapper:
    """Wrapper for managing tool descriptions with detailed information"""
    
    def __init__(self, short_description: str, detailed_description: str = None):
        self.short_description = short_description
        self.detailed_description = detailed_description or short_description
        self.current_mode = "detailed"  # "short" or "detailed"
    
    def set_mode(self, mode: str) -> None:
        """Set description mode: 'short' or 'detailed'"""
        if mode not in ["short", "detailed"]:
            raise ValueError("Mode must be 'short' or 'detailed'")
        self.current_mode = mode
    
    def get_description(self) -> str:
        """Get description based on current mode"""
        return self.detailed_description if self.current_mode == "detailed" else self.short_description
    
    def get_short_description(self) -> str:
        """Get short description"""
        return self.short_description
    
    def get_detailed_description(self) -> str:
        """Get detailed description"""
        return self.detailed_description
    
    def __str__(self) -> str:
        return self.get_description()


class ToolType(str, Enum):
    """Enumeration of tool types"""
    SHELL = "shell"
    SCRIPT = "script"
    FILES = "files"
    WEB_SEARCH = "web_search"
    WEB_PAGE = "web_page"
    GET_ALL_REFERENCES = "get_all_references"
    GET_FOLDER_STRUCTURE = "get_folder_structure"
    GOTO_DEFINITION = "goto_definition"
    ZOEKT_SEARCH = "zoekt_search"
    GET_SYMBOLS = "get_symbols"
    CODE_EDIT = "code_edit"
    MESSAGE = "message"
    EXPAND_MESSAGE = "expand_message"


class FunctionDefinition(BaseModel):
    """Function definition for OpenAI function calling"""
    name: str
    description: str
    description_wrapper: ToolDescriptionWrapper
    parameters: Dict[str, Any]
    
    model_config = {"arbitrary_types_allowed": True}
    
    def update_description_from_wrapper(self) -> None:
        """Update description field based on wrapper current mode"""
        self.description = self.description_wrapper.get_description()


class FunctionCall(BaseModel):
    """Represents a function call from LLM"""
    name: str
    arguments: Dict[str, Any]


class ToolRegistry:
    """Registry for managing function call tools"""
    
    def __init__(self):
        self._tools: Dict[str, FunctionDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
    
    def register_tool(self, 
                     name: str, 
                     description: str, 
                     parameters: Dict[str, Any], 
                     handler: Callable,
                     detailed_description: str = None) -> None:
        """Register a new tool"""
        description_wrapper = ToolDescriptionWrapper(description, detailed_description)
        self._tools[name] = FunctionDefinition(
            name=name,
            description=description,  # Store short description for compatibility
            description_wrapper=description_wrapper,
            parameters=parameters
        )
        self._handlers[name] = handler
        logger.info(f"Registered tool: {name}")
    
    def get_tools(self, use_detailed_descriptions: bool = False) -> List[Dict[str, Any]]:
        """Get all registered tools in OpenAI format"""
        # Set the mode for all tools
        mode = "detailed" if use_detailed_descriptions else "short"
        for tool in self._tools.values():
            tool.description_wrapper.set_mode(mode)
            tool.update_description_from_wrapper()
        
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,  # This will be updated based on mode
                    "parameters": tool.parameters
                }
            }
            for tool in self._tools.values()
        ]
    
    def get_tool(self, name: str) -> Optional[FunctionDefinition]:
        """Get a specific tool definition"""
        return self._tools.get(name)
    
    def execute_tool(self, function_call: FunctionCall) -> Any:
        """Execute a tool with given arguments"""
        if function_call.name not in self._handlers:
            raise ValueError(f"Unknown tool: {function_call.name}")
        
        handler = self._handlers[function_call.name]
        return handler(**function_call.arguments)


# Global tool registry instance
tool_registry = ToolRegistry()


def register_default_tools():
    """Register all default tools"""
    
    # Shell tool
    tool_registry.register_tool(
        name="shell",
        description="Execute shell commands to interact with system",
        detailed_description="""Execute shell commands safely with confirmation prompts and error handling.
        
        Features:
        - Interactive command confirmation before execution
        - Background execution support for long-running commands
        - Command output capture and display
        - Error handling and exit code reporting
        - Support for complex shell operations (pipes, redirects, etc.)
        
        Use this tool for:
        - File system operations (ls, cd, mkdir, rm, etc.)
        - Process management (ps, kill, etc.)
        - System information gathering (uname, df, etc.)
        - Network operations (ping, curl, etc.)
        """,
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "background": {
                    "type": "boolean",
                    "description": "Whether to run command in background",
                    "default": False
                }
            },
            "required": ["command"]
        },
        handler=lambda command, background=False: shell_function(command, background)
    )
    
    # Script tool
    tool_registry.register_tool(
        name="script",
        description="Create and execute scripts in various languages",
        detailed_description="""Create, execute, and manage scripts in multiple programming languages.
        
        Features:
        - Multi-language support (Python, Bash, Node.js, Ruby, etc.)
        - Script creation with syntax validation
        - Execution with argument passing
        - Timeout support for long-running scripts
        - Error handling and output capture
        
        Actions:
        - 'create': Create a new script file
        - 'execute': Run an existing script file
        - 'create_and_execute': Create and immediately run a script
        
        Supported languages: python3, bash, node, ruby, perl, php
        
        Use this tool for:
        - Automated task scripting
        - Data processing workflows
        - System administration tasks
        - Complex multi-step operations
        """,
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "execute", "create_and_execute"],
                    "description": "Action to perform on script"
                },
                "filename": {
                    "type": "string",
                    "description": "Script filename"
                },
                "content": {
                    "type": "string",
                    "description": "Script content (for create action)"
                },
                "interpreter": {
                    "type": "string",
                    "description": "Interpreter to use (e.g., python3, bash, node)",
                    "default": "bash"
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments to pass to script"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds"
                }
            },
            "required": ["action", "filename"]
        },
        handler=lambda action, filename, content=None, interpreter="bash", args=None, timeout=None: 
            script_function(action, filename, content, interpreter, args or [], timeout)
    )
    
    # Files tool
    tool_registry.register_tool(
        name="files",
        description="Perform file operations including creating, reading, updating, and deleting files",
        detailed_description="""Comprehensive file management tool with support for multiple operations.
        
        Operations:
        - 'read_file': Read contents of a text file
        - 'write_file': Create or overwrite a file with content
        - 'list_directory': List contents of a directory
        - 'file_exists': Check if a file or directory exists
        - 'delete_file': Delete a file or empty directory
        
        Features:
        - Automatic encoding detection for file reading
        - Path validation and normalization
        - Safe file operations with backup support
        - Directory traversal protection
        - Binary and text file support
        
        Use this tool for:
        - Configuration file management
        - Log file analysis
        - Code file reading and editing
        - Directory structure exploration
        - File existence validation
        """,
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read_file", "write_file", "list_directory", "file_exists", "delete_file"],
                    "description": "File operation to perform"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to file or directory"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write_file operation)"
                },
                "directory_path": {
                    "type": "string",
                    "description": "Directory path (for list_directory operation)"
                }
            },
            "required": ["operation"]
        },
        handler=lambda operation, file_path=None, content=None, directory_path=None:
            files_function(operation, file_path, content, directory_path)
    )
    
    # Web search tool
    tool_registry.register_tool(
        name="web_search",
        description="Perform a web search using DuckDuckGo",
        detailed_description="""Search the web for up-to-date information using DuckDuckGo search engine.
        
        Features:
        - Real-time web search capability
        - Result relevance ranking
        - URL and snippet extraction
        - Safe search filtering
        - Customizable result count
        
        Search capabilities:
        - Current events and news
        - Technical documentation
        - Academic papers and research
        - Product information and reviews
        - General knowledge queries
        
        Use this tool for:
        - Finding latest information
        - Technical problem research
        - API documentation lookup
        - Current events and trends
        - Supplementing knowledge cutoff limitations
        """,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        handler=lambda query, max_results=5: web_search_function(query, max_results)
    )
    
    # Web page tool
    tool_registry.register_tool(
        name="web_page",
        description="Fetch and extract content from a web page",
        detailed_description="""Fetch and extract readable content from web pages with intelligent processing.
        
        Features:
        - HTML content extraction and cleaning
        - Article and main content detection
        - JavaScript-free text extraction
        - Metadata preservation (title, author, date)
        - Link and image extraction
        - Character encoding handling
        - HTTP error handling
        
        Content extraction:
        - Main article text extraction
        - Navigation and advertisement removal
        - Code block preservation
        - Table and list formatting
        - Link URL collection
        
        Use this tool for:
        - Documentation reading
        - Article content analysis
        - API endpoint documentation
        - Tutorial and guide extraction
        - Research paper access
        """,
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        },
        handler=lambda url: web_page_function(url)
    )
    
    # Get folder structure tool
    tool_registry.register_tool(
        name="get_folder_structure",
        description="Get folder structure of a repository or directory",
        detailed_description="""Analyze and visualize the hierarchical structure of directories and repositories.
        
        Features:
        - Recursive directory traversal
        - Configurable depth limits
        - File type categorization
        - Size statistics generation
        - Project structure analysis
        - Tree visualization support
        
        Analysis capabilities:
        - File count by type (source, config, docs, etc.)
        - Directory size calculation
        - Project type detection (Python, Node.js, Java, etc.)
        - Common file pattern recognition
        - Hidden file inclusion control
        
        Use this tool for:
        - New project exploration
        - Codebase understanding
        - Architecture analysis
        - File organization assessment
        - Project documentation generation
        """,
        parameters={
            "type": "object",
            "properties": {
                "repo_dir": {
                    "type": "string",
                    "description": "Directory path to analyze",
                    "default": "."
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth to traverse",
                    "default": 3
                }
            },
            "required": []
        },
        handler=lambda repo_dir=".", max_depth=3: get_folder_structure_function(repo_dir, max_depth)
    )
    
    # Code edit tool
    tool_registry.register_tool(
        name="code_edit",
        description="Edit code files with syntax checking and formatting",
        detailed_description="""Precise code editing tool with syntax validation and content matching.
        
        Features:
        - Exact content matching for safe replacements
        - Syntax validation after edits
        - Line-based editing operations
        - File backup creation
        - Multi-language support
        - Error recovery and rollback
        
        Editing capabilities:
        - Function and method modifications
        - Variable name changes
        - Import statement updates
        - Comment additions and removals
        - Configuration file updates
        - Bug fixes and patches
        
        Supported languages:
        - Python, JavaScript, TypeScript, Java
        - C/C++, Go, Rust, Swift
        - Shell scripts, JSON, YAML, XML
        - SQL, HTML, CSS, Markdown
        
        Use this tool for:
        - Code refactoring
        - Bug fixing
        - Feature implementation
        - Configuration updates
        - Documentation comments
        """,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to edit"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (1-indexed)"
                },
                "new_content": {
                    "type": "string",
                    "description": "New content to replace specified lines"
                },
                "check_syntax": {
                    "type": "boolean",
                    "description": "Whether to check syntax after edit",
                    "default": True
                }
            },
            "required": ["file_path", "start_line", "end_line", "new_content"]
        },
        handler=lambda file_path, start_line, end_line, new_content, check_syntax=True:
            code_edit_function(file_path, start_line, end_line, new_content, check_syntax)
    )
    
    # Message tool
    tool_registry.register_tool(
        name="message",
        description="Ask user a question and wait for a response",
        detailed_description="""Interactive communication tool for gathering user input and clarifications.
        
        Features:
        - Interactive question prompts
        - Multi-line response support
        - Response validation
        - Default value suggestion
        - Required field indication
        - User-friendly prompt formatting
        
        Communication patterns:
        - Confirmation requests
        - Parameter validation
        - Decision point queries
        - Clarification requests
        - Missing information gathering
        - User preference collection
        
        Use this tool for:
        - Configuration parameter confirmation
        - User preference collection
        - Decision point clarification
        - Missing information requests
        - Action confirmation prompts
        - User input validation
        - Interactive workflows
        """,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask user"
                }
            },
            "required": ["question"]
        },
        handler=lambda question: message_function(question)
    )


# Tool function implementations (wrappers around existing tools)
def shell_function(command: str, background: bool = False) -> str:
    """Wrapper for shell command execution"""
    from terminal_agent.react.agent import shell_command_tool
    params = {"command": command, "background": background}
    return shell_command_tool(json.dumps(params))


def script_function(action: str, filename: str, content: str = None, 
                   interpreter: str = "bash", args: List[str] = None, 
                   timeout: int = None) -> str:
    """Wrapper for script operations"""
    from terminal_agent.react.tools.script_tool import script_tool
    params = {
        "action": action,
        "filename": filename,
        "interpreter": interpreter,
        "args": args or []
    }
    if content:
        params["content"] = content
    if timeout:
        params["timeout"] = timeout
    return script_tool(json.dumps(params))


def files_function(operation: str, file_path: str = None, content: str = None, 
                  directory_path: str = None) -> str:
    """Wrapper for file operations"""
    from terminal_agent.react.tools.files_tool import files_tool
    params = {"operation": operation}
    if file_path:
        params["file_path"] = file_path
    if content:
        params["content"] = content
    if directory_path:
        params["directory_path"] = directory_path
    return files_tool(json.dumps(params))


def web_search_function(query: str, max_results: int = 5) -> str:
    """Wrapper for web search"""
    from terminal_agent.react.tools.web_search_tool import web_search_tool
    params = {"query": query, "max_results": max_results}
    return web_search_tool(json.dumps(params))


def web_page_function(url: str) -> str:
    """Wrapper for web page fetching"""
    from terminal_agent.react.tools.web_page import web_page_tool
    params = {"url": url}
    return web_page_tool(json.dumps(params))


def get_folder_structure_function(repo_dir: str = ".", max_depth: int = 3) -> str:
    """Wrapper for folder structure analysis"""
    from terminal_agent.react.tools.get_folder_structure_tool import get_folder_structure_tool
    params = {"repo_dir": repo_dir, "max_depth": max_depth}
    return get_folder_structure_tool(json.dumps(params))


def code_edit_function(file_path: str, start_line: int, end_line: int, 
                      new_content: str, check_syntax: bool = True) -> str:
    """Wrapper for code editing"""
    from terminal_agent.react.tools.code_edit_tool import code_edit_tool
    params = {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "new_content": new_content,
        "check_syntax": check_syntax
    }
    return code_edit_tool(json.dumps(params))


def message_function(question: str) -> str:
    """Wrapper for user messaging"""
    from terminal_agent.react.agent import message_tool
    params = {"question": question}
    return message_tool(json.dumps(params))


# Initialize default tools on import
register_default_tools()