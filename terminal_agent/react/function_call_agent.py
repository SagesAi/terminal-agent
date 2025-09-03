#!/usr/bin/env python3
"""
Function Call ReAct Agent implementation

This module implements a ReAct agent using OpenAI's function calling feature
instead of manual JSON parsing.
"""

import json
import logging
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from pydantic import BaseModel, Field
from datetime import datetime

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.utils.command_forwarder import forwarder
from terminal_agent.utils.command_executor import execute_command, should_stop_operations, reset_stop_flag
from terminal_agent.utils.logging_config import get_logger
from terminal_agent.react.function_call_tools import openai_tool_registry

# Initialize Rich console
console = Console()

# Get logger
logger = get_logger(__name__)

# Default template path
PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "templates",
    "function_call_prompt.txt")
DEFAULT_MAX_ITERATIONS = 15


class FunctionCallAgent:
    """
    ReAct agent implementation using OpenAI function calling
    """
    
    def __init__(self,
                 llm_client: LLMClient,
                 system_info: Dict[str, Any],
                 command_analyzer: Optional[CommandAnalyzer] = None,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 template_path: str = PROMPT_TEMPLATE_PATH,
                 memory_enabled: bool = False,
                 memory_db: Optional[Any] = None,
                 user_id: str = "default_user"):
        """
        Initialize the Function Call ReAct Agent
        
        Args:
            llm_client: LLM client for API interactions
            system_info: Dictionary containing system information
            command_analyzer: Analyzer for command safety
            max_iterations: Maximum number of reasoning iterations
            template_path: Path to the prompt template file
            memory_enabled: Whether to enable memory system
            memory_db: Memory database instance
            user_id: User ID for memory system
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.command_analyzer = command_analyzer
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.template = self._load_template(template_path)
        self.model = getattr(llm_client, 'model', 'gpt-4')
        self.conversation_history = []
        self.messages = []  # Store all messages from current execute session
        
        # Memory system
        self.memory_enabled = memory_enabled
        self.user_id = user_id
        self.session_id = None
        
        if memory_enabled:
            try:
                from terminal_agent.memory.memory_database import MemoryDatabase
                from terminal_agent.memory.context_manager import ContextManager
                from terminal_agent.memory.session_manager import SessionManager
                
                if memory_db is None:
                    self.memory_db = MemoryDatabase()
                else:
                    self.memory_db = memory_db
                
                self.context_manager = ContextManager(self.memory_db, llm_client)
                self.session_manager = SessionManager(self.memory_db, self.context_manager)
                logger.info("Memory system initialized for function call agent")
            except ImportError as e:
                logger.warning(f"Failed to import memory modules: {e}")
                self.memory_enabled = False
        
        # Create template if it doesn't exist
        if not os.path.exists(template_path):
            self._create_default_template(template_path)
    
    def _load_template(self, template_path: str) -> str:
        """Load prompt template from file"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Template file not found, creating default")
            self._create_default_template(template_path)
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _create_default_template(self, template_path: str) -> None:
        """Create default function call prompt template"""
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        default_template = """
        You are a SagesAI CLI assistant that can use tools to complete tasks. 

Current system information:
- OS: {os}
- Distribution: {distribution}
- Version: {version}
- Architecture: {architecture}
- Current directory: {current_working_directory}

Use the tools available to you to complete the user's request. When you use a tool, you will receive its response which you can use to inform your next steps.

Guidelines:
1. Think step by step about what needs to be done
2. Use the appropriate tools to gather information or perform actions
3. Always provide a clear and helpful response to the user
4. If a tool fails, try to understand why and adjust your approach
5. Be efficient in your tool usage

Available tools and their usage:
- shell: Execute shell commands
- script: Create and execute scripts
- files: File operations (read, write, list, etc.)
- web_search: Search the web for information
- web_page: Fetch content from web pages
- get_folder_structure: Analyze directory structure
- code_edit: Edit code files with syntax checking
- message: Ask the user for clarification

Respond naturally to the user, incorporating tool results into your responses when appropriate."""
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(default_template)
    
    def _show_processing_message(self, message: str) -> None:
        """Show processing message to user"""
        console.print(Panel(
            f"[bold yellow]Processing...[/bold yellow]\n[dim]{message}[/dim]",
            title="[bold blue]Terminal Agent[/bold blue]",
            border_style="blue",
            expand=False
        ))
    
    def _format_tool_call_display(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Format tool call for display"""
        if tool_name == "shell":
            return f"[bold]⚡ execute: [/bold ]: {arguments.get('command', '')}"

        elif tool_name == "write_file":
            return f"✏️ [bold]write to[/bold ]: {arguments.get('file_path', '')}"
        elif tool_name == "read_file":
            file_path = arguments.get('file_path', '')
            start_line = arguments.get('offset', 1)
            limit = arguments.get('limit', 100)
            end_line = start_line + limit - 1 if limit else 'end'
      
            tree = f"📖 [bold]read[/bold ]: {file_path}\n"
            tree += f"   └── 📜 Lines: {start_line}-{end_line}"
            return tree
        elif tool_name == "web_search":
            return f"🔍 [bold]search for[/bold ]: {arguments.get('query', '')}"
        elif tool_name == "web_page":
            return f"🌐 [bold]fetch[/bold ]: {arguments.get('url', '')}"
        elif tool_name == "get_folder_structure":
            return f"📂 [bold]analyze structure of[/bold ]: {arguments.get('repo_dir', '.')}"
        elif tool_name == "edit_file":
            return f"✏️ [bold]edit[/bold ]: {arguments.get('file_path', '')}"
        elif tool_name == "message":
            return f"❓ [bold]ask[/bold ]: {arguments.get('question', '')}"
        else:
            return f"⚙️ [bold]use[/bold ] {tool_name} tool"
    
    def execute(self, query: str, preserve_history: bool = False) -> str:
        """
        Execute the agent with function calling
        
        Args:
            query: User's query
            preserve_history: Whether to preserve existing conversation history
            
        Returns:
            Final response to the user
        """
        # Reset messages for this execute session
        self.messages = []
        current_messages = []
        self.current_iteration = 0
       
        reset_stop_flag()
        
        # Update system info
        self._update_system_info()
        
        # Initialize memory session if enabled
        if self.memory_enabled:
            self.session_id = self.session_manager.get_or_create_session(self.user_id)
            
            # Store user query at the beginning for proper timing
            try:
                self.session_manager.add_message(
                    self.session_id,
                    "user",
                    query,
                    "message"
                )
            except Exception as e:
                logger.error(f"Error storing user query: {e}")

        current_messages = self._build_system_messages()
        
        if preserve_history:
           # Add preserved history to self.messages
           current_messages.extend(self.conversation_history)

        

        # Build messages
        if query:
            current_messages.append({"role": "user", "content": query})
            # Add built messages to self.messages
          
        #print(current_messages)
        #print("self.max_iterations", self.max_iterations)
        # Main execution loop
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            if should_stop_operations():
                return "Task stopped by user"
            
            try:
                # Get available tools with detailed descriptions for better LLM understanding
                # Use custom tool registry if available, otherwise fall back to global
                registry = getattr(self, 'custom_tool_registry', openai_tool_registry)
                tools = registry.get_tools()
                
                # Log which tools are available
                tool_names = [tool['function']['name'] for tool in tools]
                #logger.info(f"Available tools for this agent: {tool_names}")
                
                #print("tools", tools)
                # Call LLM with function calling
                response = self.llm_client.provider.call_with_messages_and_functions(
                    messages=current_messages,
                    tools=tools
                )
                
                # Get the message from response (handle both OpenAI object and dict formats)
                if hasattr(response, 'choices') and response.choices:
                    # OpenAI object format
                    message = response.choices[0].message
                elif isinstance(response, dict):
                    # Dictionary format from other providers
                    message = response
                else:
                    # Fallback - assume response is the message
                    message = response
                    
                logger.info(f"LLM response: {message}")
                #print("LLM response", message)
                
                # Check if we have tool calls (handle both object and dict formats)
                has_tool_calls = False
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    has_tool_calls = True
                elif isinstance(message, dict) and 'tool_calls' in message and message['tool_calls']:
                    has_tool_calls = True
                
                if has_tool_calls:
                    if hasattr(message, 'content') and message.content:
                        console.print(Panel(
                            Markdown(message.content),
                            title="[bold blue]Get it![/bold blue]",
                            border_style="blue",
                            expand=False
                        ))
                        console.print(" ")
                    # Process tool calls and continue to next iteration
                    self._process_tool_calls(message, current_messages)
                    continue
                
                # No tool calls - this is the final response
                content = None
                if hasattr(message, 'content') and message.content:
                    content = message.content
                elif isinstance(message, dict) and 'content' in message:
                    content = message['content']
                
                if content:
                    final_response = content
                    logger.info(f"Final response: {final_response}")
                    
                    # Add final assistant response to self.messages
                    assistant_response_message = {
                        "role": "assistant",
                        "content": content
                    }
                    self.messages.append(assistant_response_message)
                    
                    # Only show the result panel if this is not a subagent
                    if not getattr(self, 'subagent', False):
                        console.print(Panel(
                            Markdown(final_response),
                            title="[bold green]Result[/bold green]",
                            border_style="green",
                            expand=False
                        ))
                    
                    # Store in memory
                    if self.memory_enabled and self.session_id:
                        self._store_final_response(query, final_response)
                    
                    return final_response
            
            except Exception as e:
                logger.error(f"Error in function call loop: {str(e)}")
                self._show_processing_message(f"Retrying... ({str(e)})")
                continue
        
        # Max iterations reached
        return "Maximum iterations reached. Please try rephrasing your request."
    
    def _update_system_info(self):
        """Update system information including current directory"""
        if hasattr(forwarder, 'remote_enabled') and forwarder.remote_enabled:
            try:
                exit_code, stdout, stderr = forwarder.forward_command("pwd")
                if exit_code == 0:
                    self.system_info["current_working_directory"] = stdout.strip()
                else:
                    self.system_info["current_working_directory"] = "<unknown remote directory>"
            except Exception as e:
                logger.error(f"Error getting remote directory: {e}")
                self.system_info["current_working_directory"] = "<unknown remote directory>"
        else:
            self.system_info["current_working_directory"] = os.getcwd()

        current_date = datetime.now().strftime("%Y-%m-%d")
        self.system_info["current_date"] = f"Today's date: {current_date}"
    
    def _build_system_messages(self) -> List[Dict[str, Any]]:
        """Build message list for LLM"""
        messages = []
        
        # System message
        system_prompt = self.template.format(
            **self.system_info
        )
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add memory context if enabled
        if self.memory_enabled and self.session_id:
            try:
                memory_messages = self.session_manager.get_messages_for_llm_ctx(
                    self.user_id, model=self.model
                )
                # Ensure memory_messages is a list before extending
                if isinstance(memory_messages, dict):
                    messages.append(memory_messages)
                elif isinstance(memory_messages, list):
                    messages.extend(memory_messages)
                else:
                    logger.warning(f"Unexpected memory_messages type: {type(memory_messages)}")
                logger.debug(f"Added {len(memory_messages) if isinstance(memory_messages, list) else 1} memory messages")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
        
        # User query 
        return messages
    
    def _process_tool_calls(self, message, messages):
        """Process tool calls and add results to messages"""
        tool_call_results = []
        
        # Get tool calls from message (handle both object and dict formats)
        if hasattr(message, 'tool_calls'):
            tool_calls = message.tool_calls
        elif isinstance(message, dict) and 'tool_calls' in message:
            tool_calls = message['tool_calls']
        else:
            logger.error("No tool calls found in message")
            return tool_call_results
        
        # Process each tool call
        for tool_call in tool_calls:
            # Handle both object and dict formats for tool_call
            if hasattr(tool_call, 'function'):
                # Object format (OpenAI)
                tool_name = tool_call.function.name
                tool_id = tool_call.id
                arguments = tool_call.function.arguments
            elif isinstance(tool_call, dict) and 'function' in tool_call:
                # Dict format (other providers)
                tool_name = tool_call['function']['name']
                tool_id = tool_call.get('id', f"call_{hash(str(tool_call))}")
                arguments = tool_call['function'].get('arguments', '{}')
            else:
                logger.error(f"Invalid tool call format: {tool_call}")
                continue
                
            try:
                tool_args = json.loads(arguments)
                
                # Get the appropriate tool registry (custom or global)
                registry = getattr(self, 'custom_tool_registry', openai_tool_registry)
                
                # Check if tool exists in the registry
                if not registry.has_tool(tool_name):
                    raise ValueError(f"Tool '{tool_name}' not found in registry")
                
                # Handle compound tool names (e.g., "files.write_file:7" -> "files")
                original_tool_name = tool_name
                if '.' in tool_name:
                    # Extract the base tool name (e.g., "files" from "files.write_file:7")
                    base_tool_name = tool_name.split('.')[0]
                    
                    # For files tool, extract the operation and add it to arguments
                    if base_tool_name == "files":
                        # Extract operation from tool name (e.g., "write_file" from "files.write_file:7")
                        operation_part = tool_name.split('.')[1]
                        if ':' in operation_part:
                            operation = operation_part.split(':')[0]
                        else:
                            operation = operation_part
                        
                        # Add the operation to the arguments if not already present
                        if 'operation' not in tool_args:
                            tool_args['operation'] = operation
                        
                        # Use the base tool name for execution
                        tool_name = base_tool_name
                    else:
                        # For other compound tools, just use the base name
                        tool_name = base_tool_name
                
                # Execute the tool using the appropriate registry
                result = registry.execute_tool(tool_name, tool_args)
                logger.debug(f"Executed tool {tool_name} with args: {tool_args}")
                
                # Format the result for display
                display_result = self._format_tool_call_display(original_tool_name, tool_args)
                console.print(f"[bold blue]• Using Tool: {original_tool_name}[/bold blue]")
                console.print(display_result)
                console.print(" ")
              
                
                # Add tool call result to messages
                tool_call_results.append({
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result)
                })
                
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in tool arguments for {tool_name}: {str(e)}"
                logger.error(f"JSON decode error for {tool_name}: {e}")
                tool_call_results.append({
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Error: Invalid JSON format in tool arguments. Expected valid JSON but got: {str(e)[:200]}"
                })
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                logger.exception(f"Tool execution error for {tool_name}: {e}")
                
                # Provide more user-friendly error messages for common issues
                if "Unknown tool" in str(e):
                    user_error = f"Error: Tool '{tool_name}' is not available or not registered."
                elif "required" in str(e).lower() and "argument" in str(e).lower():
                    user_error = f"Error: Tool '{tool_name}' is missing required arguments. Please check the tool parameters."
                elif "timeout" in str(e).lower():
                    user_error = f"Error: Tool '{tool_name}' execution timed out. Please try again."
                else:
                    user_error = f"Error: Tool '{tool_name}' failed to execute: {str(e)[:200]}"
                
                tool_call_results.append({
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": user_error
                })
        
        # Add the assistant's tool calls to the message history
        assistant_message = {
            "role": "assistant",
            "content": message.get('content', '') if isinstance(message, dict) else getattr(message, 'content', ''),
            "tool_calls": [
                {
                    "id": tc.get('id') if isinstance(tc, dict) else tc.id,
                    "type": "function",
                    "function": {
                        "name": tc['function']['name'] if isinstance(tc, dict) else tc.function.name,
                        "arguments": tc['function'].get('arguments', '{}') if isinstance(tc, dict) else tc.function.arguments
                    }
                } for tc in tool_calls
            ]
        }
        messages.append(assistant_message)
        self.messages.append(assistant_message)  # Add to session messages
        
        # Add all tool call results to messages
        messages.extend(tool_call_results)
        self.messages.extend(tool_call_results)  # Add to session messages
        
        # Store in memory if enabled
        if self.memory_enabled and self.session_id:
            self._store_tool_calls_separately(message, tool_call_results)
        
        return tool_call_results
    
    def _store_tool_calls_separately(self, message, tool_call_results):
        """Store tool calls and results separately in correct chronological order"""
        if not self.memory_enabled or not self.session_id:
            return
        
        try:
            # Get content from message (handle both object and dict formats)
            assistant_content = message.get('content', '') if isinstance(message, dict) else getattr(message, 'content', '')
            message_content = assistant_content if assistant_content else "[Tool calls execution]"
            
            # Get tool calls from message (handle both object and dict formats)
            if hasattr(message, 'tool_calls'):
                tool_calls = message.tool_calls
            elif isinstance(message, dict) and 'tool_calls' in message:
                tool_calls = message['tool_calls']
            else:
                logger.error("No tool calls found in message for storage")
                return
            
            # Include tool_calls data in message metadata for proper OpenAI format reconstruction
            tool_calls_data = [
                {
                    "id": tc.get('id') if isinstance(tc, dict) else tc.id,
                    "type": "function",
                    "function": {
                        "name": tc['function']['name'] if isinstance(tc, dict) else tc.function.name,
                        "arguments": tc['function'].get('arguments', '{}') if isinstance(tc, dict) else tc.function.arguments
                    }
                } for tc in tool_calls
            ]
            
            # Store assistant message with tool_calls metadata
            assistant_message_id = self.session_manager.add_message(
                self.session_id,
                "assistant",
                message_content,
                "message",
                metadata={"tool_calls": tool_calls_data}
            )
            
            # Create a mapping from tool_call_id to result for efficient lookup
            result_map = {result['tool_call_id']: result['content'] for result in tool_call_results}
            
            # Store tool calls and results in chronological order
            for tool_call in tool_calls:
                try:
                    # Handle both object and dict formats for tool_call
                    if hasattr(tool_call, 'function'):
                        # Object format (OpenAI)
                        tool_name = tool_call.function.name
                        tool_id = tool_call.id
                        arguments = tool_call.function.arguments
                    elif isinstance(tool_call, dict) and 'function' in tool_call:
                        # Dict format (other providers)
                        tool_name = tool_call['function']['name']
                        tool_id = tool_call.get('id', f"call_{hash(str(tool_call))}")
                        arguments = tool_call['function'].get('arguments', '{}')
                    else:
                        logger.error(f"Invalid tool call format: {tool_call}")
                        continue
                    
                    tool_args = json.loads(arguments)
                    tool_result = result_map.get(tool_id, '')
                    
                    # Store tool call in standard OpenAI format
                    tool_call_data = {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    }
                    
                    # Store tool call in tool_calls table with tool_call_id
                    self.session_manager.add_tool_call(
                        assistant_message_id,
                        tool_name,
                        tool_call_data,
                        tool_result,
                        tool_id
                    )
                    
                    # Store tool result as separate message in exact OpenAI format
                    self.session_manager.add_message(
                        self.session_id,
                        "tool",
                        tool_result,
                        "tool_result",
                        metadata={"tool_call_id": tool_id}
                    )
                except Exception as e:
                    logger.error(f"Error storing tool call {tool_id}: {e}")
            
            
        except Exception as e:
            logger.error(f"Error storing tool calls separately: {e}")
    
    def _store_assistant_message_with_tool_calls(self, message, tool_call_results):
        """Legacy method - use _store_tool_calls_separately instead"""
        self._store_tool_calls_separately(message, tool_call_results)
    
    def _store_tool_call(self, function_name: str, arguments: Dict[str, Any], result: str):
        """Store individual tool call in memory (legacy method for compatibility)"""
        if not self.memory_enabled or not self.session_id:
            return
        
        try:
            # Create tool call data in standard OpenAI format
            tool_call_data = {
                "tool_name": function_name,
                "arguments": arguments,
                "result": result,
                "timestamp": time.time(),
                "message_type": "tool_call"
            }
            
            # Store as tool message
            self.session_manager.add_message(
                self.session_id,
                "tool",
                result,
                "tool_call",
                metadata=tool_call_data
            )
            
            logger.debug(f"Stored tool call in memory: {function_name}")
            
        except Exception as e:
            logger.error(f"Error storing tool call in memory: {e}")
    
    def _store_final_response(self, query: str, response: str):
        """Store final response conversation in memory with proper timing"""
        try:
            # User query is already stored at the beginning, so just store assistant response
            self.session_manager.add_message(
                self.session_id, 
                "assistant", 
                response, 
                "message"
            )
        except Exception as e:
            logger.error(f"Error storing final response conversation: {e}")
    
    def _store_conversation(self, query: str, response: str):
        """Store conversation in memory (legacy method for compatibility)"""
        try:
            self.session_manager.add_message(
                self.session_id, "user", query, "message"
            )
            self.session_manager.add_message(
                self.session_id, "assistant", response, "message"
            )
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")


def create_function_call_agent(llm_client: LLMClient,
                              system_info: Dict[str, Any],
                              command_analyzer: Optional[CommandAnalyzer] = None,
                              memory_enabled: bool = False,
                              memory_db: Optional[Any] = None,
                              user_id: str = "default_user") -> FunctionCallAgent:
    """
    Create and configure a function call ReAct agent
    
    Args:
        llm_client: LLM client for API interactions
        system_info: Dictionary containing system information
        command_analyzer: Analyzer for command safety
        memory_enabled: Whether to enable memory system
        memory_db: Memory database instance
        user_id: User ID for memory system
    
    Returns:
        FunctionCallAgent: Configured function call agent
    """
    return FunctionCallAgent(
        llm_client=llm_client,
        system_info=system_info,
        command_analyzer=command_analyzer,
        memory_enabled=memory_enabled,
        memory_db=memory_db,
        user_id=user_id
    )
