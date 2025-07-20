#!/usr/bin/env python3
"""
ReAct Agent implementation for Terminal Agent

This module implements a ReAct (Reasoning + Acting) agent that follows the
reasoning-action-observation loop to solve tasks using available tools.
"""

import json
import logging
import os
import re
import sqlite3
import sys
import time
from enum import Enum, auto
from typing import Dict, List, Callable, Union, Optional, Any, Tuple
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from pydantic import BaseModel, Field

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.utils.command_forwarder import forwarder
from terminal_agent.utils.command_executor import execute_command, should_stop_operations, reset_stop_flag
from terminal_agent.utils.logging_config import get_logger
from terminal_agent.react.tools.script_tool import script_tool
from terminal_agent.react.tools.files_tool import files_tool
from terminal_agent.react.tools.web_page import web_page_tool
from terminal_agent.react.tools.get_all_references_tool import get_all_references_tool
from terminal_agent.react.tools.get_folder_structure_tool import get_folder_structure_tool
from terminal_agent.react.tools.goto_definition_tool import goto_definition_tool
from terminal_agent.react.tools.zoekt_search_tool import zoekt_search_tool
from terminal_agent.react.tools.get_symbols_tool import get_symbols_tool
from terminal_agent.react.tools.code_edit_tool import code_edit_tool
from terminal_agent.react.tools.expand_message_tool import expand_message_tool
from terminal_agent.react.tools.web_search_tool import web_search_tool
# Do not import from terminal_agent.react.tools as it would cause circular imports
# Import expand_message_tool module when needed

# Initialize Rich console
console = Console()

# Get logger
logger = get_logger(__name__)

# Type alias for observations
Observation = Union[str, Exception]

# Default template path in the package
PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "templates",
    "react_prompt1.txt")
DEFAULT_MAX_ITERATIONS = 15


class ToolName(Enum):
    """
    Enumeration for tool names available to the agent.
    """
    SHELL = auto()
    SEARCH = auto()
    SCRIPT = auto()
    MESSAGE = auto()  # Message tool for asking questions to the user
    FILES = auto()    # File operations tool for file management
    WEB_PAGE = auto()  # Web page tool for retrieving web content
    # Get all references tool for finding all references to a symbol
    GET_ALL_REFERENCES = auto()
    # Get folder structure tool for visualizing directory structure
    GET_FOLDER_STRUCTURE = auto()
    GOTO_DEFINITION = auto()  # Go to definition tool for finding symbol definitions
    ZOEKT_SEARCH = auto()  # Zoekt search tool for powerful code search
    GET_SYMBOLS = auto()  # Get symbols tool for extracting symbols from files
    CODE_EDIT = auto()  # Code edit tool for precise code modifications with syntax checking
    # Expand message tool for viewing full content of truncated messages
    EXPAND_MESSAGE = auto()
    WEB_SEARCH = auto()  # Web search tool for searching the internet
    NONE = auto()

    def __str__(self) -> str:
        """
        String representation of the tool name.
        """
        return self.name.lower()


class Message(BaseModel):
    """
    Represents a message with sender role and content.
    """
    role: str = Field(..., description="The role of the message sender.")
    content: str = Field(..., description="The content of the message.")


class Tool:
    """
    A wrapper class for tools used by the agent, executing a function based on tool type.
    """

    def __init__(self, name: ToolName, func: Callable[[
                 str], str], description: str = ""):
        """
        Initializes a Tool with a name and an associated function.

        Args:
            name (ToolName): The name of the tool.
            func (Callable[[str], str]): The function associated with the tool.
            description (str): Description of what the tool does.
        """
        self.name = name
        self.func = func
        self.description = description

    def use(self, query: str) -> Observation:
        """
        Executes the tool's function with the provided query.

        Args:
            query (str): The input query for the tool.

        Returns:
            Observation: Result of the tool's function or an error message if an exception occurs.
        """
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return str(e)


class ReActAgent:
    """
    Defines the ReAct agent responsible for reasoning, acting, and observing to solve tasks.
    """

    def _show_processing_message(self, message: str) -> None:
        """
        Shows a user-friendly processing message panel when encountering errors.

        Args:
            message (str): The specific message to display in the panel.
        """
        console.print(Panel(
            f"[bold yellow]Processing your request...[/bold yellow]\n[dim]{
                message}[/dim]",
            title="[bold blue]Terminal Agent[/bold blue]",
            border_style="blue",
            expand=False
        ))
    
    def _generate_elegant_tool_call_text(self, tool_name: ToolName, tool_input: str) -> str:
        """
        Generates elegant tool call text for inclusion in Thinking panel.
        
        Args:
            tool_name: The tool being called
            tool_input: The input for the tool
            
        Returns:
            str: Formatted tool call text
        """
        # Parse tool input to extract relevant information
        try:
            if tool_input.startswith('{') and tool_input.endswith('}'):
                import json
                input_data = json.loads(tool_input)
            else:
                input_data = {}
        except:
            input_data = {}
        
        # Create elegant display based on tool type
        if tool_name == ToolName.FILES:
            operation = input_data.get("operation", "")
            file_path = input_data.get("file_path", "")
            
            if operation == "read_file":
                return f"📖 **I will read** `{file_path}`"
            elif operation == "write_file":
                return f"✏️ **I will write to** `{file_path}`"
            elif operation == "list_directory":
                return f"📁 **I will list contents of** `{file_path}`"
            else:
                return f"📄 **I will perform file operation on** `{file_path}`"
                
        elif tool_name == ToolName.SHELL:
            command = input_data.get("command", tool_input)
            return f"⚡ **I will execute:** `{command}`"
            
        elif tool_name == ToolName.WEB_SEARCH:
            query = input_data.get("query", tool_input)
            return f"🔍 **I will search for:** `{query}`"
            
        elif tool_name == ToolName.WEB_PAGE:
            url = input_data.get("url", tool_input)
            return f"🌐 **I will fetch:** `{url}`"
            
        elif tool_name == ToolName.GET_FOLDER_STRUCTURE:
            repo_dir = input_data.get("repo_dir", "current directory")
            return f"📂 **I will analyze structure of** `{repo_dir}`"
            
        elif tool_name == ToolName.CODE_EDIT:
            file_path = input_data.get("file_path", "")
            return f"✏️ **I will edit** `{file_path}`"
            
        elif tool_name == ToolName.GET_ALL_REFERENCES:
            word = input_data.get("word", "")
            file_path = input_data.get("relative_path", "")
            return f"🔗 **I will find references to** `{word}` **in** `{file_path}`"
            
        elif tool_name == ToolName.GOTO_DEFINITION:
            word = input_data.get("word", "")
            file_path = input_data.get("relative_path", "")
            return f"📍 **I will find definition of** `{word}` **in** `{file_path}`"
            
        elif tool_name == ToolName.ZOEKT_SEARCH:
            names = input_data.get("names", [])
            if isinstance(names, list) and names:
                return f"🔍 **I will search for:** `{', '.join(names)}`"
            else:
                return f"🔍 **I will perform code search**"
                
        elif tool_name == ToolName.GET_SYMBOLS:
            file_path = input_data.get("file_path", "")
            return f"📋 **I will extract symbols from** `{file_path}`"
            
        elif tool_name == ToolName.SCRIPT:
            action = input_data.get("action", "")
            filename = input_data.get("filename", "")
            if action == "create":
                return f"📝 **I will create script** `{filename}`"
            elif action == "execute":
                return f"▶️ **I will execute script** `{filename}`"
            else:
                return f"📝 **I will work with script** `{filename}`"
                
        elif tool_name == ToolName.MESSAGE:
            question = input_data.get("question", tool_input)
            return f"❓ **I will ask:** `{question}`"
            
        else:
            # Fallback for unknown tools
            return f"⚙️ **I will use** `{tool_name}` **tool**"
    
    def _show_elegant_tool_call(self, tool_name: ToolName, tool_input: str) -> None:
        """
        Shows an elegant tool call display similar to Cursor's experience.
        
        Args:
            tool_name: The tool being called
            tool_input: The input for the tool
        """
        # Parse tool input to extract relevant information
        try:
            if tool_input.startswith('{') and tool_input.endswith('}'):
                import json
                input_data = json.loads(tool_input)
            else:
                input_data = {}
        except:
            input_data = {}
        
        # Create elegant display based on tool type
        if tool_name == ToolName.FILES:
            operation = input_data.get("operation", "")
            file_path = input_data.get("file_path", "")
            
            if operation == "read_file":
                console.print(f"📖 I will read [bold cyan]{file_path}[/bold cyan]")
            elif operation == "write_file":
                console.print(f"✏️ I will write to [bold cyan]{file_path}[/bold cyan]")
            elif operation == "list_directory":
                console.print(f"📁 I will list contents of [bold cyan]{file_path}[/bold cyan]")
            else:
                console.print(f"📄 I will perform file operation on [bold cyan]{file_path}[/bold cyan]")
                
        elif tool_name == ToolName.SHELL:
            command = input_data.get("command", tool_input)
            console.print(f"⚡ I will execute: [bold green]{command}[/bold green]")
            
        elif tool_name == ToolName.WEB_SEARCH:
            query = input_data.get("query", tool_input)
            console.print(f"🔍 I will search for: [bold yellow]{query}[/bold yellow]")
            
        elif tool_name == ToolName.WEB_PAGE:
            url = input_data.get("url", tool_input)
            console.print(f"🌐 I will fetch: [bold blue]{url}[/bold blue]")
            
        elif tool_name == ToolName.GET_FOLDER_STRUCTURE:
            repo_dir = input_data.get("repo_dir", "current directory")
            console.print(f"📂 I will analyze structure of [bold magenta]{repo_dir}[/bold magenta]")
            
        elif tool_name == ToolName.CODE_EDIT:
            file_path = input_data.get("file_path", "")
            console.print(f"✏️ I will edit [bold cyan]{file_path}[/bold cyan]")
            
        elif tool_name == ToolName.GET_ALL_REFERENCES:
            word = input_data.get("word", "")
            file_path = input_data.get("relative_path", "")
            console.print(f"🔗 I will find references to [bold yellow]{word}[/bold yellow] in [bold cyan]{file_path}[/bold cyan]")
            
        elif tool_name == ToolName.GOTO_DEFINITION:
            word = input_data.get("word", "")
            file_path = input_data.get("relative_path", "")
            console.print(f"📍 I will find definition of [bold yellow]{word}[/bold yellow] in [bold cyan]{file_path}[/bold cyan]")
            
        elif tool_name == ToolName.ZOEKT_SEARCH:
            names = input_data.get("names", [])
            if isinstance(names, list) and names:
                console.print(f"🔍 I will search for [bold yellow]{', '.join(names)}[/bold yellow]")
            else:
                console.print(f"🔍 I will perform code search")
                
        elif tool_name == ToolName.GET_SYMBOLS:
            file_path = input_data.get("file_path", "")
            console.print(f"📋 I will extract symbols from [bold cyan]{file_path}[/bold cyan]")
            
        elif tool_name == ToolName.SCRIPT:
            action = input_data.get("action", "")
            filename = input_data.get("filename", "")
            if action == "create":
                console.print(f"📝 I will create script [bold cyan]{filename}[/bold cyan]")
            elif action == "execute":
                console.print(f"▶️ I will execute script [bold cyan]{filename}[/bold cyan]")
            else:
                console.print(f"📝 I will work with script [bold cyan]{filename}[/bold cyan]")
                
        elif tool_name == ToolName.MESSAGE:
            question = input_data.get("question", tool_input)
            console.print(f"❓ I will ask: [bold yellow]{question}[/bold yellow]")
            
        else:
            # Fallback for unknown tools
            console.print(f"⚙️ I will use {tool_name} tool")

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
        Initializes the ReAct Agent with necessary components.

        Args:
            llm_client (LLMClient): The LLM client for API interactions.
            system_info (Dict[str, Any]): Dictionary containing system information.
            command_analyzer (CommandAnalyzer, optional): Analyzer for command safety.
            max_iterations (int, optional): Maximum number of reasoning iterations.
            template_path (str, optional): Path to the prompt template file.
            memory_enabled (bool, optional): Whether to enable memory system.
            memory_db (Optional[Any], optional): Memory database instance.
            user_id (str, optional): User ID for memory system.
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.command_analyzer = command_analyzer
        self.tools: Dict[ToolName, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.template = self._load_template(template_path)
        self.model = getattr(llm_client, 'model', 'gpt-4')

        # Memory system initialization
        self.memory_enabled = memory_enabled
        self.user_id = user_id
        self.session_id = None

        if memory_enabled:
            try:
                from terminal_agent.memory.memory_database import MemoryDatabase
                from terminal_agent.memory.context_manager import ContextManager
                from terminal_agent.memory.session_manager import SessionManager

                # Initialize memory components
                if memory_db is None:
                    self.memory_db = MemoryDatabase()
                else:
                    self.memory_db = memory_db

                self.context_manager = ContextManager(
                    self.memory_db, llm_client)
                self.session_manager = SessionManager(
                    self.memory_db, self.context_manager)
                logger.info("Memory system initialized")
            except ImportError as e:
                logger.warning(f"Failed to import memory modules: {
                               e}. Memory system disabled.")
                self.memory_enabled = False
                self.memory_db = None
                self.context_manager = None
                self.session_manager = None

        # Create templates directory if it doesn't exist
        os.makedirs(os.path.dirname(template_path), exist_ok=True)

        # Create default template if it doesn't exist
        if not os.path.exists(template_path):
            self._create_default_template(template_path)
            logger.info(f"Created default template at {template_path}")

    def register_tool(self, name: ToolName, func: Callable[[
                      str], str], description: str = "") -> None:
        """
        Registers a tool to the agent.

        Args:
            name (ToolName): The name of the tool.
            func (Callable[[str], str]): The function associated with the tool.
            description (str): Description of what the tool does.
        """
        self.tools[name] = Tool(name, func, description)

    def trace(self, role: str, content: Any, display: bool = False) -> None:
        """
        Logs the message with the specified role and content.

        Args:
            role (str): The role of the message sender.
            content (Any): The content of the message, will be converted to string.
            display (bool): Whether to display the message. Defaults to True.
        """
        # 确保 content 是字符串
        if not isinstance(content, str):
            content = str(content)

        message = Message(role=role, content=content)
        self.messages.append(message)

        # Store message in memory system if enabled
        if self.memory_enabled and self.session_id and role in [
                "user", "assistant", "system"]:
            try:
                message_type = "thinking" if "Thought:" in content and role == "assistant" else "message"
                self.session_manager.add_message(
                    self.session_id, role, content, message_type)
            except Exception as e:
                logger.error(f"Error storing message in memory: {e}")

        if display:
            if role == "user":
                console.print(f"\n[bold green]User: [/bold green] {content}")
            elif role == "assistant":
                console.print(
                    f"\n[bold blue]Assistant: [/bold blue] {content}")
            elif role == "system":
                console.print(
                    f"\n[bold yellow]System: [/bold yellow] {content}")
            else:
                console.print(f"\n[bold]{role}: [/bold] {content}")

    def get_history(self) -> str:
        """
        Retrieves the conversation history.

        Returns:
            str: Formatted history of messages.
        """
        return "\n".join(
            [f"{message.role}: {message.content}" for message in self.messages])

    def think(self) -> None:
        """
        Processes the current query, decides actions, and iterates until a solution or max iteration limit is reached.
        """
        self.current_iteration += 1
        logger.info(f"Starting iteration {self.current_iteration}")

        # Check if we've reached the maximum number of iterations
        if self.current_iteration > self.max_iterations:
            logger.warning("Reached maximum iterations. Stopping.")
            self.trace(
                "assistant",
                "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations. Here's what I know so far: " +
                self.get_history())
            return

        # Check if operations should be stopped
        if should_stop_operations():
            logger.warning("Operations stopped by user.")
            self.trace("user", "Operations stopped by user.")
            return

        try:
            # 复制基本提示消息列表，避免修改原始列表
            prompt_messages = list(self.base_prompt_messages)

            # get history messages
            history_messages = self._convert_history_to_messages()

            # check if need to compress messages
            try:
                #

                # use context manager to compress messages
                if hasattr(self, 'session_manager') and hasattr(
                        self.session_manager, 'context_manager'):
                    context_manager = self.session_manager.context_manager
                    if context_manager and hasattr(
                            context_manager, 'compress_react_messages'):
                        # use context_manager's compress_react_messages method
                        compressed_history = context_manager.compress_react_messages(
                            messages=history_messages,
                            model=self.model,
                            recent_message_count=5
                        )
                        history_messages = compressed_history
                else:
                    logger.debug("no context manager")
            except Exception as e:
                logger.error(f"compress messages error: {e}")
                # continue using original messages

            # add processed history messages
            prompt_messages.extend(history_messages)

            logger.debug(f"Using {len(self.base_prompt_messages)} base messages and {
                         len(history_messages)} history messages")
            # Get the LLM's response using the message-based method
            response = self.llm_client.call_with_messages(prompt_messages)
            logger.debug(f"Thinking => {response}")

            # 记录思考过程但不显示，显示逻辑移到 decide 方法中
            self.trace("assistant", f"Thought: {response}", display=False)

            # Decide on the next action based on the response
            self.decide(response)
        except ConnectionError as e:
            # Handle connection error, exit React loop directly
            logger.error(f"Connection error in ReActAgent.think: {str(e)}")
            self.trace(
                "user",
                f"Error: Connection to LLM API failed. Please check your internet connection and API settings. Details: {
                    str(e)}",
                display=True)
            # No longer call self.think(), exit the loop directly
            console.print(
                "[bold red]Exiting ReAct loop due to connection error.[/bold red]")
            return

        except Exception as e:
            logger.debug(f"Error in ReActAgent.think: {str(e)}")
            self._show_processing_message(
                "I encountered a small hiccup but I'm trying again automatically.")
            self.trace("user", f"{str(e)}. Trying again.")
            self.think()

    def decide(self, response: str) -> None:
        """
        Processes the agent's response, deciding actions or final answers.

        Args:
            response (str): The response generated by the model.
        """
        try:
            # Try to parse the response as JSON
            parsed_response = self._parse_json_response(response)

            # 整合思考过程和工具调用到 Thinking 框中
            if "thought" in parsed_response and "final_answer" not in parsed_response:
                thought = parsed_response["thought"]
                
                # 构建完整的思考内容，包括工具调用信息
                formatted_content = thought
                
                # 如果有工具调用，添加到思考内容中
                if "action" in parsed_response:
                    action = parsed_response["action"]
                    tool_name_str = action["name"].upper()

                    # Handle the case when the tool name might be lowercase
                    try:
                        tool_name = ToolName[tool_name_str]
                    except KeyError:
                        # Try with capitalized name
                        tool_name_str = action["name"].capitalize()
                        try:
                            tool_name = ToolName[tool_name_str]
                        except KeyError:
                            logger.error(f"Unknown tool name: {action['name']}")
                            self.trace(
                                "user", f"Error: Unknown tool name '{
                                    action['name']}'")
                            self.think()
                            return

                    # Check if the tool is NONE
                    if tool_name == ToolName.NONE:
                        logger.debug(
                            "No action needed. Proceeding to final answer.")
                        self.think()
                        return
                    else:
                        # Get the tool input
                        tool_input = action.get("input", self.query)

                        # Ensure the input is serialized to a JSON string if it's a
                        # dict
                        if isinstance(tool_input, dict):
                            tool_input = json.dumps(tool_input)

                        # 生成优雅的工具调用文本
                        tool_call_text = self._generate_elegant_tool_call_text(tool_name, tool_input)
                        
                        # 将工具调用信息添加到思考内容中
                        formatted_content += f"\n\n{tool_call_text}"

                # 创建 Markdown 对象
                md = Markdown(formatted_content)

                # 创建面板
                panel = Panel(
                    md,
                    title="[bold blue]Thinking[/bold blue]",
                    border_style="blue",
                    padding=(1, 2)
                )

                # 显示面板
                console.print(panel)
                
                # 如果有工具调用，执行工具
                if "action" in parsed_response:
                    action = parsed_response["action"]
                    tool_name_str = action["name"].upper()

                    # Handle the case when the tool name might be lowercase
                    try:
                        tool_name = ToolName[tool_name_str]
                    except KeyError:
                        # Try with capitalized name
                        tool_name_str = action["name"].capitalize()
                        try:
                            tool_name = ToolName[tool_name_str]
                        except KeyError:
                            logger.error(f"Unknown tool name: {action['name']}")
                            self.trace(
                                "user", f"Error: Unknown tool name '{
                                    action['name']}'")
                            self.think()
                            return

                    # Check if the tool is NONE
                    if tool_name == ToolName.NONE:
                        logger.debug(
                            "No action needed. Proceeding to final answer.")
                        self.think()
                    else:
                        # Get the tool input
                        tool_input = action.get("input", self.query)

                        # Ensure the input is serialized to a JSON string if it's a
                        # dict
                        if isinstance(tool_input, dict):
                            tool_input = json.dumps(tool_input)

                                                # Execute the action without displaying intermediate steps
                        self.act(tool_name, tool_input)

            elif "final_answer" in parsed_response:
                # Format and display the final answer with rich formatting
                answer = parsed_response['final_answer']

                # Create a beautiful panel for the answer
                console.print("\n")  # Add some spacing
                console.print(Panel(
                    Markdown(self.safe_markdown(answer)),
                    title="[bold green]Answer[/bold green]",
                    border_style="green",
                    expand=False,
                    padding=(1, 2)
                ))

                # Record the answer in the trace but don't print it again

                self.trace(
                    "assistant",
                    f"Final Answer: {answer}",
                    display=False)

            else:
                # Handle invalid response format
                error_msg = (
                    "Invalid response format: missing 'action' or 'final_answer' field. "
                    "Please provide a valid JSON response like the following:\n\n"
                    f"{self._get_json_format_example()}"
                )
                raise ValueError(error_msg)

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Parse error in ReActAgent.decide: {str(e)}")
            self._show_processing_message(
                "I'm refining my response format and will try again.")
            self.trace("user", f"{str(e)}. Trying again.")
            self.think()

        except Exception as e:
            logger.debug(f"Error in ReActAgent.decide: {str(e)}")
            self._show_processing_message(
                "I encountered a small hiccup but I'm trying again automatically.")
            self.trace("user", f"{str(e)}. Trying again.")
            self.think()

    def act(self, tool_name: ToolName, query: str) -> None:
        """
        Executes the specified tool's function on the query and logs the result.

        Args:
            tool_name (ToolName): The tool to be used.
            query (str): The query for the tool.
        """
        # Get the tool
        tool = self.tools.get(tool_name)

        # Special handling for NONE tool (final answer)
        if tool_name == ToolName.NONE:
            # Record the final answer
            self.trace("assistant", query, display=False)
            return

        if tool:
            # Execute the tool (only show minimal information to the user)
            #console.print(f"\n[bold cyan]Executing: {tool_name}[/bold cyan]")

            # Execute the tool
            result = tool.use(query)

            # Check if this is an abort request from the message tool
            if tool_name == ToolName.MESSAGE and result == "__ABORT_TASK__":
                self.trace("user", "Task aborted by user", display=False)
                return

            # Store tool call in memory system if enabled
            if self.memory_enabled and self.session_id:
                try:
                    # Get the last message ID (should be the assistant's
                    # thinking message)
                    last_message = None
                    conn = self.memory_db.conn
                    cursor = conn.execute('''
                    SELECT id FROM messages
                    WHERE session_id = ? AND role = 'assistant'
                    ORDER BY created_at DESC LIMIT 1
                    ''', (self.session_id,))
                    row = cursor.fetchone()
                    if row:
                        last_message_id = row['id']
                        # Record the tool call
                        self.session_manager.add_tool_call(
                            last_message_id,
                            str(tool_name),
                            query,
                            str(result)
                        )
                except Exception as e:
                    logger.error(f"Error recording tool call in memory: {e}")

            # Truncate long outputs to prevent exceeding context limits
            if isinstance(result, str) and len(result) > 6000:
                truncated_result = self._truncate_long_output(result)
            else:
                truncated_result = result

            # 特殊处理 MESSAGE 工具，将其结果作为用户输入
            if tool_name == ToolName.MESSAGE:
                # 记录用户输入
                logger.debug(f"User input: {truncated_result}")
                self.trace("user", truncated_result, display=False)
            else:
                # 对于其他工具，格式化为系统观察结果
                observation = f"Observation from {
                    tool_name}: {truncated_result}"
                # 记录观察结果但不显示完整详情
                logger.debug(observation)
                self.trace("user", observation, display=False)

            # Continue thinking
            self.think()
        else:
            # Tool not found
            error_message = f"Tool '{tool_name}' not found. Available tools: {
                ', '.join([str(t) for t in self.tools.keys()])}"
            self.trace("user", f"Error: {error_message}")
            self.think()

    def safe_markdown(self, input_data):
        # if input_data is dict, convert to JSON string and wrap in Markdown
        # code block
        if isinstance(input_data, dict):
            input_data = f"```json\n{json.dumps(input_data, indent=2)}\n```"
        elif not isinstance(input_data, str):
            # if input_data is not string, convert to string format (optional:
            # you can also raise an error)
            input_data = str(input_data)
        return input_data

    def _get_json_format_example(self) -> str:
        """
        Returns a formatted example of the expected JSON response format.

        Returns:
            str: A string containing the formatted example with markdown.
        """
        return (
            'Expected JSON format example (when using a tool):\n'
            '```json\n'
            '{\n'
            '  "thought": "Your detailed step-by-step reasoning about what to do next.",\n'
            '  "action": {\n'
            '    "name": "tool_name_from_available_list",\n'
            '    "input": "Specific input for the tool"\n'
            '  }\n'
            '}\n'
            '```\n\n'
            'Or when providing a final answer (use simple string, NOT complex structures):\n'
            '```json\n'
            '{\n'
            '  "thought": "I now know the final answer. Here is my reasoning process...",\n'
            '  "final_answer": "Your answer as a simple string. For complex output, format it as markdown or plain text.\nExample:\n```\nMemory usage: 25%\nCPU usage: 10%\n```"\n'
            '}'
            '```'
        )

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the response from the LLM, supporting both JSON and XML tag formats.

        Args:
            response (str): The response string from the LLM.

        Returns:
            Dict[str, Any]: The parsed response in the standard format.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        # Clean up the response
        cleaned_response = response.strip()
        
        # First try to parse as XML tag format
        xml_result = self._parse_xml_response(cleaned_response)
        if xml_result:
            return xml_result
            
        # If XML parsing fails, try JSON format (backward compatibility)
        return self._parse_json_format(cleaned_response)
    
    def _parse_xml_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parses XML tag format responses.
        
        Args:
            response (str): The response string.
            
        Returns:
            Optional[Dict[str, Any]]: Parsed response or None if not XML format.
        """
        import re
        
        # Check if response contains XML tags
        tool_pattern = r'<tool_(\w+)>\s*({.*?})\s*</tool_\1>'
        final_answer_pattern = r'<final_answer>\s*(.*?)\s*</final_answer>'
        
        # Try to match tool usage
        tool_match = re.search(tool_pattern, response, re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1)
            parameters_str = tool_match.group(2)
            
            try:
                parameters = json.loads(parameters_str)
                return {
                    "thought": parameters.get("thought", ""),
                    "action": {
                        "name": tool_name,
                        "input": parameters.get("input", parameters)
                    }
                }
            except json.JSONDecodeError:
                # If parameters are not valid JSON, treat as simple input
                return {
                    "thought": "",
                    "action": {
                        "name": tool_name,
                        "input": parameters_str.strip()
                    }
                }
        
        # Try to match final answer
        final_answer_match = re.search(final_answer_pattern, response, re.DOTALL)
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            return {
                "thought": "I now know the final answer.",
                "final_answer": answer
            }
        
        return None
    
    def _parse_json_format(self, response: str) -> Dict[str, Any]:
        """
        Parses JSON format responses (original method logic).
        
        Args:
            response (str): The response string.
            
        Returns:
            Dict[str, Any]: The parsed JSON response.
        """
        # 使用基于栈的算法提取最长的有效 JSON 对象
        stack = []
        start_indices = []
        in_string = False
        escape_next = False
        candidates = []

        for i, char in enumerate(response):
            # Handle escape characters
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # 处理字符串
            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            # Handle brackets
            if char == '{':
                stack.append(char)
                if len(stack) == 1:
                    start_indices.append(i)
            elif char == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    start = start_indices.pop()
                    candidates.append(response[start:i + 1])

        # Try to parse each candidate JSON object
        for json_str in candidates:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

        # If no valid JSON object is found, raise an exception
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            error_msg = (
                "Invalid response format. Please provide a valid response in either XML tag format:\n\n"
                "For tool usage:\n"
                "<tool_[tool_name]>\n"
                "{\n"
                '  "thought": "Your reasoning",\n'
                '  "input": "tool input"\n'
                "}\n"
                "</tool_[tool_name]>\n\n"
                "For final answer:\n"
                "<final_answer>\n"
                "Your answer here\n"
                "</final_answer>\n\n"
                "Or in JSON format:\n\n"
                f"{self._get_json_format_example()}"
            )
            raise ValueError(error_msg) from e

    def _truncate_long_output(self, text: str, max_tokens: int = 4000) -> str:
        """
        Intelligently truncate long outputs to prevent exceeding the model's input token limit.

        Strategy:
        1. If the output does not exceed the maximum token count, return it as is
        2. If it exceeds, preserve the beginning, end, and key parts, replacing the middle with a summary

        Args:
            text (str): The text to be processed
            max_tokens (int): Maximum allowed token count, default 4000

        Returns:
            str: The processed text
        """
        # Roughly estimate token count (approximately 4 characters per token
        # for English)
        estimated_tokens = len(text) / 4

        if estimated_tokens <= max_tokens:
            return text

        # Calculate the length to preserve for head and tail (30% each)
        head_size = int(max_tokens * 0.3) * 4
        tail_size = int(max_tokens * 0.3) * 4

        # Extract head and tail
        head = text[:head_size]
        tail = text[-tail_size:]

        # Create summary information
        omitted_length = len(text) - head_size - tail_size
        omitted_tokens = int(omitted_length / 4)

        # Check if it contains tables or structured data
        has_table = '|' in text and '-+-' in text
        has_json = '{' in text and '}' in text
        has_xml = '<' in text and '>' in text

        summary = f"\n\n[...Omitted approximately {omitted_tokens} tokens..."

        if has_table:
            summary += ", containing table data"
        if has_json:
            summary += ", containing JSON data"
        if has_xml:
            summary += ", containing XML/HTML data"

        summary += "]\n\n"

        # Combine the final result
        return head + summary + tail

    def _load_template(self, template_path: str) -> str:
        """
        Loads the prompt template from a file.

        Args:
            template_path (str): Path to the template file.

        Returns:
            str: The content of the prompt template file.
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Template file not found at {
                           template_path}. Creating default template.")
            self._create_default_template(template_path)
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _create_default_template(self, template_path: str) -> None:
        """
        Creates a default template file if one doesn't exist.

        Args:
            template_path (str): Path where the template should be created.
        """
        logger.info("Creating default template")

        # Default template content
        default_template = """You are a ReAct (Reasoning and Acting) agent tasked with answering the following query:

Query: {query}

Current system information:
- OS: {os}
- Distribution: {distribution}
- Version: {version}
- Architecture: {architecture}

Previous reasoning steps and observations:
{history}

Available tools: {tools}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the available tools
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When responding, output a JSON object with the following structure:

If you need to use a tool:
{{
    "thought": "Your detailed step-by-step reasoning about what to do next",
    "action": {{
        "name": "tool_name_from_available_list",
        "input": "Specific input for the tool"
    }}
}}

If you have enough information to answer the query:
{{
    "thought": "I now know the final answer. Here is my reasoning process...",
    "final_answer": "Your comprehensive answer to the original query"
}}

Tool Usage Guidelines:
1. 'shell' tool: Use for simple commands that can be executed in a single step.
2. 'script' tool: Preferred for complex tasks requiring multiple steps or logic. Use for:
   - Data processing and transformation
   - System maintenance and automation
   - Problem diagnosis and troubleshooting
   - Tasks that might be repeated in the future

The 'script' tool accepts:
- "action": "create", "execute", or "create_and_execute"
- "filename": Script filename
- "content": Script content
- "interpreter": Optional (e.g., "python3", "bash", "node")
- "args": Optional arguments list
- "env_vars": Optional environment variables
- "timeout": Optional execution time limit in seconds

Error Handling Strategy:
- Analyze error messages carefully
- Identify root causes (dependencies, permissions, syntax)
- Fix specific errors and retry
- Try alternative approaches if multiple attempts fail

Remember:
- Think step-by-step and be thorough in your reasoning
- Use tools strategically to gather necessary information
- Base your reasoning on actual observations
- Provide a final answer only when confident
"""

        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(default_template)

        logger.info(f"Created default template at {template_path}")

    def execute(self, query: str) -> str:
        """
        Executes the agent's query-processing workflow.

        Args:
            query (str): The query to be processed.

        Returns:
            str: The final answer or last recorded message content.
        """
        # Reset the agent state
        self.query = query
        self.messages = []
        self.current_iteration = 0
        reset_stop_flag()

        # Initialize base prompt message list for reuse in each reasoning step
        self.base_prompt_messages = []

        # Ensure we always use the latest system information, especially the current working directory
        # If in remote execution mode, get the current working directory from
        # the remote system
        if hasattr(forwarder, 'remote_enabled') and forwarder.remote_enabled:
            try:
                exit_code, stdout, stderr = forwarder.forward_command("pwd")
                if exit_code == 0:
                    self.system_info["current_working_directory"] = stdout.strip()
                    logger.info(
                        f"Updated remote working directory: {
                            self.system_info['current_working_directory']}")
                else:
                    logger.warning(
                        f"Failed to get remote working directory: {stderr}")
            except Exception as e:
                logger.error(f"Error getting remote working directory: {e}")
        else:
            # Get local working directory
            self.system_info["current_working_directory"] = os.getcwd()
            logger.info(
                f"Updated local working directory: {
                    self.system_info['current_working_directory']}")

        # Create the system prompt first
        self.system_prompt = self.template.format(
            query=self.query,
            tools=', '.join(
                [f"{tool.name}: {tool.description}" for tool in self.tools.values()]),
            **self.system_info
        )
        system_message = {"role": "system", "content": self.system_prompt}

        # Initialize or get session if memory is enabled
        if self.memory_enabled:
            try:
                self.session_id = self.session_manager.get_or_create_session(
                    self.user_id)
                logger.info(
                    f"Using session {
                        self.session_id} for user {
                        self.user_id}")

                # Calculate system prompt tokens to adjust available tokens for
                # memory
                try:
                    # Get the model's context window size
                    model_context_size = self._get_model_context_size(
                        self.model)

                    # Calculate system prompt tokens
                    system_tokens = self.session_manager.context_manager.get_token_count(
                        [system_message], self.model)
                    logger.debug(f"System prompt token count: {system_tokens}")

                    # Calculate available tokens for memory messages
                    available_tokens = model_context_size - system_tokens
                    logger.debug(
                        f"Available tokens for memory: {available_tokens}")

                    # Get messages from memory for LLM context with token limit
                    memory_messages = self.session_manager.get_messages_for_llm(
                        self.user_id,
                        model=self.model,
                        available_tokens=available_tokens
                    )

                    if memory_messages:
                        logger.debug(
                            f"Loaded {
                                len(memory_messages)} messages from memory")
                        # Add memory messages to the base prompt message list
                        self.base_prompt_messages.extend(memory_messages)
                except Exception as e:
                    logger.error(f"Error loading memory messages: {e}")
                    # Fallback to loading messages without token calculation
                    try:
                        memory_messages = self.session_manager.get_messages_for_llm(
                            self.user_id, model=self.model)
                        if memory_messages:
                            self.base_prompt_messages.extend(memory_messages)
                    except Exception as inner_e:
                        logger.error(
                            f"Error in fallback memory loading: {inner_e}")
            except Exception as e:
                logger.error(f"Error initializing memory session: {e}")
                self.session_id = None

        # Add system prompt to the base prompt message list
        self.base_prompt_messages.insert(0, system_message)

        # Record the user query
        self.trace(role="user", content=query)

        self.think()

        # Return the last message content (should be the final answer)
        if self.messages:
            return self.messages[-1].content
        else:
            return "No response generated."

    def _get_model_context_size(self, model: str) -> int:
        """
        Get the context window size for a specific model

        Args:
            model: Model name

        Returns:
            Context window size in tokens
        """
        if 'sonnet' in model.lower():
            max_tokens = 200 * 1000 - 64000
        elif 'gpt' in model.lower():
            max_tokens = 128 * 1000 - 28000
        elif 'gemini' in model.lower():
            max_tokens = 1000 * 1000 - 300000
        elif 'deepseek' in model.lower():
            max_tokens = 128 * 1000 - 28000
        elif 'kimi' in model.lower():
            max_tokens = 128 * 1000 - 28000
        else:
            max_tokens = 41 * 1000 - 10000
        return max_tokens

    def _convert_history_to_messages(self) -> List[Dict[str, str]]:
        """
        Converts the internal message history to a list of message dictionaries for LLM API.

        Returns:
            List[Dict[str, str]]: List of message dictionaries with 'role' and 'content' keys.
        """
        messages = []
        for message in self.messages:
            role = "assistant" if message.role == "assistant" else "user"
            messages.append({"role": role, "content": message.content})
        return messages


def shell_command_tool(command) -> str:
    """
    Executes a shell command and returns the output.

    Args:
        command: The shell command to execute. Can be a dict or a JSON string with 'command' key and optional 'background' key.
            - 'command': The shell command string to execute (required)
            - 'background': Boolean indicating whether to run the command in the background (optional, default: False)

    Returns:
        str: The output of the command.
    """
    try:
        # Parse command if it's a JSON string
        if isinstance(command, str):
            try:
                command = json.loads(command)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format. Expected a JSON string representing a dictionary with 'command' key."

        # Ensure the command is in dictionary format
        if not isinstance(command, dict):
            return "Error: Invalid command format. Expected a dictionary with 'command' key. Example: {'command': 'ls -la', 'background': false}"

        # Check if the dictionary contains the required 'command' key
        if 'command' not in command:
            return "Error: Missing 'command' key in the input dictionary. Example: {'command': 'ls -la', 'background': false}"

        # Extract the command string and background execution flag
        cmd_str = command['command']
        background = command.get('background', False)

        # If background execution mode is enabled, modify the command format
        if background:
            # On Linux/macOS, use nohup and & to run commands in the background
            # Redirect standard output and error output to /dev/null or a
            # specified log file
            cmd_str = f"nohup {cmd_str} > /dev/null 2>&1 &"
            console.print(f"[bold cyan]Running command in background: {
                          cmd_str}[/bold cyan]")

        # Execute the command
        return_code, output, _ = execute_command(cmd_str)

        # Return the command output
        if background:
            return f"Command started in background. Return Code: {return_code}"
        else:
            return f"Return Code: {return_code}\nOutput: \n{output}"
    except Exception as e:
        return f"Error executing command: {str(e)}"


def message_tool(query: str) -> str:
    """
    A tool to ask the user questions and wait for responses.

    This tool allows the Agent to ask questions and get answers from users, suitable for:
    - Clarifying ambiguous requirements
    - Confirming before important operations
    - Gathering additional information to complete tasks
    - Offering options and requesting user preferences
    - Validating assumptions critical to task success

    Args:
        query (str): The question to ask the user, can be a simple string or JSON format.
                    If JSON format, it should contain a "question" field.
                    Example: {"question": "Which configuration option would you prefer?"}

    Returns:
        str: The user's answer.
    """
    try:
        # Handle dictionary type input
        if isinstance(query, dict):
            if "question" in query:
                question = query["question"]
            else:
                # If there is no question field in the dictionary, try to
                # convert the entire dictionary to a string
                question = str(query)
        else:
            # Handle string type input
            question = query

            # Try to parse JSON format (if it's a JSON string)
            if isinstance(query, str) and query.startswith(
                    '{') and query.endswith('}'):
                try:
                    query_data = json.loads(query)
                    if "question" in query_data:
                        question = query_data["question"]
                except json.JSONDecodeError:
                    # If parsing fails, still use the original query
                    pass

        # Display the question to the user
        console.print()
        console.print(Panel(
            Markdown(question),
            title="[bold cyan]Agent Question[/bold cyan]",
            border_style="cyan"
        ))

        # Get user input
        console.print("[bold cyan]Please enter your answer:[/bold cyan]")
        console.print("[dim](Type 'quit' or 'stop' to exit)[/dim]")
        user_response = input("> ")

        # Remove leading whitespace from user input
        user_response = user_response.lstrip()

        # Handle special keywords
        if user_response.lower() in ["quit", "exit", "stop"]:
            console.print(
                "[bold yellow]Aborting current task as requested by user[/bold yellow]")
            return "__ABORT_TASK__"

        # Return the user's answer
        return user_response
    except Exception as e:
        logger.error(f"Error in message_tool: {e}")
        return f"Error asking user: {str(e)}"


def create_react_agent(llm_client: LLMClient,
                       system_info: Dict[str, Any],
                       command_analyzer: Optional[CommandAnalyzer] = None,
                       memory_enabled: bool = False,
                       memory_db: Optional[Any] = None,
                       user_id: str = "default_user") -> ReActAgent:
    """
    Creates and configures a ReAct agent with default tools.

    Args:
        llm_client (LLMClient): The LLM client for API interactions.
        system_info (Dict[str, Any]): Dictionary containing system information.
        command_analyzer (CommandAnalyzer, optional): Analyzer for command safety.
        memory_enabled (bool, optional): Whether to enable memory system.
        memory_db (Optional[Any], optional): Memory database instance.
        user_id (str, optional): User ID for memory system.

    Returns:
        ReActAgent: A configured ReAct agent.
    """
    # Create the agent
    agent = ReActAgent(
        llm_client,
        system_info,
        command_analyzer,
        memory_enabled=memory_enabled,
        memory_db=memory_db,
        user_id=user_id)

    # Register the shell command tool
    agent.register_tool(
        ToolName.SHELL,
        shell_command_tool,
        "Execute shell commands to interact with the system. Use this for running terminal commands, file operations, system monitoring, software installation, and other shell-based tasks."
    )

    # Register the script tool
    agent.register_tool(
        ToolName.SCRIPT,
        script_tool,
        "Create and execute scripts in various languages. Send a JSON request with 'action' (create/execute/create_and_execute), 'filename', 'content' (for creation), 'interpreter' (optional, e.g., python3/bash/node), 'args' (optional, list of arguments to pass to the script), 'env_vars' (optional, dictionary of environment variables to set), and 'timeout' (optional, maximum execution time in seconds)."
    )

    # Register the message tool
    agent.register_tool(
        ToolName.MESSAGE,
        message_tool,
        "Ask the user a question and wait for a response."
    )

    # Register the files tool
    agent.register_tool(
        ToolName.FILES,
        files_tool,
        "Perform file operations including creating, reading, updating, and deleting files, as well as listing directory contents. Send a JSON request with 'operation' (create_file, read_file, update_file, delete_file, list_directory, file_exists) and operation-specific parameters. All paths are relative to the current working directory unless absolute paths are provided."
    )

    # Register the web page tool
    agent.register_tool(
        ToolName.WEB_PAGE,
        web_page_tool,
        "Crawl a web page and extract its content in a readable format. Used to retrieve information from web pages to assist in completing tasks. "
    )

    # Register new code analysis tools
    agent.register_tool(
        ToolName.GET_ALL_REFERENCES,
        get_all_references_tool,
        "Find all references to a symbol in code. Send a JSON request with 'word' (symbol to find references for), 'relative_path' (path to the file containing the symbol), 'line' (optional, line number), 'verbose' (optional, include detailed information), 'num_results' (optional, maximum number of results), and 'context_limit' (optional, number of context lines)."
    )

    agent.register_tool(
        ToolName.GET_FOLDER_STRUCTURE,
        get_folder_structure_tool,
        "Get the folder structure of a repository. Send a JSON request with 'repo_dir' (repository directory), 'max_depth' (optional, maximum depth to traverse), 'exclude_dirs' (optional, directories to exclude), 'exclude_files' (optional, file patterns to exclude), and 'pattern' (optional, file name pattern to match)."
    )

    agent.register_tool(
        ToolName.GOTO_DEFINITION,
        goto_definition_tool,
        "Find the definition of a symbol in code. Send a JSON request with 'word' (symbol to find definition for), 'line' (line number), 'relative_path' (path to the file containing the symbol), and 'verbose' (optional, include detailed information)."
    )

    agent.register_tool(
        ToolName.ZOEKT_SEARCH,
        zoekt_search_tool,
        "Perform powerful code search using Zoekt. Send a JSON request with 'names' (list of identifiers to search for), 'repo_dir' (optional, repository directory), 'language' (optional, programming language), 'num_results' (optional, maximum number of results), 'verbose' (optional, include detailed information), 'no_color' (optional, disable colored output), and 'use_cache' (optional, use cached results)."
    )

    agent.register_tool(
        ToolName.GET_SYMBOLS,
        get_symbols_tool,
        "Extract symbols from a file. Send a JSON request with 'file_path' (file to extract symbols from), 'repo_dir' (optional, repository directory), 'language' (optional, programming language), and 'keyword' (optional, filter symbols by keyword)."
    )

    # Register the web search tool
    agent.register_tool(
        ToolName.WEB_SEARCH,
        web_search_tool,
        "Perform a web search using DuckDuckGo. Send a JSON request with 'query' (search terms) and 'max_results' (optional, maximum number of results to return)."
    )

    # Register the code edit tool
    agent.register_tool(
        ToolName.CODE_EDIT,
        code_edit_tool,
        "Edit code files with proper syntax checking and formatting. Send a JSON request with 'file_path' (path to the file to edit), 'start_line' (starting line number), 'end_line' (ending line number), 'new_content' (replacement code), 'language' (optional, auto-detected from file extension), 'description' (optional), and 'check_syntax' (optional, whether to check syntax after edit)."
    )

    # Register the expand message tool if memory is enabled
    if memory_enabled and agent.session_manager:
        try:
            from terminal_agent.react.tools.expand_message_tool import init_expand_message_tool

            expand_message = init_expand_message_tool(agent.session_manager)
            agent.register_tool(
                ToolName.EXPAND_MESSAGE,
                expand_message,
                "Expand a message that was truncated due to length. Use this tool when you see a message that ends with '... (truncated)' and you need to see the full content. Provide the message_id from the truncated message."
            )
            logger.info("Expand message tool registered successfully")
        except Exception as e:
            logger.error(f"Failed to register expand message tool: {e}")

    # Return the configured agent
    return agent
