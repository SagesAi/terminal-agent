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
from terminal_agent.utils.command_executor import execute_command, should_stop_operations, reset_stop_flag
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.utils.logging_config import get_logger
from terminal_agent.react.tools.files_tool import files_tool
from terminal_agent.react.tools.script_tool import script_tool
from terminal_agent.react.tools.web_page import web_page_tool
from terminal_agent.react.tools.get_all_references_tool import get_all_references_tool
from terminal_agent.react.tools.get_folder_structure_tool import get_folder_structure_tool
from terminal_agent.react.tools.goto_definition_tool import goto_definition_tool
from terminal_agent.react.tools.zoekt_search_tool import zoekt_search_tool
from terminal_agent.react.tools.get_symbols_tool import get_symbols_tool

# Initialize Rich console
console = Console()

# Get logger
logger = get_logger(__name__)

# Type alias for observations
Observation = Union[str, Exception]

# Default template path in the package
PROMPT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "react_prompt1.txt")
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
    WEB_PAGE = auto() # Web page tool for retrieving web content
    GET_ALL_REFERENCES = auto() # Get all references tool for finding all references to a symbol
    GET_FOLDER_STRUCTURE = auto() # Get folder structure tool for visualizing directory structure
    GOTO_DEFINITION = auto() # Go to definition tool for finding symbol definitions
    ZOEKT_SEARCH = auto() # Zoekt search tool for powerful code search
    GET_SYMBOLS = auto() # Get symbols tool for extracting symbols from files
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

    def __init__(self, name: ToolName, func: Callable[[str], str], description: str = ""):
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
                    
                self.context_manager = ContextManager(self.memory_db, llm_client)
                self.session_manager = SessionManager(self.memory_db, self.context_manager)
                logger.info("Memory system initialized")
            except ImportError as e:
                logger.warning(f"Failed to import memory modules: {e}. Memory system disabled.")
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

    def register_tool(self, name: ToolName, func: Callable[[str], str], description: str = "") -> None:
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
        if self.memory_enabled and self.session_id and role in ["user", "assistant", "system"]:
            try:
                message_type = "thinking" if "Thought:" in content and role == "assistant" else "message"
                self.session_manager.add_message(self.session_id, role, content, message_type)
            except Exception as e:
                logger.error(f"Error storing message in memory: {e}")

        if display:
            if role == "user":
                console.print(f"\n[bold green]User:[/bold green] {content}")
            elif role == "assistant":
                console.print(f"\n[bold blue]Assistant:[/bold blue] {content}")
            elif role == "system":
                console.print(f"\n[bold yellow]System:[/bold yellow] {content}")
            else:
                console.print(f"\n[bold]{role}:[/bold] {content}")

    def get_history(self) -> str:
        """
        Retrieves the conversation history.

        Returns:
            str: Formatted history of messages.
        """
        return "\n".join([f"{message.role}: {message.content}" for message in self.messages])

    def think(self) -> None:
        """
        Processes the current query, decides actions, and iterates until a solution or max iteration limit is reached.
        """
        self.current_iteration += 1
        logger.info(f"Starting iteration {self.current_iteration}")

        # Check if we've reached the maximum number of iterations
        if self.current_iteration > self.max_iterations:
            logger.warning("Reached maximum iterations. Stopping.")
            self.trace("assistant", "I'm sorry, but I couldn't find a satisfactory answer within the allowed number of iterations. Here's what I know so far: " + self.get_history())
            return

        # Check if operations should be stopped
        if should_stop_operations():
            logger.warning("Operations stopped by user.")
            self.trace("user", "Operations stopped by user.")
            return

        try:
            # 复制基本提示消息列表，避免修改原始列表
            prompt_messages = list(self.base_prompt_messages)
            
            # 添加当前任务的本地消息历史
            prompt_messages.extend(self._convert_history_to_messages())
            
            logger.debug(f"Using {len(self.base_prompt_messages)} base messages and {len(self.messages)} local messages")
            # Get the LLM's response using the message-based method
            response = self.llm_client.call_with_messages(prompt_messages)
            logger.debug(f"Thinking => {response}")

            # 记录思考过程但不显示，显示逻辑移到 decide 方法中
            self.trace("assistant", f"Thought: {response}", display=False)

            # Decide on the next action based on the response
            self.decide(response)
        except ConnectionError as e:
            # 处理连接错误，直接退出 React loop
            logger.error(f"Connection error in ReActAgent.think: {str(e)}")
            self.trace("user", f"Error: Connection to LLM API failed. Please check your internet connection and API settings. Details: {str(e)}", display=True)
            # 不再调用 self.think()，直接退出循环
            console.print("[bold red]Exiting ReAct loop due to connection error.[/bold red]")
            return

    def decide(self, response: str) -> None:
        """
        Processes the agent's response, deciding actions or final answers.

        Args:
            response (str): The response generated by the model.
        """
        try:
            # Try to parse the response as JSON
            parsed_response = self._parse_json_response(response)

            # 显示思考过程，实现信息透明
            if "thought" in parsed_response and "final_answer" not in parsed_response:
                thought = parsed_response["thought"]

                # 直接使用原始思考内容，不进行额外处理
                formatted_thought = thought

                # 创建 Markdown 对象
                md = Markdown(formatted_thought)

                # 创建面板
                panel = Panel(
                    md,
                    title="[bold blue]Thinking[/bold blue]",
                    border_style="blue",
                    padding=(1, 2)
                )
                
                # 显示面板
                console.print(panel)

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
                        self.trace("user", f"Error: Unknown tool name '{action['name']}'")
                        self.think()
                        return

                # Check if the tool is NONE
                if tool_name == ToolName.NONE:
                    logger.debug("No action needed. Proceeding to final answer.")
                    self.think()
                else:
                    # Execute the action without displaying intermediate steps
                    self.act(tool_name, action.get("input", self.query))

            elif "final_answer" in parsed_response:
                # Format and display the final answer with rich formatting
                answer = parsed_response['final_answer']

                # Create a beautiful panel for the answer
                console.print("\n")  # Add some spacing
                console.print(Panel(
                    Markdown(answer),
                    title="[bold green]Answer[/bold green]",
                    border_style="green",
                    expand=False,
                    padding=(1, 2)
                ))

                # Record the answer in the trace but don't print it again
                
                self.trace("assistant", f"Final Answer: {answer}", display=False)

            else:
                # Handle invalid response format
                raise ValueError("Invalid response format: missing 'action' or 'final_answer' field")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            self.trace("user", f"Error parsing response: {str(e)}. Trying again.")
            self.think()

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            self.trace("user", f"Error: {str(e)}. Trying again.")
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
                    # Get the last message ID (should be the assistant's thinking message)
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
                observation = f"Observation from {tool_name}: {truncated_result}"
                # 记录观察结果但不显示完整详情
                logger.debug(observation)
                self.trace("user", observation, display=False)

            # Continue thinking
            self.think()
        else:
            # Tool not found
            error_message = f"Tool '{tool_name}' not found. Available tools: {', '.join([str(t) for t in self.tools.keys()])}"
            self.trace("user", f"Error: {error_message}")
            self.think()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the JSON response from the LLM.

        Args:
            response (str): The response string from the LLM.

        Returns:
            Dict[str, Any]: The parsed JSON response.

        Raises:
            json.JSONDecodeError: If the response cannot be parsed as JSON.
        """
        # Clean up the response to extract JSON
        cleaned_response = response.strip()
        
        # 使用基于栈的算法提取最长的有效 JSON 对象
        stack = []
        start_indices = []
        in_string = False
        escape_next = False
        candidates = []
        
        for i, char in enumerate(cleaned_response):
            # 处理转义字符
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
            
            # 处理括号
            if char == '{':
                stack.append(char)
                if len(stack) == 1:
                    start_indices.append(i)
            elif char == '}' and stack and stack[-1] == '{':
                stack.pop()
                if not stack:
                    start = start_indices.pop()
                    candidates.append(cleaned_response[start:i+1])
        
        # 按长度降序排列，优先尝试更长的 JSON 对象
        #candidates.sort(key=len, reverse=True)
        
        # 尝试解析每个候选 JSON 对象
        for json_str in candidates:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
        
        # 如果没有找到有效的 JSON 对象，抛出异常
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid response format: unable to parse as JSON. No valid JSON objects found in the response.") from e

        
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
            # Roughly estimate token count (approximately 4 characters per token for English)
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
            logger.warning(f"Template file not found at {template_path}. Creating default template.")
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
        "name": "Tool name from the available tools list",
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
        
        # Initialize or get session if memory is enabled
        if self.memory_enabled:
            try:
                self.session_id = self.session_manager.get_or_create_session(self.user_id)
                logger.info(f"Using session {self.session_id} for user {self.user_id}")
                
                # Load memory messages once
                try:
                    # Get messages from memory for LLM context
                    # (check_and_summarize_if_needed has already been called internally)
                    memory_messages = self.session_manager.get_messages_for_llm(self.user_id)
                    if memory_messages:
                        logger.debug(f"Loaded {len(memory_messages)} messages from memory")
                        # 将内存消息添加到基本提示消息列表中
                        self.base_prompt_messages.extend(memory_messages)
                except Exception as e:
                    logger.error(f"Error loading memory messages: {e}")
            except Exception as e:
                logger.error(f"Error initializing memory session: {e}")
                self.session_id = None
        
        # Create the system prompt once
        self.system_prompt = self.template.format(
            query=self.query,
            tools=', '.join([f"{tool.name}: {tool.description}" for tool in self.tools.values()]),
            **self.system_info
        )
        
        # Add system prompt to the base prompt message list
        self.base_prompt_messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Record the user query
        self.trace(role="user", content=query)

        self.think()

        # Return the last message content (should be the final answer)
        if self.messages:
            return self.messages[-1].content
        else:
            return "No response generated."

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

def shell_command_tool(command: str) -> str:
    """
    Executes a shell command and returns the output.

    Args:
        command (str): The shell command to execute.

    Returns:
        str: The output of the command.
    """
    try:
        # Execute the command
        return_code, output, _ = execute_command(command)
        
        # Return the command output
        return f"Return Code: {return_code}\nOutput:\n{output}"
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
        # 处理字典类型的输入
        if isinstance(query, dict):
            if "question" in query:
                question = query["question"]
            else:
                # 如果字典中没有 question 字段，尝试将整个字典转换为字符串
                question = str(query)
        else:
            # 处理字符串类型的输入
            question = query
            
            # 尝试解析 JSON 格式（如果是 JSON 字符串）
            if isinstance(query, str) and query.startswith('{') and query.endswith('}'):
                try:
                    query_data = json.loads(query)
                    if "question" in query_data:
                        question = query_data["question"]
                except json.JSONDecodeError:
                    # 如果解析失败，仍使用原始查询
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
            console.print("[bold yellow]Aborting current task as requested by user[/bold yellow]")
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
    agent = ReActAgent(llm_client, system_info, command_analyzer, memory_enabled=memory_enabled, memory_db=memory_db, user_id=user_id)

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

    # Return the configured agent
    return agent
