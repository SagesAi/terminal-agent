#!/usr/bin/env python3
"""
ReAct Agent implementation for Terminal Agent

This module implements a ReAct (Reasoning + Acting) agent that follows the 
reasoning-action-observation loop to solve tasks using available tools.
"""

import json
import logging
import os
from enum import Enum, auto
from typing import Dict, List, Callable, Union, Optional, Any, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pydantic import BaseModel, Field

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_executor import execute_command, should_stop_operations, reset_stop_flag
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.utils.logging_config import get_logger

# Initialize Rich console
console = Console()

# 获取日志记录器
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
                 template_path: str = PROMPT_TEMPLATE_PATH):
        """
        Initializes the ReAct Agent with necessary components.

        Args:
            llm_client (LLMClient): The LLM client for API interactions.
            system_info (Dict[str, Any]): Dictionary containing system information.
            command_analyzer (CommandAnalyzer, optional): Analyzer for command safety.
            max_iterations (int, optional): Maximum number of reasoning iterations.
            template_path (str, optional): Path to the prompt template file.
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
        
        # Create templates directory if it doesn't exist
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        
        # Create default template if it doesn't exist
        if not os.path.exists(template_path):
            self._create_default_template(template_path)

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

    def register_tool(self, name: ToolName, func: Callable[[str], str], description: str = "") -> None:
        """
        Registers a tool to the agent.

        Args:
            name (ToolName): The name of the tool.
            func (Callable[[str], str]): The function associated with the tool.
            description (str): Description of what the tool does.
        """
        self.tools[name] = Tool(name, func, description)

    def trace(self, role: str, content: str, display: bool = False) -> None:
        """
        Logs the message with the specified role and content.

        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
            display (bool): Whether to display the message. Defaults to True.
        """
        logger.info(f"{role}: {content}")
        self.messages.append(Message(role=role, content=content))
        if display:
            console.print(f"[bold cyan]{role}[/bold cyan]: {content}")

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
            self.trace("system", "Operations stopped by user.")
            return
        
        # Format the prompt with the current context
        
        prompt = self.template.format(
            query=self.query,
            history=self.get_history(),
            tools=', '.join([f"{tool.name}: {tool.description}" for tool in self.tools.values()]),
            **self.system_info
        )
        
        try:
            # Get the LLM's response using the simpler method
            response = self.llm_client.call_llm_with_prompt(prompt)
            logger.debug(f"Thinking => {response}")  # 降级为debug级别，不在控制台显示
            
            # Record the thinking without displaying it
            self.trace("assistant", f"Thought: {response}", display=False)
            
            # Decide on the next action based on the response
            self.decide(response)
        except ConnectionError as e:
            # 处理连接错误，直接退出 React loop
            logger.error(f"Connection error in ReActAgent.think: {str(e)}")
            self.trace("system", f"Error: Connection to LLM API failed. Please check your internet connection and API settings. Details: {str(e)}", display=True)
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
                        self.trace("system", f"Error: Unknown tool name '{action['name']}'")
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
            self.trace("system", "Error parsing response. Trying again.")
            self.think()
        
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            self.trace("system", f"Error: {str(e)}. Trying again.")
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
        
        # Try to find JSON content in the response
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_content = cleaned_response[json_start:json_end]
            return json.loads(json_content)
        else:
            # If no JSON is found, try to parse the whole response
            return json.loads(cleaned_response)

    def act(self, tool_name: ToolName, query: str) -> None:
        """
        Executes the specified tool's function on the query and logs the result.

        Args:
            tool_name (ToolName): The tool to be used.
            query (str): The query for the tool.
        """
        tool = self.tools.get(tool_name)
        
        if tool:
            # Execute the tool (only show minimal information to the user)
            console.print(f"\n[bold cyan]Executing: {tool_name}[/bold cyan]")
            
            # Execute the tool
            result = tool.use(query)
            
            # Format the observation
            observation = f"Observation from {tool_name}: {result}"
            
            # Record the observation but don't display the full details
            logger.debug(observation)
            self.trace("system", observation, display=False)
            
            # Continue the reasoning process
            self.think()
        
        else:
            logger.error(f"No tool registered for: {tool_name}")
            self.trace("system", f"Error: Tool {tool_name} not found", display=True)
            self.think()

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
        
        # Record the user query
        self.trace(role="user", content=query)
        
        # Start the reasoning process
        console.print("\n[bold blue]Starting reasoning process...[/bold blue]")
        self.think()
        
        # Return the last message content (should be the final answer)
        if self.messages:
            return self.messages[-1].content
        else:
            return "No response generated."


def shell_command_tool(command: str) -> str:
    """
    Executes a shell command and returns its output.
    
    Args:
        command (str): The shell command to execute.
        
    Returns:
        str: The command output.
    """
    return_code, output, _ = execute_command(command)
    return f"Command: {command}\nReturn Code: {return_code}\nOutput:\n{output}"


def script_tool(script_request: str) -> str:
    """
    Creates and executes a script based on the provided request.
    
    The request should be in JSON format with the following fields:
    - action: "create", "execute", or "create_and_execute"
    - filename: The name of the script file to create or execute
    - content: The content of the script (only for creation)
    - interpreter: The interpreter to use (e.g., "python", "bash", "node")
    - args: List of arguments to pass to the script (optional)
    - env_vars: Dictionary of environment variables to set (optional)
    - timeout: Maximum execution time in seconds (optional)
    
    Args:
        script_request (str): JSON string containing the script request.
        
    Returns:
        str: The result of the script operation.
    """
    try:
        # Parse the script request
        request = json.loads(script_request)
        
        # Extract fields
        action = request.get("action", "").lower()
        filename = request.get("filename", "")
        content = request.get("content", "")
        interpreter = request.get("interpreter", "")
        args = request.get("args", [])  # Extract arguments list
        env_vars = request.get("env_vars", {})  # Extract environment variables
        timeout = request.get("timeout", None)  # Extract timeout setting
        
        # Validate required fields
        if not filename:
            return "Error: Filename is required."
            
        # Create the script
        if action in ["create", "create_and_execute"]:
            if not content:
                return "Error: Script content is required for creation."
                
            # Ensure the script has the correct permissions
            try:
                with open(filename, "w") as f:
                    f.write(content)
                
                # Make the script executable
                os.chmod(filename, 0o755)
                
                console.print(f"[green]Script created: {filename}[/green]")
            except Exception as e:
                return f"Error creating script: {str(e)}"
        
        # Execute the script
        if action in ["execute", "create_and_execute"]:
            if not os.path.exists(filename):
                return f"Error: Script file {filename} does not exist."
                
            # Determine the command to run the script
            if interpreter:
                command = f"{interpreter} {filename}"
            else:
                # Use the shebang line or execute directly if executable
                command = f"./{filename}"
            
            # Add arguments to the command if provided
            if args:
                # Convert all arguments to strings and join with spaces
                args_str = " ".join([str(arg) for arg in args])
                command = f"{command} {args_str}"
            
            # Execute the script with environment variables and timeout if provided
            return_code, output, _ = execute_command(
                command, 
                env=env_vars, 
                timeout=timeout
            )
            
            return f"Script: {filename}\nCommand: {command}\nReturn Code: {return_code}\nOutput:\n{output}"
            
        return f"Script {action} completed for {filename}"
        
    except json.JSONDecodeError:
        return "Error: Invalid JSON format in script request."
    except Exception as e:
        return f"Error processing script request: {str(e)}"


def create_react_agent(llm_client: LLMClient, 
                      system_info: Dict[str, Any],
                      command_analyzer: Optional[CommandAnalyzer] = None) -> ReActAgent:
    """
    Creates and configures a ReAct agent with default tools.
    
    Args:
        llm_client (LLMClient): The LLM client for API interactions.
        system_info (Dict[str, Any]): Dictionary containing system information.
        command_analyzer (CommandAnalyzer, optional): Analyzer for command safety.
        
    Returns:
        ReActAgent: A configured ReAct agent.
    """
    # Create the agent
    agent = ReActAgent(llm_client, system_info, command_analyzer)
    
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
        "Create and execute scripts in various languages. Send a JSON request with 'action' (create/execute/create_and_execute), 'filename', 'content' (for creation), 'interpreter' (optional, e.g., python/bash/node), 'args' (optional, list of arguments to pass to the script), 'env_vars' (optional, dictionary of environment variables to set), and 'timeout' (optional, maximum execution time in seconds)."
    )
    
    # Return the configured agent
    return agent
