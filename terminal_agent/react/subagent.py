#!/usr/bin/env python3
"""
Subagent implementation for Terminal Agent

This module implements subagent functionality that allows creating specialized agents
with custom system prompts, user prompts, tool isolation, and history tracking.

Optimized with insights from Claude Code architecture analysis:
- Layered multi-agent architecture
- Intelligent concurrency control
- Advanced permission system
- Resource monitoring and limits
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Union, AsyncGenerator
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

from terminal_agent.react.function_call_agent import FunctionCallAgent, create_function_call_agent
from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.react.function_call_tools import openai_tool_registry

# Get logger
logger = logging.getLogger(__name__)


class AgentLifecycleState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = 'initializing'
    RUNNING = 'running'
    WAITING = 'waiting'
    COMPLETED = 'completed'
    FAILED = 'failed'
    ABORTED = 'aborted'


class SubagentExecutionMode(Enum):
    """Execution modes for subagents"""
    SINGLE = 'single'
    CONCURRENT = 'concurrent'
    SYNTHESIS = 'synthesis'


@dataclass
class ResourceLimits:
    """Resource limits for subagent execution"""
    max_execution_time_ms: int = 300000  # 5 minutes
    max_tokens: int = 100000
    max_tool_calls: int = 50
    max_file_operations: int = 100
    max_concurrent_tools: int = 10


@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    start_time: float = field(default_factory=time.time)
    token_count: int = 0
    tool_call_count: int = 0
    file_operations: int = 0
    network_requests: int = 0

    def check_limits(self, limits: ResourceLimits) -> List[str]:
        """Check if resource limits are exceeded"""
        violations = []
        elapsed_time = (time.time() - self.start_time) * 1000
        
        if elapsed_time > limits.max_execution_time_ms:
            violations.append(f"Execution time limit exceeded: {elapsed_time:.0f}ms > {limits.max_execution_time_ms}ms")
        
        if self.token_count > limits.max_tokens:
            violations.append(f"Token limit exceeded: {self.token_count} > {limits.max_tokens}")
        
        if self.tool_call_count > limits.max_tool_calls:
            violations.append(f"Tool call limit exceeded: {self.tool_call_count} > {limits.max_tool_calls}")
        
        if self.file_operations > limits.max_file_operations:
            violations.append(f"File operations limit exceeded: {self.file_operations} > {limits.max_file_operations}")
        
        return violations


class PermissionError(Exception):
    """Custom exception for permission violations"""
    def __init__(self, message: str, code: str = 'PERMISSION_DENIED'):
        super().__init__(message)
        self.code = code
        self.message = message


class RecursionError(Exception):
    """Custom exception for recursion detection"""
    pass


class SubagentConfig(BaseModel):
    """Configuration for a subagent"""
    name: str = Field(..., description="Name of the subagent")
    system_prompt: str = Field(..., description="System prompt for the subagent")
    user_prompt: Optional[str] = Field(None, description="Initial user prompt for the subagent (optional)")
    allowed_tools: Set[str] = Field(default_factory=set, description="Set of allowed tool names")
    denied_tools: Set[str] = Field(default_factory=set, description="Set of denied tool names")
    max_iterations: int = Field(default=15, description="Maximum iterations for the subagent")
    memory_enabled: bool = Field(default=False, description="Whether to enable memory for the subagent")
    
    # New optimization fields
    execution_mode: SubagentExecutionMode = Field(default=SubagentExecutionMode.SINGLE, description="Execution mode")
    concurrent_agents: int = Field(default=1, description="Number of concurrent agents (for concurrent mode)")
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits, description="Resource limits")
    enable_synthesis: bool = Field(default=False, description="Enable result synthesis for concurrent agents")
    
    def __post_init__(self):
        """Validate configuration"""
        if self.allowed_tools and self.denied_tools:
            raise ValueError("Cannot specify both allowed_tools and denied_tools")
        
        if self.execution_mode == SubagentExecutionMode.CONCURRENT and self.concurrent_agents < 1:
            raise ValueError("Concurrent agents must be at least 1")
        
        if self.concurrent_agents > 10:
            logger.warning(f"High concurrent agent count: {self.concurrent_agents}. Consider performance impact.")


class SubagentResult(BaseModel):
    """Result from subagent execution"""
    success: bool = Field(..., description="Whether execution was successful")
    final_answer: str = Field(..., description="Final answer from the subagent")
    history: List[Dict[str, Any]] = Field(..., description="Conversation history")
    iterations: int = Field(..., description="Number of iterations executed")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    
    # New optimization fields
    execution_time_ms: float = Field(default=0, description="Total execution time in milliseconds")
    tokens_used: int = Field(default=0, description="Total tokens used")
    resource_violations: List[str] = Field(default_factory=list, description="Resource limit violations")
    agent_results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual agent results (for concurrent mode)")
    synthesis_metadata: Optional[Dict[str, Any]] = Field(None, description="Synthesis metadata (if applicable)")


class SubagentManager:
    """Manages creation and execution of subagents with tool isolation"""
    
    def __init__(self, llm_client: LLMClient, system_info: Dict[str, Any]):
        """
        Initialize the SubagentManager
        
        Args:
            llm_client: LLM client for API interactions
            system_info: System information dictionary
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.active_subagents: Dict[str, FunctionCallAgent] = {}
        
    def create_subagent(self, config: SubagentConfig) -> FunctionCallAgent:
        """
        Create a new subagent with the specified configuration
        
        Args:
            config: Subagent configuration
            
        Returns:
            FunctionCallAgent: Configured subagent
        """
        logger.info(f"Creating subagent '{config.name}' with tool isolation")
        
        # Create a new FunctionCall agent
        subagent = create_function_call_agent(
            llm_client=self.llm_client,
            system_info=self.system_info,
            memory_enabled=config.memory_enabled,
            user_id=f"subagent_{config.name}"
        )
        
        # Override the max_iterations from config
        subagent.max_iterations = config.max_iterations
        
        # Apply tool isolation by filtering tools
        self._apply_tool_isolation(subagent, config)
        
        # Customize the system prompt
        self._customize_system_prompt(subagent, config)
        
        # Store the subagent
        self.active_subagents[config.name] = subagent
        
        logger.info(f"Subagent '{config.name}' created successfully")
        return subagent
    
    def _apply_tool_isolation(self, subagent: FunctionCallAgent, config: SubagentConfig) -> None:
        """
        Apply tool isolation by filtering allowed/denied tools
        
        Args:
            subagent: The subagent to modify
            config: Subagent configuration with tool restrictions
        """
        # Get all registered tools from the registry
        all_tools = openai_tool_registry.get_tools()
        
        # Create a custom tool registry for this subagent
        from terminal_agent.react.function_call_tools import OpenAIToolRegistry
        
        # Store the original registry for restoration if needed
        original_registry = getattr(subagent, 'custom_tool_registry', None)
        
        # Create a filtered tool registry for this subagent
        subagent.custom_tool_registry = OpenAIToolRegistry()
        
        # Apply allowlist approach if specified
        if config.allowed_tools:
            logger.info(f"Applying allowlist for subagent '{config.name}': {config.allowed_tools}")
            for tool_name_str in config.allowed_tools:
                # Find the tool in the global registry
                for tool_def in all_tools:
                    tool_name = tool_def['function']['name']
                    if tool_name == tool_name_str or tool_name_str in tool_name:
                        # Get the handler from the global registry
                        handler = openai_tool_registry._handlers.get(tool_name)
                        if handler:
                            subagent.custom_tool_registry.register_tool(tool_def, handler)
                            logger.debug(f"Allowed tool: {tool_name}")
                        else:
                            logger.warning(f"Tool '{tool_name}' not found in available tools")
                        break
                else:
                    logger.warning(f"Unknown tool name in allowlist: {tool_name_str}")
        
        # Apply denylist approach if specified (and no allowlist)
        elif config.denied_tools:
            logger.info(f"Applying denylist for subagent '{config.name}': {config.denied_tools}")
            for tool_def in all_tools:
                tool_name = tool_def['function']['name']
                if tool_name not in config.denied_tools:
                    handler = openai_tool_registry._handlers.get(tool_name)
                    if handler:
                        subagent.custom_tool_registry.register_tool(tool_def, handler)
                        logger.debug(f"Allowed tool: {tool_name}")
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in available tools")
                else:
                    logger.debug(f"Denied tool: {tool_name}")
        
        # If no restrictions, use all tools
        else:
            logger.info(f"No tool restrictions for subagent '{config.name}', allowing all tools")
            for tool_def in all_tools:
                tool_name = tool_def['function']['name']
                handler = openai_tool_registry._handlers.get(tool_name)
                if handler:
                    subagent.custom_tool_registry.register_tool(tool_def, handler)
    
    def _customize_system_prompt(self, subagent: FunctionCallAgent, config: SubagentConfig) -> None:
        """
        Customize the system prompt for the subagent
        
        Args:
            subagent: The subagent to modify
            config: Subagent configuration with custom prompts
        """
        # Create a custom system prompt that includes the configuration
        custom_prompt = f"""{config.system_prompt}"""
        
        # Replace the template in the subagent
        subagent.template = custom_prompt
        logger.debug(f"Custom system prompt set for subagent '{config.name}'")
    
    def _get_available_tools_for_display(self, subagent: FunctionCallAgent, config: Optional[SubagentConfig]) -> str:
        """
        Get a list of available tools for display purposes
        
        Args:
            subagent: The subagent
            config: Subagent configuration (optional)
            
        Returns:
            String of available tool names
        """
        # Use the custom tool registry if available, otherwise use global registry
        if hasattr(subagent, 'custom_tool_registry'):
            tools = subagent.custom_tool_registry.get_tools()
        else:
            tools = openai_tool_registry.get_tools()
        
        tool_names = [tool['function']['name'] for tool in tools]
        return ', '.join(tool_names)
    
    def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        """
        Execute a subagent with the given configuration
        
        Args:
            config: Subagent configuration
            
        Returns:
            SubagentResult: Execution result
        """
        logger.info(f"Executing subagent '{config.name}'")
        
        try:
            # Create the subagent
            subagent = self.create_subagent(config)
            
            # If the subagent has a custom tool registry, use it
            if hasattr(subagent, 'custom_tool_registry'):
                # Patch the subagent's _process_tool_calls method to use custom registry
                original_process_tool_calls = subagent._process_tool_calls
                
                def patched_process_tool_calls(message, messages):
                    # Temporarily replace the global registry for this execution
                    import terminal_agent.react.function_call_agent as function_call_module
                    original_registry = function_call_module.openai_tool_registry
                    function_call_module.openai_tool_registry = subagent.custom_tool_registry
                    
                    try:
                        result = original_process_tool_calls(message, messages)
                    finally:
                        # Always restore the original registry
                        function_call_module.openai_tool_registry = original_registry
                    
                    return result
                
                # Apply the patch
                subagent._process_tool_calls = patched_process_tool_calls
            
            # Execute the subagent with the user prompt
            final_answer = subagent.execute(config.user_prompt)
            
            # Extract tool calls
            tool_calls = self._extract_tool_calls(subagent)
            
            # Create result - use subagent.messages directly for history
            result = SubagentResult(
                success=True,
                final_answer=final_answer,
                history=subagent.messages,  # Use messages directly
                tool_calls=tool_calls,
                iterations=subagent.current_iteration,
                error=None
            )
            
            logger.info(f"Subagent '{config.name}' executed successfully in {subagent.current_iteration} iterations")
            return result
            
        except Exception as e:
            logger.error(f"Error executing subagent '{config.name}': {str(e)}")
            return SubagentResult(
                success=False,
                final_answer="",
                history=[],
                tool_calls=[],
                iterations=0,
                error=str(e)
            )
    
    def execute_subagent_with_history(self, config: SubagentConfig, conversation_history: List[Dict[str, Any]]) -> SubagentResult:
        """
        Execute a subagent with conversation history
        
        Args:
            config: Subagent configuration
            conversation_history: List of conversation messages to use as history
            
        Returns:
            SubagentResult: Execution result
        """
        logger.info(f"Executing subagent '{config.name}' with {len(conversation_history)} history messages")
        
        try:
            # Create the subagent
            subagent = self.create_subagent(config)
            
            # If subagent has a custom tool registry, use it
            if hasattr(subagent, 'custom_tool_registry'):
                # Patch the subagent's _process_tool_calls method to use custom registry
                original_process_tool_calls = subagent._process_tool_calls
                
                def patched_process_tool_calls(message, messages):
                    # Temporarily replace the global registry for this execution
                    import terminal_agent.react.function_call_agent as function_call_module
                    original_registry = function_call_module.openai_tool_registry
                    function_call_module.openai_tool_registry = subagent.custom_tool_registry
                    
                    try:
                        result = original_process_tool_calls(message, messages)
                    finally:
                        # Always restore the original registry
                        function_call_module.openai_tool_registry = original_registry
                    
                    return result
                
                # Apply the patch
                subagent._process_tool_calls = patched_process_tool_calls
            
            # Load conversation history into the subagent
            self._load_conversation_history(subagent, conversation_history)
            
            # Execute the subagent with the user prompt, preserving loaded history
            # If user_prompt is None or empty, use a default prompt or skip
            user_prompt = config.user_prompt
            
            final_answer = subagent.execute(user_prompt, preserve_history=True)
            
            # Extract tool calls
            #tool_calls = self._extract_tool_calls(subagent)
            
            # Create result - use subagent.messages directly for history
            result = SubagentResult(
                success=True,
                final_answer=final_answer,
                history=subagent.messages,  # Use messages directly
                iterations=subagent.current_iteration,
                error=None
            )
            
            logger.info(f"Subagent '{config.name}' executed successfully with history in {subagent.current_iteration} iterations")
            return result
            
        except Exception as e:
            logger.error(f"Error executing subagent '{config.name}' with history: {str(e)}")
            return SubagentResult(
                success=False,
                final_answer="",
                iterations=0,
                error=str(e)
            )
    
    def _load_conversation_history(self, subagent: FunctionCallAgent, conversation_history: List[Dict[str, Any]]) -> None:
        """
        Load conversation history into a subagent's memory
        
        Args:
            subagent: The subagent to load history into
            conversation_history: List of conversation messages to load
        """
        try:
            # Initialize conversation_history if it doesn't exist
            if not hasattr(subagent, 'conversation_history'):
                subagent.conversation_history = []
            
            # Add all messages to the history
            subagent.conversation_history.extend(conversation_history)
            logger.info(f"Loaded {len(conversation_history)} messages into subagent's conversation history")
        except Exception as e:
            logger.error(f"Error loading conversation history into subagent: {e}")
            # Re-raise the exception to handle it in the calling code if needed
            raise
    
    def _extract_tool_calls(self, subagent: FunctionCallAgent) -> List[Dict[str, Any]]:
        """
        Extract tool calls from a subagent's execution
        
        Args:
            subagent: The subagent to extract tool calls from
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
    
        # Extract from messages (primary method) - includes tool calls and assistant messages with tool_calls
        if not tool_calls and hasattr(subagent, 'messages') and subagent.messages:
            for message in subagent.messages:
                # Extract tool messages
                if message.get("role") == "tool":
                    content = message.get("content", "")
                    if content:
                        tool_calls.append({
                            "tool_name": message.get("name", "unknown"),
                            "input": "unknown",
                            "output": content,
                            "timestamp": "unknown"
                        })
                # Extract tool calls from assistant messages
                elif message.get("role") == "assistant" and "tool_calls" in message:
                    for tool_call in message.get("tool_calls", []):
                        if isinstance(tool_call, dict) and "function" in tool_call:
                            function_info = tool_call["function"]
                            tool_calls.append({
                                "tool_name": function_info.get("name", "unknown"),
                                "input": function_info.get("arguments", "{}"),
                                "output": "",
                                "timestamp": "unknown"
                            })
        
        
        return tool_calls
    
    def get_subagent(self, name: str) -> Optional[FunctionCallAgent]:
        """
        Get an active subagent by name
        
        Args:
            name: Name of the subagent
            
        Returns:
            FunctionCallAgent or None if not found
        """
        return self.active_subagents.get(name)
    
    def terminate_subagent(self, name: str) -> bool:
        """
        Terminate an active subagent
        
        Args:
            name: Name of the subagent to terminate
            
        Returns:
            bool: True if successful, False if not found
        """
        if name in self.active_subagents:
            # Clean up memory if enabled
            subagent = self.active_subagents[name]
            if subagent.memory_enabled and subagent.session_id:
                try:
                    subagent.memory_db.close()
                except Exception as e:
                    logger.error(f"Error closing memory database for subagent '{name}': {e}")
            
            # Remove from active subagents
            del self.active_subagents[name]
            logger.info(f"Subagent '{name}' terminated")
            return True
        else:
            logger.warning(f"Subagent '{name}' not found")
            return False
    
    def list_active_subagents(self) -> List[str]:
        """
        List all active subagents
        
        Returns:
            List of subagent names
        """
        return list(self.active_subagents.keys())
    
    def get_subagent_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an active subagent
        
        Args:
            name: Name of the subagent
            
        Returns:
            Dictionary with subagent information or None if not found
        """
        if name not in self.active_subagents:
            return None
        
        subagent = self.active_subagents[name]
        return {
            "name": name,
            "current_iteration": subagent.current_iteration,
            "max_iterations": subagent.max_iterations,
            "available_tools": self._get_available_tools_for_display(subagent, None),
            "memory_enabled": subagent.memory_enabled,
            "message_count": len(subagent.conversation_history),
            "session_id": subagent.session_id
        }


def create_subagent_config(
    name: str,
    system_prompt: str,
    user_prompt: str,
    allowed_tools: Optional[List[str]] = None,
    denied_tools: Optional[List[str]] = None,
    max_iterations: int = 15,
    memory_enabled: bool = False
) -> SubagentConfig:
    """
    Create a subagent configuration with validation
    
    Args:
        name: Name of the subagent
        system_prompt: System prompt for the subagent
        user_prompt: Initial user prompt for the subagent
        allowed_tools: List of allowed tool names (optional)
        denied_tools: List of denied tool names (optional)
        max_iterations: Maximum iterations (default: 15)
        memory_enabled: Whether to enable memory (default: False)
        
    Returns:
        SubagentConfig: Validated configuration
    """
    # Validate tool restrictions
    if allowed_tools is not None and denied_tools is not None:
        raise ValueError("Cannot specify both allowed_tools and denied_tools")
    
    # Convert to sets
    allowed_set = set(allowed_tools) if allowed_tools else set()
    denied_set = set(denied_tools) if denied_tools else set()
    
    return SubagentConfig(
        name=name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        allowed_tools=allowed_set,
        denied_tools=denied_set,
        max_iterations=max_iterations,
        memory_enabled=memory_enabled
    )