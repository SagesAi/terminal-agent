#!/usr/bin/env python3
"""
History Message Compression Agent

This module implements a specialized agent for compressing conversation history
using the subagent framework with restricted tools (readfile only).
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

from terminal_agent.react.subagent import SubagentManager, SubagentConfig, SubagentResult
from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.react.function_call_tools import openai_tool_registry

# Get logger
logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for history compression"""
    max_history_length: int = 50
    compression_threshold: int = 100
    preserve_system_messages: bool = True
    preserve_tool_results: bool = False
    summary_length_target: int = 20


class HistoryCompressionAgent:
    """
    Specialized agent for compressing conversation history
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        system_info: Dict[str, Any],
        compression_config: Optional[CompressionConfig] = None
    ):
        """
        Initialize the History Compression Agent
        
        Args:
            llm_client: LLM client for API interactions
            system_info: System information dictionary
            compression_config: Configuration for compression behavior
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.compression_config = compression_config or CompressionConfig()
        
        # Create subagent manager
        self.subagent_manager = SubagentManager(llm_client, system_info)
        
        # Initialize the compression subagent
        self.compression_subagent = self._create_compression_subagent()
        
        # Initialize tool tracking
        self.tool_calls_history = []
        self.tool_results_history = []
        
        logger.info("History Compression Agent initialized")
    
    def _create_compression_subagent(self):
        """Create the specialized compression subagent"""
        config = SubagentConfig(
            name="history_compression",
            system_prompt="You are SagesAI, a terminal agent for devops.\nYou are a helpful AI assistant tasked with summarizing conversations.",
            user_prompt="Please compress and summarize the conversation history to focus on key information and reduce redundancy.",
            allowed_tools={"read_file"},  # Only allow readfile tool
            max_iterations=30,
            memory_enabled=False
        )
        
        return self.subagent_manager.create_subagent(config)
    

    def parse_compression_result(self, result: str) -> str:
        """
        Extract content between <summary> and </summary> tags from the result string.
        
        Args:
            result: The string containing the content to parse
            
        Returns:
            The extracted content between the tags, or an empty string if not found
        """
        if not result or not isinstance(result, str):
            return ""
        
        start_tag = "<summary>"
        end_tag = "</summary>"
        
        start_idx = result.find(start_tag)
        if start_idx == -1:
            return ""
        
        start_idx += len(start_tag)
        end_idx = result.find(end_tag, start_idx)
        
        if end_idx == -1:
            return ""
        
        return result[start_idx:end_idx].strip()
    
    def compress_history(
        self,
        history: List[Dict[str, Any]],
        compression_config: Optional[CompressionConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Compress conversation history using the specialized subagent
        
        Args:
            history: List of conversation messages to compress
            compression_config: Optional compression configuration
            
        Returns:
            Compressed list of messages
        """
        if not history:
            return []
        
        
        try:
            # Create the compression conversation: system prompt + context + user prompt
            system_prompt = "You are SagesAI, a terminal agent for devops.\nYou are a helpful AI assistant tasked with summarizing conversations."
            context_messages = history  # The context to be compressed
            user_prompt = self._create_compression_prompt(history)

            
            # Execute the compression subagent with conversation history
            result = self.subagent_manager.execute_subagent_with_history(
                SubagentConfig(
                    name="history_compression",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    allowed_tools={"read_file"},
                    max_iterations=10,
                    memory_enabled=False
                ),
                context_messages
            )
            
            if result.success:
                # Parse the compressed history from the result
                # Extract tool results from the agent's execution history
                
                # Build structured message with extracted tool information
                # Pass the full messages to _generate_tool_call_reminder_from_calls for direct extraction
                compressed_history = self._build_structured_message_with_tools(
                    self.parse_compression_result(result.final_answer),
                    result.history,  # Pass full messages instead of extracted calls
                )
                logger.info(f"History compressed from {len(history)} to {len(compressed_history)} messages")
                return compressed_history
            else:
                logger.error(f"Compression failed: {result.error}")
                # Return original history if compression fails
                return history
                
        except Exception as e:
            logger.error(f"Error during history compression: {e}")
            # Return original history if compression fails
            return history
    
    def load_template(self, template_name: str) -> str:
        """Load a template file from the templates directory.
        
        Args:
            template_name: Name of the template file to load
            
        Returns:
            str: The content of the template file
        """
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the templates directory path
        templates_dir = os.path.join(current_dir, 'templates')
        # Get the full path to the template file
        template_path = os.path.join(templates_dir, template_name)
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Template file not found: {template_path}")
            raise
    
    def _create_compression_prompt(
        self,
        history: List[Dict[str, Any]]
    ) -> str:
        """Create the compression prompt for the subagent"""
        
        try:
            # Load the compression prompt template
            compression_prompt = self.load_template("compact_prompt.txt")
            return compression_prompt
        except Exception as e:
            logger.error(f"Error loading compression prompt template: {e}")
            # Fallback to a simple prompt if the template can't be loaded
            return f"Please summarize the following conversation history:\n\n{json.dumps(history, indent=2)}"
    

    
    def _build_structured_message_with_tools(
        self,
        raw_result: str,
        messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build structured message format with paired tool calls and results from history"""
        
        # Build the compression summary text
        compression_summary = f"This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:\n{raw_result}"
        
        # Create the base structured message content
        structured_content = [
            {
                "type": "text",
                "text": "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# important-instruction-reminders\nDo what has been asked; nothing more, nothing less.\nNEVER create files unless they're absolutely necessary for achieving your goal.\nALWAYS prefer editing an existing file to creating a one.\nNEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.\n\nIMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.\n</system-reminder>"
            },
            {
                "type": "text", 
                "text": compression_summary
            }
        ]
        
        # Generate paired tool call and result reminders
        tool_pairs = self._generate_paired_tool_reminders(messages)
        
        # Add tool pairs as alternating call and result sections
        for i, (call_reminder, result_reminder) in enumerate(tool_pairs, 1):
            # Add tool call section
            structured_content.append({
                "type": "text",
                "text": f'<system-reminder>\n{call_reminder}\n</system-reminder>'
            })
            
            # Add corresponding tool result section
            structured_content.append({
                "type": "text",
                "text": f'<system-reminder>\n{result_reminder}\n</system-reminder>'
            })
        
        # Create the final structured message
        structured_message = {
            "role": "user",
            "content": structured_content
        }
        
        return structured_message
    
    def _generate_paired_tool_reminders(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[tuple]:
        """Generate paired tool call and result reminders"""
        
        # Extract tool calls from messages
        tool_calls = self._extract_tool_calls_from_messages(messages)
        
        # Create pairs of tool calls and their corresponding results
        pairs = []
        
        # First, try to pair by tool_call_id
        for call in tool_calls:
            call_id = call.get('id', '')
            tool_name = call.get('function', {}).get('name', 'unknown_tool')
            tool_args = call.get('function', {}).get('arguments', '{}')
            
            # Find corresponding tool result
            result_content = call.get('result', '')
            # Generate reminders for this pair
            call_reminder = self._generate_single_tool_call_reminder(tool_name, tool_args)
            result_reminder = self._generate_single_tool_result_reminder(result_content, tool_name)
            
            pairs.append((call_reminder, result_reminder))
        
        return pairs
    
    def _extract_tool_calls_from_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool calls and their results from messages"""
        tool_calls = []
        tool_call_id_to_result = {}
    
        # First pass: collect all tool results
        for message in messages:
            if message.get("role") == "tool" and "tool_call_id" in message:
                tool_call_id_to_result[message["tool_call_id"]] = message.get("content", "")
    
        # Second pass: match tool calls with their results
        for message in messages:
            # Handle assistant messages with tool_calls
            if message.get("role") == "assistant" and "tool_calls" in message:
                for tool_call in message.get("tool_calls", []):
                    if not isinstance(tool_call, dict) or "function" not in tool_call:
                        continue
                        
                    tool_call_id = tool_call.get("id", "")
                    function = tool_call.get("function", {})
                    
                    tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function.get("name", "unknown"),
                            "arguments": function.get("arguments", "{}")
                        },
                        "result": tool_call_id_to_result.get(tool_call_id, "")
                    })
        
        return tool_calls


    
    def _generate_single_tool_call_reminder(self, tool_name: str, tool_args: str) -> str:
        """Generate a single tool call reminder"""
        try:
            import json
            args_dict = json.loads(tool_args)
            args_str = ', '.join(f'{k}: {v}' for k, v in args_dict.items())
            return f'Called the {tool_name} tool with the following input: {args_str}'
        except:
            return f'Called the {tool_name} tool with the following input: {tool_args}'
    
    def _generate_single_tool_result_reminder(self, result_content: str, tool_name: str) -> str:
        """Generate a single tool result reminder"""
        # Truncate long content for display
        if len(result_content) > 6000:
            result_content = result_content[:6000] + '... [truncated]'
        
        return f'Result of calling the {tool_name} tool: "{result_content}"'
    

def create_history_compression_agent(
    llm_client: LLMClient,
    system_info: Dict[str, Any],
    max_history_length: int = 50,
    compression_threshold: int = 100,
    preserve_system_messages: bool = True,
    preserve_tool_results: bool = False,
    summary_length_target: int = 20
) -> HistoryCompressionAgent:
    """
    Create a configured history compression agent
    
    Args:
        llm_client: LLM client for API interactions
        system_info: System information dictionary
        max_history_length: Maximum history length to maintain
        compression_threshold: Threshold at which to apply compression
        preserve_system_messages: Whether to preserve system messages
        preserve_tool_results: Whether to preserve tool results
        summary_length_target: Target length for compressed history
        
    Returns:
        Configured HistoryCompressionAgent
    """
    config = CompressionConfig(
        max_history_length=max_history_length,
        compression_threshold=compression_threshold,
        preserve_system_messages=preserve_system_messages,
        preserve_tool_results=preserve_tool_results,
        summary_length_target=summary_length_target
    )
    
    return HistoryCompressionAgent(llm_client, system_info, config)