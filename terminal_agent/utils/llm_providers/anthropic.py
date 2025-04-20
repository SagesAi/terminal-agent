"""
Anthropic Claude Provider Implementation

This module implements the Anthropic Claude provider for the LLM client.
"""

import os
from typing import List, Dict, Any, Optional, Union
import logging
from rich.console import Console

from .base import BaseLLMProvider

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider implementation.
    """
    
    # Available models
    AVAILABLE_MODELS = [
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-2.1",
        "claude-2.0"
    ]
    
    # Default model
    DEFAULT_MODEL = "claude-3-sonnet"
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Anthropic Claude provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
            model: Model identifier to use (defaults to DEFAULT_MODEL)
            api_base: Base URL for API (optional, for custom endpoints)
            **kwargs: Additional provider-specific parameters
        """
        # Import here to avoid loading the library if not used
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package is not installed. Please install it with 'pip install anthropic'.")
        
        # Get API key from parameters or environment
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Provide it as a parameter or set the ANTHROPIC_API_KEY environment variable.")
        
        # Set model
        self.model = model or self.DEFAULT_MODEL
        
        # Initialize client
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        logger.debug(f"Initialized Anthropic Claude provider with model: {self.model}")
    
    def get_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        return "anthropic"
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models for this provider.
        
        Returns:
            List[str]: List of model identifiers
        """
        return self.AVAILABLE_MODELS
    
    @staticmethod
    def get_default_model() -> str:
        """
        Get the default model for this provider.
        
        Returns:
            str: Default model identifier
        """
        return AnthropicProvider.DEFAULT_MODEL
    
    def _convert_to_claude_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert standard messages format to Claude's format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            List[Dict[str, Any]]: Messages in Claude format
        """
        claude_messages = []
        system_content = None
        
        # Extract system message if present
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                claude_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                claude_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
        
        return claude_messages, system_content
    
    def call_with_messages(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """
        Call the Claude API with a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            str: The model's response text
            
        Raises:
            ConnectionError: When there's a connection issue with the API
            Exception: For other errors
        """
        try:
            # Convert messages to Claude format
            claude_messages, system_content = self._convert_to_claude_messages(messages)
            
            # Create message parameters
            params = {
                "model": self.model,
                "messages": claude_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add system content if present
            if system_content:
                params["system"] = system_content
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Call the API
            response = self.client.messages.create(**params)
            
            # Return the response text
            return response.content[0].text
            
        except Exception as e:
            # Check if it's a connection error
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with Anthropic API: {str(e)}")
                raise ConnectionError(f"Unable to connect to Anthropic API: {str(e)}")
            
            # Re-raise other exceptions
            logger.error(f"Error calling Anthropic API: {str(e)}")
            raise
    
    def call_with_prompt(self, 
                        prompt: str, 
                        temperature: float = 0.2,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the Claude API with a single prompt string.
        
        Args:
            prompt: The complete prompt to send to the LLM
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            str: The model's response text
            
        Raises:
            ConnectionError: When there's a connection issue with the API
            Exception: For other errors
        """
        # Convert the prompt to a messages array
        messages = [{"role": "user", "content": prompt}]
        
        # Call the API with the messages
        return self.call_with_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
