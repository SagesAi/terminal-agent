"""
OpenAI Provider Implementation

This module implements the OpenAI provider for the LLM client.
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

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider implementation.
    """
    
    # Available models
    AVAILABLE_MODELS = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]
    
    # Default model
    DEFAULT_MODEL = "gpt-4"
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model identifier to use (defaults to DEFAULT_MODEL)
            api_base: Base URL for API (optional, for custom endpoints)
            **kwargs: Additional provider-specific parameters
        """
        # Import here to avoid loading the library if not used
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package is not installed. Please install it with 'pip install openai'.")
        
        # Get API key from parameters or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it as a parameter or set the OPENAI_API_KEY environment variable.")
        
        # Set model
        self.model = model or self.DEFAULT_MODEL
        
        # Set API base URL if provided
        self.api_base = api_base
        
        # Initialize client
        client_kwargs = {}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
            
        self.client = OpenAI(api_key=self.api_key, **client_kwargs)
        
        logger.debug(f"Initialized OpenAI provider with model: {self.model}")
    
    def get_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        return "openai"
    
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
        return OpenAIProvider.DEFAULT_MODEL
    
    def call_with_messages(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """
        Call the OpenAI API with a list of messages.
        
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
            # Create additional parameters dictionary
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Call the API
            response = self.client.chat.completions.create(**params)
            
            # Return the response text
            return response.choices[0].message.content
            
        except Exception as e:
            # Check if it's a connection error
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with OpenAI API: {str(e)}")
                raise ConnectionError(f"Unable to connect to OpenAI API: {str(e)}")
            
            # Re-raise other exceptions
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def call_with_messages_and_functions(self,
                                        messages: List[Dict[str, Any]],
                                        tools: List[Dict[str, Any]],
                                        temperature: float = 0.2,
                                        max_tokens: int = 2000,
                                        **kwargs) -> Any:
        """
        Call the OpenAI API with function calling support.
        
        Args:
            messages: List of message dictionaries
            tools: List of tool definitions for function calling
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Any: The response object with potential function_call
            
        Raises:
            ConnectionError: When there's a connection issue with the API
            Exception: For other errors
        """
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",  # Let the model decide when to use tools
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            params.update(kwargs)
            
            response = self.client.chat.completions.create(**params)
            
            return response.choices[0].message
            
        except Exception as e:
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with OpenAI API: {str(e)}")
                raise ConnectionError(f"Unable to connect to OpenAI API: {str(e)}")
            
            logger.error(f"Error calling OpenAI API with tools: {str(e)}")
            raise
        """
        Call the OpenAI API with a list of messages.
        
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
            # Create additional parameters dictionary
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Call the API
            response = self.client.chat.completions.create(**params)
            
            # Return the response text
            return response.choices[0].message.content
            
        except Exception as e:
            # Check if it's a connection error
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with OpenAI API: {str(e)}")
                raise ConnectionError(f"Unable to connect to OpenAI API: {str(e)}")
            
            # Re-raise other exceptions
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def call_with_prompt(self, 
                        prompt: str, 
                        temperature: float = 0.2,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the OpenAI API with a single prompt string.
        
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
