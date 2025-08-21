"""
Kimi Provider Implementation

This module implements the Kimi provider for the LLM client.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import logging
from rich.console import Console

from .base import BaseLLMProvider

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

class KimiProvider(BaseLLMProvider):
    """
    Kimi API provider implementation.
    """

    # Available models
    AVAILABLE_MODELS = [
        "kimi-k2-0711-preview",
    ]

    # Default model
    DEFAULT_MODEL = "kimi-k2-0711-preview"

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Kimi provider.

        Args:
            api_key: Kimi API key (defaults to KIMI_API_KEY environment variable)
            model: Model identifier to use (defaults to DEFAULT_MODEL)
            api_base: Base URL for API (optional, for custom endpoints)
            **kwargs: Additional provider-specific parameters
        """
        # Import here to avoid loading the library if not used
        try:
            import requests
        except ImportError:
            raise ImportError("Requests package is not installed. Please install it with 'pip install requests'.")

        # Get API key from parameters or environment
        self.api_key = api_key or os.environ.get("KIMI_API_KEY")
        if not self.api_key:
            raise ValueError("Kimi API key is required. Provide it as a parameter or set the KIMI_API_KEY environment variable.")

        # Set model
        self.model = model or self.DEFAULT_MODEL

        # Set API base URL if provided
        self.api_base = api_base or "https://api.moonshot.cn/v1"

        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        logger.debug(f"Initialized Kimi provider with model: {self.model}")

    def get_name(self) -> str:
        """
        Get the name of the provider.

        Returns:
            str: Provider name
        """
        return "kimi"

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
        return KimiProvider.DEFAULT_MODEL

    def call_with_messages(self,
                          messages: List[Dict[str, str]],
                          temperature: float = 0.2,
                          max_tokens: int = 4096,
                          **kwargs) -> str:
        """
        Call the Kimi API with a list of messages.

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
        url = f"{self.api_base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling Kimi API: {str(e)}")
            raise

    def call_with_prompt(self,
                        prompt: str,
                        temperature: float = 0.3,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the Kimi API with a single prompt string.

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
        messages = [{"role": "user", "content": prompt}]
        return self.call_with_messages(messages, temperature, max_tokens, **kwargs)
    
    def call_with_messages_and_functions(self,
                                        messages: List[Dict[str, Any]],
                                        tools: List[Dict[str, Any]],
                                        temperature: float = 0.2,
                                        max_tokens: int = 4096,
                                        **kwargs) -> Any:
        """
        Call Kimi API with function calling support.
        
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
        url = f"{self.api_base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": "auto",
            **kwargs
        }

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Create response object compatible with OpenAI format
            message = result["choices"][0]["message"]
            
            response_obj = {
                "role": "assistant",
                "content": message.get("content", "")
            }
            
            # Add tool calls if present
            if "tool_calls" in message:
                response_obj["tool_calls"] = message["tool_calls"]
            
            return response_obj
            
        except Exception as e:
            logger.error(f"Error calling Kimi API with tools: {str(e)}")
            raise
