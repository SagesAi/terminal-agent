"""
DeepSeek Provider Implementation

This module implements the DeepSeek provider for the LLM client.
"""

import os
import httpx
from typing import List, Dict, Any, Optional, Union
import logging
from rich.console import Console

from .base import BaseLLMProvider

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek API provider implementation.
    """
    
    # Available models
    AVAILABLE_MODELS = [
        "deepseek-chat",
        "deepseek-coder"
    ]
    
    # Default model
    DEFAULT_MODEL = "deepseek-chat"
    
    # Default API base URL
    DEFAULT_API_BASE = "https://api.deepseek.com/v1"
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY environment variable)
            model: Model identifier to use (defaults to DEFAULT_MODEL)
            api_base: Base URL for API (defaults to DEFAULT_API_BASE)
            **kwargs: Additional provider-specific parameters
        """
        # Get API key from parameters or environment
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Provide it as a parameter or set the DEEPSEEK_API_KEY environment variable.")
        
        # Set model
        self.model = model or self.DEFAULT_MODEL
        
        # Set API base URL
        self.api_base = api_base or self.DEFAULT_API_BASE
        
        logger.debug(f"Initialized DeepSeek provider with model: {self.model}")
    
    def get_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        return "deepseek"
    
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
        return DeepSeekProvider.DEFAULT_MODEL
    
    def call_with_messages(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """
        Call the DeepSeek API with a list of messages.
        
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
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            payload.update(kwargs)
            
            # Call the API
            try:
                with httpx.Client(timeout=60.0*3) as client:
                    response = client.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    # Check for errors
                    if response.status_code != 200:
                        logger.error(f"DeepSeek API Error: Status {response.status_code}")
                        logger.error(f"Response: {response.text}")
                        response.raise_for_status()
                    
                    # Parse the response
                    data = response.json()
                
                # Return the response text
                return data["choices"][0]["message"]["content"]
                
            except httpx.ConnectError as e:
                logger.error(f"Connection error with DeepSeek API: {str(e)}")
                raise ConnectionError(f"Unable to connect to DeepSeek API: {str(e)}")
            
        except ConnectionError:
            # Re-raise connection errors
            raise
        except Exception as e:
            # Log and re-raise other exceptions
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            raise
    
    def call_with_prompt(self, 
                        prompt: str, 
                        temperature: float = 0.2,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the DeepSeek API with a single prompt string.
        
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
