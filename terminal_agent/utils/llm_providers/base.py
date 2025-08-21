"""
Base LLM Provider

This module defines the abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface that all LLM provider implementations must follow.
    """
    
    @abstractmethod
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the provider (defaults to environment variable)
            model: Model identifier to use
            api_base: Base URL for API (optional, for custom endpoints)
            **kwargs: Additional provider-specific parameters
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models for this provider.
        
        Returns:
            List[str]: List of model identifiers
        """
        return []
    
    @abstractmethod
    def call_with_messages(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """
        Call the LLM API with a list of messages.
        
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
        pass
    
    @abstractmethod
    def call_with_messages_and_functions(self,
                                        messages: List[Dict[str, Any]],
                                        tools: List[Dict[str, Any]],
                                        temperature: float = 0.2,
                                        max_tokens: int = 2000,
                                        **kwargs) -> Any:
        """
        Call the LLM API with function calling support.
        
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
        pass
        """
        Call the LLM API with a list of messages.
        
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
        pass
    
    @abstractmethod
    def call_with_prompt(self, 
                        prompt: str, 
                        temperature: float = 0.2,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the LLM API with a single prompt string.
        
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
        pass
    
    @staticmethod
    def get_default_model() -> str:
        """
        Get the default model for this provider.
        
        Returns:
            str: Default model identifier
        """
        return ""
    
    def format_system_prompt(self, system_prompt: str) -> Dict[str, str]:
        """
        Format a system prompt according to the provider's requirements.
        
        Args:
            system_prompt: The system prompt text
            
        Returns:
            Dict[str, str]: Formatted system message
        """
        return {"role": "system", "content": system_prompt}
    
    def format_user_prompt(self, user_prompt: str) -> Dict[str, str]:
        """
        Format a user prompt according to the provider's requirements.
        
        Args:
            user_prompt: The user prompt text
            
        Returns:
            Dict[str, str]: Formatted user message
        """
        return {"role": "user", "content": user_prompt}
    
    def format_assistant_message(self, assistant_message: str) -> Dict[str, str]:
        """
        Format an assistant message according to the provider's requirements.
        
        Args:
            assistant_message: The assistant message text
            
        Returns:
            Dict[str, str]: Formatted assistant message
        """
        return {"role": "assistant", "content": assistant_message}
