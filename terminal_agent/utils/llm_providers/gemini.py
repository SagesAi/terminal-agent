"""
Google Gemini Provider Implementation

This module implements the Google Gemini provider for the LLM client.
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

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider implementation.
    """
    
    # Available models
    AVAILABLE_MODELS = [
        "gemini-pro",
        "gemini-ultra"
    ]
    
    # Default model
    DEFAULT_MODEL = "gemini-pro"
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Gemini API key (defaults to GOOGLE_API_KEY environment variable)
            model: Model identifier to use (defaults to DEFAULT_MODEL)
            api_base: Base URL for API (optional, for custom endpoints)
            **kwargs: Additional provider-specific parameters
        """
        # Import here to avoid loading the library if not used
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package is not installed. Please install it with 'pip install google-generativeai'.")
        
        # Get API key from parameters or environment
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Provide it as a parameter or set the GOOGLE_API_KEY environment variable.")
        
        # Set model
        self.model = model or self.DEFAULT_MODEL
        
        # Initialize the client
        genai.configure(api_key=self.api_key)
        
        # Set up the model
        self.genai = genai
        self.model_obj = genai.GenerativeModel(self.model)
        
        logger.debug(f"Initialized Gemini provider with model: {self.model}")
    
    def get_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        return "gemini"
    
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
        return GeminiProvider.DEFAULT_MODEL
    
    def format_system_prompt(self, system_prompt: str) -> Dict[str, str]:
        """
        Format a system prompt according to Gemini's requirements.
        
        Gemini uses a different format for system prompts compared to OpenAI.
        
        Args:
            system_prompt: The system prompt text
            
        Returns:
            Dict[str, str]: Formatted system message
        """
        return {"role": "system", "content": system_prompt}
    
    def _convert_to_gemini_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert standard messages format to Gemini's format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            List[Dict[str, str]]: Messages in Gemini format
        """
        gemini_messages = []
        system_content = None
        
        # Extract system message if present
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                gemini_messages.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [{"text": msg["content"]}]
                })
        
        # If there's a system message, prepend it to the first user message
        if system_content and gemini_messages and gemini_messages[0]["role"] == "user":
            gemini_messages[0]["parts"][0]["text"] = f"{system_content}\n\n{gemini_messages[0]['parts'][0]['text']}"
        
        return gemini_messages
    
    def call_with_messages(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """
        Call the Gemini API with a list of messages.
        
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
            # Convert messages to Gemini format
            gemini_messages = self._convert_to_gemini_messages(messages)
            
            # Create generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40)
            }
            
            # Handle chat history if there are multiple messages
            if len(gemini_messages) > 1:
                chat = self.model_obj.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(
                    gemini_messages[-1]["parts"][0]["text"],
                    generation_config=generation_config
                )
            else:
                # Single message case
                response = self.model_obj.generate_content(
                    gemini_messages[0]["parts"][0]["text"],
                    generation_config=generation_config
                )
            
            # Return the response text
            return response.text
            
        except Exception as e:
            # Check if it's a connection error
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with Gemini API: {str(e)}")
                raise ConnectionError(f"Unable to connect to Gemini API: {str(e)}")
            
            # Re-raise other exceptions
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def call_with_messages_and_functions(self,
                                        messages: List[Dict[str, Any]],
                                        tools: List[Dict[str, Any]],
                                        temperature: float = 0.2,
                                        max_tokens: int = 2000,
                                        **kwargs) -> Any:
        """
        Call Gemini API with function calling support.
        
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
            # Convert messages to Gemini format
            gemini_messages = self._convert_to_gemini_messages(messages)
            
            # Convert OpenAI tools format to Gemini function declarations
            function_declarations = []
            for tool in tools:
                if "function" in tool:
                    func = tool["function"]
                    function_declaration = {
                        "name": func["name"],
                        "description": func.get("description", "")
                    }
                    if "parameters" in func:
                        function_declaration["parameters"] = func["parameters"]
                    function_declarations.append(function_declaration)
            
            # Create generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40)
            }
            
            # Create tool config if functions are provided
            tool_config = None
            if function_declarations:
                self.model_obj._tools = function_declarations
                tool_config = {
                    "function_calling_config": {
                        "mode": "AUTO"
                    }
                }
            
            # Handle chat history if there are multiple messages
            if len(gemini_messages) > 1:
                chat = self.model_obj.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(
                    gemini_messages[-1]["parts"][0]["text"],
                    generation_config=generation_config,
                    tools=tool_config
                )
            else:
                # Single message case
                response = self.model_obj.generate_content(
                    gemini_messages[0]["parts"][0]["text"],
                    generation_config=generation_config,
                    tools=tool_config
                )
            
            # Create response object compatible with OpenAI format
            result = {
                "role": "assistant",
                "content": response.text if hasattr(response, 'text') else ""
            }
            
            # Handle function calls
            if hasattr(response, 'function_call') and response.function_call:
                tool_calls = [{
                    "id": f"call_{hash(str(response.function_call))}",
                    "type": "function",
                    "function": {
                        "name": response.function_call.name,
                        "arguments": json.dumps(response.function_call.args) if hasattr(response.function_call, 'args') else "{}"
                    }
                }]
                result["tool_calls"] = tool_calls
            
            return result
            
        except Exception as e:
            # Check if it's a connection error
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with Gemini API: {str(e)}")
                raise ConnectionError(f"Unable to connect to Gemini API: {str(e)}")
            
            # Re-raise other exceptions
            logger.error(f"Error calling Gemini API with tools: {str(e)}")
            raise
    
    def call_with_prompt(self, 
                        prompt: str, 
                        temperature: float = 0.2,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the Gemini API with a single prompt string.
        
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
        try:
            # Create generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40)
            }
            
            # Call the API directly with the prompt
            response = self.model_obj.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Return the response text
            return response.text
            
        except Exception as e:
            # Check if it's a connection error
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with Gemini API: {str(e)}")
                raise ConnectionError(f"Unable to connect to Gemini API: {str(e)}")
            
            # Re-raise other exceptions
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def call_with_messages_and_functions(self,
                                        messages: List[Dict[str, Any]],
                                        tools: List[Dict[str, Any]],
                                        temperature: float = 0.2,
                                        max_tokens: int = 2000,
                                        **kwargs) -> Any:
        """
        Call Gemini API with function calling support.
        
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
            # Convert messages to Gemini format
            gemini_messages = self._convert_to_gemini_messages(messages)
            
            # Convert OpenAI tools format to Gemini function declarations
            function_declarations = []
            for tool in tools:
                if "function" in tool:
                    func = tool["function"]
                    function_declaration = {
                        "name": func["name"],
                        "description": func.get("description", "")
                    }
                    if "parameters" in func:
                        function_declaration["parameters"] = func["parameters"]
                    function_declarations.append(function_declaration)
            
            # Create generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 40)
            }
            
            # Create tool config if functions are provided
            tool_config = None
            if function_declarations:
                self.model_obj._tools = function_declarations
                tool_config = {
                    "function_calling_config": {
                        "mode": "AUTO"
                    }
                }
            
            # Handle chat history if there are multiple messages
            if len(gemini_messages) > 1:
                chat = self.model_obj.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(
                    gemini_messages[-1]["parts"][0]["text"],
                    generation_config=generation_config,
                    tools=tool_config
                )
            else:
                # Single message case
                response = self.model_obj.generate_content(
                    gemini_messages[0]["parts"][0]["text"],
                    generation_config=generation_config,
                    tools=tool_config
                )
            
            # Create response object compatible with OpenAI format
            result = {
                "role": "assistant",
                "content": response.text if hasattr(response, 'text') else ""
            }
            
            # Handle function calls
            if hasattr(response, 'function_call') and response.function_call:
                tool_calls = [{
                    "id": f"call_{hash(str(response.function_call))}",
                    "type": "function",
                    "function": {
                        "name": response.function_call.name,
                        "arguments": json.dumps(response.function_call.args) if hasattr(response.function_call, 'args') else "{}"
                    }
                }]
                result["tool_calls"] = tool_calls
            
            return result
            
        except Exception as e:
            # Check if it's a connection error
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"Connection error with Gemini API: {str(e)}")
                raise ConnectionError(f"Unable to connect to Gemini API: {str(e)}")
            
            # Re-raise other exceptions
            logger.error(f"Error calling Gemini API with tools: {str(e)}")
            raise
