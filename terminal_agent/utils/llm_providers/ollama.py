"""
Ollama Provider Implementation

This module implements the Ollama provider for the LLM client.
Ollama is a local LLM server that provides API-compatible endpoints for running
open-source models like Llama, Mistral, etc.
"""

import os
import json
import httpx
from typing import List, Dict, Any, Optional, Union
import logging
from rich.console import Console

from .base import BaseLLMProvider

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize console
console = Console()

class OllamaProvider(BaseLLMProvider):
    """
    Ollama API provider implementation.
    
    This provider allows using locally hosted models through Ollama's API.
    """
    
    # Default available models (can be extended based on what's installed in Ollama)
    AVAILABLE_MODELS = [
        "llama3",
        "llama2",
        "mistral",
        "mixtral",
        "phi",
        "gemma",
        "codellama",
        "vicuna",
        "orca-mini"
    ]
    
    # Default model
    DEFAULT_MODEL = "llama3"
    
    def __init__(self, 
                 api_key: Optional[str] = None,  # Not used for Ollama but kept for interface consistency
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Ollama provider.
        
        Args:
            api_key: Not used for Ollama, but kept for interface consistency
            model: Model identifier to use (defaults to DEFAULT_MODEL)
            api_base: Base URL for API (defaults to http://localhost:11434)
            **kwargs: Additional provider-specific parameters
        """
        # Set model
        self.model = model or self.DEFAULT_MODEL
        
        # Set API base URL
        self.api_base = api_base or os.environ.get("OLLAMA_API_BASE") or "http://localhost:11434"
        
        # Remove trailing slash if present
        self.api_base = self.api_base.rstrip('/')
        
        # Initialize HTTP client
        self.client = httpx.Client(timeout=60.0)  # Longer timeout for local inference
        
        # Try to fetch available models from Ollama
        try:
            self._fetch_available_models()
        except Exception as e:
            logger.warning(f"Could not fetch available models from Ollama: {str(e)}")
            logger.warning("Using default model list. Make sure Ollama is running.")
        
        logger.debug(f"Initialized Ollama provider with model: {self.model}")
    
    def _fetch_available_models(self):
        """
        Fetch available models from Ollama server.
        Updates the AVAILABLE_MODELS list with models installed on the server.
        """
        try:
            response = self.client.get(f"{self.api_base}/api/tags")
            if response.status_code == 200:
                data = response.json()
                if "models" in data:
                    # Extract model names from the response
                    models = [model["name"] for model in data["models"]]
                    if models:
                        # Update available models
                        self.AVAILABLE_MODELS = models
                        logger.debug(f"Updated available models from Ollama: {models}")
        except Exception as e:
            logger.warning(f"Error fetching models from Ollama: {str(e)}")
    
    def check_service_availability(self):
        """
        Check if Ollama service is available and the specified model exists.
        
        Returns:
            tuple: (is_available, available_models, error_message)
                is_available: True if service is available
                available_models: List of available model names
                error_message: Error message if service is not available
        """
        try:
            # Try to connect to Ollama server
            response = self.client.get(f"{self.api_base}/api/tags")
            
            if response.status_code != 200:
                return False, [], f"Could not connect to Ollama server at {self.api_base}: Status {response.status_code}"
            
            # Get available models
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]
            
            if not available_models:
                return False, [], f"No models found in Ollama server at {self.api_base}"
            
            # Check if the specified model is available
            if self.model not in available_models:
                return True, available_models, f"Model '{self.model}' not found in Ollama server. Available models: {', '.join(available_models)}"
            
            return True, available_models, None
            
        except Exception as e:
            return False, [], f"Could not connect to Ollama server at {self.api_base}: {str(e)}"
    
    @staticmethod
    def get_installation_instructions():
        """
        Get installation instructions for Ollama.
        
        Returns:
            list: List of installation instruction strings
        """
        return [
            "1. Install Ollama from https://ollama.com",
            "2. Start the Ollama server with 'ollama serve'",
            "3. Pull a model with 'ollama pull llama3' (or any other model)"
        ]
    
    def get_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        return "ollama"
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models for this provider.
        
        Returns:
            List[str]: List of model identifiers
        """
        # 如果还没有获取过可用模型，尝试从服务器获取
        if not self.AVAILABLE_MODELS:
            self._fetch_available_models()
        return self.AVAILABLE_MODELS
    
    @staticmethod
    def get_default_model() -> str:
        """
        Get the default model for this provider.
        
        Returns:
            str: Default model identifier
        """
        return OllamaProvider.DEFAULT_MODEL
    
    def call_with_messages(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """
        Call the Ollama API with a list of messages.
        
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
            # Format messages for Ollama (which uses OpenAI-compatible format)
            formatted_messages = []
            for msg in messages:
                # Ensure role is one of: system, user, assistant
                role = msg.get("role", "user")
                if role not in ["system", "user", "assistant"]:
                    role = "user"
                
                formatted_messages.append({
                    "role": role,
                    "content": msg.get("content", "")
                })
            
            # Create request payload
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False  # Don't stream the response
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value
            
            # Call the API
            response = self.client.post(
                f"{self.api_base}/api/chat",
                json=payload
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"Ollama API error: Status {response.status_code}, Response: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse the response
            data = response.json()
            
            # Return the response text
            return data.get("message", {}).get("content", "")
            
        except httpx.ConnectError as e:
            error_msg = f"Unable to connect to Ollama API at {self.api_base}: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        except Exception as e:
            # Re-raise other exceptions
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise
    
    def call_with_prompt(self, 
                        prompt: str, 
                        temperature: float = 0.2,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the Ollama API with a single prompt string.
        
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
            # For simple prompts, we can use the /api/generate endpoint
            # which is more efficient for single-turn interactions
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False  # Don't stream the response
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value
            
            # Call the API
            response = self.client.post(
                f"{self.api_base}/api/generate",
                json=payload
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"Ollama API error: Status {response.status_code}, Response: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse the response
            data = response.json()
            
            # Return the response text
            return data.get("response", "")
            
        except httpx.ConnectError as e:
            error_msg = f"Unable to connect to Ollama API at {self.api_base}: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        except Exception as e:
            # If there's an error with the generate endpoint, fall back to the chat endpoint
            logger.warning(f"Error using Ollama generate API, falling back to chat API: {str(e)}")
            
            # Convert the prompt to a messages array
            messages = [{"role": "user", "content": prompt}]
            
            # Call the API with the messages
            return self.call_with_messages(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
