"""
VLLM Provider Implementation

This module implements the VLLM provider for the LLM client.
VLLM is a high-throughput and memory-efficient inference engine for LLMs,
which provides an OpenAI-compatible API server.
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

class VLLMProvider(BaseLLMProvider):
    """
    VLLM API provider implementation.
    
    This provider allows using models hosted with VLLM's OpenAI-compatible server.
    """
    
    # Default available models (these will be overridden if the server provides a models list)
    AVAILABLE_MODELS = [
        "llama-2-7b",
        "llama-2-13b",
        "llama-2-70b",
        "mistral-7b",
        "mixtral-8x7b",
        "phi-2",
        "gemma-7b",
        "gemma-2b"
    ]
    
    # Default model
    DEFAULT_MODEL = "llama-2-7b"
    
    def __init__(self, 
                 api_key: Optional[str] = None,  # Optional for some VLLM deployments
                 model: str = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize the VLLM provider.
        
        Args:
            api_key: API key (optional, some VLLM deployments may require it)
            model: Model identifier to use (defaults to DEFAULT_MODEL)
            api_base: Base URL for API (defaults to http://localhost:8000)
            **kwargs: Additional provider-specific parameters
        """
        # Set API key
        self.api_key = api_key or os.environ.get("VLLM_API_KEY")
        
        # Set model
        self.model = model or self.DEFAULT_MODEL
        
        # Set API base URL
        self.api_base = api_base or os.environ.get("VLLM_API_BASE") or "http://localhost:8000"
        
        # Remove trailing slash if present
        self.api_base = self.api_base.rstrip('/')
        
        # Initialize HTTP client with headers
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.Client(timeout=60.0*3, headers=headers)  # Longer timeout for inference
        
        # Try to fetch available models from VLLM server
        try:
            self._fetch_available_models()
        except Exception as e:
            logger.warning(f"Could not fetch available models from VLLM server: {str(e)}")
            logger.warning("Using default model list. Make sure VLLM server is running.")
        
        logger.debug(f"Initialized VLLM provider with model: {self.model}")
    
    def _fetch_available_models(self):
        """
        Fetch available models from VLLM server.
        Updates the AVAILABLE_MODELS list with models available on the server.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        try:
            response = self.client.get(f"{self.api_base}/v1/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    # Extract model IDs from the response
                    models = [model["id"] for model in data["data"]]
                    if models:
                        # Update available models
                        self.AVAILABLE_MODELS = models
                        logger.debug(f"Updated available models from VLLM: {models}")
        except Exception as e:
            logger.warning(f"Error fetching models from VLLM: {str(e)}")
    
    def check_service_availability(self):
        """
        Check if VLLM service is available and the specified model exists.
        
        Returns:
            tuple: (is_available, available_models, error_message)
                is_available: True if service is available
                available_models: List of available model names
                error_message: Error message if service is not available
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        try:
            # Try to connect to VLLM server
            response = self.client.get(f"{self.api_base}/v1/models", headers=headers)
            
            if response.status_code != 200:
                return False, [], f"Could not connect to VLLM server at {self.api_base}: Status {response.status_code}"
            
            # Get available models
            data = response.json()
            available_models = [model["id"] for model in data.get("data", [])]
            
            if not available_models:
                # VLLM might not have models listed but still work with specific model
                # So we don't treat this as an error
                return True, [], None
            
            # Check if the specified model is available
            if self.model not in available_models:
                return True, available_models, f"Model '{self.model}' not found in VLLM server. Available models: {', '.join(available_models)}"
            
            return True, available_models, None
            
        except Exception as e:
            return False, [], f"Could not connect to VLLM server at {self.api_base}: {str(e)}"
    
    @staticmethod
    def get_installation_instructions():
        """
        Get installation instructions for VLLM.
        
        Returns:
            list: List of installation instruction strings
        """
        return [
            "1. Install VLLM using pip: 'pip install vllm'",
            "2. Start the VLLM server: 'python -m vllm.entrypoints.openai.api_server --model <your-model>'",
            "3. Make sure the server is accessible at the specified API base URL"
        ]
    
    def get_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            str: Provider name
        """
        return "vllm"
    
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
        return VLLMProvider.DEFAULT_MODEL
    
    def call_with_messages(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """
        Call the VLLM API with a list of messages.
        
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
            # Create request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value
            
            # Call the API
            response = self.client.post(
                f"{self.api_base}/v1/chat/completions",
                json=payload
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"VLLM API error: Status {response.status_code}, Response: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse the response
            data = response.json()
            
            # Return the response text
            return data["choices"][0]["message"]["content"]
            
        except httpx.ConnectError as e:
            error_msg = f"Unable to connect to VLLM API at {self.api_base}: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        except Exception as e:
            # Re-raise other exceptions
            logger.error(f"Error calling VLLM API: {str(e)}")
            raise
    
    def call_with_prompt(self, 
                        prompt: str, 
                        temperature: float = 0.2,
                        max_tokens: int = 2000,
                        **kwargs) -> str:
        """
        Call the VLLM API with a single prompt string.
        
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
            # Try to use the completions endpoint first
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value
            
            # Call the API
            response = self.client.post(
                f"{self.api_base}/v1/completions",
                json=payload
            )
            
            # Check for errors
            if response.status_code != 200:
                # If completions endpoint fails, we'll fall back to chat completions
                logger.warning(f"VLLM completions API error, falling back to chat API: {response.status_code}, {response.text}")
                raise Exception("Completions endpoint failed")
            
            # Parse the response
            data = response.json()
            
            # Return the response text
            return data["choices"][0]["text"]
            
        except Exception as e:
            # If there's an error with the completions endpoint, fall back to the chat endpoint
            logger.warning(f"Error using VLLM completions API, falling back to chat API: {str(e)}")
            
            # Convert the prompt to a messages array
            messages = [{"role": "user", "content": prompt}]
            
            # Call the API with the messages
            return self.call_with_messages(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
