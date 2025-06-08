#!/usr/bin/env python3
"""
Get all references tool for Terminal Agent.
Provides the ability to find all references to a symbol in code.
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, Optional, Union, List

from terminal_agent.react.lsp.multilspy_toolkit import LSPToolKit

# Configure logging
logger = logging.getLogger(__name__)

# Cache for LSPToolKit instances
_lsp_toolkit_cache = {}

def get_lsp_toolkit(repo_dir: str, language: str = None) -> LSPToolKit:
    """
    Get or create an LSPToolKit instance for the given repository directory and language.
    
    Args:
        repo_dir: Repository directory path
        language: Programming language (default: python)
        
    Returns:
        LSPToolKit instance
    """
    if language is None:
        language = "python"
    
    # Normalize language name
    language = language.lower()
    
    # Map language aliases
    language_aliases = {
        "cpp": "c++",
        "js": "javascript",
        "ts": "typescript"
    }
    language = language_aliases.get(language, language)
    
    # Create a cache key based on repo_dir and language
    cache_key = f"{repo_dir}:{language}"
    logger.debug(f"Looking for LSPToolKit with cache key: {cache_key}")
    
    # Check if we already have a cached toolkit for this repo_dir and language
    if cache_key in _lsp_toolkit_cache:
        logger.debug(f"Using cached LSPToolKit instance for {repo_dir} with language {language}")
        return _lsp_toolkit_cache[cache_key]
    
    # Create a new toolkit instance
    logger.debug(f"Creating new LSPToolKit instance for {repo_dir} with language {language}")
    toolkit = LSPToolKit(repo_dir, language)
    
    # Cache the toolkit
    _lsp_toolkit_cache[cache_key] = toolkit
    
    return toolkit

def get_all_references_tool(query: Union[str, Dict]) -> Union[Dict, List]:
    """
    Find all references to a symbol in a code file.
    
    Args:
        query: JSON string or dictionary with the following fields:
            - word: Symbol to find references for
            - relative_path: Path to the file containing the symbol, relative to the repository root
            - line: (optional) Line number (0-based) where the symbol is located
            - verbose: (optional) Whether to include detailed information in the result (default: True)
            - num_results: (optional) Maximum number of results to return (default: 10)
            - context_limit: (optional) Number of lines to show before and after each reference (default: 10)
            
    Returns:
        List of references or a dictionary with error information
    """
    logger.info(f"get_all_references_tool called with query: {query}")
    
    try:
        # Parse query if it's a string, otherwise use it directly
        if isinstance(query, str):
            try:
                query_data = json.loads(query)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON query: {e}")
                return {"error": f"Invalid JSON format: {str(e)}"}
        else:
            query_data = query
            
        logger.debug(f"Parsed query data: {query_data}")
        
        # Extract parameters
        word = query_data.get("word")
        relative_path = query_data.get("relative_path")
        line = query_data.get("line")
        verbose = query_data.get("verbose", True)
        num_results = query_data.get("num_results", 10)
        context_limit = query_data.get("context_limit", 10)
        repo_dir = query_data.get("repo_dir", os.getcwd())
        
        # Validate parameters
        if not word:
            logger.error("Missing required parameter: word")
            return {"error": "Missing required parameter: word"}
        
        if not relative_path:
            logger.error("Missing required parameter: relative_path")
            return {"error": "Missing required parameter: relative_path"}
        
        # Convert relative_path to absolute file_path
        file_path = os.path.abspath(os.path.join(repo_dir, relative_path))
        logger.debug(f"Converted relative path '{relative_path}' to absolute path '{file_path}'")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}
        
        # Check if path is a directory
        if os.path.isdir(file_path):
            logger.error(f"Path is a directory, not a file: {file_path}")
            return {"error": f"Path is a directory, not a file: {file_path}. Please specify a file path."}
        
        # Determine language based on file extension
        _, file_ext = os.path.splitext(relative_path)
        language_map = {
            ".py": "python",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".rs": "rust",
            ".js": "javascript",
            ".ts": "typescript"
        }
        language = query_data.get("language") or language_map.get(file_ext.lower(), "python")
        
        logger.debug(f"Extracted parameters: word={word}, line={line}, "
                    f"relative_path={relative_path}, verbose={verbose}, language={language}")
        
        try:
            # Get LSPToolKit instance
            logger.debug(f"Getting LSPToolKit instance for repo_dir={repo_dir}, language={language}")
            toolkit = get_lsp_toolkit(repo_dir, language)
            
            # Get references
            references_result = toolkit.get_references(
                word, 
                relative_path, 
                line_number=line, 
                verbose=verbose,
                context_limit=context_limit
            )
            
            # Limit the number of results if specified
            if isinstance(references_result, list) and num_results > 0:
                references_result = references_result[:num_results]
            
            logger.debug(f"References result: {references_result}")
            return references_result
            
        except Exception as e:
            error_msg = f"Error getting references: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON query: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

# Example usage
if __name__ == "__main__":
    # Configure console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    
    # Example 1: Find references to a Python function
    python_query = json.dumps({
        "word": "get_lsp_toolkit",
        "relative_path": "terminal_agent/react/tools/get_all_references_tool.py",
        "line": 20,
        "verbose": True
    })
    print("\nExample 1: Python function references\n")
    print(get_all_references_tool(python_query))
    
    # Example 2: Find references with verbose=False
    python_query_simple = json.dumps({
        "word": "get_lsp_toolkit",
        "relative_path": "terminal_agent/react/tools/get_all_references_tool.py",
        "verbose": False
    })
    print("\nExample 2: Simple output format\n")
    print(get_all_references_tool(python_query_simple))
    
    # Example 3: Error handling - file not found
    error_query = json.dumps({
        "word": "nonexistent_function",
        "relative_path": "nonexistent_file.py",
        "verbose": True
    })
    print("\nExample 3: Error handling\n")
    print(get_all_references_tool(error_query))
