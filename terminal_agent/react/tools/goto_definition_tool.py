#!/usr/bin/env python3
"""
Go to definition tool for Terminal Agent.
Provides the ability to find the definition of a symbol in code.
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, Optional, Union

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

def goto_definition_tool(query: Union[str, Dict]) -> str:
    """
    Find the definition of a symbol in a code file.
    
    Args:
        query: JSON string or dictionary with the following fields:
            - word: Symbol to find definition for
            - line: Line number (0-based)
            - relative_path: Path to the file containing the symbol, relative to the repository root
            - verbose: (optional) Whether to include detailed information in the result (default: True)
            
    Returns:
        JSON string with the definition information or error message
    """
    logger.info(f"goto_definition_tool called with query: {query}")
    
    try:
        # Parse query if it's a string, otherwise use it directly
        if isinstance(query, str):
            try:
                query_data = json.loads(query)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON query: {e}")
                return json.dumps({"error": f"Invalid JSON format: {str(e)}"})
        else:
            query_data = query
            
        logger.debug(f"Parsed query data: {query_data}")
        
        # Extract parameters
        word = query_data.get("word")
        line = query_data.get("line", 0)
        relative_path = query_data.get("relative_path")
        verbose = query_data.get("verbose", True)
        repo_dir = os.getcwd()
        #

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
        language = language_map.get(file_ext.lower(), "python")  # Default to Python if extension not recognized
        
        logger.debug(f"Extracted parameters: word={word}, line={line}, "
                    f"relative_path={relative_path}, verbose={verbose}, language={language}")
        
        # Validate parameters
        if not word:
            logger.error("Missing required parameter: word")
            return json.dumps({"error": "Missing required parameter: word"})
        
        if not relative_path:
            logger.error("Missing required parameter: relative_path")
            return json.dumps({"error": "Missing required parameter: relative_path"})
        
        # Convert relative_path to absolute file_path
        file_path = os.path.abspath(os.path.join(repo_dir, relative_path))
        logger.debug(f"Converted relative path '{relative_path}' to absolute path '{file_path}'")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return json.dumps({"error": f"File not found: {file_path}"})
        
        # Ensure repo_dir is an absolute path
        repo_dir = os.path.abspath(repo_dir)
        logger.debug(f"Using repo_dir: '{repo_dir}'")
        
        try:
            # Get LSPToolKit instance
            logger.debug(f"Getting LSPToolKit instance for repo_dir={repo_dir}, language={language}")
            toolkit = get_lsp_toolkit(repo_dir, language)
            # Convert absolute file_path back to relative path for LSPToolKit
            relative_file_path = os.path.relpath(file_path, repo_dir)
            
            try:
                # 直接返回 get_definition 的结果
                definition_result = toolkit.get_definition(word, relative_file_path, line=line, verbose=verbose)
                logger.debug(f"Definition result: {definition_result}")
                return definition_result
                
            except Exception as e:
                logger.error(f"Error in get_definition: {str(e)}")
                logger.error(traceback.format_exc())
                return {"error": f"Error getting definition: {str(e)}"}
                
        except Exception as e:
            error_msg = f"Error in goto_definition_tool: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON query: {str(e)}"
        logger.error(error_msg)
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
    
    # Example 1: Find definition of a Python function
    python_query = json.dumps({
        "word": "goto_definition_tool",
        "line": 64,  # Line where the function is defined
        "relative_path": "terminal_agent/react/tools/goto_definition_tool.py",
        "verbose": True
    })
    print("\nExample 1: Python function definition\n")
    print(goto_definition_tool(python_query))
    
    # Example 2: Find definition with verbose=False
    python_query_simple = json.dumps({
        "word": "goto_definition_tool",
        "line": 64,
        "relative_path": "terminal_agent/react/tools/goto_definition_tool.py",
        "verbose": False
    })
    print("\nExample 2: Simple output format\n")
    print(goto_definition_tool(python_query_simple))
    
    # Example 3: Error handling - file not found
    error_query = json.dumps({
        "word": "nonexistent_function",
        "line": 10,
        "relative_path": "nonexistent_file.py",
        "verbose": True
    })
    print("\nExample 3: Error handling\n")
    print(goto_definition_tool(error_query))
    
    # Example 4: Cross-file definition lookup (Go example)
    go_query = json.dumps({
        "word": "StringUtils",
        "line": 8,
        "relative_path": "tests/go_test_project/utils/stringutils.go",
        "verbose": True
    })
    print("\nExample 4: Go cross-file definition lookup\n")
    print(goto_definition_tool(go_query))
    
    # Example 5: Custom query
    custom_query = json.dumps({
        "word": "YourSymbol",
        "line": 10,
        "relative_path": "path/to/your/file.py",
        "verbose": True
    })
    print("\nExample 5: Custom query (commented out to avoid errors)\n")
    # Uncomment the line below to run your custom query
    # print(goto_definition_tool(custom_query))
