#!/usr/bin/env python3
"""
Zoekt-based code search tool for Terminal Agent.
Provides a powerful code search capability using the Zoekt search engine.
"""

import os
import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Union

from terminal_agent.react.search.code_search import search_code_with_zoekt
from terminal_agent.react.search.zoekt_server import ZoektServer

# Configure logging
logger = logging.getLogger(__name__)

# Cache for ZoektServer instances
_zoekt_server_cache = {}

# Cache for search results
_search_results_cache = {}

def get_zoekt_server(repo_path: str, language: Optional[str] = None) -> ZoektServer:
    """
    Get or create a ZoektServer instance
    
    Args:
        repo_path: Repository path
        language: Programming language (optional)
        
    Returns:
        ZoektServer instance
    """
    cache_key = f"{repo_path}:{language or 'generic'}"
    if cache_key not in _zoekt_server_cache:
        # Create index path for each repository
        index_path = os.path.join("/tmp/zoekt_indexes", os.path.basename(repo_path))
        server = ZoektServer(language or "generic", repo_path=repo_path, index_path=index_path)
        
        # Setup the index
        server.setup_index(repo_path)
        _zoekt_server_cache[cache_key] = server
    
    return _zoekt_server_cache[cache_key]

def cached_code_search(names: List[str], repo_path: str, language: Optional[str] = None, 
                      num_result: int = 10, verbose: bool = False, max_age: int = 300) -> Any:
    """
    Code search with caching
    
    Args:
        names: List of symbol names to search for
        repo_path: Repository path
        language: Programming language (optional)
        num_result: Maximum number of results per symbol
        verbose: Whether to return detailed information
        max_age: Maximum cache age in seconds
        
    Returns:
        Search results
    """
    cache_key = f"{repo_path}:{language or 'generic'}:{','.join(sorted(names))}:{num_result}:{verbose}"
    
    # Check cache
    if cache_key in _search_results_cache:
        timestamp, results = _search_results_cache[cache_key]
        if time.time() - timestamp < max_age:
            logger.info(f"Using cached search results for {names}")
            return results
    
    # Get ZoektServer instance
    server = get_zoekt_server(repo_path, language)
    
    # Execute search
    results = search_code_with_zoekt(names, server, num_result=num_result, verbose=verbose)
    
    # Update cache
    _search_results_cache[cache_key] = (time.time(), results)
    
    return results

def zoekt_search_tool(query: str) -> str:
    """
    Code search using Zoekt
    
    Args:
        query: JSON string containing the following fields:
            - names: List of identifiers to search for
            - repo_dir: Repository directory (optional, defaults to current directory)
            - language: Programming language (optional)
            - num_results: Maximum number of results per identifier (optional, defaults to 10)
            - verbose: Whether to return detailed information (optional, defaults to False)
            - no_color: Whether to disable colored output (optional, defaults to False)
            - use_cache: Whether to use cache (optional, defaults to True)
            
    Returns:
        JSON string with search results
    """
    try:
        # Parse query
        query_data = json.loads(query) if isinstance(query, str) else query
        
        # Extract parameters
        names = query_data.get("names", [])
        repo_dir = query_data.get("repo_dir", os.getcwd())
        language = query_data.get("language")
        num_results = query_data.get("num_results", 10)
        verbose = query_data.get("verbose", True)
        no_color = query_data.get("no_color", False)
        use_cache = query_data.get("use_cache", True)
        
        if not names:
            return json.dumps({"error": "Missing required parameter 'names'"})
        
        # Ensure repo_dir is an absolute path
        repo_dir = os.path.abspath(repo_dir)
        
        # Set environment variables to control color output
        original_term = os.environ.get('TERM')
        if no_color:
            os.environ['TERM'] = 'dumb'  # Disable colors
        
        # Execute search
        if use_cache:
            search_result = cached_code_search(
                names, repo_dir, language, 
                num_result=num_results, 
                verbose=verbose
            )
        else:
            server = get_zoekt_server(repo_dir, language)
            search_result = search_code_with_zoekt(
                names, server, 
                num_result=num_results, 
                verbose=verbose
            )
        
        # Restore environment variables
        if no_color and original_term is not None:
            os.environ['TERM'] = original_term
        elif no_color and original_term is None:
            os.environ.pop('TERM', None)
    
        
        # 直接返回搜索结果，对 LLM 更友好
        return search_result
        
    except Exception as e:
        logger.error(f"Error in zoekt_search_tool: {e}")
        return {"error": f"Error in code search: {str(e)}"}
