#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool for retrieving repository directory structure
"""

import os
import json
import logging
import re
from pathlib import Path
import sys
from typing import Dict, Union

# Add project root directory to Python path
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, repo_dir)

from terminal_agent.react.search.get_repo_struct import DisplayablePath, visualize_tree

# Configure logger for current module only, without affecting global configuration
logger = logging.getLogger(__name__)

def get_folder_structure_tool(query_str: Union[str, Dict]) -> str:
    """
    Tool for retrieving repository directory structure.
    
    Args:
        query_str: JSON string or dictionary containing query parameters, supporting the following parameters:
            - repo_dir (str): Repository root directory path, can be relative or absolute path
            - max_depth (int, optional): Maximum traversal depth, default is 3
            - exclude_dirs (list, optional): List of directories to exclude
            - exclude_files (list, optional): List of file patterns to exclude
            - pattern (str, optional): File name matching pattern (regular expression)
    
    Returns:
        str: String representation of the directory structure tree
    """
    logger.info(f"get_folder_structure_tool called with query: {query_str}")
    
    try:
        # Parse query parameters, if it's a string parse as JSON, otherwise use directly
        if isinstance(query_str, str):
            try:
                query = json.loads(query_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON query: {e}")
                return json.dumps({"error": f"Invalid JSON format: {str(e)}"}, ensure_ascii=False)
        else:
            query = query_str
        
        # Extract parameters
        repo_dir_path = query.get("repo_dir", repo_dir)
        max_depth = query.get("max_depth", 3)
        exclude_dirs = query.get("exclude_dirs", [".git", "__pycache__", "node_modules", ".idea", ".vscode", "venv", "env", ".pytest_cache"])
        exclude_files = query.get("exclude_files", ["*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.class"])
        pattern = query.get("pattern", None)
        
        logger.debug(f"Parameters: repo_dir={repo_dir_path}, max_depth={max_depth}, "
                    f"exclude_dirs={exclude_dirs}, exclude_files={exclude_files}, pattern={pattern}")
        
        # Ensure path is absolute
        if not os.path.isabs(repo_dir_path):
            # Use current working directory for relative paths instead of project root
            repo_dir_path = os.path.abspath(repo_dir_path)
        
        if not os.path.exists(repo_dir_path):
            return json.dumps({"error": f"Directory not found: {repo_dir_path}"})
        
        # Custom filter criteria
        def criteria(path):
            # Exclude directories
            if path.is_dir() and path.name in exclude_dirs:
                return False
            
            # Exclude files
            if path.is_file():
                # Check if file matches exclusion pattern
                for exclude_pattern in exclude_files:
                    if re.match(exclude_pattern.replace("*", ".*"), path.name):
                        return False
                
                # If pattern is specified, check if file matches
                if pattern and not re.search(pattern, path.name):
                    return False
            
            return True
        
        # Use visualize_tree to generate directory structure
        tree_output = visualize_tree(
            dir_path=repo_dir_path,
            level=max_depth,
            limit_to_directories=False
        )
        
        return tree_output
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing query JSON: {e}")
        return json.dumps({"error": f"Invalid JSON format: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in get_folder_structure_tool: {e}")
        return json.dumps({"error": f"Error generating folder structure: {str(e)}"})

if __name__ == "__main__":
    # Test example
    test_query = json.dumps({
        "repo_dir": ".",
        "max_depth": 2,
        "exclude_dirs": [".git", "__pycache__"],
        "exclude_files": ["*.pyc"],
        "pattern": None
    })
    
    result = get_folder_structure_tool(test_query)
    print(result)
