#!/usr/bin/env python3
"""
File operations tool for Terminal Agent.
Provides a set of functions for file management and operations.
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Configure logging
logger = logging.getLogger(__name__)
console = Console()

# Define excluded files and directories for safety
EXCLUDED_FILES = [".env", "config.json", "credentials.json", "secret.key"]
EXCLUDED_DIRS = [".git", "node_modules", "__pycache__", ".venv", "venv"]
EXCLUDED_EXT = [".pem", ".key", ".crt", ".p12", ".pfx"]

def should_exclude_file(file_path: str) -> bool:
    """
    Check if a file should be excluded based on path, name, or extension.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file should be excluded, False otherwise
    """
    path = Path(file_path)
    
    # Check if any part of the path is in excluded directories
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return True
    
    # Check filename
    if path.name in EXCLUDED_FILES:
        return True
    
    # Check extension
    if path.suffix in EXCLUDED_EXT:
        return True
    
    return False

def clean_path(path: str) -> str:
    """
    Clean and normalize a file path.
    
    Args:
        path: The path to clean
        
    Returns:
        str: Cleaned and normalized path
    """
    # Convert to absolute path if not already
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    # Normalize path (resolve .. and . components)
    return os.path.normpath(path)

def files_tool(query: str) -> str:
    """
    File operations tool for Terminal Agent.
    
    This tool provides file management capabilities including:
    - Creating files
    - Reading file contents
    - Updating files
    - Deleting files
    - Listing directory contents
    - Checking if files exist
    - Comparing files
    
    Args:
        query: JSON string containing the operation and parameters
               Format: {"operation": "operation_name", ...parameters}
               
    Returns:
        str: Result of the operation as a string
    """
    try:
        # Parse the query as JSON if it's a string, or use it directly if it's already a dict
        try:
            if isinstance(query, dict):
                query_data = query
            else:
                query_data = json.loads(query)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format in query"
        
        # Extract the operation
        operation = query_data.get("operation", "").lower()
        
        # Dispatch to the appropriate operation
        if operation == "create_file":
            return create_file(query_data)
        elif operation == "read_file":
            return read_file(query_data)
        elif operation == "update_file":
            return update_file(query_data)
        elif operation == "delete_file":
            return delete_file(query_data)
        elif operation == "list_directory":
            return list_directory(query_data)
        elif operation == "file_exists":
            return file_exists(query_data)
        elif operation == "compare_files":
            return compare_files(query_data)
        else:
            return f"Error: Unknown operation '{operation}'. Supported operations: create_file, read_file, update_file, delete_file, list_directory, file_exists, compare_files"
            
    except Exception as e:
        logger.error(f"Error in files_tool: {e}")
        return f"Error in files operation: {str(e)}"

def create_file(params: Dict) -> str:
    """
    Create a new file with the specified content.
    
    Args:
        params: Dictionary containing:
            - file_path: Path to the file to create
            - content: Content to write to the file
            - overwrite (optional): Whether to overwrite if file exists
            
    Returns:
        str: Success or error message
    """
    file_path = params.get("file_path")
    content = params.get("content", "")
    overwrite = params.get("overwrite", False)
    
    if not file_path:
        return "Error: Missing required parameter 'file_path'"
    
    try:
        # Clean and normalize the path
        file_path = clean_path(file_path)
        
        # Check if the file should be excluded
        if should_exclude_file(file_path):
            return f"Error: Cannot create file '{file_path}' - access to this file type is restricted"
        
        # Check if file exists and handle overwrite
        if os.path.exists(file_path) and not overwrite:
            return f"Error: File '{file_path}' already exists. Use 'overwrite: true' to replace it or use update_file operation"
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Success: File '{file_path}' created successfully"
        
    except Exception as e:
        logger.error(f"Error creating file: {e}")
        return f"Error creating file: {str(e)}"

def read_file(params: Dict) -> str:
    """
    Read the content of a file.
    
    Args:
        params: Dictionary containing:
            - file_path: Path to the file to read
            - start_line (optional): Starting line number (1-based indexing)
            - end_line (optional): Ending line number (inclusive)
            
    Returns:
        str: File content or error message
    """
    file_path = params.get("file_path")
    start_line = params.get("start_line")
    end_line = params.get("end_line")
    
    if not file_path:
        return "Error: Missing required parameter 'file_path'"
    
    try:
        # Clean and normalize the path
        file_path = clean_path(file_path)
        
        # Check if the file should be excluded
        if should_exclude_file(file_path):
            return f"Error: Cannot read file '{file_path}' - access to this file type is restricted"
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        # Check if it's a file
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file"
        
        # Read the file
        if start_line is not None or end_line is not None:
            # Read specific line range
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Convert to 0-based indexing for Python
            start_idx = (int(start_line) - 1) if start_line is not None else 0
            end_idx = int(end_line) if end_line is not None else len(lines)
            
            # Validate line range
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(lines):
                end_idx = len(lines)
            
            # Extract the requested lines
            content = ''.join(lines[start_idx:end_idx])
        else:
            # Read the entire file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        return content
        
    except UnicodeDecodeError:
        return f"Error: '{file_path}' appears to be a binary file and cannot be read as text"
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {str(e)}"

def update_file(params: Dict) -> str:
    """
    Update an existing file with new content, replace specific text, or append content.
    
    Args:
        params: Dictionary containing:
            - file_path: Path to the file to update
            - content: New content (for full rewrite or append)
            - old_str (optional): Text to replace
            - new_str (optional): Replacement text
            - mode (optional): Update mode - 'write' (default), 'append', or 'replace'
            
    Returns:
        str: Success or error message
    """
    file_path = params.get("file_path")
    content = params.get("content")
    old_str = params.get("old_str")
    new_str = params.get("new_str")
    mode = params.get("mode", "write").lower()
    
    if not file_path:
        return "Error: Missing required parameter 'file_path'"
    
    # Validate parameters
    if mode == "replace" and (old_str is None or new_str is None):
        return "Error: Both 'old_str' and 'new_str' must be provided for replace mode"
    elif mode in ["write", "append"] and content is None:
        return f"Error: 'content' must be provided for {mode} mode"
    
    try:
        # Clean and normalize the path
        file_path = clean_path(file_path)
        
        # Check if the file should be excluded
        if should_exclude_file(file_path):
            return f"Error: Cannot update file '{file_path}' - access to this file type is restricted"
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        # Check if it's a file
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file"
        
        # Full rewrite
        if mode == "write":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Success: File '{file_path}' completely updated"
        
        # String replacement
        elif mode == "replace":
            # Read the current content
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            # Check if old_str exists
            if old_str not in current_content:
                return f"Error: Text to replace not found in '{file_path}'"
            
            # Count occurrences
            occurrences = current_content.count(old_str)
            if occurrences > 1:
                return f"Warning: '{old_str}' appears {occurrences} times in the file. All occurrences will be replaced."
            
            # Perform replacement
            new_content = current_content.replace(old_str, new_str)
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return f"Success: Replaced '{old_str}' with '{new_str}' in '{file_path}'"
        
        # Append content
        elif mode == "append":
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Success: Content appended to '{file_path}'"
        
    except Exception as e:
        logger.error(f"Error updating file: {e}")
        return f"Error updating file: {str(e)}"

def delete_file(params: Dict) -> str:
    """
    Delete a file or directory.
    
    Args:
        params: Dictionary containing:
            - file_path: Path to the file or directory to delete
            - recursive (optional): Whether to recursively delete directories
            
    Returns:
        str: Success or error message
    """
    file_path = params.get("file_path")
    recursive = params.get("recursive", False)
    
    if not file_path:
        return "Error: Missing required parameter 'file_path'"
    
    try:
        # Clean and normalize the path
        file_path = clean_path(file_path)
        
        # Check if the file should be excluded
        if should_exclude_file(file_path):
            return f"Error: Cannot delete '{file_path}' - access to this file type is restricted"
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: '{file_path}' does not exist"
        
        # Handle directory
        if os.path.isdir(file_path):
            if not recursive:
                return f"Error: '{file_path}' is a directory. Use 'recursive: true' to delete directories"
            
            import shutil
            shutil.rmtree(file_path)
            return f"Success: Directory '{file_path}' and its contents deleted"
        
        # Handle file
        os.remove(file_path)
        return f"Success: File '{file_path}' deleted"
        
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return f"Error deleting file: {str(e)}"

def list_directory(params: Dict) -> str:
    """
    List the contents of a directory.
    
    Args:
        params: Dictionary containing:
            - directory_path: Path to the directory to list
            - include_hidden (optional): Whether to include hidden files
            
    Returns:
        str: JSON string with directory contents or error message
    """
    directory_path = params.get("directory_path", ".")
    include_hidden = params.get("include_hidden", False)
    
    try:
        # Clean and normalize the path
        directory_path = clean_path(directory_path)
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist"
        
        # Check if it's a directory
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory"
        
        # List contents
        contents = []
        for item in os.listdir(directory_path):
            # Skip hidden files if not included
            if not include_hidden and item.startswith('.'):
                continue
            
            item_path = os.path.join(directory_path, item)
            
            # Skip excluded files
            if should_exclude_file(item_path):
                continue
            
            # Get item info
            is_dir = os.path.isdir(item_path)
            size = os.path.getsize(item_path) if os.path.isfile(item_path) else 0
            
            contents.append({
                "name": item,
                "is_directory": is_dir,
                "size": size,
                "path": item_path
            })
        
        # Sort: directories first, then files
        contents.sort(key=lambda x: (not x["is_directory"], x["name"]))
        
        return json.dumps(contents, indent=2)
        
    except Exception as e:
        logger.error(f"Error listing directory: {e}")
        return f"Error listing directory: {str(e)}"

def file_exists(params: Dict) -> str:
    """
    Check if a file or directory exists.
    
    Args:
        params: Dictionary containing:
            - file_path: Path to check
            
    Returns:
        str: JSON string with existence information
    """
    file_path = params.get("file_path")
    
    if not file_path:
        return "Error: Missing required parameter 'file_path'"
    
    try:
        # Clean and normalize the path
        file_path = clean_path(file_path)
        
        exists = os.path.exists(file_path)
        is_file = os.path.isfile(file_path) if exists else False
        is_dir = os.path.isdir(file_path) if exists else False
        
        result = {
            "exists": exists,
            "is_file": is_file,
            "is_directory": is_dir,
            "path": file_path
        }
        
        if exists and is_file:
            result["size"] = os.path.getsize(file_path)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error checking file existence: {e}")
        return f"Error checking file existence: {str(e)}"

def compare_files(params: Dict) -> str:
    """
    Compare the contents of two files and return the differences.
    
    Args:
        params: Dictionary containing:
            - file_path1: Path to the first file
            - file_path2: Path to the second file
            - context_lines (optional): Number of context lines to show (default: 3)
            
    Returns:
        str: Differences between the files or error message
    """
    file_path1 = params.get("file_path1")
    file_path2 = params.get("file_path2")
    context_lines = params.get("context_lines", 3)
    
    if not file_path1 or not file_path2:
        return "Error: Missing required parameters 'file_path1' and/or 'file_path2'"
    
    try:
        # Clean and normalize the paths
        file_path1 = clean_path(file_path1)
        file_path2 = clean_path(file_path2)
        
        # Check if the files should be excluded
        if should_exclude_file(file_path1):
            return f"Error: Cannot read file '{file_path1}' - access to this file type is restricted"
        if should_exclude_file(file_path2):
            return f"Error: Cannot read file '{file_path2}' - access to this file type is restricted"
        
        # Check if files exist
        if not os.path.exists(file_path1):
            return f"Error: File '{file_path1}' does not exist"
        if not os.path.exists(file_path2):
            return f"Error: File '{file_path2}' does not exist"
        
        # Check if they are files
        if not os.path.isfile(file_path1):
            return f"Error: '{file_path1}' is not a file"
        if not os.path.isfile(file_path2):
            return f"Error: '{file_path2}' is not a file"
        
        # Read the files
        try:
            with open(file_path1, 'r', encoding='utf-8') as f:
                content1 = f.readlines()
        except UnicodeDecodeError:
            return f"Error: '{file_path1}' appears to be a binary file and cannot be compared as text"
            
        try:
            with open(file_path2, 'r', encoding='utf-8') as f:
                content2 = f.readlines()
        except UnicodeDecodeError:
            return f"Error: '{file_path2}' appears to be a binary file and cannot be compared as text"
        
        # Compare the files using difflib
        import difflib
        diff = difflib.unified_diff(
            content1, 
            content2,
            fromfile=file_path1,
            tofile=file_path2,
            n=int(context_lines)
        )
        
        # Format the differences
        diff_text = ''.join(diff)
        if not diff_text:
            return f"Files '{file_path1}' and '{file_path2}' are identical"
        
        return diff_text
        
    except Exception as e:
        logger.error(f"Error comparing files: {e}")
        return f"Error comparing files: {str(e)}"
