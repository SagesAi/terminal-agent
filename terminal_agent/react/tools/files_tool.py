#!/usr/bin/env python3
"""
File operations tool for Terminal Agent.
Provides a set of functions for file management and operations.
Supports both local and remote file operations.
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path

# Import command forwarder and decorators for remote file operations
from terminal_agent.utils.command_forwarder import forwarder, remote_file_operation

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

@remote_file_operation
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
        
        # Local file operations
        # Check if file exists
        if os.path.exists(file_path) and not overwrite:
            return f"Error: File '{file_path}' already exists. Use 'overwrite: true' to overwrite"
        
        # Create parent directory (if it doesn't exist)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write file content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully created file: {file_path}"
        
    except Exception as e:
        logger.error(f"Error creating file: {e}")
        return f"Error creating file: {str(e)}"

@remote_file_operation
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
    start_line = params.get("start_line", 1)
    end_line = params.get("end_line", None)
    
    if not file_path:
        return "Error: Missing required parameter 'file_path'"
    
    try:
        # Clean and normalize the path
        file_path = clean_path(file_path)
        
        # Check if the file should be excluded
        if should_exclude_file(file_path):
            return f"Error: Cannot read file '{file_path}' - access to this file type is restricted"
        
        # Local file operations
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        # Check if it's a file
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file"
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            if start_line > 1 or end_line is not None:
                # Read specific lines
                lines = f.readlines()
                
                # Adjust to 1-based indexing
                start_idx = start_line - 1
                end_idx = end_line if end_line is None else end_line - 1
                
                # Validate line range
                if start_idx < 0:
                    start_idx = 0
                if end_idx is not None and end_idx >= len(lines):
                    end_idx = len(lines) - 1
                
                # Get requested lines
                if end_idx is None:
                    content = ''.join(lines[start_idx:])
                else:
                    content = ''.join(lines[start_idx:end_idx+1])
            else:
                # Read entire file
                content = f.read()
        
        return content
        
    except UnicodeDecodeError:
        return f"Error: '{file_path}' appears to be a binary file and cannot be read as text"
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {str(e)}"

@remote_file_operation
def update_file(params: Dict) -> str:
    """
    Update an existing file with new content, replace specific text, or append content.
    
    Args:
        params: Dictionary containing:
            - file_path: Path to the file to update
            - content (optional): New content for the file
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
        
    # Validate parameters based on mode
    if mode == "write" and content is None:
        return "Error: 'content' parameter is required for 'write' mode"
    elif mode == "append" and content is None:
        return "Error: 'content' parameter is required for 'append' mode"
    elif mode == "replace" and (old_str is None or new_str is None):
        return "Error: 'old_str' and 'new_str' parameters are required for 'replace' mode"
    
    try:
        # Clean and normalize the path
        file_path = clean_path(file_path)
        
        # Check if the file should be excluded
        if should_exclude_file(file_path):
            return f"Error: Cannot update file '{file_path}' - access to this file type is restricted"
        
        # Local file operations
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist"
        
        # Check if it's a file
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file"
        
        # Read current content
        with open(file_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Update content based on mode
        if mode == "write":
            new_content = content
        elif mode == "append":
            new_content = current_content + content
        elif mode == "replace":
            new_content = current_content.replace(old_str, new_str)
        else:
            return f"Error: Invalid mode '{mode}'. Supported modes: write, append, replace"
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return f"Success: File '{file_path}' updated successfully"
        
    except UnicodeDecodeError:
        return f"Error: '{file_path}' appears to be a binary file and cannot be updated as text"
    except Exception as e:
        logger.error(f"Error updating file: {e}")
        return f"Error updating file: {str(e)}"

@remote_file_operation
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
        
        # Local file operations
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: '{file_path}' does not exist"
        
        # Determine if it's a file or directory
        if os.path.isfile(file_path):
            # Delete file
            os.remove(file_path)
            return f"Success: File '{file_path}' deleted successfully"
        else:
            # Delete directory
            if recursive:
                shutil.rmtree(file_path)
                return f"Success: Directory '{file_path}' and its contents deleted successfully"
            else:
                try:
                    os.rmdir(file_path)
                    return f"Success: Directory '{file_path}' deleted successfully"
                except OSError as e:
                    if "not empty" in str(e).lower():
                        return f"Error: Directory '{file_path}' is not empty. Use 'recursive: true' to delete non-empty directories"
                    else:
                        return f"Error: '{file_path}' is neither a file nor a directory"
        
    except Exception as e:
        logger.error(f"Error deleting file or directory: {e}")
        return f"Error deleting file or directory: {str(e)}"

@remote_file_operation
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
    directory_path = params.get("directory_path")
    include_hidden = params.get("include_hidden", False)
    
    if not directory_path:
        return "Error: Missing required parameter 'directory_path'"
    
    try:
        # Clean and normalize the path
        directory_path = clean_path(directory_path)
        
        # Local file operations
        # Check directory exists
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist"
        
        # Check if it's a directory
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory"
        
        # List directory contents
        contents = []
        for item in os.listdir(directory_path):
            # Skip hidden files (if needed)
            if not include_hidden and item.startswith('.'):
                continue
                
            item_path = os.path.join(directory_path, item)
            item_type = "directory" if os.path.isdir(item_path) else "file"
            item_size = os.path.getsize(item_path) if os.path.isfile(item_path) else None
            
            contents.append({
                "name": item,
                "type": item_type,
                "size": item_size,
                "path": item_path
            })
        
        # Sort by type and name
        contents.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"]))
        
        result = {
            "directory": directory_path,
            "contents": contents,
            "count": len(contents)
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error listing directory: {e}")
        return f"Error listing directory: {str(e)}"

@remote_file_operation
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
        
        # Local file operations
        # Check if file exists
        exists = os.path.exists(file_path)
        
        # Get additional information (if exists)
        result = {
            "exists": exists,
            "path": file_path,
            "remote": forwarder.remote_enabled if hasattr(forwarder, 'remote_enabled') else False
        }
        
        if exists:
            result["is_file"] = os.path.isfile(file_path)
            result["is_directory"] = os.path.isdir(file_path)
            
            if result["is_file"]:
                result["size"] = os.path.getsize(file_path)
                result["extension"] = os.path.splitext(file_path)[1]
        
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
