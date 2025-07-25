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
import shutil
import tempfile

# Import command forwarder and decorators for remote file operations
from terminal_agent.utils.command_forwarder import forwarder, remote_file_operation

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

try:
    import portalocker
    HAS_LOCK = True
except ImportError:
    HAS_LOCK = False

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
    max_lines = params.get("max_lines", 100)  # Default maximum of 100 lines per read

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

        # Get total line count (optional)
        total_lines = None
        try:
            # Only count total lines for files smaller than 10MB
            if os.path.getsize(file_path) < 10 * 1024 * 1024:  # Only calculate total lines for files smaller than 10MB
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    total_lines = sum(1 for _ in f)
        except Exception as e:
            logger.debug(f"Failed to count total lines: {e}")

        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Skip lines before the specified start line
            if start_line > 1:
                for _ in range(start_line - 1):
                    next(f, None)

            # Calculate how many lines to read
            if end_line is not None:
                lines_to_read = min(end_line - start_line + 1, max_lines)
            else:
                lines_to_read = max_lines

            # Read the specified number of lines
            lines = []
            for i in range(lines_to_read):
                line = next(f, None)
                if line is None:
                    break
                lines.append(line)

            # If no lines were read
            if not lines:
                return f"No content found at line {start_line}"

            # Build content
            content = ''.join(lines)

            # Add line range information
            actual_end_line = start_line + len(lines) - 1
            line_range_info = f"[Lines {start_line}-{actual_end_line}"

            if total_lines:
                line_range_info += f" of {total_lines}]"
            else:
                line_range_info += "]"

            # Check if there are more lines
            next_line = next(f, None)
            if next_line is not None:
                line_range_info += f" (More lines available, continue from line {actual_end_line + 1})"

            # Add line range information to the beginning of content
            content = line_range_info + "\n\n" + content

        return content

    except UnicodeDecodeError:
        return f"Error: '{file_path}' appears to be a binary file and cannot be read as text"
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {str(e)}"

@remote_file_operation
def update_file(params: Dict) -> str:
    """
    Update an existing file by appending content or replacing specific content.

    Modes:
      - append:  Append content to the end of the file
      - replace: Replace specific content occurrences with new content

    Args:
        params: {
            file_path: str,           # required
            mode: str,                # 'append' or 'replace'
            content: str,             # required for append
            old_content: str,         # required for replace
            new_content: str,         # required for replace
            all_occurrences: bool,    # optional, default False
            create_backup: bool       # optional, default True
        }

    Returns:
        str: Success or error message
    """
    # Parameter validation
    file_path = params.get("file_path")
    mode = params.get("mode")
    create_backup = params.get("create_backup", True)

    if not file_path:
        return "Error: Missing required parameter 'file_path'"
    if mode not in ("append", "replace"):
        return "Error: 'mode' must be either 'append' or 'replace'"

    if mode == "append":
        content = params.get("content")
        if not isinstance(content, str):
            return "Error: 'content' must be a string and is required for 'append' mode"
    else:
        old_content = params.get("old_content")
        new_content = params.get("new_content")
        if not isinstance(old_content, str) or not isinstance(new_content, str):
            return "Error: 'old_content' and 'new_content' must be strings and are required for 'replace' mode"
        all_occurrences = params.get("all_occurrences", False)

    # Normalize and check file path
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist"
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a file"

    # Create backup if requested
    if create_backup:
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)

    # Acquire exclusive lock if portalocker is available
    lock_handle = None
    if HAS_LOCK:
        lock_handle = open(file_path, 'r+')
        portalocker.lock(lock_handle, portalocker.LOCK_EX)

    try:
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines(keepends=True)

        if mode == "append":
            # Ensure newline at end
            lines.append(content if content.endswith('\n') else content + '\n')
        else:
            # replace mode
            full_text = ''.join(lines)
            if all_occurrences:
                # Replace all occurrences and split back into lines
                replaced_text = full_text.replace(old_content, new_content)
                lines = replaced_text.splitlines(keepends=True)
            else:
                # Replace first occurrence only
                pos = full_text.find(old_content)
                if pos == -1:
                    return "Error: 'old_content' not found in file"
                # Determine start and end line indices based on character position
                cum_len = 0
                start_idx = end_idx = None
                for idx, line in enumerate(lines):
                    cum_len += len(line)
                    if start_idx is None and cum_len > pos:
                        start_idx = idx
                    if start_idx is not None and cum_len >= pos + len(old_content):
                        end_idx = idx
                        break
                if start_idx is None or end_idx is None:
                    return "Error: Unable to determine replace range"
                # Prepare new content lines
                new_lines = new_content.splitlines(keepends=True)
                # Perform slice replacement
                lines = lines[:start_idx] + new_lines + lines[end_idx+1:]

        # Atomic write to file
        dir_name = os.path.dirname(file_path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name)
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
            tmp_file.writelines(lines)
        os.replace(tmp_path, file_path)

    finally:
        # Release lock if held
        if lock_handle:
            portalocker.unlock(lock_handle)
            lock_handle.close()

    return f"Success: File '{file_path}' updated in '{mode}' mode" + (f" (Backup: {backup_path})" if create_backup else "")


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
