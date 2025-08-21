#!/usr/bin/env python3
"""
Read file tool for Terminal Agent.

Provides file reading functionality with offset and limit support.
Follows the exact JSON schema specification for tool registration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


READ_FILE_DESCRIPTION = """
Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. 
It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- For large files, the tool automatically reads in chunks of max_lines (default 100 lines)
- To read next chunk, use the suggested line number as offset in your next call
"""

def read_file(file_path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
    """
    Read content from a file with optional offset and limit.
    
    Args:
        file_path: The absolute path to the file to read (required)
        offset: The line number to start reading from (1-based, optional)
        limit: The number of lines to read (optional)
        
    Returns:
        str: File content or error message
    """
    try:
        # Validate file path
        if not file_path:
            return "Error: file_path is required"
        
        path = Path(file_path)
        if not path.is_absolute():
            return f"Error: file_path must be an absolute path, got: {file_path}"
        
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        if not path.is_file():
            return f"Error: '{file_path}' is not a regular file"
        
        if not os.access(path, os.R_OK):
            return f"Error: File '{file_path}' is not readable"
        
        # Validate offset and limit
        if offset is not None:
            if not isinstance(offset, int) or offset < 1:
                return "Error: offset must be a positive integer"
        
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                return "Error: limit must be a positive integer"
        
        # Set defaults
        start_line = offset if offset is not None else 1
        max_lines = limit if limit is not None else 1000
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Skip lines before offset
                if start_line > 1:
                    for _ in range(start_line - 1):
                        next(f, None)
                
                # Read lines
                lines = []
                lines_read = 0
                
                for line in f:
                    if limit is not None and lines_read >= max_lines:
                        break
                    lines.append(line)
                    lines_read += 1
                
                if not lines:
                    if start_line > 1:
                        return f"No content found at line {start_line}"
                    else:
                        return "File is empty"
                
                content = ''.join(lines)
                
                # Calculate actual range read
                end_line = start_line + lines_read - 1
                
                # Check if there's more content
                try:
                    next_line = next(f, None)
                    has_more = next_line is not None
                except StopIteration:
                    has_more = False
                
                # Format result
                result_parts = []
                result_parts.append(f"📄 File: {file_path}")
                result_parts.append(f"📖 Lines {start_line}-{end_line}")
                
                if has_more and limit is not None:
                    result_parts.append(f"➡️  More content available from line {end_line + 1}")
                
                result_parts.append("")
                result_parts.append("─" * 50)
                result_parts.append("")
                result_parts.append(content)
                
                return '\n'.join(result_parts)
                
        except UnicodeDecodeError:
            return f"Error: '{file_path}' appears to be a binary file and cannot be read as text"
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error in read_file: {e}")
        return f"Error in read_file: {str(e)}"


def read_file_tool_handler(query: Union[str, Dict[str, Any]]) -> str:
    """
    Tool handler function that processes the query according to the JSON schema.
    
    Args:
        query: JSON string or dictionary containing the parameters
        
    Returns:
        str: File content or error message
    """
    try:
        # Parse query if it's a string
        if isinstance(query, str):
            try:
                query_data = json.loads(query)
            except json.JSONDecodeError:
                return "Error: Invalid JSON format in query"
        else:
            query_data = query
        
        # Extract parameters according to schema
        file_path = query_data.get("file_path")
        offset = query_data.get("offset")
        limit = query_data.get("limit")
        
        # Call the main function
        return read_file(file_path=file_path, offset=offset, limit=limit)
        
    except Exception as e:
        logger.error(f"Error in read_file_tool_handler: {e}")
        return f"Error processing query: {str(e)}"


# Tool registration schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read content from a file with optional offset and limit",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read"
                },
                "offset": {
                    "type": "number",
                    "description": "The line number to start reading from. Only provide if the file is too large to read at once"
                },
                "limit": {
                    "type": "number",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once."
                }
            },
            "required": ["file_path"],
            "additionalProperties": False
        }
    }
}

# Export for easy access
TOOL_DEFINITION = TOOL_SCHEMA["function"]


# Example usage
if __name__ == "__main__":
    import tempfile
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a test file
    test_content = """#!/usr/bin/env python3
"Test file for read_file tool"

import os
import sys

def main():
    print("Hello, World!")
    return 42

if __name__ == "__main__":
    main()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        print("=" * 60)
        print("🧪 Testing read_file tool")
        print("=" * 60)
        
        # Test 1: Basic reading
        print("\n1️⃣ Basic file reading:")
        result1 = read_file_tool_handler({"file_path": test_file})
        print(result1)
        
        # Test 2: With offset
        print("\n2️⃣ Reading from line 3:")
        result2 = read_file_tool_handler({"file_path": test_file, "offset": 3})
        print(result2)
        
        # Test 3: With offset and limit
        print("\n3️⃣ Reading lines 3-5:")
        result3 = read_file_tool_handler({"file_path": test_file, "offset": 3, "limit": 3})
        print(result3)
        
        # Test 4: Error handling
        print("\n4️⃣ Error handling - nonexistent file:")
        result4 = read_file_tool_handler({"file_path": "/nonexistent/file.txt"})
        print(result4)
        
    finally:
        # Clean up
        os.unlink(test_file)