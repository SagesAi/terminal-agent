#!/usr/bin/env python3
"""
Write file tool for Terminal Agent.

Provides file writing functionality with absolute path validation.
Follows the exact JSON schema specification for tool registration.
"""

import os
import json
import logging
from typing import Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

WRITE_FILE_DESCRIPTION = """
Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.
"""


def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file with absolute path validation.
    
    Args:
        file_path: The absolute path to the file to write (must be absolute, not relative)
        content: The content to write to the file
        
    Returns:
        str: Success message or error message
    """
    try:
        # Validate file path
        if not file_path:
            return "❌ Error: file_path is required"
        
        path = Path(file_path)
        
        # Ensure absolute path
        if not path.is_absolute():
            return f"❌ Error: file_path must be an absolute path, got: {file_path}"
        
        # Create parent directories if they don't exist
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"❌ Error: Failed to create parent directories: {e}"
        
        # Check if parent directory is writable
        if not os.access(path.parent, os.W_OK):
            return f"❌ Error: Directory '{path.parent}' is not writable"
        
        # Check if file exists and is not writable
        if path.exists() and not os.access(path, os.W_OK):
            return f"❌ Error: File '{file_path}' is not writable"
        
        # Write content to file
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Get file info for confirmation
            file_size = path.stat().st_size
            is_new = not path.exists() or file_size == 0
            
            action = "created" if is_new else "updated"
            
            return f"✅ Successfully {action} file: {file_path}\n📄 Size: {file_size} bytes"
            
        except UnicodeEncodeError as e:
            return f"❌ Error: Unicode encoding error: {e}"
        except IOError as e:
            return f"❌ Error: I/O error while writing file: {e}"
        except Exception as e:
            return f"❌ Error: Failed to write file: {e}"
            
    except Exception as e:
        logger.error(f"Error in write_file: {e}")
        return f"❌ Error in write_file: {str(e)}"


def write_file_tool_handler(query: Union[str, Dict[str, Any]]) -> str:
    """
    Tool handler function that processes the query according to the JSON schema.
    
    Args:
        query: JSON string or dictionary containing the parameters
        
    Returns:
        str: Success message or error message
    """
    try:
        # Parse query if it's a string
        if isinstance(query, str):
            try:
                query_data = json.loads(query)
            except json.JSONDecodeError:
                return "❌ Error: Invalid JSON format in query"
        else:
            query_data = query
        
        # Extract required parameters
        file_path = query_data.get("file_path")
        content = query_data.get("content")
        
        # Validate required parameters
        if not file_path:
            return "❌ Error: file_path is required"
        if content is None:
            return "❌ Error: content is required"
        
        # Call the main function
        return write_file(file_path=file_path, content=content)
        
    except Exception as e:
        logger.error(f"Error in write_file_tool_handler: {e}")
        return f"❌ Error processing query: {str(e)}"


# Tool registration schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": WRITE_FILE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative)"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["file_path", "content"],
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
    
    print("=" * 60)
    print("🧪 Testing write_file tool")
    print("=" * 60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_write.txt")
        
        # Test 1: Basic file creation
        print("\n1️⃣ Creating new file:")
        content1 = """Hello, World!
This is a test file created by the write_file tool.
It has multiple lines and special characters: àáâãäåæçèéêë"""
        
        result1 = write_file_tool_handler({
            "file_path": test_file,
            "content": content1
        })
        print(result1)
        
        # Verify file was created
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                print("📖 File content:")
                print(f.read())
        
        # Test 2: Updating existing file
        print("\n2️⃣ Updating existing file:")
        content2 = "Updated content with new information"
        
        result2 = write_file_tool_handler({
            "file_path": test_file,
            "content": content2
        })
        print(result2)
        
        # Test 3: Error handling - relative path
        print("\n3️⃣ Error handling - relative path:")
        result3 = write_file_tool_handler({
            "file_path": "relative/path.txt",
            "content": "This should fail"
        })
        print(result3)
        
        # Test 4: Error handling - missing parameters
        print("\n4️⃣ Error handling - missing parameters:")
        result4 = write_file_tool_handler({
            "file_path": test_file
            # Missing content parameter
        })
        print(result4)
        
        # Test 5: Create file in nested directory
        print("\n5️⃣ Creating file in nested directory:")
        nested_file = os.path.join(temp_dir, "nested", "deep", "file.txt")
        content5 = "File in nested directory"
        
        result5 = write_file_tool_handler({
            "file_path": nested_file,
            "content": content5
        })
        print(result5)
        
        # Verify nested file was created
        if os.path.exists(nested_file):
            print("✅ Nested file created successfully")