#!/usr/bin/env python3
"""
Edit File Tool for Terminal Agent.

Provides file editing functionality through string replacement.
Follows the exact JSON schema specification for tool registration.
"""

import os
import json
import logging
from typing import Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


EDIT_FILE_DESCRIPTION = """
Performs exact string replacements in files. 

Usage:
- You must use your `read_file` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. 
- When editing text from `read_file` tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`. 
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.
"""


def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """
    Edit a file by replacing strings with exact matching.
    
    Args:
        file_path: The absolute path to the file to modify (required)
        old_string: The text to replace (required)
        new_string: The text to replace it with (must be different from old_string)
        replace_all: Replace all occurrences of old_string (default false)
        
    Returns:
        str: Success message with replacement details or error message
    """
    try:
        # Validate parameters
        if not file_path:
            return "❌ Error: file_path is required"
        
        if old_string is None:
            return "❌ Error: old_string is required"
        
        if new_string is None:
            return "❌ Error: new_string is required"
        
        if old_string == new_string:
            return "❌ Error: old_string and new_string must be different"
        
        # Validate file path
        path = Path(file_path)
        if not path.is_absolute():
            return f"❌ Error: file_path must be an absolute path, got: {file_path}"
        
        if not path.exists():
            return f"❌ Error: File '{file_path}' does not exist"
        
        if not path.is_file():
            return f"❌ Error: '{file_path}' is not a regular file"
        
        if not os.access(path, os.R_OK):
            return f"❌ Error: File '{file_path}' is not readable"
        
        if not os.access(path, os.W_OK):
            return f"❌ Error: File '{file_path}' is not writable"
        
        # Read file content
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"❌ Error: '{file_path}' appears to be a binary file and cannot be edited as text"
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"
        
        # Count occurrences before replacement
        occurrences = content.count(old_string)
        if occurrences == 0:
            return f"❌ Error: old_string not found in file"
        
        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements_made = occurrences
        else:
            # Replace only the first occurrence
            new_content = content.replace(old_string, new_string, 1)
            replacements_made = 1
        
        # Check if any changes were made
        if new_content == content:
            return "❌ Error: No changes were made (old_string not found or identical to new_string)"
        
        # Write updated content back to file
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            return f"❌ Error writing file: {str(e)}"
        
        # Calculate changes
        lines_changed = len(new_content.splitlines()) - len(content.splitlines())
        
        # Format success message
        result_parts = []
        result_parts.append(f"✅ Successfully edited file: {file_path}")
        result_parts.append(f"📄 Replacements made: {replacements_made}")
        result_parts.append(f"📊 Total occurrences found: {occurrences}")
        
        if replacements_made < occurrences and not replace_all:
            result_parts.append(f"ℹ️  Use replace_all=true to replace all {occurrences} occurrences")
        
        if lines_changed != 0:
            result_parts.append(f"📏 Lines changed: {lines_changed:+d}")
        
        # Show preview of changes
        if len(old_string) <= 100 and len(new_string) <= 100:
            result_parts.append("")
            result_parts.append("─" * 50)
            result_parts.append("🔄 Changes:")
            result_parts.append(f"From: {repr(old_string)}")
            result_parts.append(f"To:   {repr(new_string)}")
            result_parts.append("─" * 50)
        
        return '\n'.join(result_parts)
        
    except Exception as e:
        logger.error(f"Error in edit_file: {e}")
        return f"❌ Error in edit_file: {str(e)}"


def edit_file_tool_handler(query: Union[str, Dict[str, Any]]) -> str:
    """
    Tool handler function that processes the query according to the JSON schema.
    
    Args:
        query: JSON string or dictionary containing the parameters
        
    Returns:
        str: Success message with replacement details or error message
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
        
        # Extract parameters according to schema
        file_path = query_data.get("file_path")
        old_string = query_data.get("old_string")
        new_string = query_data.get("new_string")
        replace_all = query_data.get("replace_all", False)
        
        # Call the main function
        return edit_file(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all
        )
        
    except Exception as e:
        logger.error(f"Error in edit_file_tool_handler: {e}")
        return f"❌ Error processing query: {str(e)}"


# Tool registration schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Edit a file by replacing strings with exact matching",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify"
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)"
                },
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace all occurrences of old_string (default false)"
                }
            },
            "required": ["file_path", "old_string", "new_string"],
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
    print("🧪 Testing edit_file tool")
    print("=" * 60)
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        test_content = '''#!/usr/bin/env python3
"""Test file for edit_file tool."""

import os
import sys

def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return 42

if __name__ == "__main__":
    hello_world()
'''
        f.write(test_content)
        test_file = f.name
    
    try:
        # Test 1: Single replacement
        print("\n1️⃣ Single replacement:")
        result1 = edit_file_tool_handler({
            "file_path": test_file,
            "old_string": "Hello, World!",
            "new_string": "Hello, Universe!",
            "replace_all": False
        })
        print(result1)
        
        # Test 2: Multiple occurrences with replace_all
        print("\n2️⃣ Adding multiple occurrences:")
        # Add another occurrence
        with open(test_file, 'a') as f:
            f.write("\nprint('Hello, World!')\n")
        
        result2 = edit_file_tool_handler({
            "file_path": test_file,
            "old_string": "Hello, World!",
            "new_string": "Hello, Everyone!",
            "replace_all": True
        })
        print(result2)
        
        # Test 3: Function replacement
        print("\n3️⃣ Function replacement:")
        result3 = edit_file_tool_handler({
            "file_path": test_file,
            "old_string": "    return 42",
            "new_string": "    return 100",
            "replace_all": False
        })
        print(result3)
        
        # Test 4: Error handling - string not found
        print("\n4️⃣ Error handling - string not found:")
        result4 = edit_file_tool_handler({
            "file_path": test_file,
            "old_string": "nonexistent_string",
            "new_string": "replacement",
            "replace_all": False
        })
        print(result4)
        
        # Test 5: Error handling - relative path
        print("\n5️⃣ Error handling - relative path:")
        result5 = edit_file_tool_handler({
            "file_path": "relative/path.py",
            "old_string": "test",
            "new_string": "replacement",
            "replace_all": False
        })
        print(result5)
        
        # Test 6: Show final file content
        print("\n6️⃣ Final file content:")
        with open(test_file, 'r') as f:
            print(f.read())
        
    finally:
        # Clean up
        os.unlink(test_file)