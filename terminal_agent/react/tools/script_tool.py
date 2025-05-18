#!/usr/bin/env python3
"""
Script tool for Terminal Agent.
Provides functionality to create and execute scripts.
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional, Any, Tuple

from rich.console import Console

from terminal_agent.utils.command_executor import execute_command
from terminal_agent.utils.command_forwarder import forwarder, remote_file_operation
from terminal_agent.react.tools.files_tool import create_file, file_exists, read_file

# Configure logging
logger = logging.getLogger(__name__)
console = Console()

def script_tool(script_request: str) -> str:
    """
    Creates and executes a script based on the provided request.

    The request should be in JSON format with the following fields:
    - action: "create", "execute", or "create_and_execute"
    - filename: The name of the script file to create or execute
    - content: The content of the script (only for creation)
    - interpreter: The interpreter to use (e.g., "python3", "bash", "node")
    - args: List of arguments to pass to the script (optional)
    - env_vars: Dictionary of environment variables to set (optional)
    - timeout: Maximum execution time in seconds (optional)

    Args:
        script_request (str): JSON string containing the script request.

    Returns:
        str: The result of the script operation.
    """
    try:
        # Import command safety module
        from terminal_agent.utils.command_safety import check_script_safety, display_script_safety_warning
        # Parse the script request
        if isinstance(script_request, dict):
            request = script_request
        else:
            request = json.loads(script_request)
        # Extract fields
        action = request.get("action", "").lower()
        filename = request.get("filename", "")
        content = request.get("content", "")
        interpreter = request.get("interpreter", "")
        args = request.get("args", [])  # Extract arguments list
        env_vars = request.get("env_vars", {})  # Extract environment variables
        timeout = request.get("timeout", None)  # Extract timeout setting
        # Validate required fields
        if not filename:
            return "Error: Filename is required."

        # Create the script
        if action in ["create", "create_and_execute"]:
            if not content:
                return "Error: Script content is required for creation."

            # Check script safety
            is_safe, warnings = check_script_safety(content)

            # Display safety warnings if any
            if warnings:
                display_script_safety_warning(content, warnings)

            # Ensure the script has the correct permissions
            try:
                # 使用 files_tool 创建文件，支持远程模式
                create_result = create_file({
                    "file_path": filename,
                    "content": content,
                    "overwrite": True
                })
                
                if "Error" in create_result:
                    return create_result
                
                # 设置可执行权限
                if forwarder.remote_enabled:
                    # 在远程主机上设置权限
                    _, _, _ = forwarder.forward_command(f"chmod 755 {filename}")
                else:
                    # 在本地设置权限
                    os.chmod(filename, 0o755)

                console.print(f"[green]Script created: {filename}[/green]")
            except Exception as e:
                return f"Error creating script: {str(e)}"
        # Execute the script
        if action in ["execute", "create_and_execute"]:
            # 检查文件是否存在，支持远程模式
            exists_result = file_exists({"file_path": filename})
            if "Error" in exists_result or "does not exist" in exists_result:
                return f"Error: Script file {filename} does not exist."

            # If we're just executing (not creating), check the script content for safety
            if action == "execute":
                try:
                    # 使用 files_tool 读取文件，支持远程模式
                    script_content = read_file({"file_path": filename})
                    
                    if "Error" in script_content:
                        return script_content

                    # Check script safety
                    is_safe, warnings = check_script_safety(script_content)

                    # Display safety warnings if any
                    if warnings:
                        display_script_safety_warning(script_content, warnings)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not check script safety: {str(e)}[/yellow]")

            # Determine the command to run the script
            if interpreter:
                command = f"{interpreter} {filename}"
            else:
                # Use the shebang line or execute directly if executable
                command = f"./{filename}"

            # Add arguments to the command if provided
            if args:
                # Convert all arguments to strings and join with spaces
                args_str = " ".join([str(arg) for arg in args])
                command = f"{command} {args_str}"

            # Execute the script with environment variables and timeout if provided
            return_code, output, _ = execute_command(
                command,
                env=env_vars,
                timeout=timeout
            )

            # 添加一行日志，记录实际的返回代码
            logger.debug(f"Script execution return code: {return_code}")

            # 修复 bug：确保在输出中正确反映返回代码
            if return_code == 0:
                result = f"Command: {command}\nReturn Code: {return_code} (Success)\nOutput:\n{output}"
            else:
                result = f"Command: {command}\nReturn Code: {return_code} (FAILED)\nOutput:\n{output}"
            
            return result

        # Handle unknown action
        return f"Error: Unknown action '{action}'. Must be 'create', 'execute', or 'create_and_execute'."

    except json.JSONDecodeError:
        return "Error: Invalid JSON format in script request."
    except Exception as e:
        return f"Error processing script request: {str(e)}"
