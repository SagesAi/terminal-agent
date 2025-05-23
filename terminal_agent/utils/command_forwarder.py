#!/usr/bin/env python3
"""
Command Forwarder - Responsible for forwarding commands to local or remote execution
"""

import os
import tempfile
import logging
from typing import Tuple, Optional, Dict, Any, List, Callable
from functools import wraps
from dotenv import load_dotenv, find_dotenv

from terminal_agent.utils.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

# Load environment variables at module level
# First try to load from user's home directory
home_env = os.path.join(os.path.expanduser("~"), ".terminal_agent", ".env")
if os.path.exists(home_env):
    load_dotenv(home_env)
    logger.debug(f"Loaded environment from: {home_env}")
else:
    # Try to load from current directory or parent directory
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
        logger.debug(f"Loaded environment from: {dotenv_path}")
    else:
        logger.debug("No .env file found. Using system environment variables.")

try:
    import paramiko
except ImportError:
    logger.warning("paramiko library not installed, remote execution functionality will not be available")
    paramiko = None

class CommandForwarder:
    """Command forwarder, determines whether commands are executed locally or remotely"""
    
    def __init__(self):
        """Initialize command forwarder"""
        self.client = None
        self.connected = False
        # Set default values
        self._initialized = False
        self.remote_enabled = False
        self.host = ""
        self.user = ""
        self.port = 22
        self.auth_type = "key"
        self.key_path = "~/.ssh/id_rsa"
        self.password = ""
        self.sudo_enabled = False
        
        # Load configuration immediately, rather than lazy loading
        self._load_config()
    
    def _load_config(self):
        """Load remote execution configuration"""
        if self._initialized:
            return
            
        # Read configuration from environment variables
        self.remote_enabled = os.getenv("REMOTE_EXECUTION_ENABLED", "false").lower() == "true"
        
        if self.remote_enabled:
            if paramiko is None:
                logger.error("Remote execution is enabled, but paramiko library is not installed. Please install paramiko: pip install paramiko")
                self.remote_enabled = False
                return
                
            self.host = os.getenv("REMOTE_HOST", "")
            self.user = os.getenv("REMOTE_USER", "")
            self.port = int(os.getenv("REMOTE_PORT", "22"))
            self.auth_type = os.getenv("REMOTE_AUTH_TYPE", "key")
            self.key_path = os.getenv("REMOTE_KEY_PATH", "~/.ssh/id_rsa")
            self.password = os.getenv("REMOTE_PASSWORD", "")
            self.sudo_enabled = os.getenv("REMOTE_SUDO_ENABLED", "false").lower() == "true"
        
        # Mark as initialized
        self._initialized = True
    
    def _connect(self) -> bool:
        """Establish SSH connection"""
        # Ensure configuration is loaded
        if not self._initialized:
            self._load_config()
            
        if not self.remote_enabled or self.connected:
            return self.connected
            
        # Set connection parameters
        connect_params = {
            "hostname": self.host,
            "username": self.user,
            "port": self.port,
            "timeout": 10,  # 10 seconds timeout
            "allow_agent": False,  # Don't use SSH agent
            "look_for_keys": False  # Don't look for other keys
        }
        
        # Add authentication parameters based on authentication type
        if self.auth_type == "key":
            key_path = os.path.expanduser(self.key_path)
            connect_params["key_filename"] = key_path
        else:
            connect_params["password"] = self.password
        
        # Try to connect, retry up to 3 times
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.client.connect(**connect_params)
                
                self.connected = True
                logger.info(f"Successfully connected to remote host {self.host}")
                return True
                
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(f"Failed to connect to remote host (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    # Wait for a while before retrying
                    import time
                    time.sleep(2)  # Wait for 2 seconds
        
        # All retries failed
        logger.error(f"Failed to connect to remote host after {max_retries} attempts: {str(last_error)}")
        self.connected = False
        return False
    
    def forward_command(self, command: str, real_time_output: bool = True, callback=None, cwd=None) -> Tuple[int, str, str]:
        """
        Forward command to remote host for execution
        
        Args:
            command: Command to be executed
            real_time_output: Whether to provide real-time output
            callback: Optional callback function to process real-time output
            cwd: Working directory for command execution (on remote host)
            
        Returns:
            Tuple[int, str, str]: (Return code, Standard output, Standard error)
        """
        # Ensure configuration is loaded
        if not self._initialized:
            self._load_config()
            
        if not self.remote_enabled:
            raise ValueError("Remote execution is not enabled")
            
        # Try to connect or reconnect
        if not self.connected and not self._connect():
            raise ConnectionError("Failed to connect to remote host")
        
        # Handle working directory if specified
        original_command = command
        
        # Handle working directory and sudo permissions
        if cwd:
            # First cd to the target directory
            cd_command = f"cd {cwd} && "
        else:
            cd_command = ""
            
        # Then apply sudo if needed
        if self.sudo_enabled and not command.strip().startswith("sudo"):
            # Add sudo to the actual command, not to the cd part
            sudo_command = f"sudo {command}"
            # Combine with cd if needed
            if cwd:
                # Use bash -c to execute the combined command
                command = f"bash -c '{cd_command}{sudo_command}'"
            else:
                command = sudo_command
        elif cwd:  # No sudo but has cwd
            # Use bash -c to execute the command in the specified directory
            command = f"bash -c '{cd_command}{command}'"
            
        # Try to execute command, if it fails, try to reconnect once
        try:
            # Execute the command
            stdin, stdout, stderr = self.client.exec_command(command, timeout=30)  # 30 seconds timeout
            
            # For real-time output processing
            stdout_data = []
            stderr_data = []
            
            if real_time_output:
                # Get the channel to monitor for data
                channel = stdout.channel
                
                # Process output in real-time
                import select
                while not channel.exit_status_ready():
                    if channel.recv_ready():
                        data = channel.recv(1024).decode('utf-8')
                        stdout_data.append(data)
                        if callback:
                            callback('stdout', data)
                        else:
                            print(data, end='', flush=True)
                    
                    if channel.recv_stderr_ready():
                        data = channel.recv_stderr(1024).decode('utf-8')
                        stderr_data.append(data)
                        if callback:
                            callback('stderr', data)
                        else:
                            print(data, end='', flush=True)
                    
                    # Small sleep to prevent CPU hogging
                    if not (channel.recv_ready() or channel.recv_stderr_ready()):
                        import time
                        time.sleep(0.1)
                
                # Get any remaining data
                while channel.recv_ready():
                    data = channel.recv(1024).decode('utf-8')
                    stdout_data.append(data)
                    if callback:
                        callback('stdout', data)
                    else:
                        print(data, end='', flush=True)
                        
                while channel.recv_stderr_ready():
                    data = channel.recv_stderr(1024).decode('utf-8')
                    stderr_data.append(data)
                    if callback:
                        callback('stderr', data)
                    else:
                        print(data, end='', flush=True)
                        
                # Get exit status
                exit_code = channel.recv_exit_status()
                stdout_str = ''.join(stdout_data)
                stderr_str = ''.join(stderr_data)
            else:
                # Original behavior for non-real-time output
                exit_code = stdout.channel.recv_exit_status()
                stdout_str = stdout.read().decode('utf-8')
                stderr_str = stderr.read().decode('utf-8')
            
            return exit_code, stdout_str, stderr_str
            
        except Exception as e:
            logger.warning(f"Command execution failed, attempting to reconnect: {str(e)}")
            
            # Try to reconnect
            self.connected = False
            if self._connect():
                try:
                    # Try to execute command again but don't use recursion
                    stdin, stdout, stderr = self.client.exec_command(command, timeout=30)
                    
                    # Process output based on real-time setting
                    if real_time_output:
                        # Similar to the original code but without recursion
                        channel = stdout.channel
                        stdout_data = []
                        stderr_data = []
                        
                        # Get exit status and output
                        exit_code = channel.recv_exit_status()
                        stdout_str = stdout.read().decode('utf-8')
                        stderr_str = stderr.read().decode('utf-8')
                        
                        # If callback is provided, call it
                        if callback and stdout_str:
                            callback('stdout', stdout_str)
                        if callback and stderr_str:
                            callback('stderr', stderr_str)
                        elif callback is None:  # 如果没有提供回调，直接打印
                            if stdout_str:
                                print(stdout_str, end='', flush=True)
                            if stderr_str:
                                print(stderr_str, end='', flush=True)
                                
                        return exit_code, stdout_str, stderr_str
                    else:
                        # Original behavior for non-real-time output
                        exit_code = stdout.channel.recv_exit_status()
                        stdout_str = stdout.read().decode('utf-8')
                        stderr_str = stderr.read().decode('utf-8')
                        return exit_code, stdout_str, stderr_str
                except Exception as e2:
                    logger.error(f"Failed to execute command after reconnection: {str(e2)}")
                    return 1, "", f"Command execution failed: {str(e2)}"
            else:
                logger.error(f"Failed to execute command on remote host: {str(e)}")  
                return 1, "", f"Command execution failed: {str(e)}"
    
    def forward_file_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Forward file operation to remote host
        
        Args:
            operation: Operation type (read, write, exists, etc.)
            **kwargs: Operation parameters
            
        Returns:
            Dict[str, Any]: Operation result
        """
        # Ensure configuration is loaded
        if not self._initialized:
            self._load_config()
            
        if not self.remote_enabled:
            raise ValueError("Remote execution is not enabled")
            
        if not self.connected and not self._connect():
            raise ConnectionError("Failed to connect to remote host")
            
        try:
            sftp = self.client.open_sftp()
            
            if operation == "read":
                path = kwargs.get("path", "")
                with sftp.open(path, 'r') as f:
                    content = f.read().decode('utf-8')
                sftp.close()
                return {"success": True, "content": content}
                
            elif operation == "write":
                path = kwargs.get("path", "")
                content = kwargs.get("content", "")
                with sftp.open(path, 'w') as f:
                    f.write(content)
                sftp.close()
                return {"success": True}
                
            elif operation == "exists":
                path = kwargs.get("path", "")
                try:
                    stat_result = sftp.stat(path)
                    # Check if it's a directory
                    # According to SFTP protocol, directory's st_mode bitwise AND with 0o40000 (16384) is non-zero
                    is_directory = bool(stat_result.st_mode & 0o40000)
                    is_file = not is_directory
                    
                    sftp.close()
                    return {
                        "success": True, 
                        "exists": True, 
                        "is_file": is_file,
                        "is_directory": is_directory
                    }
                except FileNotFoundError:
                    sftp.close()
                    return {"success": True, "exists": False, "is_file": False, "is_directory": False}
                    
            else:
                sftp.close()
                return {"success": False, "error": f"Unsupported operation: {operation}"}
                
        except Exception as e:
            logger.error(f"Failed to perform file operation on remote host: {str(e)}")
            return {"success": False, "error": str(e)}

# Create global command forwarder instance
forwarder = CommandForwarder()

def remote_aware(func):
    """
    Decorator that makes functions remote-aware.
    If remote execution is enabled, forwards commands to remote host.
    Otherwise executes locally.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # If remote execution is not enabled, call the original function directly
        if not forwarder.remote_enabled:
            return func(*args, **kwargs)
            
        # Determine how to handle based on function name
        func_name = func.__name__
        
        if func_name == "execute_command_single" or func_name == "execute_command":
            # Extract command parameters
            command = args[0] if args else kwargs.get("command", "")
            show_output = args[1] if len(args) > 1 else kwargs.get("show_output", True)
            need_confirmation = args[2] if len(args) > 2 else kwargs.get("need_confirmation", True)
            
            # Extract working directory if provided
            cwd = kwargs.get("cwd", None)
            
            # If user confirmation is needed, show remote execution prompt
            if need_confirmation:
                from rich.console import Console
                from rich.text import Text
                
                console = Console()
                if show_output:
                    console.print(f"[bold cyan]Executing command on remote host {forwarder.host}:[/bold cyan] {command}")
                
                # User confirmation logic could be added here, but since the original function handles confirmation, we just show the prompt
            
            # Execute remote command with real-time output
            try:
                # Define a callback function for real-time output
                def output_callback(stream_type, data):
                    if show_output:
                        from rich.console import Console
                        console = Console()
                        # Use different colors for stdout and stderr
                        if stream_type == 'stderr':
                            console.print(data, end='', style="bold red")
                        else:
                            console.print(data, end='')
                
                # Use real-time output if show_output is True
                exit_code, stdout, stderr = forwarder.forward_command(
                    command, 
                    real_time_output=show_output,
                    callback=output_callback if show_output else None,
                    cwd=cwd
                )
                output = stdout + stderr
                
                # If real-time output was used, we don't need to print the output again
                # Only print if not using real-time output but show_output is True
                if show_output and output and not show_output:
                    from rich.console import Console
                    from rich.text import Text
                    console = Console()
                    console.print(Text(output))
                    
                return exit_code, output, False  # Return format consistent with original function
            except Exception as e:
                logger.error(f"Failed to execute command on remote host: {str(e)}")
                return 1, str(e), False
        
        # For other functions, remote execution is not yet supported, so call the original function
        # Additional handling logic can be added as needed
        return func(*args, **kwargs)
        
    return wrapper


def remote_file_operation(func):
    """
    Decorator that makes file operation functions remote-aware.
    If remote execution is enabled, forwards file operations to remote host.
    Otherwise executes locally.
    """
    @wraps(func)
    def wrapper(params: Dict):
        # If remote execution is not enabled, call the original function directly
        if not forwarder.remote_enabled:
            return func(params)
        
        # Determine file operation type based on function name
        func_name = func.__name__
        file_path = params.get("file_path") or params.get("directory_path")
        
        if not file_path:
            return "Error: Missing required parameter 'file_path' or 'directory_path'"
        
        # Clean and normalize path
        try:
            from terminal_agent.react.tools.files_tool import clean_path, should_exclude_file
            file_path = clean_path(file_path)
            
            # Check if file should be excluded
            if should_exclude_file(file_path):
                return f"Error: Cannot access '{file_path}' - access to this file type is restricted"
        except Exception as e:
            logger.error(f"Failed to process path: {str(e)}")
            return f"Error processing path: {str(e)}"
        
        try:
            # Determine file operation type based on function name
            if func_name == "create_file":
                content = params.get("content", "")
                overwrite = params.get("overwrite", False)
                
                # Check if file exists
                exists_result = forwarder.forward_file_operation("exists", path=file_path)
                if exists_result.get("success", False) and exists_result.get("exists", False) and not overwrite:
                    return f"Error: File '{file_path}' already exists on remote host. Use 'overwrite: true' to overwrite"
                
                # Create file
                result = forwarder.forward_file_operation("write", path=file_path, content=content)
                if result.get("success", False):
                    return f"Successfully created file on remote host: {file_path}"
                else:
                    return f"Error creating file on remote host: {result.get('error', 'Unknown error')}"
                    
            elif func_name == "read_file":
                start_line = params.get("start_line", 1)
                end_line = params.get("end_line", None)
                
                # Check if file exists
                exists_result = forwarder.forward_file_operation("exists", path=file_path)
                if not exists_result.get("success", False) or not exists_result.get("exists", False):
                    return f"Error: File '{file_path}' does not exist on remote host"
                
                if not exists_result.get("is_file", False):
                    return f"Error: '{file_path}' is not a file on remote host"
                
                # Read file content
                result = forwarder.forward_file_operation("read", path=file_path)
                if not result.get("success", False):
                    return f"Error reading file on remote host: {result.get('error', 'Unknown error')}"
                
                content = result.get("content", "")
                
                # Handle line range
                if start_line > 1 or end_line is not None:
                    lines = content.splitlines(True)
                    
                    # Adjust to 1-based index
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
                
                return content
                
            elif func_name == "update_file":
                content = params.get("content")
                old_str = params.get("old_str")
                new_str = params.get("new_str")
                mode = params.get("mode", "write").lower()
                
                # Check if file exists
                exists_result = forwarder.forward_file_operation("exists", path=file_path)
                if not exists_result.get("success", False) or not exists_result.get("exists", False):
                    return f"Error: File '{file_path}' does not exist on remote host"
                
                if not exists_result.get("is_file", False):
                    return f"Error: '{file_path}' is not a file on remote host"
                
                # Read current content (for append and replace modes)
                current_content = ""
                if mode in ["append", "replace"]:
                    read_result = forwarder.forward_file_operation("read", path=file_path)
                    if not read_result.get("success", False):
                        return f"Error reading file on remote host: {read_result.get('error', 'Unknown error')}"
                    current_content = read_result.get("content", "")
                
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
                result = forwarder.forward_file_operation("write", path=file_path, content=new_content)
                if not result.get("success", False):
                    return f"Error updating file on remote host: {result.get('error', 'Unknown error')}"
                
                return f"Success: File '{file_path}' updated successfully on remote host"
                
            elif func_name == "delete_file":
                recursive = params.get("recursive", False)
                
                # Check if file exists
                exists_result = forwarder.forward_file_operation("exists", path=file_path)
                if not exists_result.get("success", False) or not exists_result.get("exists", False):
                    return f"Error: File or directory '{file_path}' does not exist on remote host"
                
                # Execute delete command
                if exists_result.get("is_file", False):
                    cmd = f"rm '{file_path}'"
                    exit_code, stdout, stderr = forwarder.forward_command(cmd)
                    if exit_code == 0:
                        return f"Success: File '{file_path}' deleted successfully on remote host"
                    else:
                        return f"Error deleting file on remote host: {stderr}"
                elif exists_result.get("is_directory", False):
                    if recursive:
                        cmd = f"rm -rf '{file_path}'"
                    else:
                        cmd = f"rmdir '{file_path}'"
                    
                    exit_code, stdout, stderr = forwarder.forward_command(cmd)
                    if exit_code == 0:
                        if recursive:
                            return f"Success: Directory '{file_path}' and all its contents deleted successfully on remote host"
                        else:
                            return f"Success: Directory '{file_path}' deleted successfully on remote host"
                    else:
                        if "Directory not empty" in stderr or "Not a directory" in stderr:
                            return f"Error: Directory '{file_path}' is not empty on remote host. Use 'recursive: true' to delete non-empty directories"
                        else:
                            return f"Error deleting directory on remote host: {stderr}"
                else:
                    return f"Error: '{file_path}' is neither a file nor a directory on remote host"
                    
            elif func_name == "list_directory":
                include_hidden = params.get("include_hidden", False)
                
                # Check if directory exists
                exists_result = forwarder.forward_file_operation("exists", path=file_path)
                if not exists_result.get("success", False) or not exists_result.get("exists", False):
                    return f"Error: Directory '{file_path}' does not exist on remote host"
                
                if not exists_result.get("is_directory", False):
                    return f"Error: '{file_path}' is not a directory on remote host"
                
                # Execute ls command to get directory contents
                cmd = f"ls -la '{file_path}' | tail -n +2"  # Skip first line (total)
                exit_code, stdout, stderr = forwarder.forward_command(cmd)
                
                if exit_code != 0:
                    return f"Error listing directory on remote host: {stderr}"
                
                # Parse directory contents
                import json
                contents = []
                for line in stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                        
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                        
                    permissions = parts[0]
                    size = int(parts[4])
                    name = ' '.join(parts[8:])
                    
                    # Skip current directory and parent directory
                    if name in [".", ".."]:
                        continue
                        
                    # Skip hidden files (if needed)
                    if not include_hidden and name.startswith('.'):
                        continue
                    
                    item_path = os.path.join(file_path, name)
                    item_type = "directory" if permissions.startswith('d') else "file"
                    
                    contents.append({
                        "name": name,
                        "type": item_type,
                        "size": size,
                        "path": item_path
                    })
                
                # Sort by type and name
                contents.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"]))
                
                result = {
                    "directory": file_path,
                    "contents": contents,
                    "count": len(contents)
                }
                
                return json.dumps(result, indent=2)
                
            elif func_name == "file_exists":
                # Check if file exists on remote host
                result = forwarder.forward_file_operation("exists", path=file_path)
                
                if not result.get("success", False):
                    return f"Error checking file existence on remote host: {result.get('error', 'Unknown error')}"
                
                # Build result
                import json
                response = {
                    "exists": result.get("exists", False),
                    "path": file_path,
                    "remote": True
                }
                
                if response["exists"]:
                    response["is_file"] = result.get("is_file", False)
                    response["is_directory"] = result.get("is_directory", False)
                    
                    if response["is_file"] and "size" in result:
                        response["size"] = result.get("size", 0)
                        response["extension"] = os.path.splitext(file_path)[1]
                
                return json.dumps(response, indent=2)
            
            # 对于其他函数，调用原函数
            return func(params)
            
        except Exception as e:
            logger.error(f"Remote file operation failed: {str(e)}")
            return f"Error in remote file operation: {str(e)}"
    
    return wrapper
