#!/usr/bin/env python3
"""
Command Safety Checker for Terminal Agent

This module provides safety checking functionality for shell commands and scripts.
It identifies potentially dangerous commands and provides warnings or blocks execution.
"""

import re
import os
from typing import Dict, List, Tuple, Optional, Set
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialize Rich console
console = Console()

# Define dangerous command patterns
DANGEROUS_COMMANDS = {
    # File system dangers
    "rm": {
        "pattern": r"\brm\s+(-[rf]+\s+|\s+-[rf]+|\s+--recursive|\s+--force)+",
        "description": "Recursive or forced file deletion",
        "severity": "high"
    },
    "rmdir": {
        "pattern": r"\brmdir\b",
        "description": "Directory removal",
        "severity": "medium"
    },
    "dd": {
        "pattern": r"\bdd\b",
        "description": "Direct disk operations that can overwrite data",
        "severity": "high"
    },
    "mkfs": {
        "pattern": r"\bmkfs\b",
        "description": "Formatting file systems",
        "severity": "high"
    },
    
    # System modification dangers
    "chmod": {
        "pattern": r"\bchmod\s+777\b|\bchmod\s+-R\b|\bchmod\s+--recursive\b",
        "description": "Recursive or insecure permission changes",
        "severity": "medium"
    },
    "chown": {
        "pattern": r"\bchown\s+-R\b|\bchown\s+--recursive\b",
        "description": "Recursive ownership changes",
        "severity": "medium"
    },
    
    # Network dangers
    "wget": {
        "pattern": r"\bwget\s+-O\b|\bwget\s+--output-document\b",
        "description": "Downloading and directly saving files",
        "severity": "medium"
    },
    "curl": {
        "pattern": r"\bcurl\s+-o\b|\bcurl\s+--output\b|\bcurl\s+\|\s*bash\b",
        "description": "Downloading and executing scripts directly",
        "severity": "high"
    },
    
    # Process dangers
    "kill": {
        "pattern": r"\bkill\s+-9\b|\bkillall\b",
        "description": "Forcefully killing processes",
        "severity": "medium"
    },
    
    # Dangerous shell operations
    "eval": {
        "pattern": r"\beval\b",
        "description": "Evaluating strings as code",
        "severity": "high"
    },
    "exec": {
        "pattern": r"\bexec\b",
        "description": "Replacing current process with another",
        "severity": "medium"
    },
    
    # Pipe to shell
    "pipe_to_shell": {
        "pattern": r"\|\s*(bash|sh|zsh|ksh|csh)\b",
        "description": "Piping content directly to a shell",
        "severity": "high"
    },
    
    # Sudo and privilege escalation
    "sudo": {
        "pattern": r"\bsudo\b",
        "description": "Executing commands with elevated privileges",
        "severity": "medium"
    },
    "su": {
        "pattern": r"\bsu\b",
        "description": "Switching user, potentially to root",
        "severity": "medium"
    }, 
    # system power commands
    "reboot": {
        "pattern": r"\breboot\b",
        "description": "Restart the system, which will interrupt all running programs and services",
        "severity": "high"
    },
    "shutdown": {
        "pattern": r"\bshutdown\b",
        "description": "Shut down the system, which will interrupt all running programs and services",
        "severity": "high"
    },
    "halt": {
        "pattern": r"\bhalt\b",
        "description": "Stop the system, similar to shutdown",
        "severity": "high"
    },
    "poweroff": {
        "pattern": r"\bpoweroff\b",
        "description": "Turn off system power",
        "severity": "high"
    },
    "init": {
        "pattern": r"\binit\s+[06]\b",
        "description": "Change system runlevel to shutdown(0) or restart(6)",
        "severity": "high"
    },
}

def check_command_safety(command: str) -> Tuple[bool, List[Dict[str, str]]]:
    """
    Check if a command contains potentially dangerous operations.
    
    Args:
        command (str): The command to check
        
    Returns:
        Tuple[bool, List[Dict[str, str]]]: (is_safe, list of warnings)
    """
    warnings = []
    is_safe = True
    
    # Check for dangerous commands
    for cmd_name, cmd_info in DANGEROUS_COMMANDS.items():
        if re.search(cmd_info["pattern"], command, re.IGNORECASE):
            warnings.append({
                "command": cmd_name,
                "description": cmd_info["description"],
                "severity": cmd_info["severity"]
            })
            if cmd_info["severity"] == "high":
                is_safe = False
    
    return is_safe, warnings

def display_safety_warning(command: str, warnings: List[Dict[str, str]]) -> None:
    """
    Display a warning about potentially dangerous commands.
    
    Args:
        command (str): The command that was checked
        warnings (List[Dict[str, str]]): List of warnings
    """
    if not warnings:
        return
    
    # Create warning text
    warning_text = Text()
    warning_text.append("⚠️ This command contains potentially dangerous operations:\n\n")
    
    for warning in warnings:
        severity_color = "yellow" if warning["severity"] == "medium" else "red"
        warning_text.append(f"• ", style="bold")
        warning_text.append(warning["command"], style=f"bold {severity_color}")
        warning_text.append(f": {warning['description']}\n")
    
    warning_text.append("\nPlease review carefully before execution.")
    
    # Create and display panel
    panel = Panel(
        warning_text,
        title="[bold red]⚠️ SECURITY WARNING ⚠️[/bold red]",
        border_style="red",
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)
    console.print("\n")

def check_script_safety(script_content: str) -> Tuple[bool, List[Dict[str, str]]]:
    """
    Check if a script contains potentially dangerous operations.
    
    Args:
        script_content (str): The content of the script to check
        
    Returns:
        Tuple[bool, List[Dict[str, str]]]: (is_safe, list of warnings)
    """
    warnings = []
    is_safe = True
    
    # Split script into lines for analysis
    lines = script_content.split('\n')
    
    for line in lines:
        # Skip comments and empty lines
        if line.strip().startswith('#') or not line.strip():
            continue
        
        # Check each line for dangerous commands
        line_safe, line_warnings = check_command_safety(line)
        
        # Add line number to warnings
        for warning in line_warnings:
            warning["line"] = lines.index(line) + 1
            warnings.append(warning)
        
        if not line_safe:
            is_safe = False
    
    return is_safe, warnings

def display_script_safety_warning(script_content: str, warnings: List[Dict[str, str]]) -> None:
    """
    Display a warning about potentially dangerous script operations.
    
    Args:
        script_content (str): The script content that was checked
        warnings (List[Dict[str, str]]): List of warnings
    """
    if not warnings:
        return
    
    # Create warning text
    warning_text = Text()
    warning_text.append("⚠️ This script contains potentially dangerous operations:\n\n")
    
    for warning in warnings:
        severity_color = "yellow" if warning["severity"] == "medium" else "red"
        warning_text.append(f"• Line {warning.get('line', '?')}: ", style="bold")
        warning_text.append(warning["command"], style=f"bold {severity_color}")
        warning_text.append(f": {warning['description']}\n")
    
    warning_text.append("\nPlease review the script carefully before execution.")
    
    # Create and display panel
    panel = Panel(
        warning_text,
        title="[bold red]⚠️ SCRIPT SECURITY WARNING ⚠️[/bold red]",
        border_style="red",
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(panel)
    console.print("\n")
