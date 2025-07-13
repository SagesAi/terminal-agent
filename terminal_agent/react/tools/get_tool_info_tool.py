#!/usr/bin/env python3
"""
Tool for retrieving information about other tools.

This module provides a tool for the agent to query information about other tools,
including their parameters, examples, and best practices.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

# Get logger
logger = logging.getLogger(__name__)

def load_tools_metadata() -> Dict[str, Any]:
    """
    Load tool metadata from the JSON file.
    
    Returns:
        Dict[str, Any]: Dictionary containing tool metadata
    """
    metadata_path = os.path.join(
        os.path.dirname(__file__), 
        "tools_metadata.json"
    )
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load tools metadata: {e}")
        return {}

def get_tool_info_tool(input_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed information about a specific tool.
    
    Args:
        input_json: Dictionary containing:
            - tool_name (str): Name of the tool to get information about
            - detail_level (str, optional): Level of detail to return ("basic" or "full")
    
    Returns:
        Dict[str, Any]: Tool information or error message
    """
    # Parse input
    if isinstance(input_json, str):
        try:
            input_json = json.loads(input_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
    
    # Get tool name and detail level
    tool_name = input_json.get("tool_name")
    detail_level = input_json.get("detail_level", "basic")
    
    if not tool_name:
        return {
            "error": "Missing required parameter: tool_name",
            "available_tools": list(load_tools_metadata().keys())
        }
    
    # Load tool metadata
    tools_metadata = load_tools_metadata()
    
    # Check if tool exists
    if tool_name not in tools_metadata:
        return {
            "error": f"Tool '{tool_name}' not found",
            "available_tools": list(tools_metadata.keys())
        }
    
    # Get tool info
    tool_info = tools_metadata[tool_name]
    
    # Return based on detail level
    if detail_level == "basic":
        # Basic information includes name, description, parameters, and one example
        return {
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": tool_info["parameters"],
            "example": tool_info["examples"][0] if tool_info.get("examples") else None
        }
    else:
        # Full information includes everything
        return tool_info

def list_available_tools() -> Dict[str, Any]:
    """
    List all available tools and their basic descriptions.
    
    Returns:
        Dict[str, Any]: Dictionary containing tool names and descriptions
    """
    tools_metadata = load_tools_metadata()
    
    return {
        "available_tools": [
            {
                "name": name,
                "description": info["description"]
            }
            for name, info in tools_metadata.items()
        ]
    }
