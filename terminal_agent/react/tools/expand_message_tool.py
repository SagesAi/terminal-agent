"""
Expand message tool for Terminal Agent.

This tool allows the agent to expand truncated messages and view their full content.
"""

import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global session manager reference
_session_manager = None

def expand_message_tool(query: str) -> str:
    """
    Expand a message that was truncated due to length.
    
    Args:
        query: A JSON string containing the message_id to expand.
               Format: {"message_id": "uuid-of-message"}
               
    Returns:
        The full content of the message in JSON format.
    """
    global _session_manager
    
    try:
        # Parse the query as JSON
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return json.dumps({
                "success": False,
                "error": "Invalid JSON query. Expected format: {\"message_id\": \"uuid-of-message\"}"
            })
        
        # Check if session manager is initialized
        if not _session_manager:
            logger.error("Session manager not initialized for expand_message_tool")
            return json.dumps({
                "success": False,
                "error": "Session manager not initialized"
            })
        
        # Get message_id from params
        message_id = params.get("message_id")
        if not message_id:
            return json.dumps({
                "success": False,
                "error": "message_id is required"
            })
        
        # Get the expanded message from the session manager
        result = _session_manager.expand_message(message_id)
        
        if result.get("status") == "error":
            return json.dumps({
                "success": False,
                "error": result.get("message", "Unknown error expanding message")
            })
        
        # Return the result as JSON string
        return json.dumps({
            "success": True,
            "message_id": message_id,
            "role": result.get("role", ""),
            "content": result.get("message", ""),
            "created_at": result.get("created_at", "")
        })
        
    except Exception as e:
        logger.error(f"Error executing expand_message tool: {e}")
        return json.dumps({
            "success": False,
            "error": f"Error expanding message: {str(e)}"
        })

def init_expand_message_tool(session_manager):
    """
    Initialize the expand message tool with a session manager.
    
    Args:
        session_manager: Session manager instance
        
    Returns:
        The expand_message_tool function
    """
    global _session_manager
    _session_manager = session_manager
    return expand_message_tool
