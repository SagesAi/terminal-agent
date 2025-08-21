"""
Session Management

Manages user sessions, session timeouts, and interaction history.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import uuid
import logging

from terminal_agent.memory.memory_database import MemoryDatabase
from terminal_agent.memory.context_manager import ContextManager

# Get logger
logger = logging.getLogger(__name__)

# Default session timeout (seconds)
DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour

class SessionManager:
    """Manages user sessions and session history"""
    
    def __init__(self, db: MemoryDatabase, context_manager: ContextManager, session_timeout: int = DEFAULT_SESSION_TIMEOUT):
        """Initialize session manager
        
        Args:
            db: Memory database instance
            context_manager: Context manager instance
            session_timeout: Session timeout in seconds
        """
        self.db = db
        self.context_manager = context_manager
        self.session_timeout = session_timeout
    
    def get_or_create_session(self, user_id: str) -> str:
        """Get user's active session, create new session if none exists
        
        Args:
            user_id: User ID
            
        Returns:
            Session ID
        """
        conn = self.db.conn
        now = datetime.now().isoformat()
        
        try:
            # Find user's latest session
            cursor = conn.execute('''
            SELECT id, updated_at FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC LIMIT 1
            ''', (user_id,))
            
            session_row = cursor.fetchone()
            
            if session_row:
                session_id = session_row['id']
                updated_at = datetime.fromisoformat(session_row['updated_at'])
                
                # Check if session has timed out
                if (datetime.now() - updated_at).total_seconds() < self.session_timeout:
                    # Update session's last activity time
                    conn.execute('''
                    UPDATE sessions SET updated_at = ? WHERE id = ?
                    ''', (now, session_id))
                    conn.commit()
                    logger.debug(f"Using existing session: {session_id} (user: {user_id})")
                    return session_id
            
            # Create new session
            session_id = str(uuid.uuid4())
            conn.execute('''
            INSERT INTO sessions (id, user_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''', (session_id, user_id, now, now, '{}'))
            conn.commit()
            
            logger.info(f"Created new session for user {user_id}: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error getting or creating session: {e}")
            # Create a temporary session ID on error
            return f"temp_{str(uuid.uuid4())}"
    
    def add_message(self, session_id: str, role: str, content: Any, message_type: str = "message", metadata: Dict[str, Any] = None) -> str:
        """Add message to session
        
        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            message_type: Message type (message, thinking, summary, etc.)
            metadata: Additional metadata for the message
            
        Returns:
            Message ID
        """
        conn = self.db.conn
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        try:
            # Convert content to JSON string if not already a string
            if not isinstance(content, str):
                content = json.dumps(content)
                
            # Convert metadata to JSON string if provided
            metadata_json = None
            if metadata:
                metadata_json = json.dumps(metadata)
                
            conn.execute('''
            INSERT INTO messages (id, session_id, role, content, type, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (message_id, session_id, role, content, message_type, now, metadata_json))
            
            # Update session's last activity time
            conn.execute('''
            UPDATE sessions SET updated_at = ? WHERE id = ?
            ''', (now, session_id))
            
            conn.commit()
            logger.debug(f"Added message to session {session_id}: role={role}, type={message_type}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return ""
    
    def add_tool_call(self, message_id: str, tool_name: str, input_data: Any, output_data: Any = None, tool_call_id: str = None) -> str:
        """Add tool call record
        
        Args:
            message_id: Associated message ID
            tool_name: Tool name
            input_data: Input data
            output_data: Output data
            tool_call_id: OpenAI tool call ID for linking results
            
        Returns:
            Tool call ID
        """
        conn = self.db.conn
        tool_call_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        try:
            # Convert input and output to JSON strings
            if not isinstance(input_data, str):
                input_data = json.dumps(input_data)
            
            if output_data is not None and not isinstance(output_data, str):
                output_data = json.dumps(output_data)
                
            conn.execute('''
            INSERT INTO tool_calls (id, message_id, tool_name, input, output, created_at, tool_call_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (tool_call_id, message_id, tool_name, input_data, output_data, now, tool_call_id))
            
            conn.commit()
            logger.debug(f"Added tool call record: tool={tool_name}, messageID={message_id}")
            return tool_call_id
            
        except Exception as e:
            logger.error(f"Error adding tool call: {e}")
            return ""

    def get_messages_for_llm(self, user_id: str, model: str = "gpt-4", available_tokens: Optional[int] = None) -> List[Dict]:
        """Get messages for LLM context
        
        This method will:
        1. Get or create user session
        2. Check if summary generation is needed
        3. Return messages suitable for LLM context
        
        Args:
            user_id: User ID
            model: LLM model name for token counting and compression
            available_tokens: Optional token limit after accounting for system prompt
            
        Returns:
            Message list for LLM context
        """
        try:
            # Get or create session
            session_id = self.get_or_create_session(user_id)
            
            # Check if summary generation is needed
            #self.context_manager.check_and_summarize_if_needed(session_id, model)
            
            # Get context messages with compression if needed
            # If available_tokens is provided, use it to adjust compression
            if available_tokens:
                # Get the default token threshold from context manager
                default_threshold = self.context_manager.token_threshold
                
                # Temporarily adjust token threshold based on available tokens
                # This ensures we don't exceed the available token space
                adjusted_threshold = min(default_threshold, available_tokens)
                original_threshold = self.context_manager.token_threshold
                
                try:
                    # Temporarily set the adjusted threshold
                    self.context_manager.token_threshold = adjusted_threshold
                    messages = self.context_manager.get_messages_for_context(session_id, model)
                finally:
                    # Restore original threshold
                    self.context_manager.token_threshold = original_threshold
            else:
                # Use default threshold
                messages = self.context_manager.get_messages_for_context(session_id, model)
            
            # Convert to format required by LLM
            llm_messages = []
            for msg in messages:
                # Handle tool messages with OpenAI format
                if msg["role"] == "tool" and msg.get("metadata") and msg["metadata"].get("tool_call_id"):
                    # Reconstruct tool result in OpenAI format
                    llm_message = {
                        "role": "tool",
                        "tool_call_id": msg["metadata"]["tool_call_id"],
                        "content": str(msg["content"]) if not isinstance(msg["content"], str) else msg["content"]
                    }
                elif msg["role"] == "assistant" and msg.get("metadata") and msg["metadata"].get("tool_calls"):
                    # Reconstruct assistant message with tool calls
                    llm_message = {
                        "role": "assistant",
                        "content": str(msg["content"]) if not isinstance(msg["content"], str) else msg["content"],
                        "tool_calls": msg["metadata"]["tool_calls"]
                    }
                else:
                    # Handle regular messages
                    if isinstance(msg["content"], dict) and msg["content"].get("_truncated_content"):
                        # For truncated content, use the message or a default message
                        content = msg["content"].get("_message", "[Content was truncated due to length]")
                    else:
                        content = str(msg["content"]) if not isinstance(msg["content"], str) else msg["content"]
                    
                    llm_message = {
                        "role": msg["role"],
                        "content": content
                    }
                
                llm_messages.append(llm_message)
                
            token_count = self.context_manager.get_token_count(llm_messages, model)
            logger.info(f"Retrieved {len(llm_messages)} LLM context messages for user {user_id} with {token_count} tokens")
            return llm_messages
            
        except Exception as e:
            logger.error(f"Error getting LLM messages: {e}")
            return []

    def get_messages_for_llm_dropped(self, user_id: str) -> List[Dict]:
        """Get messages for LLM context
        
        This method will:
        1. Get or create user session
        2. Check if summary generation is needed
        3. Return messages suitable for LLM context
        
        Args:
            user_id: User ID
            
        Returns:
            Message list for LLM context
        """
        try:
            # Get or create session
            session_id = self.get_or_create_session(user_id)
            
            # Check if summary generation is needed
            self.context_manager.check_and_summarize_if_needed(session_id)
            
            # Get context messages
            messages = self.context_manager.get_messages_for_context(session_id)
            
            # Convert to format required by LLM
            llm_messages = []
            for msg in messages:
                llm_message = {
                    "role": msg["role"],
                    "content": str(msg["content"]) if not isinstance(msg["content"], str) else msg["content"]
                }
                llm_messages.append(llm_message)
                
            logger.debug(f"Retrieved {len(llm_messages)} LLM context messages for user {user_id}")
            return llm_messages
            
        except Exception as e:
            logger.error(f"Error getting LLM messages: {e}")
            return []
    
    def clear_inactive_sessions(self) -> int:
        """Clear inactive sessions
        
        Returns:
            Number of sessions cleared
        """
        conn = self.db.conn
        now = datetime.now()
        timeout_threshold = now.timestamp() - self.session_timeout
        timeout_time = datetime.fromtimestamp(timeout_threshold).isoformat()
        
        try:
            # Find timed out sessions
            cursor = conn.execute('''
            SELECT id FROM sessions
            WHERE updated_at < ?
            ''', (timeout_time,))
            
            inactive_sessions = [row['id'] for row in cursor.fetchall()]
            
            if not inactive_sessions:
                return 0
                
            # Delete tool calls and messages for timed out sessions
            for session_id in inactive_sessions:
                # Get all messages in the session
                cursor = conn.execute('''
                SELECT id FROM messages
                WHERE session_id = ?
                ''', (session_id,))
                
                message_ids = [row['id'] for row in cursor.fetchall()]
                
                # Delete tool calls associated with messages
                for message_id in message_ids:
                    conn.execute('''
                    DELETE FROM tool_calls
                    WHERE message_id = ?
                    ''', (message_id,))
                
                # Delete session messages
                conn.execute('''
                DELETE FROM messages
                WHERE session_id = ?
                ''', (session_id,))
            
            # Delete timed out sessions
            conn.execute('''
            DELETE FROM sessions
            WHERE id IN ({})
            '''.format(','.join(['?'] * len(inactive_sessions))), inactive_sessions)
            
            conn.commit()
            
            logger.info(f"Cleared {len(inactive_sessions)} inactive sessions")
            return len(inactive_sessions)
            
        except Exception as e:
            logger.error(f"Error clearing inactive sessions: {e}")
            return 0
