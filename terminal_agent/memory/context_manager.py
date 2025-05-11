"""
Context Management

Manages session context, including token counting, summary generation, and context window management.
References the Suna project's implementation, but uses SQLite instead of Supabase for data storage.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from terminal_agent.memory.memory_database import MemoryDatabase

# Get logger
logger = logging.getLogger(__name__)

# Token management constants
DEFAULT_TOKEN_THRESHOLD = 3000  # Default token threshold
SUMMARY_TARGET_TOKENS = 1000    # Summary target token count
RESERVE_TOKENS = 500            # Reserved tokens for new messages

class ContextManager:
    """Manages session context, including token counting and summary generation"""
    
    def __init__(self, db: MemoryDatabase, llm_client=None, token_threshold: int = DEFAULT_TOKEN_THRESHOLD):
        """Initialize context manager
        
        Args:
            db: Memory database instance
            llm_client: LLM client for generating summaries
            token_threshold: Token threshold, exceeding this will trigger summary generation
        """
        self.db = db
        self.llm_client = llm_client
        self.token_threshold = token_threshold
    
    def get_token_count(self, messages: List[Dict]) -> int:
        """Estimate token count for a message list
        
        Args:
            messages: Message list
            
        Returns:
            Estimated token count
        """
        # Simple estimation: each English word is about 0.3 tokens
        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            total_tokens += len(str(content).split()) * 0.3
        return int(total_tokens)
    
    def should_summarize(self, messages: List[Dict]) -> bool:
        """Determine if summary should be generated
        
        Args:
            messages: Message list
            
        Returns:
            True if summary should be generated, False otherwise
        """
        return self.get_token_count(messages) > self.token_threshold
    
    def generate_summary(self, messages: List[Dict]) -> Optional[Dict]:
        """Generate conversation summary
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary message dictionary, None if generation fails
        """
        if not self.llm_client or not messages:
            return None
            
        # Build summary prompt
        system_message = {
            "role": "system",
            "content": """You are a professional summary assistant. Your task is to create a concise but comprehensive summary of the conversation history.

The summary should:
1. Retain all key information, including decisions, conclusions, and important context
2. Include tools used and their results
3. Maintain chronological order of events
4. Present as a list of key points with section headings
5. Only include factual information from the conversation (don't add new information)
6. Be concise but detailed enough to continue the conversation

Very important: This summary will replace earlier parts of the conversation in the LLM context window, so ensure it contains all key information and the latest state of the conversation - this is how we'll know how to continue the conversation."""
        }
        
        # Prepare message history
        message_history = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            message_history += f"{role}: {content}\n\n"
        
        user_message = {
            "role": "user",
            "content": f"""Please generate a summary for the following conversation:

==================== CONVERSATION HISTORY ====================
{message_history}
==================== END OF CONVERSATION HISTORY ====================

Please provide the summary now."""
        }
        
        # Call LLM to generate summary
        try:
            response = self.llm_client.call_with_messages([system_message, user_message])
            
            # Format summary
            formatted_summary = f"""
======== CONVERSATION HISTORY SUMMARY ========

{response}

======== END OF SUMMARY ========

Above is a summary of the conversation history. The conversation continues below.
"""
            
            return {
                "role": "system",
                "content": formatted_summary,
                "type": "summary"
            }
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def get_messages_for_context(self, session_id: str) -> List[Dict]:
        """Get messages for current context
        
        Following Suna project's implementation, get the latest summary and all messages after it,
        or all messages if no summary exists
        
        Args:
            session_id: Session ID
            
        Returns:
            Message list for context
        """
        conn = self.db.conn
        messages = []
        
        try:
            # Find the latest summary message
            cursor = conn.execute('''
            SELECT id, created_at FROM messages
            WHERE session_id = ? AND type = 'summary' AND is_llm_message = 1
            ORDER BY created_at DESC LIMIT 1
            ''', (session_id,))
            
            summary_row = cursor.fetchone()
            
            if summary_row:
                # Found summary, get summary and all subsequent messages
                latest_summary_id = summary_row['id']
                latest_summary_time = summary_row['created_at']
                logger.debug(f"Found latest summary: {latest_summary_id}, time: {latest_summary_time}")
                
                cursor = conn.execute('''
                SELECT id, role, content, type, created_at FROM messages
                WHERE session_id = ? AND is_llm_message = 1
                AND (id = ? OR created_at > ?)
                ORDER BY created_at
                ''', (session_id, latest_summary_id, latest_summary_time))
            else:
                # No summary, get all messages
                logger.debug(f"No summary found, getting all messages for session {session_id}")
                cursor = conn.execute('''
                SELECT id, role, content, type, created_at FROM messages
                WHERE session_id = ? AND is_llm_message = 1
                ORDER BY created_at
                ''', (session_id,))
            
            # Process query results
            for row in cursor:
                content = row['content']
                # Try to parse JSON content
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass
                
                message = {
                    "id": row['id'],
                    "role": row['role'],
                    "content": content,
                    "type": row['type'],
                    "created_at": row['created_at']
                }
                
                messages.append(message)
            
            logger.debug(f"Retrieved {len(messages)} context messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting context messages: {e}")
            return []
    
    def check_and_summarize_if_needed(self, session_id: str) -> Optional[Dict]:
        """Check if summary generation is needed, and generate and store if needed
        
        Args:
            session_id: Session ID
            
        Returns:
            Summary message if generated, None otherwise
        """
        try:
            # Get all current session messages
            messages = self.get_messages_for_context(session_id)
            
            # Check if token count exceeds threshold
            token_count = self.get_token_count(messages)
        
            if token_count < self.token_threshold:
                logger.debug(f"Session {session_id} token count ({token_count}) below threshold ({self.token_threshold}), no summary needed")
                return None
                
            logger.info(f"Session {session_id} token count ({token_count}) exceeds threshold ({self.token_threshold}), generating summary")
            
            # If too few messages, don't generate summary
            if len(messages) < 3:
                logger.info(f"Session {session_id} has too few messages ({len(messages)}), not generating summary")
                return None
                
            # Generate summary
            summary = self.generate_summary(messages)
            if not summary:
                return None
                
            # Store summary in database
            conn = self.db.conn
            summary_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            conn.execute('''
            INSERT INTO messages (id, session_id, role, content, type, created_at, is_llm_message)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ''', (
                summary_id,
                session_id,
                summary["role"],
                json.dumps(summary["content"]),
                summary["type"],
                now
            ))
            
            conn.commit()
            
            logger.info(f"Generated and stored summary for session {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error checking and generating summary: {e}")
            return None
