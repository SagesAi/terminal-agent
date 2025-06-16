"""
Context Management

Manages session context, including token counting, summary generation, and context window management.
References the Suna project's implementation, but uses SQLite instead of Supabase for data storage.
"""

import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import logging

from terminal_agent.memory.memory_database import MemoryDatabase

# Get logger
logger = logging.getLogger(__name__)

# Token management constants
DEFAULT_TOKEN_THRESHOLD = 8192  # Default token threshold
SUMMARY_TARGET_TOKENS = 1000    # Summary target token count
RESERVE_TOKENS = 500            # Reserved tokens for new messages
DEFAULT_USER_MESSAGE_THRESHOLD = 2048  # Default threshold for user message compression
DEFAULT_ASSISTANT_MESSAGE_THRESHOLD = 4096  # Default threshold for assistant message compression
DEFAULT_MAX_ITERATIONS = 6  # Default max iterations for recursive compression

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

    def get_token_count(self, messages: List[Dict], model: str = None) -> int:
        """Estimate token count for a message list

        If litellm is installed, use it for accurate counting; otherwise use simple heuristic

        Args:
            messages: Message list
            model: Model name (for litellm)

        Returns:
            Estimated token count
        """
        try:
            # Try to use litellm for accurate token counting
            if model:
                try:
                    import litellm
                    return litellm.token_counter(model=model, messages=messages)
                except (ImportError, Exception) as e:
                    logger.warning(f"Cannot use litellm for token counting: {e}, falling back to heuristic method")
        except Exception as e:
            logger.warning(f"Error in token counting: {e}, falling back to heuristic method")

        # Simple estimation: each English word is about 0.6 tokens
        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            total_tokens += len(str(content).split()) * 0.6
        return int(total_tokens)

    def should_summarize(self, messages: List[Dict]) -> bool:
        """Determine if summary should be generated

        Args:
            messages: Message list

        Returns:
            True if summary should be generated, False otherwise
        """
        return self.get_token_count(messages) > self.token_threshold

    def compress_messages(self, messages: List[Dict], model: str = "gpt-4", max_tokens: Optional[int] = None) -> List[Dict]:
        """Compress message list

        Args:
            messages: Message list to compress
            model: LLM model name
            max_tokens: Optional maximum token count, if None will be determined based on model

        Returns:
            Compressed message list
        """
        # Get token count before compression
        uncompressed_token_count = self.get_token_count(messages, model)
        logger.info(f"Starting message compression: current token count {uncompressed_token_count}")

        # Call compression logic
        compressed_messages = self._compress_messages(messages, model, max_tokens=max_tokens)

        # Get token count after compression
        compressed_token_count = self.get_token_count(compressed_messages, model)
        logger.info(f"Message compression result: {uncompressed_token_count} -> {compressed_token_count} tokens")

        return compressed_messages

    def _compress_messages(self, messages: List[Dict], model: str, max_tokens: int = 41000,
                          token_threshold: int = DEFAULT_USER_MESSAGE_THRESHOLD,
                          max_iterations: int = DEFAULT_MAX_ITERATIONS) -> List[Dict]:
        """Compress messages, recursively lower threshold if still over max_tokens

        Args:
            messages: Message list
            model: LLM model
            max_tokens: Maximum token count
            token_threshold: Token threshold for compression
            max_iterations: Maximum iterations for recursive compression

        Returns:
            Compressed message list
        """

        if max_iterations <= 0:
            logger.warning("Compress messages: reached max iterations, returning uncompressed messages")
            return messages

        result = messages  # Create copy to avoid modifying original messages

        # Get token count before compression
        uncompressed_total_token_count = self.get_token_count(messages, model)

        # Compress by message type
        result_after_user = self._compress_user_messages(result, model, max_tokens, token_threshold)
        token_count_after_user = self.get_token_count(result_after_user, model)
        logger.debug(f"After user message compression: {uncompressed_total_token_count} -> {token_count_after_user}")

        result_after_assistant = self._compress_assistant_messages(result_after_user, model, max_tokens, token_threshold)
        token_count_after_assistant = self.get_token_count(result_after_assistant, model)
        logger.debug(f"After assistant message compression: {token_count_after_user} -> {token_count_after_assistant}")

        result = result_after_assistant
        compressed_token_count = token_count_after_assistant

        logger.info(f"Compressed messages: {uncompressed_total_token_count} -> {compressed_token_count}")

        # If still over max_tokens, recursively compress with lower threshold
        if compressed_token_count > max_tokens:
            # If no compression effect, directly lower the threshold and try again
            logger.warning(f"No compression effect, trying with lower threshold: {compressed_token_count} > {max_tokens}")
            result = self._compress_messages(messages, model, max_tokens, int(token_threshold / 2), max_iterations - 1)

        return result

    def _compress_user_messages(self, messages: List[Dict], model: str, max_tokens: int,
                              token_threshold: int = DEFAULT_USER_MESSAGE_THRESHOLD) -> List[Dict]:
        """Compress user messages

        Args:
            messages: Message list
            model: LLM model
            max_tokens: Maximum token count
            token_threshold: Token threshold for compression

        Returns:
            Compressed message list
        """
        result = messages.copy()

        # First check total token count
        uncompressed_total_token_count = self.get_token_count(result, model)
        logger.debug(f"Total token count before user message compression: {uncompressed_total_token_count}")

        # If total token count exceeds the maximum limit, perform compression
        if uncompressed_total_token_count > max_tokens:
            # Start traversing from the newest messages
            user_message_count = 0
            compressed_count = 0

            for msg in reversed(result):
                if msg.get("role") == "user":
                    user_message_count += 1
                    msg_id = msg.get("id", "unknown")
                    content = msg.get("content", "")

                    if isinstance(content, str):
                        # Calculate token count for the entire message
                        msg_token_count = self.get_token_count([msg], model)
                        logger.debug(f"User message {msg_id}: {msg_token_count} tokens, threshold: {token_threshold}")

                        # Only compress messages that are not the most recent
                        if user_message_count > 1 and msg_token_count > token_threshold:
                            msg["content"] = self._compress_message(msg_content=content, message_id=msg_id, max_tokens=token_threshold)
                            compressed_count += 1

            logger.info(f"Compressed {compressed_count} user messages with threshold {token_threshold}")
        else:
            logger.debug(f"No need to compress user messages, token count {uncompressed_total_token_count} <= {max_tokens}")

        return result

    def _compress_assistant_messages(self, messages: List[Dict], model: str, max_tokens: int,
                               token_threshold: int = DEFAULT_ASSISTANT_MESSAGE_THRESHOLD) -> List[Dict]:
        """Compress assistant messages

        Args:
            messages: Message list
            model: LLM model
            max_tokens: Maximum token count
            token_threshold: Token threshold for compression

        Returns:
            Compressed message list
        """
        result = messages.copy()

        # First check total token count
        uncompressed_total_token_count = self.get_token_count(result, model)
        logger.debug(f"Total token count before assistant message compression: {uncompressed_total_token_count}")

        # If total token count exceeds the maximum limit, perform compression
        if uncompressed_total_token_count > max_tokens:
            # Start traversing from the newest messages
            assistant_message_count = 0
            compressed_count = 0

            for msg in reversed(result):
                if msg.get("role") == "assistant":
                    assistant_message_count += 1
                    msg_id = msg.get("id", "unknown")
                    content = msg.get("content", "")

                    if isinstance(content, str):
                        # Calculate token count for the entire message
                        msg_token_count = self.get_token_count([msg], model)
                        logger.debug(f"Assistant message {msg_id}: {msg_token_count} tokens, threshold: {token_threshold}")

                        # Only compress messages that are not the most recent
                        if assistant_message_count > 1 and msg_token_count > token_threshold:
                            msg["content"] = self._compress_message(msg_content=content, message_id=msg_id, max_tokens=token_threshold)
                            compressed_count += 1

            logger.info(f"Compressed {compressed_count} assistant messages with threshold {token_threshold}")
        else:
            logger.debug(f"No need to compress assistant messages, token count {uncompressed_total_token_count} <= {max_tokens}")

        return result

    def _compress_message(self, msg_content: Union[str, dict], message_id: Optional[str] = None,
                        max_tokens: int = 3000, chars_per_token: float = 3.0) -> Union[str, dict]:
        """Compress single message content

        Args:
            msg_content: Message content
            message_id: Message ID
            max_tokens: Maximum token count for the message
            chars_per_token: Estimated characters per token ratio (default: 3.0 for English)

        Returns:
            Compressed message content
        """
        # 将token阈值转换为估计的字符长度
        max_length = int(max_tokens * chars_per_token)
        # Handle None content
        if msg_content is None:
            return ""

        # Handle string content
        if isinstance(msg_content, str):
            if len(msg_content) > max_length:
                truncated = msg_content[:max_length]
                # Ensure we don't cut in the middle of a Unicode character
                try:
                    truncated.encode('utf-8')
                except UnicodeEncodeError:
                    # If encoding fails, remove characters until it's valid UTF-8
                    while len(truncated) > 0:
                        try:
                            truncated.encode('utf-8')
                            break
                        except UnicodeEncodeError:
                            truncated = truncated[:-1]

                return truncated + "... (truncated)" + f"\n\nThis message is hidden, use the expand-message tool with message_id \"{message_id}\" to see the full message"
            else:
                return msg_content

        # Handle dictionary content
        elif isinstance(msg_content, dict):
            try:
                json_content = json.dumps(msg_content)
                if len(json_content) > max_length:
                    return json_content[:max_length] + "... (truncated)" + f"\n\nThis message is hidden, use the expand-message tool with message_id \"{message_id}\" to see the full message"
                else:
                    return msg_content
            except (TypeError, ValueError):
                # If JSON serialization fails, convert to string and truncate
                str_content = str(msg_content)
                if len(str_content) > max_length:
                    return str_content[:max_length] + "... (truncated)" + f"\n\nThis message is hidden, use the expand-message tool with message_id \"{message_id}\" to see the full message"
                else:
                    return str_content

        # Handle other types by converting to string
        else:
            str_content = str(msg_content)
            if len(str_content) > max_length:
                return str_content[:max_length] + "... (truncated)" + f"\n\nThis message is hidden, use the expand-message tool with message_id \"{message_id}\" to see the full message"
            else:
                return str_content

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

    def get_full_message(self, message_id: str) -> Optional[Dict]:
        """Get the full content of a message by its ID

        Args:
            message_id: Message ID

        Returns:
            Full message content or None if not found
        """
        conn = self.db.conn

        try:
            cursor = conn.execute('''
            SELECT id, role, content, type, created_at FROM messages
            WHERE id = ?
            ''', (message_id,))

            row = cursor.fetchone()
            if not row:
                logger.warning(f"Message with ID {message_id} not found")
                return None

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

            return message

        except Exception as e:
            logger.error(f"Error getting full message: {e}")
            return None

    def get_messages_for_context(self, session_id: str, model: str = "gpt-4", max_messages: int = 50) -> List[Dict]:
        """Get messages for current context

        Following Suna project's implementation, get the latest summary and all messages after it,
        or all messages if no summary exists, then apply message compression if needed.
        Limit to max_messages most recent messages to improve performance.

        Args:
            session_id: Session ID
            model: LLM model name for token counting and compression
            max_messages: Maximum number of messages to retrieve (default: 50)

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

                # Get the summary and the most recent messages after it
                cursor = conn.execute('''
                WITH recent_messages AS (
                    SELECT id, role, content, type, created_at
                    FROM messages
                    WHERE session_id = ? AND is_llm_message = 1
                    AND (id = ? OR created_at > ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                SELECT * FROM recent_messages
                ORDER BY created_at ASC
                ''', (session_id, latest_summary_id, latest_summary_time, max_messages))
            else:
                # No summary, get the most recent messages
                logger.debug(f"No summary found, getting most recent {max_messages} messages for session {session_id}")
                cursor = conn.execute('''
                WITH recent_messages AS (
                    SELECT id, role, content, type, created_at FROM messages
                    WHERE session_id = ? AND is_llm_message = 1
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                SELECT * FROM recent_messages
                ORDER BY created_at ASC
                ''', (session_id, max_messages))

            # Process query results (already in chronological order)
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

            logger.info(f"Retrieved {len(messages)} context messages for session {session_id}")

            filtered = []
            for msg in messages:
                content = msg.get("content")
                #if msg.get("role") == "user" and isinstance(content, str) and "Error" in content:
                if msg.get("role") == "user" and isinstance(content, str) and "Error" in content:
                    logger.info(f"Filtered out message {msg.get('id')} containing 'Error'")
                    continue
                filtered.append(msg)
            messages = filtered
            logger.info(f"{len(messages)} messages remain after filtering 'Error' messages")

            # Apply compression if token count exceeds threshold
            token_count = self.get_token_count(messages, model)
            logger.debug(f"Context messages token count: {token_count}")

            if token_count > self.token_threshold:
                logger.info(f"Message token count ({token_count}) exceeds threshold ({self.token_threshold}), applying compression")
                messages = self.compress_messages(messages, model, self.token_threshold)

            return messages

        except Exception as e:
            logger.error(f"Error getting context messages: {e}")
            return []

    def check_and_summarize_if_needed(self, session_id: str, model: str = "gpt-4") -> Optional[Dict]:
        """Check if summary generation is needed, and generate and store if needed

        Args:
            session_id: Session ID
            model: LLM model name for token counting

        Returns:
            Summary message if generated, None otherwise
        """
        try:
            # Get all current session messages
            messages = self.get_messages_for_context(session_id, model)

            # Check if token count exceeds threshold
            token_count = self.get_token_count(messages, model)

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
