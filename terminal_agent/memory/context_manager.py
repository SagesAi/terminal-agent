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
import sys
import warnings

# Suppress pkg_resources deprecation warning from litellm
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources is deprecated.*')

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

from terminal_agent.memory.memory_database import MemoryDatabase
from terminal_agent.react.history_compression_agent import HistoryCompressionAgent

# Get logger
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()

# Token management constants
DEFAULT_TOKEN_THRESHOLD = 8192  # Default token threshold
SUMMARY_TARGET_TOKENS = 1000    # Summary target token count
RESERVE_TOKENS = 500            # Reserved tokens for new messages
DEFAULT_USER_MESSAGE_THRESHOLD = 2048  # Default threshold for user message compression
DEFAULT_ASSISTANT_MESSAGE_THRESHOLD = 4096  # Default threshold for assistant message compression
DEFAULT_MAX_ITERATIONS = 8  # Default max iterations for recursive compression
MIN_TOKEN_THRESHOLD = 50  # Minimum threshold to prevent excessive compression

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
        self.compression_agent = HistoryCompressionAgent(llm_client, system_info={"name": "CompressAgent"})

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
                    # Preprocess messages to ensure all content is properly stringified
                    processed_messages = []
                    for msg in messages:
                        processed_msg = msg.copy()
                        content = msg.get("content", "")
                        if isinstance(content, dict):
                            processed_msg["content"] = json.dumps(content)
                        elif not isinstance(content, str):
                            processed_msg["content"] = str(content)
                        processed_messages.append(processed_msg)

                    return litellm.token_counter(model=model, messages=processed_messages)
                except (ImportError, Exception) as e:
                    logger.warning(f"Cannot use litellm for token counting: {e}, falling back to heuristic method")
        except Exception as e:
            logger.warning(f"Error in token counting: {e}, falling back to heuristic method")

        # Simple estimation: each English word is about 0.3 tokens
        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            elif not isinstance(content, str):
                content = str(content)
            total_tokens += len(content.split()) * 0.3
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
        """Compress messages using while loop until token count is within limit

        Args:
            messages: Message list
            model: LLM model
            max_tokens: Maximum token count
            token_threshold: Token threshold for compression
            max_iterations: Maximum iterations for recursive compression

        Returns:
            Compressed message list
        """
        # check if initial token count is within limit
        initial_token_count = self.get_token_count(messages, model)
        logger.info(f"Initial token count: {initial_token_count}, max_tokens: {max_tokens}")
        if initial_token_count <= max_tokens:
            return messages

        # initialize state variables - always start from original messages
        current_threshold = token_threshold
        current_messages = messages.copy()
        current_token_count = initial_token_count

        # 设置初始进度为10%，确保用户能看到进度条在移动
        with Progress(
            "[cyan]{task.description}",
            SpinnerColumn("dots"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            "[bold]{task.fields[status]}",
            TimeElapsedColumn(),
            console=console,
            transient=False,  # Set to False to keep progress visible
            expand=True
        ) as progress:
            task = progress.add_task("[bold blue]Optimizing context", total=100, status="")
            # 立即更新进度到10%，确保用户看到进度条在移动
            progress.update(task, completed=10, status="[cyan]Starting...[/cyan]")

            # core loop: continue trying compression as long as token count exceeds limit
            while current_token_count > max_tokens:
                # check if threshold has reached minimum
                if current_threshold < MIN_TOKEN_THRESHOLD:
                    progress.update(task, status="[yellow]Optimizing...[/yellow]")
                    logger.warning(
                        f"Token threshold {current_threshold} below minimum {MIN_TOKEN_THRESHOLD}, "
                        "stopping compression"
                    )
                    return current_messages

                # important: always start from original messages to apply new threshold
                # compress user messages
                user_compressed = self._compress_user_messages(
                    messages, model, max_tokens, current_threshold  # use original messages
                )
                user_token_count = self.get_token_count(user_compressed, model)
                logger.debug(f"After user compression: {initial_token_count} -> {user_token_count}")

                # Update progress - ensure at least 20% progress
                progress_percent = max(20, min(100, int((1 - (user_token_count / initial_token_count)) * 50)))
                progress.update(task, completed=progress_percent, status="[cyan]Optimizing...[/cyan]")

                # compress assistant messages
                assistant_compressed = self._compress_assistant_messages(
                    user_compressed, model, max_tokens, current_threshold
                )
                assistant_token_count = self.get_token_count(assistant_compressed, model)
                logger.debug(f"After assistant compression: {user_token_count} -> {assistant_token_count}")

                # Update progress - ensure at least 50% progress
                progress_percent = max(50, min(100, int((1 - (assistant_token_count / initial_token_count)) * 100)))
                progress.update(task, completed=progress_percent, status="[magenta]Optimizing...[/magenta]")

                # update state
                current_messages = assistant_compressed
                current_token_count = assistant_token_count

                # Check if compression was successful
                if current_token_count <= max_tokens:
                    progress.update(task, completed=100, status="[green]Complete[/green]")
                    # Don't show compression completion panel
                    logger.info(f"Compression successful: {current_token_count} tokens ≤ {max_tokens}")
                    return current_messages

                # lower threshold for next iteration
                logger.debug(
                    f"Compression insufficient: {current_token_count} > {max_tokens}, "
                    f"lowering threshold to {current_threshold//2}"
                )
                current_threshold = current_threshold // 2

        # theoretically will not reach here
        return current_messages


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

    def compress_react_messages(self, messages: List[Dict], model: str = "gpt-4",
                          max_tokens_threshold: Optional[int] = None,
                          recent_message_count: int = 5) -> List[Dict]:
        """
        Compress message history during React execution process, replacing older messages with summaries

        Args:
            messages: List of message history
            model: LLM model name
            max_tokens_threshold: Token threshold, uses default threshold if None
            recent_message_count: Number of recent messages to preserve

        Returns:
            Compressed message list
        """
        # If no threshold specified, use default threshold
        if max_tokens_threshold is None:
            max_tokens_threshold = self.token_threshold

        # Estimate current message token count
        current_tokens = self.get_token_count(messages, model)

        # If under threshold, return original messages
        if current_tokens <= max_tokens_threshold:
            return messages

        logger.info(f"Message history exceeds token threshold ({current_tokens} > {max_tokens_threshold}). Compressing...")

        # Preserve system messages
        system_messages = [m for m in messages if m.get("role") == "system"]

        # Separate user and assistant messages
        non_system_messages = [m for m in messages if m.get("role") != "system"]

        # If too few messages, no need to compress
        if len(non_system_messages) <= recent_message_count:
            return messages

        # Split messages: preserve recent messages, compress older ones
        older_messages, recent_messages = non_system_messages[:-recent_message_count], non_system_messages[-recent_message_count:]

        # If no messages to compress, return original messages
        if not older_messages:
            return messages

        # Build summary prompt
        summary_prompt = (
            "Provide a detailed but concise summary of our conversation above. "
            "Focus on information that would be helpful for continuing the conversation, "
            "including what we did, what we're doing, which files we're working on, and what we're going to do next."
        )

        # Convert messages to be compressed into text
        messages_text = ""
        for msg in older_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            messages_text += f"{role.upper()}: {content}\n\n"

        # Use LLM to generate summary
        if self.llm_client:
            try:
                # void progress bar when calling LLM to generate summary to avoid conflict with ReAct execution process
                summary_response = self.llm_client.call_with_messages([
                    {"role": "user", "content": messages_text + "\n" + summary_prompt}
                ], show_progress=False)

                # Create new message list with summary
                compressed_messages = system_messages.copy()

                # Add summary as assistant message
                summary_message = {
                    "role": "assistant",
                    "content": f"Previous conversation summary: {summary_response}"
                }
                compressed_messages.append(summary_message)

                # Add recent messages
                compressed_messages.extend(recent_messages)

                # Calculate token count after compression
                compressed_tokens = self.get_token_count(compressed_messages, model)
                logger.info(f"Compressed {len(messages)} messages to {len(compressed_messages)} messages ({current_tokens} -> {compressed_tokens} tokens)")

                return compressed_messages
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                # Return original messages if compression fails
                return messages
        else:
            logger.warning("No LLM client available for summary generation, returning original messages")
            return messages

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

    def generate_summary(self, messages: List[Dict]) -> List[Dict[str, Any]]:
        """Generate conversation summary

        Args:
            messages: Messages to summarize

        Returns:
            Summary message dictionary, None if generation fails
        """
        if not self.llm_client or not messages:
            return []


        compressed = self.compression_agent.compress_history(messages)

        return compressed


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

    def get_messages_for_summary(self, session_id: str, model: str = "gpt-4") -> List[Dict]:
        """Get messages for summary

        Args:
            session_id: Session ID (required)
            model: LLM model name for token counting and compression (default: "gpt-4")

        Returns:
            List[Dict]: List of message dictionaries formatted for LLM processing

        Raises:
            ValueError: If session_id is invalid
            sqlite3.Error: If there's a database error
            Exception: For any other unexpected errors
        """
        if not session_id or not isinstance(session_id, str):
            logger.error("Invalid session_id: must be a non-empty string")
            raise ValueError("session_id must be a non-empty string")

        if not hasattr(self, 'db') or not hasattr(self.db, 'conn'):
            logger.error("Database connection not initialized")
            raise RuntimeError("Database connection not available")

        conn = self.db.conn
        messages = []
        cursor = None

        try:
            if conn is None:
                logger.error("Database connection is None")
                raise RuntimeError("Database connection is not available")

            # Find the latest summary message
            try:
                cursor = conn.execute('''
                SELECT id, created_at FROM messages
                WHERE session_id = ? AND type = 'summary' AND is_llm_message = 1
                ORDER BY created_at DESC LIMIT 1
                ''', (session_id,))
                summary_row = cursor.fetchone()
                cursor.close()

                if summary_row:
                    # Found summary, get summary and all subsequent messages
                    latest_summary_id = summary_row['id']
                    latest_summary_time = summary_row['created_at']
                    logger.debug(f"Found latest summary: {latest_summary_id}, time: {latest_summary_time}")

                    cursor = conn.execute('''
                    SELECT id, role, content, type, created_at, metadata
                    FROM messages
                    WHERE session_id = ? AND is_llm_message = 1
                    AND (id = ? OR created_at > ?)
                    ORDER BY created_at ASC
                    ''', (session_id, latest_summary_id, latest_summary_time))
                else:
                    # No summary, get the most recent messages
                    logger.debug(f"No summary found, getting all messages for session {session_id}")
                    cursor = conn.execute('''
                    SELECT id, role, content, type, created_at, metadata
                    FROM messages
                    WHERE session_id = ? AND is_llm_message = 1
                    ORDER BY created_at ASC
                    ''', (session_id,))
            except sqlite3.Error as e:
                logger.error(f"Database error: {str(e)}")
                raise

            # Process query results with error handling
            rows = cursor.fetchall()
            if not rows:
                logger.info(f"No messages found for session {session_id}")
                return []

            for row in rows:
                try:
                    content = row['content']
                    metadata = row['metadata']

                    # Safely parse JSON content
                    if content and isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.debug(f"Could not parse message content as JSON: {str(e)}")

                    # Safely parse metadata
                    parsed_metadata = None
                    if metadata and isinstance(metadata, str):
                        try:
                            parsed_metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.debug(f"Could not parse metadata as JSON: {str(e)}")
                            parsed_metadata = metadata

                    message = {
                        "id": row['id'],
                        "role": row['role'],
                        "content": content,
                        "type": row['type'],
                        "created_at": row['created_at'],
                        "metadata": parsed_metadata
                    }
                    messages.append(message)

                except Exception as e:
                    logger.error(f"Error processing message row: {str(e)}")
                    continue

            # Convert to LLM message format with error handling
            llm_messages = []
            for msg in messages:
                try:
                    # Handle tool messages with OpenAI format
                    if msg.get("role") == "tool" and msg.get("metadata") and msg["metadata"].get("tool_call_id"):
                        llm_message = {
                            "role": "tool",
                            "tool_call_id": msg["metadata"]["tool_call_id"],
                            "content": str(msg.get("content", "")) if not isinstance(msg.get("content"), str) else msg["content"]
                        }
                    # Handle assistant messages with tool calls
                    elif msg.get("role") == "assistant" and msg.get("metadata") and msg["metadata"].get("tool_calls"):
                        llm_message = {
                            "role": "assistant",
                            "content": str(msg.get("content", "")) if not isinstance(msg.get("content"), str) else msg["content"],
                            "tool_calls": msg["metadata"].get("tool_calls", [])
                        }
                    # Handle regular messages
                    else:
                        content = msg.get("content", "")
                        if isinstance(content, dict) and content.get("_truncated_content"):
                            content = content.get("_message", "[Content was truncated due to length]")
                        elif not isinstance(content, str):
                            content = str(content)

                        llm_message = {
                            "role": msg.get("role", "user"),
                            "content": content
                        }

                    llm_messages.append(llm_message)

                except Exception as e:
                    logger.error(f"Error formatting message for LLM: {str(e)}")
                    continue

            return llm_messages

        except sqlite3.Error as e:
            logger.error(f"Database error in get_messages_for_summary: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in get_messages_for_summary: {str(e)}")
            raise

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.error(f"Error closing cursor: {str(e)}")



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
                SELECT id, role, content, type, created_at, metadata
                FROM messages
                WHERE session_id = ? AND is_llm_message = 1
                AND (id = ? OR created_at > ?)
                ORDER BY created_at ASC
                ''', (session_id, latest_summary_id, latest_summary_time))
            else:
                # No summary, get the most recent messages
                logger.debug(f"No summary found, getting all messages for session {session_id}")
                cursor = conn.execute('''
                SELECT id, role, content, type, created_at, metadata
                FROM messages
                WHERE session_id = ? AND is_llm_message = 1
                ORDER BY created_at ASC
                ''', (session_id,))

            # Process query results (already in chronological order)
            for row in cursor:
                content = row['content']
                metadata = row['metadata']

                # Try to parse JSON content
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

                # Try to parse JSON metadata
                if metadata:
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        pass

                message = {
                    "id": row['id'],
                    "role": row['role'],
                    "content": content,
                    "type": row['type'],
                    "created_at": row['created_at'],
                    "metadata": metadata
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
                console.print(f"[bold blue]Optimizing context[/bold blue]")
                logger.info(f"Message token count ({token_count}) exceeds threshold ({self.token_threshold}), applying compression")
                messages = self.compress_messages(messages, model, self.token_threshold)

            return messages

        except Exception as e:
            logger.error(f"Error getting context messages: {e}")
            return []

    def check_and_summarize_if_needed(self, session_id: str, model: str = "gpt-4") -> List[Dict[str, Any]]:
        """Check if summary generation is needed, and generate and store if needed

        Args:
            session_id: Session ID
            model: LLM model name for token counting

        Returns:
            List of message dictionaries. If a summary was generated, returns a list
            containing the summary message. If no summary was needed, returns the
            original list of messages. Returns an empty list on error.
        """
        try:
            # Get all current session messages
            messages = self.get_messages_for_summary(session_id, model)

            # Check if token count exceeds threshold
            token_count = self.get_token_count(messages, model)

            if token_count < self.token_threshold:
                logger.debug(f"Session {session_id} token count ({token_count}) below threshold ({self.token_threshold}), no summary needed")
                return messages

            console.print(f"[bold yellow]Generating summary...[/bold yellow]")
            logger.info(f"Session {session_id} token count ({token_count}) exceeds threshold ({self.token_threshold}), generating summary")

            # If too few messages, don't generate summary
            #if len(messages) < 3:
            #    logger.info(f"Session {session_id} has too few messages ({len(messages)}), not generating summary")
            #    return messages

            # Generate summary
            summary = self.generate_summary(messages)
            if not summary:
                logger.warning("Failed to generate summary, returning original messages")
                return messages

            try:
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
                    json.dumps(summary["content"] if isinstance(summary["content"], str) else summary["content"]),
                    "summary",
                    now
                ))
                conn.commit()

                logger.info(f"Generated and stored summary for session {session_id}")
                # Return the summary as a single-item list to maintain consistent return type
                return [summary]
                
            except Exception as db_error:
                logger.error(f"Error storing summary in database: {db_error}")
                # Return original messages if we couldn't store the summary
                return messages

        except Exception as e:
            logger.error(f"Error checking and generating summary: {e}")
            return []
