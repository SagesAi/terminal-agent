#!/usr/bin/env python3
"""
Efficient Summary-Based Compression Strategy for Terminal Agent

This module provides a high-performance compression strategy that uses intelligent
summarization to preserve important information while significantly reducing token usage.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SummaryCompressionResult:
    """Result of summary-based compression"""
    content: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    summary_type: str
    metadata: Dict[str, Any]


class SummaryCompressor:
    """
    High-performance summary-based message compressor
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # Important patterns to preserve
        self.preservation_patterns = {
            'tool_calls': r'(📖|⚡|🔍|📁|✏️|🔗|📍|📋|📝|❓|I will)',
            'file_paths': r'[\/\\][\w\-\.\/\\]+',
            'commands': r'`[^`]+`',
            'errors': r'(error|Error|ERROR|exception|Exception|EXCEPTION|warning|Warning|WARNING)',
            'urls': r'https?://[^\s]+',
            'code_blocks': r'```[\s\S]*?```',
            'success': r'(success|Success|SUCCESS|completed|Completed|COMPLETED)',
        }
    
    def compress_messages(self, messages: List[Dict], model: str = "gpt-4", 
                         max_tokens: int = 4096, preserve_recent: int = 3) -> List[Dict]:
        """
        Compress a list of messages using summary-based strategy
        
        Args:
            messages: List of message dictionaries
            model: LLM model name
            max_tokens: Maximum token limit
            preserve_recent: Number of recent messages to preserve
            
        Returns:
            Compressed message list
        """
        logger.info(f"[SUMMARY_COMPRESSION] Starting message list compression: {len(messages)} messages, "
                   f"max_tokens={max_tokens}, preserve_recent={preserve_recent}")
        
        if not messages:
            logger.info("[SUMMARY_COMPRESSION] No messages to compress")
            return messages
        
        if len(messages) <= preserve_recent:
            logger.info(f"[SUMMARY_COMPRESSION] Messages count ({len(messages)}) <= preserve_recent ({preserve_recent}), "
                       f"no compression needed")
            return messages
        
        # Separate recent messages from older ones
        recent_messages = messages[-preserve_recent:]
        older_messages = messages[:-preserve_recent]
        
        logger.info(f"[SUMMARY_COMPRESSION] Split messages: {len(older_messages)} older + {len(recent_messages)} recent")
        
        if not older_messages:
            logger.info("[SUMMARY_COMPRESSION] No older messages to summarize")
            return messages
        
        # Create summary of older messages
        logger.info(f"[SUMMARY_COMPRESSION] Creating summary for {len(older_messages)} older messages")
        summary_message = self._create_summary_message(older_messages, model, max_tokens)
        
        # Combine summary with recent messages
        compressed_messages = [summary_message] + recent_messages
        
        # Log compression results
        original_tokens = self._estimate_tokens(messages)
        compressed_tokens = self._estimate_tokens(compressed_messages)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        logger.info(f"[SUMMARY_COMPRESSION] Compression completed: {len(messages)} -> {len(compressed_messages)} messages "
                   f"({original_tokens} -> {compressed_tokens} tokens, ratio: {compression_ratio:.2f})")
        
        if compression_ratio < 0.8:
            logger.info(f"[SUMMARY_COMPRESSION] ✅ Good compression achieved: {((1 - compression_ratio) * 100):.1f}% reduction")
        elif compression_ratio < 1.0:
            logger.info(f"[SUMMARY_COMPRESSION] ⚠️ Moderate compression: {((1 - compression_ratio) * 100):.1f}% reduction")
        else:
            logger.warning(f"[SUMMARY_COMPRESSION] ❌ No compression achieved: ratio = {compression_ratio:.2f}")
        
        return compressed_messages
    
    def compress_single_message(self, content: str, message_id: Optional[str] = None,
                               max_tokens: int = 3000) -> SummaryCompressionResult:
        """
        Compress a single message using summary strategy
        
        Args:
            content: Message content
            message_id: Optional message ID
            max_tokens: Maximum token limit
            
        Returns:
            Compression result
        """
        logger.info(f"[SINGLE_COMPRESSION] Starting single message compression: "
                   f"length={len(content)}, max_tokens={max_tokens}, message_id={message_id}")
        
        if not content:
            logger.info("[SINGLE_COMPRESSION] Empty content, no compression needed")
            return SummaryCompressionResult("", 0, 0, 1.0, "empty", {})
        
        original_length = len(content)
        estimated_tokens = self._estimate_tokens([{"content": content}])
        
        logger.info(f"[SINGLE_COMPRESSION] Content analysis: {original_length} chars, {estimated_tokens:.1f} estimated tokens")
        
        # If already under limit, return as-is
        if estimated_tokens <= max_tokens:
            logger.info(f"[SINGLE_COMPRESSION] Content within limits ({estimated_tokens:.1f} <= {max_tokens}), no compression needed")
            return SummaryCompressionResult(
                content, original_length, original_length, 1.0, 
                "no_compression_needed", {"estimated_tokens": estimated_tokens}
            )
        
        # Choose compression strategy based on content type
        logger.info(f"[SINGLE_COMPRESSION] Content exceeds limits, choosing compression strategy...")
        
        if self._has_code_blocks(content):
            logger.info("[SINGLE_COMPRESSION] Detected code blocks, using code-preserving compression")
            compressed_content = self._compress_code_content(content, max_tokens)
            summary_type = "code_preserving"
        elif self._has_tool_calls(content):
            logger.info("[SINGLE_COMPRESSION] Detected tool calls, using tool-preserving compression")
            compressed_content = self._compress_tool_content(content, max_tokens)
            summary_type = "tool_preserving"
        elif self._has_errors(content):
            logger.info("[SINGLE_COMPRESSION] Detected errors/warnings, using error-preserving compression")
            compressed_content = self._compress_error_content(content, max_tokens)
            summary_type = "error_preserving"
        else:
            logger.info("[SINGLE_COMPRESSION] Using general compression strategy")
            compressed_content = self._compress_general_content(content, max_tokens)
            summary_type = "general"
        
        compressed_length = len(compressed_content)
        compression_ratio = compressed_length / original_length
        
        logger.info(f"[SINGLE_COMPRESSION] Compression completed: {original_length} -> {compressed_length} chars "
                   f"(ratio: {compression_ratio:.2f}, type: {summary_type})")
        
        preserved_elements = self._count_preserved_elements(content)
        logger.info(f"[SINGLE_COMPRESSION] Preserved elements: {preserved_elements}")
        
        return SummaryCompressionResult(
            compressed_content, original_length, compressed_length, compression_ratio,
            summary_type, {
                "estimated_tokens": estimated_tokens,
                "message_id": message_id,
                "preserved_elements": preserved_elements
            }
        )
    
    def _create_summary_message(self, messages: List[Dict], model: str, max_tokens: int) -> Dict:
        """Create a summary message from older messages"""
        
        logger.info(f"[SUMMARY_CREATION] Creating summary for {len(messages)} messages, max_tokens={max_tokens}")
        
        # Extract key information from messages
        logger.info("[SUMMARY_CREATION] Extracting key information from messages...")
        key_info = self._extract_key_information(messages)
        
        # Log extracted information
        for key, value in key_info.items():
            if isinstance(value, list) and value:
                logger.info(f"[SUMMARY_CREATION] Extracted {len(value)} {key}: {value[:3]}{'...' if len(value) > 3 else ''}")
        
        # If we have LLM client, use it for intelligent summary
        if self.llm_client and len(messages) > 2:
            logger.info("[SUMMARY_CREATION] Using LLM for intelligent summary generation")
            try:
                summary_content = self._generate_llm_summary(messages, key_info, max_tokens)
                logger.info(f"[SUMMARY_CREATION] LLM summary generated: {len(summary_content)} characters")
            except Exception as e:
                logger.warning(f"[SUMMARY_CREATION] LLM summary failed: {e}, falling back to pattern-based summary")
                summary_content = self._generate_pattern_summary(key_info, max_tokens)
        else:
            logger.info("[SUMMARY_CREATION] Using pattern-based summary generation")
            summary_content = self._generate_pattern_summary(key_info, max_tokens)
        
        logger.info(f"[SUMMARY_CREATION] Summary created: {len(summary_content)} characters")
        
        return {
            "role": "system",
            "content": summary_content,
            "type": "summary"
        }
    
    def _extract_key_information(self, messages: List[Dict]) -> Dict[str, Any]:
        """Extract key information from messages"""
        key_info = {
            "tool_calls": [],
            "file_paths": [],
            "commands": [],
            "errors": [],
            "successes": [],
            "urls": [],
            "code_blocks": [],
            "key_points": []
        }
        
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            
            # Extract tool calls
            tool_calls = re.findall(self.preservation_patterns['tool_calls'], content)
            key_info["tool_calls"].extend(tool_calls)
            
            # Extract file paths
            file_paths = re.findall(self.preservation_patterns['file_paths'], content)
            key_info["file_paths"].extend(file_paths)
            
            # Extract commands
            commands = re.findall(self.preservation_patterns['commands'], content)
            key_info["commands"].extend(commands)
            
            # Extract errors and warnings
            errors = re.findall(self.preservation_patterns['errors'], content)
            key_info["errors"].extend(errors)
            
            # Extract success messages
            successes = re.findall(self.preservation_patterns['success'], content)
            key_info["successes"].extend(successes)
            
            # Extract URLs
            urls = re.findall(self.preservation_patterns['urls'], content)
            key_info["urls"].extend(urls)
            
            # Extract code blocks
            code_blocks = re.findall(self.preservation_patterns['code_blocks'], content)
            key_info["code_blocks"].extend(code_blocks)
            
            # Extract key sentences
            sentences = content.split('. ')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in 
                      ['success', 'error', 'file', 'command', 'tool', 'result', 'found', 'created', 'analyzed']):
                    key_info["key_points"].append(sentence.strip())
        
        # Remove duplicates and limit counts
        for key in key_info:
            if isinstance(key_info[key], list):
                key_info[key] = list(set(key_info[key]))[:10]  # Limit to 10 items
        
        return key_info
    
    def _generate_llm_summary(self, messages: List[Dict], key_info: Dict, max_tokens: int) -> str:
        """Generate summary using LLM"""
        
        # Prepare message history for LLM
        message_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            message_text += f"{role.upper()}: {content}\n\n"
        
        # Create summary prompt
        summary_prompt = f"""
Please provide a concise but comprehensive summary of the conversation above.
Focus on:
1. Key actions and decisions made
2. Important files and paths mentioned
3. Tool calls and their purposes
4. Error messages and warnings
5. Successful operations and results
6. Commands executed

Keep the summary under {max_tokens * 3} characters while preserving all essential information.
Format the summary with clear sections and bullet points.

Conversation to summarize:
{message_text}
"""
        
        # Generate summary
        summary = self.llm_client.call_with_messages([
            {"role": "user", "content": summary_prompt}
        ], show_progress=False)
        
        return f"""
======== CONVERSATION SUMMARY ========

{summary}

======== END OF SUMMARY ========

Above is a summary of the previous conversation. The conversation continues below.
"""
    
    def _generate_pattern_summary(self, key_info: Dict, max_tokens: int) -> str:
        """Generate summary using pattern extraction"""
        
        summary_parts = []
        
        # Add tool calls
        if key_info["tool_calls"]:
            summary_parts.append("**Tool Calls:**")
            for tool in key_info["tool_calls"][:5]:
                summary_parts.append(f"- {tool}")
        
        # Add file paths
        if key_info["file_paths"]:
            summary_parts.append("**Files:**")
            for file_path in key_info["file_paths"][:8]:
                summary_parts.append(f"- {file_path}")
        
        # Add commands
        if key_info["commands"]:
            summary_parts.append("**Commands:**")
            for cmd in key_info["commands"][:5]:
                summary_parts.append(f"- {cmd}")
        
        # Add errors and warnings
        if key_info["errors"]:
            summary_parts.append("**Issues:**")
            for error in key_info["errors"][:3]:
                summary_parts.append(f"- {error}")
        
        # Add successes
        if key_info["successes"]:
            summary_parts.append("**Successes:**")
            for success in key_info["successes"][:3]:
                summary_parts.append(f"- {success}")
        
        # Add key points
        if key_info["key_points"]:
            summary_parts.append("**Key Points:**")
            for point in key_info["key_points"][:5]:
                summary_parts.append(f"- {point}")
        
        # Add URLs
        if key_info["urls"]:
            summary_parts.append("**References:**")
            for url in key_info["urls"][:3]:
                summary_parts.append(f"- {url}")
        
        if not summary_parts:
            summary_parts.append("**Summary:** Previous conversation contained general discussion.")
        
        summary_content = "\n\n".join(summary_parts)
        
        return f"""
======== CONVERSATION SUMMARY ========

{summary_content}

======== END OF SUMMARY ========

Above is a summary of the previous conversation. The conversation continues below.
"""
    
    def _compress_code_content(self, content: str, max_tokens: int) -> str:
        """Compress content with code blocks"""
        # Preserve code blocks and compress surrounding text
        code_blocks = re.findall(self.preservation_patterns['code_blocks'], content)
        
        # Remove code blocks from content for compression
        content_without_code = re.sub(self.preservation_patterns['code_blocks'], '[CODE_BLOCK]', content)
        
        # Compress the non-code content
        compressed_text = self._compress_text_preserving_patterns(content_without_code, max_tokens // 2)
        
        # Reinsert code blocks
        result = compressed_text
        for i, code_block in enumerate(code_blocks[:3]):  # Limit to 3 code blocks
            result = result.replace('[CODE_BLOCK]', code_block, 1)
        
        # Remove remaining placeholders
        result = result.replace('[CODE_BLOCK]', '')
        
        return result
    
    def _compress_tool_content(self, content: str, max_tokens: int) -> str:
        """Compress content with tool calls"""
        # Extract tool calls
        tool_calls = re.findall(r'(📖|⚡|🔍|📁|✏️|🔗|📍|📋|📝|❓|I will)[^\\n]*', content)
        
        # Create compressed content focusing on tools
        compressed_parts = []
        
        if tool_calls:
            compressed_parts.append("**Tool Operations:**")
            for tool in tool_calls[:5]:
                compressed_parts.append(f"- {tool.strip()}")
        
        # Add other important patterns
        compressed_parts.extend(self._extract_other_patterns(content))
        
        return "\n\n".join(compressed_parts)
    
    def _compress_error_content(self, content: str, max_tokens: int) -> str:
        """Compress content with errors and warnings"""
        # Extract errors and warnings
        errors = re.findall(r'(error|Error|ERROR|exception|Exception|EXCEPTION|warning|Warning|WARNING)[^\\n]*', content)
        
        compressed_parts = []
        
        if errors:
            compressed_parts.append("**Issues Found:**")
            for error in errors[:5]:
                compressed_parts.append(f"- {error.strip()}")
        
        # Add other important patterns
        compressed_parts.extend(self._extract_other_patterns(content))
        
        return "\n\n".join(compressed_parts)
    
    def _compress_general_content(self, content: str, max_tokens: int) -> str:
        """Compress general content"""
        return self._compress_text_preserving_patterns(content, max_tokens)
    
    def _compress_text_preserving_patterns(self, content: str, max_tokens: int) -> str:
        """Compress text while preserving important patterns"""
        max_length = max_tokens * 3  # Convert tokens to characters
        
        if len(content) <= max_length:
            return content
        
        # Extract important patterns first
        important_parts = []
        
        # Extract file paths
        file_paths = re.findall(self.preservation_patterns['file_paths'], content)
        if file_paths:
            important_parts.append("**Files:** " + ", ".join(file_paths[:5]))
        
        # Extract commands
        commands = re.findall(self.preservation_patterns['commands'], content)
        if commands:
            important_parts.append("**Commands:** " + ", ".join(commands[:3]))
        
        # Extract URLs
        urls = re.findall(self.preservation_patterns['urls'], content)
        if urls:
            important_parts.append("**References:** " + ", ".join(urls[:3]))
        
        # Smart truncation with important parts preserved
        head_size = int(max_length * 0.4)
        tail_size = int(max_length * 0.3)
        middle_size = max_length - head_size - tail_size
        
        head = content[:head_size]
        tail = content[-tail_size:]
        
        # Combine parts
        result_parts = [head]
        
        if important_parts:
            result_parts.append("\n\n**Key Information:**\n" + "\n".join(important_parts))
        
        result_parts.append(f"\n\n[... {len(content) - head_size - tail_size} characters omitted ...]\n\n")
        result_parts.append(tail)
        
        return "".join(result_parts)
    
    def _extract_other_patterns(self, content: str) -> List[str]:
        """Extract other important patterns from content"""
        patterns = []
        
        # File paths
        file_paths = re.findall(self.preservation_patterns['file_paths'], content)
        if file_paths:
            patterns.append(f"**Files:** {', '.join(file_paths[:3])}")
        
        # Commands
        commands = re.findall(self.preservation_patterns['commands'], content)
        if commands:
            patterns.append(f"**Commands:** {', '.join(commands[:3])}")
        
        # Success messages
        successes = re.findall(self.preservation_patterns['success'], content)
        if successes:
            patterns.append(f"**Success:** {', '.join(successes[:2])}")
        
        return patterns
    
    def _has_code_blocks(self, content: str) -> bool:
        """Check if content has code blocks"""
        return bool(re.search(self.preservation_patterns['code_blocks'], content))
    
    def _has_tool_calls(self, content: str) -> bool:
        """Check if content has tool calls"""
        return bool(re.search(self.preservation_patterns['tool_calls'], content))
    
    def _has_errors(self, content: str) -> bool:
        """Check if content has errors or warnings"""
        return bool(re.search(self.preservation_patterns['errors'], content))
    
    def _count_preserved_elements(self, content: str) -> Dict[str, int]:
        """Count preserved elements in content"""
        counts = {}
        for pattern_name, pattern in self.preservation_patterns.items():
            matches = re.findall(pattern, content)
            counts[pattern_name] = len(matches)
        return counts
    
    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """Estimate token count for messages"""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            total_chars += len(str(content))
        return int(total_chars / 4)  # Rough estimation: 4 chars per token


# Convenience function for direct integration
def compress_messages_with_summary(messages: List[Dict], model: str = "gpt-4", 
                                  max_tokens: int = 4096, preserve_recent: int = 3,
                                  llm_client=None) -> List[Dict]:
    """
    Compress messages using summary-based strategy
    
    Args:
        messages: List of message dictionaries
        model: LLM model name
        max_tokens: Maximum token limit
        preserve_recent: Number of recent messages to preserve
        llm_client: Optional LLM client for intelligent summaries
        
    Returns:
        Compressed message list
    """
    compressor = SummaryCompressor(llm_client)
    return compressor.compress_messages(messages, model, max_tokens, preserve_recent)


def compress_single_message_with_summary(content: str, message_id: Optional[str] = None,
                                        max_tokens: int = 3000, llm_client=None) -> str:
    """
    Compress a single message using summary strategy
    
    Args:
        content: Message content
        message_id: Optional message ID
        max_tokens: Maximum token limit
        llm_client: Optional LLM client for intelligent summaries
        
    Returns:
        Compressed content
    """
    compressor = SummaryCompressor(llm_client)
    result = compressor.compress_single_message(content, message_id, max_tokens)
    return result.content 