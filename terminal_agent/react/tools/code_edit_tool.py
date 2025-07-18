#!/usr/bin/env python3
"""
Code Edit Tool for Terminal Agent.
Provides the ability to edit code files with proper syntax checking and formatting.
"""

import os
import json
import logging
import traceback
import subprocess
import tempfile
import ast
import difflib
import re
from typing import Dict, Any, Optional, Union, List, Tuple


# Import command forwarder for remote file operations
from terminal_agent.utils.command_forwarder import forwarder, remote_file_operation

# Import diff generation and rendering components
try:
    from terminal_agent.diff.diff_generator import DiffGenerator
    from terminal_agent.diff.diff_renderer import DiffRenderer
    HAS_DIFF_TOOLS = True
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported diff tools")
except ImportError as e:
    HAS_DIFF_TOOLS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import diff tools: {str(e)}")

# Configure logging
logger = logging.getLogger(__name__)

class CodeEditError(Exception):
    """Base exception for code editing errors"""
    pass

class MatchNotFoundError(CodeEditError):
    """Raised when content cannot be found in the file"""
    pass

class AmbiguousMatchError(CodeEditError):
    """Raised when multiple matches are found in the file"""
    pass

class CodeEditor:
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize the code editor with configurable settings.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider a match
        """
        self.similarity_threshold = similarity_threshold
        self.strategies = [
            self._try_exact_match,
            self._try_normalized_match,
            self._try_relative_indent_match,
            self._try_fuzzy_line_match,
        ]
    
    def find_best_match(self, file_content: str, search_text: str) -> Tuple[int, int, float]:
        """
        Find the best match of search_text in file_content using multiple matching strategies.
        
        Tries different matching strategies in order of specificity, from most exact to most fuzzy.
        
        Returns:
            Tuple of (start_line, end_line, similarity_score)
            
        Raises:
            MatchNotFoundError: If no good match is found with any strategy
        """
        if not search_text.strip():
            raise ValueError("Search text cannot be empty or whitespace only")
            
        file_lines = file_content.splitlines(keepends=True)
        search_lines = search_text.splitlines(keepends=True)
        
        # Try each strategy in order until we find a match
        for strategy in self.strategies:
            try:
                result = strategy(file_content, file_lines, search_text, search_lines)
                if result:
                    start_line, end_line, score = result
                    if score >= self.similarity_threshold:
                        return start_line, end_line, score
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {str(e)}")
                continue
        
        # If we get here, no strategy found a good enough match
        best_ratio = self._calculate_best_ratio(file_content, search_text)
        raise MatchNotFoundError(
            f"Could not find a good match (best similarity: {best_ratio:.2f}, threshold: {self.similarity_threshold}).\n"
            "To improve matching, try:\n"
            "1. Ensure the code block includes more unique identifiers (function/class names, unique strings)\n"
            "2. Include surrounding context (2-3 lines before/after the change)\n"
            "3. Check for consistent indentation and whitespace\n"
            "4. Provide the exact code block from the file, including all whitespace"
        )
    
    def _try_exact_match(self, file_content: str, file_lines: List[str], 
                        search_text: str, search_lines: List[str]) -> Optional[Tuple[int, int, float]]:
        """Strategy 1: Exact character-by-character match"""
        if search_text in file_content:
            start_char = file_content.index(search_text)
            start_line, end_line = self._char_pos_to_line_range(file_content, start_char, len(search_text))
            return start_line, end_line, 1.0
        return None
    
    def _try_normalized_match(self, file_content: str, file_lines: List[str], 
                            search_text: str, search_lines: List[str]) -> Optional[Tuple[int, int, float]]:
        """Strategy 2: Normalized whitespace match"""
        normalized_file = self._normalize_whitespace(file_content)
        normalized_search = self._normalize_whitespace(search_text)
        
        if normalized_search in normalized_file:
            start_char = normalized_file.index(normalized_search)
            # Create a mapping from normalized to original positions
            norm_to_orig = self._create_position_mapping(file_content, normalized_file)
            if start_char in norm_to_orig:
                orig_start = norm_to_orig[start_char]
                orig_end = norm_to_orig.get(start_char + len(normalized_search), len(file_content))
                start_line, end_line = self._char_pos_to_line_range(file_content, orig_start, orig_end - orig_start)
                return start_line, end_line, 0.95  # Slightly lower score than exact match
        return None
    
    def _try_relative_indent_match(self, file_content: str, file_lines: List[str], 
                                 search_text: str, search_lines: List[str]) -> Optional[Tuple[int, int, float]]:
        """Strategy 3: Match relative indentation patterns"""
        if len(search_lines) < 2:  # Need multiple lines for relative indentation
            return None
            
        # Calculate relative indentation for search text
        search_indents = [len(line) - len(line.lstrip()) for line in search_lines if line.strip()]
        if not search_indents:
            return None
            
        base_indent = search_indents[0]
        search_rel_indents = [indent - base_indent for indent in search_indents]
        
        # Search through the file for matching indentation patterns
        for i in range(len(file_lines) - len(search_lines) + 1):
            window = file_lines[i:i + len(search_lines)]
            window_indents = [len(line) - len(line.lstrip()) for line in window if line.strip()]
            
            if len(window_indents) != len(search_indents):
                continue
                
            window_base = window_indents[0]
            window_rel_indents = [indent - window_base for indent in window_indents]
            
            if window_rel_indents == search_rel_indents:
                # Verify content similarity
                window_text = ''.join(window)
                ratio = difflib.SequenceMatcher(None, search_text, window_text).ratio()
                if ratio >= self.similarity_threshold:
                    return i + 1, i + len(search_lines), ratio
        
        return None
    
    def _try_fuzzy_line_match(self, file_content: str, file_lines: List[str], 
                            search_text: str, search_lines: List[str]) -> Optional[Tuple[int, int, float]]:
        """Strategy 4: Fuzzy line-based matching"""
        if not search_lines:
            return None
            
        best_match = None
        best_ratio = 0.0
        search_text = ''.join(search_lines)
        search_len = len(search_lines)
        
        # Try different window sizes around the expected size
        for window in range(max(1, search_len - 2), min(search_len + 3, len(file_lines) + 1)):
            for i in range(len(file_lines) - window + 1):
                chunk = file_lines[i:i+window]
                chunk_text = ''.join(chunk)
                
                # Calculate similarity with line-based normalization
                ratio = self._calculate_similarity(search_text, chunk_text)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (i + 1, i + window, ratio)  # 1-based line numbers
        
        return best_match if best_ratio >= self.similarity_threshold else None
    
    def _calculate_best_ratio(self, file_content: str, search_text: str) -> float:
        """Calculate the best similarity ratio between search_text and any part of file_content"""
        file_lines = file_content.splitlines(keepends=True)
        search_lines = search_text.splitlines(keepends=True)
        best_ratio = 0.0
        
        # Try different matching strategies to find the best ratio
        for strategy in [self._try_exact_match, self._try_normalized_match, 
                        self._try_relative_indent_match, self._try_fuzzy_line_match]:
            try:
                result = strategy(file_content, file_lines, search_text, search_lines)
                if result and result[2] > best_ratio:
                    best_ratio = result[2]
            except Exception:
                continue
                
        return best_ratio
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts with normalization"""
        # Normalize whitespace for better matching
        norm1 = self._normalize_whitespace(text1)
        norm2 = self._normalize_whitespace(text2)
        
        # Use SequenceMatcher with different strategies
        matcher = difflib.SequenceMatcher(None, norm1, norm2)
        ratio = matcher.ratio()
        
        # If ratio is low, try with different parameters
        if ratio < 0.5:
            matcher = difflib.SequenceMatcher(lambda x: x in " \t\n", norm1, norm2)
            ratio = max(ratio, matcher.ratio() * 0.9)  # Slightly penalize whitespace-only matches
            
        return ratio
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for better matching"""
        # Replace all whitespace sequences with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing space
        return text.strip()
    
    def _char_pos_to_line_range(self, text: str, start_char: int, length: int) -> Tuple[int, int]:
        """Convert character positions to line numbers (1-based)"""
        lines = text.splitlines(keepends=True)
        char_count = 0
        start_line = end_line = 1
        
        for i, line in enumerate(lines, 1):
            line_length = len(line)
            if char_count <= start_char < char_count + line_length:
                start_line = i
            if char_count < start_char + length <= char_count + line_length:
                end_line = i
                break
            char_count += line_length
        
        return start_line, end_line
    
    def _create_position_mapping(self, original: str, normalized: str) -> Dict[int, int]:
        """Create a mapping from normalized string positions to original string positions"""
        mapping = {}
        orig_pos = norm_pos = 0
        
        while orig_pos < len(original) and norm_pos < len(normalized):
            if original[orig_pos].isspace() and normalized[norm_pos] == ' ':
                # Skip all whitespace in original until we hit the next non-whitespace
                while orig_pos < len(original) and original[orig_pos].isspace():
                    orig_pos += 1
                norm_pos += 1
            elif orig_pos < len(original) and norm_pos < len(normalized) and original[orig_pos] == normalized[norm_pos]:
                mapping[norm_pos] = orig_pos
                orig_pos += 1
                norm_pos += 1
            else:
                # This should not happen if normalization is consistent
                orig_pos += 1
        
        return mapping


def clean_path(path: str) -> str:
    """
    Clean and normalize a file path.

    Args:
        path: The path to clean

    Returns:
        str: Cleaned and normalized path
    """
    # Convert to absolute path if not already
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # Normalize path (resolve .. and . components)
    return os.path.normpath(path)


def find_matching_file_path(base_dir: str, relative_path: str) -> Optional[str]:
    """
    Find a file path that matches the given relative path, even if the case doesn't match.
    
    Args:
        base_dir: Base directory to search in
        relative_path: Relative path to find
        
    Returns:
        str: Absolute path to the matching file, or None if not found
    """
    # First try direct match
    abs_path = os.path.join(base_dir, relative_path)
    if os.path.exists(abs_path):
        return abs_path
        
    # Try case-insensitive match
    try:
        # Get the directory part and filename part
        dir_path, filename = os.path.split(relative_path)
        search_dir = os.path.join(base_dir, dir_path) if dir_path else base_dir
        
        if not os.path.exists(search_dir):
            return None
            
        # Look for a case-insensitive match
        for entry in os.listdir(search_dir):
            if entry.lower() == filename.lower():
                # Return the path with the original filename to maintain the expected case
                # This is important for tests that expect the original case to be preserved
                return os.path.join(search_dir, filename)
    except Exception:
        pass
        
    return None


def add_line_numbers(text: str, start_line: int = 1) -> str:
    """
    Add line numbers to text.
    
    Args:
        text: Text to add line numbers to
        start_line: Starting line number
        
    Returns:
        str: Text with line numbers
    """
    lines = text.splitlines()
    max_line_num = start_line + len(lines) - 1
    line_width = len(str(max_line_num))
    
    numbered_lines = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:{line_width}} | {line}")
        
    return "\n".join(numbered_lines)

def code_edit_tool(query: Union[str, Dict]) -> str:
    """
    Edit code files with proper syntax checking and formatting.
    
    This tool provides the ability to edit code files by replacing specific
    sections of code with new content. It includes syntax checking and formatting
    to ensure the edited code is valid.
    
    Args:
        query: JSON string or dictionary with the following fields:
            - file_path: Path to the file to edit (required)
            - model: Mode of operation, either "replace" or "add" (optional, default: "replace")
            - new_content: The new code to add or replace (required)
            - old_content: The existing code to be replaced (required for "replace" mode)
                          Should exactly match the content in the file, including whitespace and indentation
            - start_line: Starting line number (1-based indexing) (required for "add" mode, optional for "replace" mode)
            - end_line: Ending line number (1-based indexing, inclusive) (optional for "replace" mode)
            - language: Programming language of the file (optional, default: auto-detect)
            - description: Description of the edit (optional)
            - check_syntax: Whether to check syntax after edit (optional, default: True)
            
    Returns:
        JSON string with the result of the operation or error message
    """
    logger.info(f"code_edit_tool called with query: {query}")
    
    try:
        # Parse query if it's a string, otherwise use it directly
        if isinstance(query, str):
            try:
                query_data = json.loads(query)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON query: {e}")
                return json.dumps({"error": f"Invalid JSON format: {str(e)}"})
        else:
            query_data = query
            
        logger.debug(f"Parsed query data: {query_data}")
        
        # Extract parameters
        file_path = query_data.get("file_path")
        start_line = query_data.get("start_line")
        end_line = query_data.get("end_line")
        new_content = query_data.get("new_content")
        old_content = query_data.get("old_content")
        language = query_data.get("language")
        description = query_data.get("description", "Code edit")
        check_syntax = query_data.get("check_syntax", True)
        model = query_data.get("model", "replace")
        
        logger.debug(f"Extracted parameters: file_path={file_path}, start_line={start_line}, "
                    f"end_line={end_line}, language={language}, check_syntax={check_syntax}, model={model}")
        
        # Validate parameters
        if not file_path:
            logger.error("Missing required parameter: file_path")
            return json.dumps({"error": "Missing required parameter: file_path"})
        
        if new_content is None:
            logger.error("Missing required parameter: new_content")
            return json.dumps({"error": "Missing required parameter: new_content"})
            
        # Validate model parameter
        if model not in ["replace", "add"]:
            logger.error(f"Invalid model parameter: {model}. Must be 'replace' or 'add'")
            return json.dumps({"error": f"Invalid model parameter: {model}. Must be 'replace' or 'add'"})
        
        # Validate parameters based on the model type
        if model == "add":
            # For add mode, only start_line is required
            if start_line is None:
                logger.error("Missing required parameter: start_line for add mode")
                return json.dumps({"error": "Missing required parameter: start_line for add mode"})
        else:  # replace mode
            # For replace mode, only old_content and new_content are required
            if old_content is None:
                logger.error("Missing required parameter: old_content for replace mode")
                error_msg = "Missing required parameter: old_content for replace mode. "
                error_msg += "Please provide the exact code block to be replaced, including proper indentation and whitespace."
                return json.dumps({"error": error_msg})
            
            # start_line and end_line are completely optional in replace mode
            # old_content matching is the only way to identify the code to replace
            # No additional validation needed here as they'll be checked later if used
        
        # Clean and normalize the file path
        file_path = clean_path(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Try to find a matching file path
            base_dir = os.path.dirname(os.path.dirname(file_path))
            rel_path = os.path.relpath(file_path, base_dir)
            matching_path = find_matching_file_path(base_dir, rel_path)
            
            if matching_path:
                file_path = matching_path
                logger.info(f"Found matching file: {file_path}")
            else:
                logger.error(f"File not found: {file_path}")
                return json.dumps({"error": f"File not found: {file_path}"})
        
        # Check if it's a file
        if not os.path.isfile(file_path):
            logger.error(f"Not a file: {file_path}")
            return json.dumps({"error": f"Not a file: {file_path}"})
        
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            logger.error(f"File {file_path} appears to be a binary file")
            return json.dumps({"error": f"File {file_path} appears to be a binary file and cannot be edited as text"})
        
        # If old_content is provided, try to find the matching content in the file
        warning_message = ""
        if old_content is not None:
            # Normalize line endings in old_content
            if old_content and not old_content.endswith('\n'):
                old_content += '\n'
                
            # Find the content in the file using CodeEditor
            try:
                editor = CodeEditor()
                start_line, end_line, similarity = editor.find_best_match(''.join(lines), old_content)
                logger.info(f"Found matching content at lines {start_line}-{end_line} (similarity: {similarity:.2f})")
                
            except MatchNotFoundError as e:
                logger.warning(f"Could not find old_content in file {file_path}")
                
                # Return a helpful error message
                error_msg = "Could not find the content to replace in the file.\n\n"
                error_msg += "To fix this issue, please ensure that:\n"
                error_msg += "1. The content to replace (old_content) is accurate and appears in the file\n"
                error_msg += "2. The indentation and whitespace match exactly\n"
                error_msg += "3. You include enough context (like function/class definitions) to uniquely identify the location\n"
                error_msg += f"\nBest match similarity: {getattr(e, 'best_similarity', 0):.2f} (threshold: {getattr(editor, 'similarity_threshold', 0.8)})\n"
                return json.dumps({"error": error_msg})
                
            except Exception as e:
                logger.error(f"Error while finding content match: {str(e)}")
                return json.dumps({"error": f"Error while processing file: {str(e)}"})
        
        # Validate line numbers if they were provided or determined from old_content
        if start_line < 1 or start_line > len(lines) + 1:
            logger.error(f"Invalid start_line: {start_line}. File has {len(lines)} lines.")
            return json.dumps({"error": f"Invalid start_line: {start_line}. File has {len(lines)} lines."})
        
        if model == "replace" and (end_line < start_line or end_line > len(lines)):
            logger.error(f"Invalid end_line: {end_line}. File has {len(lines)} lines.")
            return json.dumps({"error": f"Invalid end_line: {end_line}. File has {len(lines)} lines."})
        
        # Convert to 0-based indexing for internal use
        start_index = start_line - 1
        end_index = end_line if model == "replace" else start_index
        
        # Make a backup of the file
        backup_path = f"{file_path}.bak"
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.debug(f"Created backup at {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
        
        # Prepare the new content
        # Ensure the new content ends with a newline if it doesn't already
        if new_content and not new_content.endswith('\n'):
            new_content += '\n'
            
        # Verify old_content if provided and we didn't already set warning_message
        if old_content is not None and not warning_message and model == "replace":
            # Get the actual content being replaced
            actual_content = ''.join(lines[start_index:end_index])
            
            # Compare with provided old_content (already normalized above)
            if actual_content.strip() != old_content.strip():
                logger.warning(f"Old content verification failed. The content at lines {start_line}-{end_line} doesn't match the provided old_content.")
                logger.debug(f"Actual content:\n{actual_content}")
                logger.debug(f"Expected content:\n{old_content}")
                
                # Return warning but continue with the edit
                warning_message = f"Warning: The content at lines {start_line}-{end_line} doesn't match the provided old_content. "
                warning_message += "This might indicate incorrect line numbers or outdated content reference. "
                warning_message += "The edit was still applied, but please verify the result."
            else:
                logger.debug("Old content verification passed.")
        
        # Create the updated content based on the model
        if model == "replace":
            # Replace mode: replace the content between start_index and end_index
            updated_lines = lines[:start_index] + [new_content] + lines[end_index:]
        else:  # add mode
            # Add mode: insert the new content at start_index without removing anything
            updated_lines = lines[:start_index] + [new_content] + lines[start_index:]
        
        # Generate context preview for better understanding
        context_before = lines[max(0, start_index-5):start_index]
        context_after = lines[end_index:min(end_index+5, len(lines))]
        
        original_block = context_before + lines[start_index:end_index] + context_after
        modified_block = context_before + [new_content] + context_after
        
        # Add line numbers for better readability
        original_with_lines = add_line_numbers("\n".join(original_block), max(1, start_line-5))
        modified_with_lines = add_line_numbers("\n".join(modified_block), max(1, start_line-5))
        
        logger.debug(f"Original code:\n{original_with_lines}")
        logger.debug(f"Modified code:\n{modified_with_lines}")
        
        # Auto-detect language if not specified
        if not language:
            _, ext = os.path.splitext(file_path)
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.c': 'c',
                '.cpp': 'cpp',
                '.h': 'cpp',
                '.hpp': 'cpp',
                '.go': 'go',
                '.rs': 'rust',
                '.rb': 'ruby',
                '.php': 'php',
                '.sh': 'bash',
                '.html': 'html',
                '.css': 'css',
                '.json': 'json',
                '.md': 'markdown',
                '.xml': 'xml',
                '.yaml': 'yaml',
                '.yml': 'yaml',
            }
            language = language_map.get(ext.lower(), 'text')
            logger.debug(f"Auto-detected language: {language}")
        
        # Create a temporary file for syntax checking
        temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file_path)[1])
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:
                temp_file.writelines(updated_lines)
        except Exception as e:
            logger.error(f"Error writing to temporary file: {str(e)}")
            os.unlink(temp_path)
            raise
        
        logger.debug(f"Created temporary file at {temp_path}")
        
        # Check syntax if requested
        syntax_check_passed = True
        syntax_check_output = ""
        
        if check_syntax:
            if language == 'python':
                # First try to auto-format with autopep8 if available
                try:
                    autopep_result = subprocess.run(['autopep8', '--in-place', '--aggressive', temp_path],
                                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                  universal_newlines=True)
                    if autopep_result.returncode != 0:
                        logger.warning(f"autopep8 formatting failed: {autopep_result.stderr}")
                    else:
                        logger.debug("Applied autopep8 formatting")
                        # Re-read the file after formatting
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            updated_lines = f.readlines()
                except FileNotFoundError:
                    logger.debug("autopep8 not found, skipping auto-formatting")
                
                # Run flake8 for syntax checking and linting
                try:
                    # Check for syntax errors and critical issues
                    flake_result = subprocess.run(
                        ['flake8', '--isolated', '--select=F821,F822,F831,E111,E112,E113,E999,E902', temp_path],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                        universal_newlines=True
                    )
                    
                    if flake_result.returncode != 0:
                        syntax_check_passed = False
                        syntax_check_output = flake_result.stdout or flake_result.stderr
                        logger.warning(f"Flake8 check found critical issues: {syntax_check_output}")
                        return json.dumps({
                            "success": False,
                            "error": "Flake8 check found critical issues",
                            "details": syntax_check_output,
                            "file_path": file_path,
                            "start_line": start_line,
                            "end_line": end_line
                        }, indent=2)
                except FileNotFoundError:
                    logger.debug("flake8 not found, skipping lint check")
            
            elif language == 'javascript' or language == 'typescript':
                # Check JS/TS syntax if node is available
                try:
                    result = subprocess.run(['node', '--check', temp_path], 
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                            universal_newlines=True)
                    
                    if result.returncode != 0:
                        syntax_check_passed = False
                        syntax_check_output = result.stderr
                        logger.warning(f"JavaScript/TypeScript syntax check failed: {syntax_check_output}")
                except FileNotFoundError:
                    logger.debug("node not found, skipping syntax check")
            
            elif language == 'go':
                # Check Go syntax using gofmt
                # gofmt can check syntax even for incomplete code blocks
                try:
                    # Use gofmt to check syntax
                    gofmt_result = subprocess.run(['gofmt', '-e', '-w', temp_path], 
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                                universal_newlines=True)
                    
                    if gofmt_result.returncode != 0:
                        syntax_check_passed = False
                        syntax_check_output = gofmt_result.stderr
                        logger.warning(f"Go syntax check failed: {syntax_check_output}")
                    else:
                        logger.debug("Go syntax check passed")
                        
                        # Try golangci-lint for more comprehensive linting
                        try:
                            # Use golangci-lint run with default enabled linters
                            golangci_result = subprocess.run(
                                ['golangci-lint', 'run', '--no-config', 
                                 '--enable=errcheck,govet,ineffassign,staticcheck', temp_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True,
                                timeout=5  # Add timeout to prevent hanging
                            )
                            
                            if golangci_result.returncode != 0:
                                # This is just a warning, not a failure
                                lint_output = golangci_result.stdout or golangci_result.stderr
                                logger.warning(f"golangci-lint found issues: {lint_output}")
                                warning_message = f"Warning: golangci-lint found potential issues: {lint_output}"
                            else:
                                logger.debug("golangci-lint check passed")
                        except (subprocess.SubprocessError, FileNotFoundError) as e:
                            logger.debug(f"golangci-lint check skipped: {str(e)}")
                    
                    # Note: We're not using go vet or go build here because they require
                    # complete, compilable files, which might not be the case when editing
                    # individual functions or code blocks
                except FileNotFoundError:
                    logger.debug("gofmt not found, skipping syntax check")
            
            # Add more language-specific syntax checkers as needed
        
        # Apply the changes if syntax check passed or if syntax check was not requested
        if not check_syntax or syntax_check_passed:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            logger.info(f"Successfully edited {file_path} from line {start_line} to {end_line}")
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Create preview of changes
            original_preview = "\n".join(lines[max(0, start_index-3):min(end_index+3, len(lines))])
            # Calculate the number of lines in the new content (for preview)
            new_content_line_count = len(new_content.splitlines())
            updated_preview = "\n".join(updated_lines[max(0, start_index-3):min(start_index + new_content_line_count + 3, len(updated_lines))])
            
            result = {
                "success": True,
                "message": f"Successfully edited {file_path} from line {start_line} to {end_line}",
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "description": description,
                "preview": {
                    "original": add_line_numbers(original_preview, max(1, start_line-3)),
                    "modified": add_line_numbers(updated_preview, max(1, start_line-3))
                }
            }
            
            # Generate enhanced diff visualization if diff tools are available
            logger.debug(f"HAS_DIFF_TOOLS status: {HAS_DIFF_TOOLS}")
            if HAS_DIFF_TOOLS:
                try:
                    # Get file extension for language detection
                    _, ext = os.path.splitext(file_path)
                    language = ext[1:] if ext else None
                    logger.info(f"Detected language from file extension: {language}")
                    
                    # Create diff generator and renderer
                    logger.info("Initializing DiffGenerator and DiffRenderer")
                    diff_gen = DiffGenerator()
                    diff_renderer = DiffRenderer()
                    logger.info("Successfully initialized diff tools")
                    
                    # Generate unified diff - use basename to avoid creating files with full paths
                    file_basename = os.path.basename(file_path)
                    unified_diff = diff_gen.generate_unified_diff(
                        "\n".join(lines),
                        "\n".join(updated_lines),
                        f"{file_basename} (before)",
                        f"{file_basename} (after)",
                        ignore_whitespace_only_changes=True
                    )
                    
                    # Render the diff
                    rendered_diff = diff_renderer.render_unified_diff(unified_diff, language=language)
                    
                    # Get diff statistics
                    diff_stats = diff_gen.get_diff_stats("\n".join(lines), "\n".join(updated_lines))
                    
                    # Display the diff directly to console for immediate feedback using Rich
                    try:
                        from rich.console import Console
                        from rich.panel import Panel
                        from rich.text import Text
                        from rich.table import Table
                        
                        console = Console()
                        
                        # Print title without panel border
                        title = Text(f"Changes to {file_path}", style="bold blue")
                        
                        # Create the diff text
                        diff_text = Text(rendered_diff)
                        
                        # Create a table for statistics
                        stats_table = Table(show_header=False, box=None, padding=(0, 1))
                        stats_table.add_column("Stat", style="cyan")
                        stats_table.add_column("Value", style="green")
                        stats_table.add_row("Added", str(diff_stats['added']))
                        stats_table.add_row("Removed", str(diff_stats['removed']))
                        stats_table.add_row("Changed", str(diff_stats['changed']))
                        
                        # Display everything
                        console.print("\n")
                        console.print(title)
                        console.print(diff_text)
                        console.print("[bold cyan]Diff Statistics:[/bold cyan]")
                        console.print(stats_table)
                        console.print("\n")
                    except ImportError:
                        # Fallback to simple print if Rich is not available
                        print("\n" + "=" * 40)
                        print(f"Changes to {file_path}:")
                        print("=" * 40)
                        print(rendered_diff)
                        print(f"Stats: {diff_stats['added']} lines added, {diff_stats['removed']} lines removed, {diff_stats['changed']} lines changed")
                        print("=" * 40 + "\n")
                    
                    # Add enhanced diff to result (for API consumers)
                    result["enhanced_diff"] = {
                        "unified": rendered_diff,
                        "stats": diff_stats
                    }
                    
                    # Generate side-by-side diff for logging purposes only
                    # We don't include it in the result as it can be too verbose for terminal output
                    try:
                        # Use basename for side-by-side diff too
                        if not 'file_basename' in locals():
                            file_basename = os.path.basename(file_path)
                            
                        diff_renderer.render_side_by_side_diff(
                            original_preview,
                            updated_preview,
                            f"{file_basename} (before)",
                            f"{file_basename} (after)",
                            language=language
                        )
                        # Note: We intentionally don't add side-by-side diff to the result
                        # as it can make the output too verbose and hard to read in terminals
                    except Exception as e:
                        logger.debug(f"Side-by-side diff generation failed: {str(e)}")
                except Exception as e:
                    logger.warning(f"Enhanced diff generation failed: {str(e)}")
                    logger.warning(f"Exception traceback: {traceback.format_exc()}")
                    # This is non-critical, so we continue without enhanced diff
            
            # Add warning message if old_content verification failed
            if warning_message:
                result["warning"] = warning_message
                
            return json.dumps(result, indent=2)
        else:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return json.dumps({
                "success": False,
                "error": "Syntax check failed",
                "details": syntax_check_output,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line
            }, indent=2)
    
    except Exception as e:
        logger.error(f"Error in code_edit_tool: {str(e)}")
        logger.error(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error": f"Error in code_edit_tool: {str(e)}",
            "details": traceback.format_exc()
        }, indent=2)

# Example usage
if __name__ == "__main__":
    # Configure console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    
    # Example: Edit a Python file
    python_query = json.dumps({
        "file_path": "example.py",
        "start_line": 10,
        "end_line": 15,
        "new_content": "def new_function():\n    print('This is a new function')\n    return True\n",
        "language": "python",
        "description": "Replace old function with new implementation"
    })
    
    print(code_edit_tool(python_query))
