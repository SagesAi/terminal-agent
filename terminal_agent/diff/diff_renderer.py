#!/usr/bin/env python3
"""
Diff Renderer for Terminal Agent.
Provides functionality for rendering diffs with syntax highlighting in the terminal.
"""

import os
import re
import sys
import unicodedata
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.measure import Measurement
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class DiffRenderer:
    """
    Renders diffs with syntax highlighting for terminal display.
    
    This class provides methods to render unified diffs and side-by-side diffs
    with various formatting options including syntax highlighting.
    """
    
    def __init__(self, theme: str = "monokai", width: int = 160, tab_size: int = 4, convert_tabs: bool = True, show_whitespace: bool = False):
        """
        Initialize the DiffRenderer.
        
        Args:
            theme: Syntax highlighting theme (default: "monokai")
            width: Maximum width for rendering (default: 160)
            tab_size: Number of spaces to convert tabs to (default: 4)
            convert_tabs: Whether to convert tabs to spaces (default: True)
            show_whitespace: Whether to show whitespace as visible characters (default: False)
        """
        self.theme = theme
        self.width = width
        self.tab_size = tab_size
        self.convert_tabs = convert_tabs
        self.show_whitespace = show_whitespace
        
        # Define colors for diff markers
        self.colors = {
            "added": "green",
            "removed": "red",
            "changed": "yellow",
            "header": "blue",
            "line_number": "dim",
            "whitespace": "dim cyan",
        }
        
        # Initialize Rich console if available
        if HAS_RICH:
            self.console = Console(width=width, highlight=True)
        
    def render_unified_diff(
        self, 
        diff_text: str, 
        language: Optional[str] = None,
        show_line_numbers: bool = True,
        tab_size: Optional[int] = None,
        convert_tabs: Optional[bool] = None,
        show_whitespace: Optional[bool] = None
    ) -> str:
        """
        Render a unified diff with syntax highlighting.
        
        Args:
            diff_text: Unified diff text to render
            language: Programming language for syntax highlighting (auto-detect if None)
            show_line_numbers: Whether to show line numbers (default: True)
            tab_size: Number of spaces to convert tabs to (overrides instance setting)
            convert_tabs: Whether to convert tabs to spaces (overrides instance setting)
            show_whitespace: Whether to show whitespace as visible characters (overrides instance setting)
            
        Returns:
            str: Rendered diff as a string
        """
        if not diff_text:
            return "No differences found."
            
        if not HAS_RICH:
            # Fallback to plain text if Rich is not available
            return diff_text
        
        # Use method parameters if provided, otherwise use instance settings
        tab_size = tab_size if tab_size is not None else self.tab_size
        convert_tabs = convert_tabs if convert_tabs is not None else self.convert_tabs
        show_whitespace = show_whitespace if show_whitespace is not None else self.show_whitespace
            
        # Split the diff into lines
        lines = diff_text.splitlines()
        
        # Calculate maximum line width, considering Chinese character width
        max_width = self.width - 4  # Subtract width for borders and padding
        
        # Process the diff lines
        result = []
        for line in lines:
            # Process the line content (convert tabs, etc.)
            processed_line = self._process_line(line, tab_size, convert_tabs, show_whitespace)
            
            # Check line width, if it contains Chinese characters it may need truncation
            if self._get_string_width(processed_line) > max_width:
                # Truncate line to ensure display width doesn't exceed maximum width
                truncated_line = ""
                current_width = 0
                
                # Preserve special characters at the beginning of the line (+, -, @@, etc.)
                prefix = ""
                if processed_line.startswith(("+", "-", "@", " ")):
                    prefix = processed_line[0]
                    processed_line = processed_line[1:]
                    current_width = 1  # Prefix occupies one character width
                
                for char in processed_line:
                    char_width = 2 if unicodedata.east_asian_width(char) in ('F', 'W', 'A') else 1
                    if current_width + char_width <= max_width:
                        truncated_line += char
                        current_width += char_width
                    else:
                        break
                
                processed_line = prefix + truncated_line
            
            # Preserve exact spaces by using no_wrap=True and ensuring spaces are preserved
            if processed_line.startswith("+++") or processed_line.startswith("---"):
                # Header lines
                text = Text(processed_line, style=self.colors["header"], no_wrap=True)
                result.append(text)
            elif processed_line.startswith("@@"):
                # Hunk header
                text = Text(processed_line, style=self.colors["changed"], no_wrap=True)
                result.append(text)
            elif processed_line.startswith("+"):
                # Added line
                text = Text(processed_line, style=self.colors["added"], no_wrap=True)
                result.append(text)
            elif processed_line.startswith("-"):
                # Removed line
                text = Text(processed_line, style=self.colors["removed"], no_wrap=True)
                result.append(text)
            else:
                # Context line
                result.append(Text(processed_line, no_wrap=True))
                
        # Render the result using Rich's capture feature with options to preserve spaces
        with self.console.capture() as capture:
            for text in result:
                # Use print_line to avoid any automatic formatting that might affect spaces
                self.console.print(text, highlight=False, soft_wrap=False)
                
        # Get the captured output as a string
        return capture.get()
        
    def render_side_by_side_diff(
        self,
        old_content: str,
        new_content: str,
        old_title: str = "Old",
        new_title: str = "New",
        language: Optional[str] = None,
        show_line_numbers: bool = True
    ) -> str:
        """
        Render a side-by-side diff of two content strings.
        
        Args:
            old_content: Original content
            new_content: New content
            old_title: Title for the old content panel
            new_title: Title for the new content panel
            language: Programming language for syntax highlighting (auto-detect if None)
            show_line_numbers: Whether to show line numbers (default: True)
            
        Returns:
            str: Rendered side-by-side diff as a string
        """
        if not HAS_RICH:
            # Fallback to unified diff if Rich is not available
            from terminal_agent.diff.diff_generator import DiffGenerator
            diff_gen = DiffGenerator()
            return diff_gen.generate_unified_diff(old_content, new_content, old_title, new_title)
        
        # Process content to ensure each line's width won't cause border misalignment
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        # Calculate the maximum width for each column
        column_width = (self.width // 2) - 4  # Subtract width for borders and padding
        
        # Process Chinese characters that might cause border misalignment
        processed_old_lines = []
        processed_new_lines = []
        
        for line in old_lines:
            # If the line contains Chinese characters, it may need to be truncated or padded
            if self._get_string_width(line) > column_width:
                # Truncate the line to ensure display width doesn't exceed column width
                truncated_line = ""
                current_width = 0
                for char in line:
                    char_width = 2 if unicodedata.east_asian_width(char) in ('F', 'W', 'A') else 1
                    if current_width + char_width <= column_width:
                        truncated_line += char
                        current_width += char_width
                    else:
                        break
                processed_old_lines.append(truncated_line)
            else:
                processed_old_lines.append(line)
        
        for line in new_lines:
            # Process new content similarly
            if self._get_string_width(line) > column_width:
                truncated_line = ""
                current_width = 0
                for char in line:
                    char_width = 2 if unicodedata.east_asian_width(char) in ('F', 'W', 'A') else 1
                    if current_width + char_width <= column_width:
                        truncated_line += char
                        current_width += char_width
                    else:
                        break
                processed_new_lines.append(truncated_line)
            else:
                processed_new_lines.append(line)
        
        # Recombine the processed content
        processed_old_content = "\n".join(processed_old_lines)
        processed_new_content = "\n".join(processed_new_lines)
            
        # Create syntax objects for both sides
        old_syntax = Syntax(
            processed_old_content, 
            language or "text", 
            theme=self.theme,
            line_numbers=show_line_numbers,
            word_wrap=False  # Disable auto word wrap as we handle it manually
        )
        
        new_syntax = Syntax(
            processed_new_content, 
            language or "text", 
            theme=self.theme,
            line_numbers=show_line_numbers,
            word_wrap=False  # Disable auto word wrap as we handle it manually
        )
        
        # Create panels for both sides
        old_panel = Panel(old_syntax, title=old_title, border_style="red", width=column_width + 4)
        new_panel = Panel(new_syntax, title=new_title, border_style="green", width=column_width + 4)
        
        # Create columns layout with fixed width for each column
        columns = Columns([old_panel, new_panel], equal=True)
        
        # Render to string
        with self.console.capture() as capture:
            self.console.print(columns)
            
        return capture.get()
        
    def _get_string_width(self, text: str) -> int:
        """
        Calculate the display width of a string, considering that Chinese characters and other full-width characters
        occupy two character widths.
        
        Args:
            text: The text to calculate the width for
            
        Returns:
            int: The display width of the text
        """
        width = 0
        # Remove ANSI escape sequences, they don't occupy display width
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        
        for char in text:
            # Use unicodedata.east_asian_width to determine character width
            # F, W, A (in most terminals) are considered wide characters (occupying two character positions)
            eaw = unicodedata.east_asian_width(char)
            if eaw in ('F', 'W', 'A'):
                width += 2
            else:
                width += 1
        return width
    
    def _process_line(self, line: str, tab_size: int, convert_tabs: bool, show_whitespace: bool) -> str:
        """
        Process a line of text, converting tabs to spaces and/or showing whitespace as visible characters.
        
        Args:
            line: Line to process
            tab_size: Number of spaces to convert tabs to
            convert_tabs: Whether to convert tabs to spaces
            show_whitespace: Whether to show whitespace as visible characters
            
        Returns:
            str: Processed line
        """
        if not line:
            return ""
            
        # Convert tabs to spaces if requested
        if convert_tabs:
            line = line.replace("\t", " " * tab_size)
            
        # Show whitespace as visible characters if requested
        if show_whitespace:
            # Use Unicode characters for whitespace
            # · (U+00B7) for space, → (U+2192) for tab
            line = line.replace(" ", "·")
            if not convert_tabs:  # Only show tabs if not converting them
                line = line.replace("\t", "→")
                
        return line
        
    def highlight_diff_line(self, line: str, tab_size: Optional[int] = None, convert_tabs: Optional[bool] = None, show_whitespace: Optional[bool] = None) -> str:
        """
        Add color highlighting to a single diff line.
        
        Args:
            line: Diff line to highlight
            tab_size: Number of spaces to convert tabs to (overrides instance setting)
            convert_tabs: Whether to convert tabs to spaces (overrides instance setting)
            show_whitespace: Whether to show whitespace as visible characters (overrides instance setting)
            
        Returns:
            str: Highlighted line
        """
        if not line:
            return ""
            
        if not HAS_RICH:
            return line
        
        # Use method parameters if provided, otherwise use instance settings
        tab_size = tab_size if tab_size is not None else self.tab_size
        convert_tabs = convert_tabs if convert_tabs is not None else self.convert_tabs
        show_whitespace = show_whitespace if show_whitespace is not None else self.show_whitespace
        
        # Process the line content (convert tabs, etc.)
        processed_line = self._process_line(line, tab_size, convert_tabs, show_whitespace)
            
        if processed_line.startswith("+"):
            with self.console.capture() as capture:
                self.console.print(Text(processed_line, style=self.colors["added"], no_wrap=True), highlight=False, soft_wrap=False)
            return capture.get()
        elif processed_line.startswith("-"):
            with self.console.capture() as capture:
                self.console.print(Text(processed_line, style=self.colors["removed"], no_wrap=True), highlight=False, soft_wrap=False)
            return capture.get()
        elif processed_line.startswith("@@"):
            with self.console.capture() as capture:
                self.console.print(Text(processed_line, style=self.colors["changed"], no_wrap=True), highlight=False, soft_wrap=False)
            return capture.get()
        elif processed_line.startswith("+++") or processed_line.startswith("---"):
            with self.console.capture() as capture:
                self.console.print(Text(processed_line, style=self.colors["header"], no_wrap=True), highlight=False, soft_wrap=False)
            return capture.get()
        else:
            return processed_line


# Example usage
if __name__ == "__main__":
    # Example content
    old_content = """def example():
    print("Hello")
    # Old comment
    return True
"""

    new_content = """def example():
    print("Hello")
    # New improved comment
    return True
"""

    # Create a diff renderer
    renderer = DiffRenderer()
    
    # Generate a unified diff
    from terminal_agent.diff.diff_generator import DiffGenerator
    diff_gen = DiffGenerator()
    unified_diff = diff_gen.generate_unified_diff(
        old_content, 
        new_content,
        "example.py (before)",
        "example.py (after)"
    )
    
    # Render the unified diff
    rendered_diff = renderer.render_unified_diff(unified_diff, language="python")
    print("Rendered Unified Diff:")
    print(rendered_diff)
    
    # Render a side-by-side diff
    side_by_side = renderer.render_side_by_side_diff(
        old_content,
        new_content,
        "example.py (before)",
        "example.py (after)",
        language="python"
    )
    print("\nSide-by-Side Diff:")
    print(side_by_side)
