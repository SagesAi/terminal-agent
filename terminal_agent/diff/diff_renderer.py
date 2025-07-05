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
    from rich.console import Console, Group
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.measure import Measurement
    from rich.box import SQUARE
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
        
        # Extract git diff header and content
        header_lines = []
        content_lines = []
        in_header = True
        
        for line in lines:
            # Process the line content (convert tabs, etc.)
            processed_line = self._process_line(line, tab_size, convert_tabs, show_whitespace)
            
            # Truncate line to ensure display width doesn't exceed maximum width
            processed_line = self._truncate_to_width(processed_line, max_width)
            
            # Separate header from content
            if in_header and (processed_line.startswith('diff --git') or 
                             processed_line.startswith('---') or 
                             processed_line.startswith('+++')):  
                header_lines.append(processed_line)
            else:
                if in_header and header_lines:  # First non-header line
                    in_header = False
                content_lines.append(processed_line)
        
        # Format header
        header_result = []
        for line in header_lines:
            if line.startswith("diff --git"):
                header_result.append(Text(line, style="bold blue", no_wrap=True))
            else:  # --- or +++ lines
                header_result.append(Text(line, style=self.colors["header"], no_wrap=True))
        
        # Format content with improved indentation and function highlighting
        content_result = []
        current_function = None
        indent_level = 0
        
        for line in content_lines:
            # Detect function context in hunk headers
            if line.startswith("@@"):
                # Extract function name if present
                function_match = re.search(r'@@ .+ @@ (.+)', line)
                if function_match:
                    current_function = function_match.group(1).strip()
                    # Highlight function name in hunk header
                    line = re.sub(r'(@@ .+ @@ )(.+)', r'\1[bold magenta]\2[/bold magenta]', line)
                text = Text.from_markup(line)
                text.style = self.colors["changed"]
                text.no_wrap = True
                content_result.append(text)
                continue
            
            # Handle indentation for better code structure visibility
            if line.startswith("+") or line.startswith("-"):
                prefix = line[0]
                code_line = line[1:]
                
                # Calculate indentation level based on spaces/tabs at start of line
                if code_line.strip():
                    leading_space = len(code_line) - len(code_line.lstrip())
                    indent_level = leading_space // tab_size
                
                # Style based on line type
                style = self.colors["added"] if prefix == "+" else self.colors["removed"]
                
                # Highlight keywords for better code readability
                if language and (language.lower() in ["python", "javascript", "typescript", "java", "c", "cpp"]):
                    # Highlight common programming keywords
                    for keyword in ["def ", "class ", "function ", "if ", "else ", "for ", "while ", "return ", "import "]:
                        if keyword in code_line:
                            code_line = code_line.replace(keyword, f"[bold]{keyword}[/bold]")
                
                # Create styled text with prefix
                text = Text.from_markup(f"{prefix}{code_line}")
                text.style = style
                text.no_wrap = True
                content_result.append(text)
            else:
                # Context line
                content_result.append(Text(line, no_wrap=True))
        
        # Render the header and content
        with self.console.capture() as capture:
            # Print header without panel
            for text in header_result:
                self.console.print(text, highlight=False, soft_wrap=False)
            
            # Print content with panel for better visual separation
            content_panel = Panel(
                Group(*content_result),
                box=SQUARE,  # Use SQUARE box for better CJK character alignment
                padding=(0, 1),
                title="" if not current_function else f"[bold magenta]{current_function}[/bold magenta]",
                border_style="dim",
                expand=False
            )
            self.console.print(content_panel)
        
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
            # Truncate line to ensure display width doesn't exceed column width
            processed_old_lines.append(self._truncate_to_width(line, column_width))
        
        for line in new_lines:
            # Process new content similarly
            processed_new_lines.append(self._truncate_to_width(line, column_width))
        
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
        
        # Create panels for both sides with exact width control
        # Use box.SQUARE for more consistent border rendering with CJK characters
        exact_width = column_width + 4  # Add padding for borders
        
        old_panel = Panel(
            old_syntax, 
            title=old_title, 
            border_style="red", 
            width=exact_width,
            box=SQUARE,
            padding=(0, 1)
        )
        
        new_panel = Panel(
            new_syntax, 
            title=new_title, 
            border_style="green", 
            width=exact_width,
            box=SQUARE,
            padding=(0, 1)
        )
        
        # Create columns layout with fixed width and no spacing between columns
        columns = Columns([old_panel, new_panel], equal=True, padding=0)
        
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
    
    def _truncate_to_width(self, text: str, max_width: int, add_ellipsis: bool = True) -> str:
        """
        Truncate text to ensure it doesn't exceed max_width display characters,
        considering full-width characters like CJK.
        
        Args:
            text: Text to truncate
            max_width: Maximum display width
            add_ellipsis: Whether to add ellipsis (...) when truncating
            
        Returns:
            str: Truncated text
        """
        # If text is already within max width, return it as is
        if self._get_string_width(text) <= max_width:
            return text
        
        # Extract prefix if present (for diff markers)
        prefix = ""
        if text and text[0] in ("+", "-", "@", " "):
            prefix = text[0]
            text = text[1:]
            max_width -= 1  # Adjust max_width for prefix
        
        # Reserve space for ellipsis if needed
        ellipsis_width = 3 if add_ellipsis else 0
        effective_max_width = max_width - ellipsis_width
        
        result = ""
        current_width = 0
        
        for char in text:
            char_width = 2 if unicodedata.east_asian_width(char) in ('F', 'W', 'A') else 1
            if current_width + char_width <= effective_max_width:
                result += char
                current_width += char_width
            else:
                break
        
        # Add ellipsis if requested and truncation occurred
        if add_ellipsis and len(result) < len(text):
            result += "..."
        
        return prefix + result
    
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
