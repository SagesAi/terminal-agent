"""
Utility functions for LSP toolkit.
"""

import re
from typing import Dict, List, Any, Optional, Union, Tuple



def word_to_position(source: str, word: str, line = None, offset: int = 0):
    """
    Find the position of a word in a source.

    Args:
        source (str): The source string to search in.
        word (str): The word to find the position of.
        line (None|int|list, optional): The line number(s) to search in. Defaults to None.
        offset (int, optional): The offset to adjust the line number(s) by. Defaults to 0.

    Returns:
        dict: A dictionary containing the line and column position of the word.
              The line position is 0-based, while the column position is 1-based.
              Returns None if the word is not found.
    """
    if isinstance(line, list):
        line = line[0]
    lines = source.splitlines()
    try:
        for i, _line in enumerate(lines):
            if word in _line:
                return {"line": (line + offset) if line else (i + offset), "column": lines[line].index(word)+1 if line else (_line.index(word) + 1)} ## +1 because the position is 0-based
    except ValueError:
        for i, _line in enumerate(lines):
            if word in _line:
                return {"line": i, "column": lines[i].index(word)+1}
    except IndexError:
        for i, _line in enumerate(lines):
            if word in _line:
                return {"line": i, "column": _line.index(word)+1}
    return None



def get_text(doc, range):
    """
    Retrieves the text within the specified range in the given document.

    Args:
        doc (str): The document to extract text from.
        range (dict): The range object specifying the start and end positions.

    Returns:
        str: The extracted text within the specified range.
    """
    return doc[offset_at_position(doc, range["start"]):offset_at_position(doc, range["end"])]


def offset_at_position(doc, position):
    """
    Calculates the offset at a given position in a document.

    Args:
        doc (str): The document content.
        position (dict): The position object containing the line and character.

    Returns:
        int: The offset at the given position.
    """
    return position["character"] + len("".join(doc.splitlines(True)[: position["line"]]))

def add_num_line(text: str, start_line: int = 0) -> str:
    """
    Add line numbers to text.
    
    Args:
        text: The text to add line numbers to
        start_line: The starting line number (0-based, default is 0)
        
    Returns:
        The text with line numbers
    """
    lines = text.splitlines()
    result = []
    
    for i, line in enumerate(lines):
        result.append(f"{i + start_line} {line}")
    
    return "\n".join(result)


def matching_symbols(symbols, object):
    """
    Find a matching symbol based on line range.

    Args:
        symbols (list): List of symbols to search through.
        object (dict): The object to match against.

    Returns:
        dict or None: The matching symbol if found, None otherwise.
    """
    for symbol in symbols:
        ## approximat matching only is strong enough
        if "location" not in symbol:
            if symbol["range"]["start"]["line"] == object["range"]["start"]["line"]:
                return symbol
            else:
                continue
        if symbol["location"]["range"]["start"]["line"] == object["range"]["start"]["line"]:
            return symbol
    return None



def matching_kind_symbol(symbol: Dict[str, Any]) -> str:
    """
    Get a human-readable description of a symbol kind.
    
    Args:
        symbol: The symbol
        
    Returns:
        A string describing the symbol kind
    """
    kind_map = {
        1: "File",
        2: "Module",
        3: "Namespace",
        4: "Package",
        5: "Class",
        6: "Method",
        7: "Property",
        8: "Field",
        9: "Constructor",
        10: "Enum",
        11: "Interface",
        12: "Function",
        13: "Variable",
        14: "Constant",
        15: "String",
        16: "Number",
        17: "Boolean",
        18: "Array",
        19: "Object",
        20: "Key",
        21: "Null",
        22: "EnumMember",
        23: "Struct",
        24: "Event",
        25: "Operator",
        26: "TypeParameter"
    }
    
    kind = symbol.get("kind")
    if kind and kind in kind_map:
        return kind_map[kind]
    
    return "Unknown"


def get_language_from_file_extension(file_path: str) -> Optional[str]:
    """
    Determine the programming language based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The programming language as a string, or None if unknown
    """
    if not file_path:
        return None
        
    # 获取文件扩展名（小写）
    import os
    _, ext = os.path.splitext(file_path.lower())
    
    # 映射文件扩展名到编程语言
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "shell"
    }
    
    return extension_map.get(ext)


def search_in_document(document: str, search_term: str) -> List[Dict[str, Any]]:
    """
    Search for a term in a document and return all occurrences.
    
    Args:
        document (str): The document text to search in
        search_term (str): The term to search for
        
    Returns:
        List[Dict[str, Any]]: A list of matches, each with line number and content
    """
    if not document or not search_term:
        return []
    
    results = []
    lines = document.splitlines()
    
    # Create a regex pattern that matches whole words
    pattern = r'\b' + re.escape(search_term) + r'\b'
    
    # Search each line for the pattern
    for line_num, line_content in enumerate(lines):
        matches = list(re.finditer(pattern, line_content))
        if matches:
            for match in matches:
                results.append({
                    "line": line_num,
                    "character": match.start(),
                    "content": line_content.strip()
                })
    
    return results
