#!/usr/bin/env python3
"""
Get symbols tool for Terminal Agent.
Provides the ability to find all symbols in a code file.
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, Optional, Union, List

from terminal_agent.react.lsp.multilspy_toolkit import LSPToolKit
from multilspy.lsp_protocol_handler.lsp_types import SymbolKind

# Configure logging
logger = logging.getLogger(__name__)


# Cache for LSPToolKit instances
_lsp_toolkit_cache = {}

def get_lsp_toolkit(repo_dir: str, language: str = None) -> LSPToolKit:
    """
    Get or create an LSPToolKit instance for the given repository directory and language.
    
    Args:
        repo_dir: Repository directory path
        language: Programming language (default: python)
        
    Returns:
        LSPToolKit instance
    """
    if language is None:
        language = "python"
    
    # Normalize language name
    language = language.lower()
    
    # Map language aliases
    language_aliases = {
        "cpp": "c++",
        "js": "javascript",
        "ts": "typescript"
    }
    language = language_aliases.get(language, language)
    
    # Create a cache key based on repo_dir and language
    cache_key = f"{repo_dir}:{language}"
    logger.debug(f"Looking for LSPToolKit with cache key: {cache_key}")
    
    # Check if we already have a cached toolkit for this repo_dir and language
    if cache_key in _lsp_toolkit_cache:
        logger.debug(f"Using cached LSPToolKit instance for {repo_dir} with language {language}")
        return _lsp_toolkit_cache[cache_key]
    
    # Create a new toolkit instance
    logger.debug(f"Creating new LSPToolKit instance for {repo_dir} with language {language}")
    toolkit = LSPToolKit(repo_dir, language)
    
    # Cache the toolkit
    _lsp_toolkit_cache[cache_key] = toolkit
    
    return toolkit

def get_symbols_tool(query: Union[str, Dict]) -> str:
    """
    Find all symbols in a code file.
    
    Args:
        query: JSON string or dictionary with the following fields:
            - relative_path: Path to the file to get symbols from, relative to the repository root
            - preview_size: (optional) Number of lines to preview for each symbol (default: 10)
            - verbose: (optional) Whether to include detailed information in the result (default: True)
            - language: (optional) Programming language of the file (default: inferred from file extension)
            
    Returns:
        JSON string with the symbols information or error message
    """
    logger.info(f"get_symbols_tool called with query: {query}")
    
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
        relative_path = query_data.get("relative_path")
        preview_size = query_data.get("preview_size", 10)
        verbose = query_data.get("verbose", True)
        repo_dir = query_data.get("repo_dir", os.getcwd())
        
        # Validate parameters
        if not relative_path:
            logger.error("Missing required parameter: relative_path")
            return json.dumps({"error": "Missing required parameter: relative_path"})
        
        # Determine language based on file extension
        _, file_ext = os.path.splitext(relative_path)
        language_map = {
            ".py": "python",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".rs": "rust",
            ".js": "javascript",
            ".ts": "typescript"
        }
        language = query_data.get("language") or language_map.get(file_ext.lower(), "python")
        
        logger.debug(f"Extracted parameters: relative_path={relative_path}, preview_size={preview_size}, "
                    f"verbose={verbose}, language={language}, repo_dir={repo_dir}")
        
        # Convert relative_path to absolute file_path
        file_path = os.path.abspath(os.path.join(repo_dir, relative_path))
        logger.debug(f"Converted relative path '{relative_path}' to absolute path '{file_path}'")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}
        
        # Ensure repo_dir is an absolute path
        repo_dir = os.path.abspath(repo_dir)
        logger.debug(f"Using repo_dir: '{repo_dir}'")
        
        try:
            # Get LSPToolKit instance
            logger.debug(f"Getting LSPToolKit instance for repo_dir={repo_dir}, language={language}")
            toolkit = get_lsp_toolkit(repo_dir, language)
            
            # Convert absolute file_path back to relative path for LSPToolKit
            relative_file_path = os.path.relpath(file_path, repo_dir)
            
            try:
                # 直接使用 LSP 服务器的 document_symbols 请求获取符号信息
                with toolkit.server.start_server():
                    try:
                        # 获取原始符号信息
                        raw_symbols = toolkit.server.request_document_symbols(relative_file_path)
                        logger.debug(f"Raw symbols result: {raw_symbols}")
                        
                        # 如果没有符号，返回空列表
                        if not raw_symbols or len(raw_symbols) == 0:
                            return []
                            
                        # 打印原始符号类型和内容（调试用）
                        logger.debug(f"\n\n原始符号类型: {type(raw_symbols)}")
                        logger.debug(f"原始符号内容示例: {str(raw_symbols)[:200]}...")
                        
                        # 直接处理所有原始符号，不再提取第一组
                        logger.debug("开始处理所有符号...")
                        processed_symbols = process_symbols(raw_symbols)
                        logger.debug(f"处理后的符号结果类型: {type(processed_symbols)}")
                        
                        # 转换为列表格式
                        if isinstance(processed_symbols, list):
                            formatted_symbols = processed_symbols
                            logger.debug(f"使用列表结果，长度: {len(formatted_symbols)}")
                        elif processed_symbols:
                            formatted_symbols = [processed_symbols]
                            logger.debug("使用单个结果")
                        else:
                            formatted_symbols = []
                            logger.debug("没有处理结果，返回空列表")
                        
                        # 打印符号的 kind 类型统计
                        kind_types = set()
                        def check_kind_types(symbols):
                            if isinstance(symbols, list):
                                for s in symbols:
                                    check_kind_types(s)
                            elif isinstance(symbols, dict) and 'kind' in symbols:
                                kind_value = symbols['kind']
                                kind_types.add(type(kind_value).__name__)
                                logger.debug(f"Symbol: {symbols.get('name', 'unnamed')}, kind: {kind_value} ({type(kind_value).__name__})")
                                if 'children' in symbols and symbols['children']:
                                    check_kind_types(symbols['children'])
                        
                        check_kind_types(formatted_symbols)
                        logger.debug(f"符号 kind 字段类型汇总: {kind_types}")
                        
                        # 打印部分格式化符号示例
                        if formatted_symbols:
                            logger.debug(f"格式化符号示例: {str(formatted_symbols[0])[:200]}...")
                        
                        # 直接返回 Python 对象，而不是 JSON 字符串
                        return formatted_symbols
                        
                    except Exception as e:
                        logger.error(f"Error getting document symbols: {str(e)}")
                        logger.error(traceback.format_exc())
                        return {"error": f"Error getting document symbols: {str(e)}"}
                        
                
            except Exception as e:
                logger.error(f"Error in get_symbols: {str(e)}")
                logger.error(traceback.format_exc())
                return {"error": f"Error in get_symbols: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Error creating LSPToolKit: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error creating LSPToolKit: {str(e)}"}
            
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing query JSON: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error parsing query JSON: {str(e)}"}
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Unexpected error: {str(e)}"}

def process_symbols(symbols):
    """Process a list of symbols, converting them into a formatted data structure
        
    Args:
        symbols: Symbol list or a single symbol to process
        
    Returns:
        Processed symbol list or a single symbol
    """
    logger.debug(f"Processing symbols of type: {type(symbols)}")
    
    if symbols is None:
        return None
    
    # If it's a dictionary, process a single symbol
    if isinstance(symbols, dict):
        # Create a new result dictionary, instead of modifying the original symbol
        result = {}
        
        # Process each field
        for key, value in symbols.items():
            if key == 'kind':
                # Convert kind field to string
                if isinstance(value, int):
                    result[key] = _get_symbol_kind_name(value)
                    logger.debug(f"Converted kind {value} to {result[key]} for symbol {symbols.get('name', 'unnamed')}")
                else:
                    # If it's already a string, copy it directly
                    result[key] = value
            elif key == 'children' and value is not None:
                # Recursively process child symbols
                result[key] = process_symbols(value)
            else:
                # Copy other fields directly
                result[key] = value
        
        return result
    
    # If it's a tuple, process each element
    if isinstance(symbols, tuple):
        logger.debug(f"Processing tuple with {len(symbols)} items")
        result = []
        for i, item in enumerate(symbols):
            logger.debug(f"Processing tuple item {i} of type {type(item)}")
            processed = process_symbols(item)  # Recursively process each element
            if isinstance(processed, list):
                result.extend(processed)
            elif processed is not None:
                result.append(processed)
        return result
    
    # If it's a list, process each element
    if isinstance(symbols, list):
        logger.debug(f"Processing list with {len(symbols)} items")
        result = []
        for i, item in enumerate(symbols):
            logger.debug(f"Processing list item {i} of type {type(item)}")
            processed = process_symbols(item)  # Recursively process each element
            if isinstance(processed, list):
                result.extend(processed)
            elif processed is not None:
                result.append(processed)
        return result
    
    # Return other types directly
    return symbols

def process_single_symbol(symbol):
    """
    Process a single symbol information
    
    Args:
        symbol: Single symbol information
        
    Returns:
        Processed symbol information dictionary, or None if processing fails
    """
    # Ensure symbol is a dictionary type
    if not isinstance(symbol, dict):
        return None
        
    # Extract basic information
    name = symbol.get('name', '')
    kind_value = symbol.get('kind', 0)
    kind = _get_symbol_kind_name(kind_value)
    
    # Add debug log
    logger.debug(f"Processing symbol: {name}, kind_value: {kind_value}, kind_name: {kind}")
    
    # Extract location information
    location = {}
    try:
        if 'location' in symbol:
            loc = symbol['location']
            if isinstance(loc, dict) and 'range' in loc:
                range_info = loc['range']
                if isinstance(range_info, dict):
                    location = {
                        'start': {
                            'line': range_info.get('start', {}).get('line', 0),
                            'character': range_info.get('start', {}).get('character', 0)
                        },
                        'end': {
                            'line': range_info.get('end', {}).get('line', 0),
                            'character': range_info.get('end', {}).get('character', 0)
                        }
                    }
        elif 'range' in symbol:
            range_info = symbol['range']
            if isinstance(range_info, dict):
                location = {
                    'start': {
                        'line': range_info.get('start', {}).get('line', 0),
                        'character': range_info.get('start', {}).get('character', 0)
                    },
                    'end': {
                        'line': range_info.get('end', {}).get('line', 0),
                        'character': range_info.get('end', {}).get('character', 0)
                    }
                }
    except Exception as e:
        # If extracting location information fails, use an empty dictionary
        logger.debug(f"Error extracting location info: {e}")
        location = {}
    
    # Create formatted symbol information
    formatted_symbol = {
        'name': name,
        'kind': kind,
        'location': location
    }
    
    # Add detailed information (if available)
    if 'detail' in symbol and symbol['detail']:
        formatted_symbol['detail'] = symbol['detail']
        
    # Process child symbols
    if 'children' in symbol and symbol['children']:
        children = process_symbols(symbol['children'])
        if children:
            formatted_symbol['children'] = children
            
    return formatted_symbol


def _get_symbol_kind_name(kind_value):
    """
    Convert a symbol kind numeric value to a readable string name
    
    Args:
        kind_value: int - Numeric value of the symbol kind
        
    Returns:
        String name of the symbol kind
    """
    # Symbol kind mapping as defined in the SymbolKind class
    symbol_kinds = {
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
    
    try:
        # Try to get the symbol kind name from the mapping
        return symbol_kinds.get(kind_value, f"Unknown({kind_value})")
    except Exception:
        return f"Unknown({kind_value})"
