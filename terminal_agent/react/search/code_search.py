"""
Code search implementation using Zoekt.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import jedi

from terminal_agent.react.search.zoekt_server import ZoektServer
from terminal_agent.react.lsp.utils import add_num_line, get_language_from_file_extension
from codetext.utils import parse_code
from codetext.parser import PythonParser, CsharpParser, RustParser, JavaParser, GoParser, CppParser

logging.getLogger('codetext').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

def get_node_text(start_byte: int, end_byte: int, code: str) -> str:
    """
    Extract a substring from the given code.

    Args:
        start_byte (int): The starting index from where to begin the extraction.
        end_byte (int): The ending index where to stop the extraction.
        code (str): The string from which to extract the substring.

    Returns:
        str: The extracted substring from the start_byte to end_byte.
    """
    return code[start_byte:end_byte]

def get_parser(language: str):
    """
    Get a parser corresponding to the given language.

    Args:
        language (str): The given programming language.

    Returns:
        Parser corresponding to the given language.

    Raises:
        NotImplementedError: If the language is not supported.
    """
    if language == "python":
        return PythonParser()
    elif language == "csharp":
        return CsharpParser()
    elif language == "rust":
        return RustParser()
    elif language == "java":
        return JavaParser()
    elif language == "go":
        return GoParser()
    elif language == "cpp" or language == "c":
        return CppParser()
    else:
        raise NotImplementedError(f"Language {language} is not supported yet")

def get_code_jedi(definition: jedi.Script, verbose: bool=False) -> str:
    """
    Fetch and possibly format code from a jedi.Script definition.

    This function gets the code for a definition and optionally adds line numbers to it.

    Args:
        definition (jedi.Script): The jedi.Script instance where the code lives.
        verbose (bool, optional): If true, line numbers are appended before each code line. Defaults to False.

    Returns:
        str: The raw or line-numbered code as a string.
    """
    raw = definition.get_line_code(after=definition.get_definition_end_position()[0]-definition.get_definition_start_position()[0])
    start_num_line = definition.get_definition_start_position()[0] - 2 # jedi start from 1
    if not verbose:
        return raw
    else:
        results = []
        splited_raw = raw.split("\n")
        for _, line in enumerate(splited_raw):
            new_line = str(start_num_line + 1) + " " + line
            results.append(new_line)
            start_num_line += 1
        return "\n".join(results)

    
def search_code_with_zoekt(names: list, backend: object, num_result: int = 10, verbose: bool = True) -> dict:
    """
    Search for elements inside a project using the Zoekt search engine.

    Args:
        names (list): List of names to be searched in files.
        backend (object): Backend that provides search functionality.
        num_result (int, optional): Maximum number of search results to return. Defaults to 10.
        verbose (bool, optional): If set to True, returns detailed output. Defaults to True.

    Returns:
        dict: A dictionary containing structured results, text output, and raw documents.
    """
    # 检查语言支持
    if backend.language == "generic":
        raise ValueError("Generic language is not supported for code search. Please specify a specific language.")
    
    # 定义需要简化搜索的语言列表
    simplified_search_languages = ["go", "rust", "c", "cpp"]
    use_simplified_search = backend.language in simplified_search_languages
    
    # 获取语言解析器
    parser = None
    if not use_simplified_search:
        try:
            parser = get_parser(backend.language)
        except NotImplementedError as e:
            logging.warning(f"Parser not available for {backend.language}, falling back to simplified search: {str(e)}")
            use_simplified_search = True
    
    # 初始化搜索结果
    search_results = {name: [] for name in names}
    
    # 获取当前语言的文件扩展名
    language_extensions = {
        "python": [".py", ".pyi", ".pyx"],
        "go": [".go"],
        "rust": [".rs"],
        "c": [".c", ".h"],
        "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"],
        "java": [".java"],
        "csharp": [".cs"],
    }
    current_extensions = language_extensions.get(backend.language, [])
    if not current_extensions:
        raise ValueError(f"No file extensions defined for language '{backend.language}'")
    
    # 执行搜索
    with backend.start_server():
        zoekt_results = backend.search([f"{name}" for name in names], num_result=num_result)
    
    # 处理每个搜索词的结果
    for name in names:
        # 检查搜索结果是否存在
        if f'{name}' not in zoekt_results or "result" not in zoekt_results[f'{name}'] or "FileMatches" not in zoekt_results[f'{name}']["result"]:
            logging.warning(f"No results found for keyword '{name}'")
            continue
        
        files = zoekt_results[f'{name}']["result"]["FileMatches"]
        if not files:
            continue
        
        # 按语言过滤文件
        filtered_files = [file for file in files 
                         if os.path.splitext(file["FileName"])[1].lower() in current_extensions]
        
        if not filtered_files:
            logging.info(f"No {backend.language} files found for keyword '{name}'")
            continue
        
        # 处理每个匹配的文件
        for file in filtered_files:
            try:
                file_path = os.path.join(backend.repo_path, file["FileName"])
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                
                # 简化搜索模式 - 直接按行搜索
                if use_simplified_search:
                    process_simplified_search(name, file, source, search_results)
                    continue
                
                # 语法解析模式 - 查找函数和类
                try:
                    root_node = parse_code(source, backend.language).root_node
                    process_syntax_search(name, file, source, root_node, parser, search_results)
                except Exception as e:
                    logging.error(f"Syntax parsing error in {file['FileName']}: {e}")
                    # 解析失败时回退到简化搜索
                    process_simplified_search(name, file, source, search_results)
            except Exception as e:
                logging.error(f"Error processing file {file['FileName']}: {e}")
                lines = source.split("\n")
                for line_idx, line in enumerate(lines):
                    if name in line:
                        result = {
                            "file": file["FileName"],
                            "implementation": add_num_line("\n".join(lines[max(0, line_idx-2): min(line_idx+2, len(lines))]), max(0, line_idx-2))
                        }
                        search_results[name].append(result)
    
    search_results = {name: search_results[name][:num_result] for name in names}
    
    # 如果需要详细输出，直接返回格式化的文本字符串
    if verbose:
        # 使用辅助函数格式化结果
        return format_search_results(names, search_results)
    
    # 如果不需要详细输出，只返回原始搜索结果
    return search_results

def process_simplified_search(name, file, source, search_results):
    """
    执行简化的基于行的搜索。
    
    Args:
        name: 要搜索的关键词
        file: 文件信息字典
        source: 文件源代码
        search_results: 搜索结果字典，将被修改
    """
    lines = source.split("\n")
    for line_idx, line in enumerate(lines):
        if name in line:
            context_start = max(0, line_idx - 2)
            context_end = min(line_idx + 3, len(lines))
            context = "\n".join(lines[context_start:context_end])
            
            result = {
                "file": file["FileName"],
                "line": line_idx + 1,
                "implementation": add_num_line(context, context_start)
            }
            search_results[name].append(result)
            break  # 每个文件只添加一个结果


def process_syntax_search(name, file, source, root_node, parser, search_results):
    """
    使用语法解析器执行基于函数和类的搜索。
    
    Args:
        name: 搜索关键词
        file: 文件信息字典
        source: 文件源代码
        root_node: 语法树根节点
        parser: 语言解析器
        search_results: 搜索结果字典，将被修改
    """
    found_match = False
    
    # 处理函数
    try:
        function_list = parser.get_function_list(root_node)
        for func in function_list:
            try:
                metadata = parser.get_function_metadata(func, source)
                if name.lower() in metadata["identifier"].lower():
                    func_source = get_node_text(func.start_byte, func.end_byte, source)
                    result = {
                        "file": file["FileName"],
                        "name": metadata["identifier"],
                        "type": "function",
                        "line": func.start_point[0],
                        "end_line": func.end_point[0] + 1,
                        "documentation": parser.get_docstring(func, source),
                        "implementation": add_num_line(func_source, func.start_point[0])
                    }
                    search_results[name].append(result)
                    found_match = True
            except Exception as e:
                logging.error(f"Error processing function in {file['FileName']}: {e}")
    except Exception as e:
        logging.error(f"Error getting function list in {file['FileName']}: {e}")
    
    # 处理类
    try:
        class_list = parser.get_class_list(root_node)
        for cls in class_list:
            try:
                metadata = parser.get_class_metadata(cls, source)
                if name.lower() in metadata["identifier"].lower():
                    cls_source = get_node_text(cls.start_byte, cls.end_byte, source)
                    result = {
                        "file": file["FileName"],
                        "name": metadata["identifier"],
                        "type": "class",
                        "line": cls.start_point[0],
                        "end_line": cls.end_point[0] + 1,
                        "documentation": parser.get_docstring(cls, source),
                        "implementation": add_num_line(cls_source, cls.start_point[0])
                    }
                    search_results[name].append(result)
                    found_match = True
            except Exception as e:
                logging.error(f"Error processing class in {file['FileName']}: {e}")
    except Exception as e:
        logging.error(f"Error getting class list in {file['FileName']}: {e}")
    
    # 如果没有找到函数或类匹配，执行行级搜索
    if not found_match:
        process_simplified_search(name, file, source, search_results)


def format_search_results(names, search_results):
    """
    将搜索结果格式化为可读的文本。
    
    Args:
        names: 搜索关键词列表
        search_results: 搜索结果字典
        
    Returns:
        str: 格式化的文本输出
    """
    out_str = ""
    for name in names:
        out_str += f"Results for '{name}':\n"
        out_str += f"{'='*50}\n"
        
        if not search_results[name]:
            out_str += "No results found.\n\n"
            continue
        
        for result in search_results[name]:
            out_str += f"File: {result['file']}\n"
            
            if "name" in result:
                out_str += f"Name: {result['name']}\n"
            
            if "type" in result:
                out_str += f"Type: {result['type']}\n"
            
            if "line" in result:
                if "end_line" in result:
                    out_str += f"Lines: {result['line']} - {result['end_line']}\n"
                else:
                    out_str += f"Line: {result['line']}\n"
            
            if "range" in result:
                out_str += f"Line Range: {result['range']}\n" 
                
            if "documentation" in result and result["documentation"]:
                out_str += f"Documentation:\n{result['documentation']}\n"
            
            if "implementation" in result:
                out_str += f"Implementation:\n{result['implementation']}\n"
            
            out_str += f"{'-'*50}\n"
        
        out_str += "\n"
    
    return out_str


def search_code(names, backend, verbose, language=None):
    """
    Search for code elements using the appropriate search method.
    
    Args:
        names: List of names to search for
        backend: Backend object that provides search functionality
        verbose: Whether to include verbose output
        language: Optional language override
        
    Returns:
        Search results
    """
    return search_code_with_zoekt(names, backend, num_result=10, verbose=verbose)
