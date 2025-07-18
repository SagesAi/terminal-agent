"""
LSP Toolkit based on Microsoft's multilspy library.
Provides a natural language interface for interacting with Language Server Protocol.
"""

import os
import logging
import re
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

import logging
from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from multilspy.lsp_protocol_handler.lsp_types import SymbolKind

from .utils import word_to_position, get_language_from_file_extension, add_num_line, matching_symbols, matching_kind_symbol, get_text

# 定义一个自定义的日志过滤器，过滤掉 INFO 级别的日志
class InfoFilter(logging.Filter):
    def filter(self, record):
        # 只允许 WARNING 及以上级别通过
        return record.levelno > logging.INFO

class LSPToolKit:
    """
        This class serves as a natural language interface for LLM to interact with Language Server Protocol.
        It provides functionalities like opening files, accessing text, finding definitions and finding symbols
        related to words in specified documents.

        root_path: str - the root path of your codebase,
        language: str, optional - the language of your code (default is 'python')
    """
    def __init__(self, root_path, language="python"):
        """
        Creating a language server with root path and language configuration provided by user

        MultilspyConfig: A class that helps to configure Language Server for multiple languages
        MultilspyLogger: A class that helps to Log LSP operations and Debugging
        """
        # 配置 multilspy 日志
        multilspy_logger = logging.getLogger('multilspy')
        multilspy_logger.setLevel(logging.WARNING)  # 设置为 WARNING 级别
        
        # 添加过滤器，过滤掉 INFO 级别的日志
        for handler in multilspy_logger.handlers[:]:
            multilspy_logger.removeHandler(handler)
        multilspy_logger.addFilter(InfoFilter())
        
        self.root_path = root_path
        self.language = language
        self.server = SyncLanguageServer.create(MultilspyConfig(code_language=language), MultilspyLogger(), root_path)

    def open_file(self, relative_path):
        """
        Open a file using the file's relative path to the root path of code base

        Args:
            relative_path: str - relative path of file to codebase root path.

        Returns:
            the file text if successful, Else returns an error message string.
        """
        with self.server.start_server():
            with self.server.open_file(relative_path):
                result = self.server.get_open_file_text(relative_path)

        return result

    def get_definition(self, word, relative_path, line=None, offset=0, verbose=False):
        """
        Get the definition of a word in a file
        
        Args:
            word (str): word to find definition for
            relative_path (str): relative path to the file
            line (int, optional): line number of the word. Defaults to None.
            offset (int, optional): offset to add to line number. Defaults to 0.
            verbose (bool, optional): print detailed_output if set to True. Defaults to False.
            
        Returns:
            str or dict: definition of the word or error message
        """
        doc = self.open_file(relative_path)
        cursor_pos = word_to_position(doc, word, line=line, offset=offset)
        
        # Verifying if the cursor position exists and then getting the definition
        if cursor_pos is not None:
            try:
                with self.server.start_server():
                    output = self.server.request_definition(relative_path, **cursor_pos)
                    
                # Handle None response from language server
                if output is None:
                    logging.warning(f"Language server returned None for definition of '{word}' in {relative_path}")
                    return {"error": f"No definition found for '{word}' at line {line if line is not None else 'unknown'}"}
            except Exception as e:
                logging.error(f"Error requesting definition: {str(e)}")
                return {"error": f"Error requesting definition: {str(e)}"}
        else:
            return "The tool cannot find the word in the file"
        
        # If verbose setting is true, then gives detailed information for first symbol from output. In case the symbol doe not have a location attribute then shows the rang attribute.
        if verbose and len(output) > 0:
            symbols = self.get_symbols(output[0]["relativePath"], verbose=False)
            symbol = matching_symbols(symbols, output[0])
            if symbol is None:
                return "Please try again with semantic or code search tool"
            symbol_type = matching_kind_symbol(symbol)
            definition = ""
            if "location" not in symbol:
                if symbol["range"]["end"]["line"] - symbol["range"]["start"]["line"] > 100:
                    definition = "Too long to display, you should use open_file tool to view partial content step by step."
                    symbol["range"]["end"]["line"] = symbol["range"]["start"]["line"] + 100
                definition += add_num_line(get_text(self.open_file(output[0]["relativePath"]), symbol["range"]), symbol["range"]["start"]["line"])
            else:
                if symbol["location"]["range"]["end"]["line"] - symbol["location"]["range"]["start"]["line"] > 100:
                    definition = "Too long to display, you should use open_file tool to view partial content step by step."
                    symbol["location"]["range"]["end"]["line"] = symbol["location"]["range"]["start"]["line"] + 100
                definition = add_num_line(get_text(self.open_file(output[0]["relativePath"]), symbol["location"]["range"]), symbol["location"]["range"]["start"]["line"])
            output = "Name: " + str(symbol["name"]) + "\n" + "Type: " + str(symbol_type) + "\n" + "Definition: " + definition
        
        return output
    
    def get_symbols(self, file_path: str, preview_size: int = 10, verbose: bool = True) -> list:
        """
        Get all symbols in a file

        Args:
            file_path (str): relative path to the file
            preview_size (int, optional): only preview a set number of lines of definitions to save number of tokens. Defaults to 10.
            verbose (bool, optional): print detailed_output if set to True. Defaults to True.

        Returns:
            list: Returns either a list of symbols or a detailed list of symbols based on the detailed_output flag
        """
        with self.server.start_server():
            file_symbols = self.server.request_document_symbols(file_path)[0]


        if not verbose:
            return file_symbols

        file_source = self.open_file(file_path)
        detailed_list = []

        with self.server.start_server():
            for symbol in file_symbols:
                symbol_definition = get_text(file_source, symbol["range"])
                # TODO: Add more primary symbols depending on the language
                primary_symbols = [SymbolKind.Class, SymbolKind.Function, SymbolKind.Struct]
                primary_symbols = [int(symbol_kind) for symbol_kind in primary_symbols]
                if symbol["kind"] in primary_symbols:
                    symbol_line_location = next((line_num for line_num, line in enumerate(symbol_definition.split("\n")) if symbol["name"] in line), 0)
                    try:
                        character_index = symbol_definition.split("\n")[symbol_line_location].index(symbol["name"])
                        mouse_over_info = self.server.request_hover(file_path, symbol["range"]["start"]["line"], character_index)
                        hover_documentation = mouse_over_info["contents"] if mouse_over_info else "None"

                        if "value" not in hover_documentation:
                            hover_documentation = "None"
                            # definition_preview = "\n".join(symbol_definition.split("\n")[:preview_size+4])
                        # else:
                            # definition_preview = "\n".join(symbol_definition.split("\n")[:preview_size])
                        if "import" in symbol_definition:
                            continue
                        definition_with_line_numbers = add_num_line(symbol_definition, symbol["range"]["start"]["line"])
                        output_item = "\n".join([
                            "Name: " + str(symbol["name"]),
                            "Type: " + str(matching_kind_symbol(symbol)),
                            "Definition: " + str(definition_with_line_numbers),
                            "Documentation: " + str(hover_documentation)
                        ])
                        detailed_list.append(output_item)
                    except ValueError:
                        pass
        file_symbols = [symbol_item for symbol_item in detailed_list if symbol_item is not None]
        return file_symbols
    
    def get_references(
        self,
        search_word: str,
        file_path: str,
        line_number: int = None,
        offset_value: int = 0,
        verbose: bool = False,
        context_limit: int = 10
    ) -> str:
        """
        This function is used to get references of a particular identifier in a codebase. It can also provide detailed
        output if verbose argument is set to true.

        Args:
            search_word (str): The identifier to be searched in the codebase.
            file_path (str): Path of the file in which to search the identifier.
            line_number (int, optional): Line number to start the search from. Defaults to None.
            offset_value (int, optional): The number of positions to ignore from the start of the line. Defaults to 0.
            verbose (bool, optional): If set to True, detailed output will be returned. Defaults to False.
            context_limit (int, optional): Defines the number of lines to print before and after the matched line in verbose mode. Defaults to 10.

        Returns:
            str: This function returns a string consisting of locations of the search identifier in the document.
                In verbose mode, this string contains additional information at each location such as implementation code.
        """
        document_contents = self.open_file(file_path)
        
        if line_number is None:
            # Handle cases where the line number is either incorrect or not known
            cursor_position = word_to_position(document_contents, search_word, line=None, offset=offset_value)
        else:
            try:
                cursor_position = word_to_position(document_contents, search_word, line=line_number, offset=offset_value)
            except:
                cursor_position = word_to_position(document_contents, search_word, line=None, offset=offset_value)

        if cursor_position is None:
            return "The tool cannot find the word in the file"

        with self.server.start_server():
            references_output = self.server.request_references(file_path, **cursor_position)

        if verbose:
            detailed_output = []
            for reference_item in references_output:
                document_item = self.open_file(reference_item["relativePath"])
                reference_item["range"]["start"]["line"] = max(0, reference_item["range"]["start"]["line"] - context_limit)
                reference_item["range"]["end"]["line"] = min(len(document_item.splitlines(True)), reference_item["range"]["end"]["line"] + context_limit)
                reference_item["range"]["start"]["character"] = 0
                reference_item["range"]["end"]["character"] = len(document_item.splitlines(True)[reference_item["range"]["end"]["line"]-1])
                reference_code = get_text(document_item, reference_item["range"])
                formatted_results = []
                for index, code_line in enumerate(reference_code.split("\n")):
                    code_line = str(index + reference_item["range"]["start"]["line"]) + " " + code_line
                    formatted_results.append(code_line)

                reference_code = "\n".join(formatted_results)

                reference_info = "File Name: " + str(reference_item["relativePath"]) + "\n" + "Implementation: " + str(reference_code) + "\n"
                detailed_output.append(reference_info)
            references_output = detailed_output
        return references_output
