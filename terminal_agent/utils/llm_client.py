#!/usr/bin/env python3
"""
LLM API client for Terminal Agent
Supporting multiple LLM providers including OpenAI, DeepSeek, Gemini, and Claude
"""

import os
from typing import List, Dict, Any, Optional, Literal, Union, Tuple
import logging
import re
import json
from rich.console import Console
from rich.progress import Progress

from terminal_agent.utils.logging_config import get_logger
from terminal_agent.utils.llm_providers import (
    BaseLLMProvider,
    OpenAIProvider,
    DeepSeekProvider,
    GeminiProvider,
    AnthropicProvider,
    OllamaProvider,
    VLLMProvider
)

# 获取日志记录器
logger = get_logger(__name__)

# 初始化Rich控制台
console = Console()

# 延迟导入command_context，避免循环导入
_command_context = None

def get_command_context():
    """获取命令上下文管理器实例，延迟导入避免循环依赖"""
    global _command_context
    if _command_context is None:
        from terminal_agent.utils.command_context import command_context as ctx
        _command_context = ctx
    return _command_context

def should_stop_operations():
    """检查是否应该停止所有操作"""
    try:
        cmd_ctx = get_command_context()
        return getattr(cmd_ctx, "stop_requested", False)
    except Exception:
        # 如果出现任何错误，默认不停止操作
        return False

class LLMClient:
    """Client for interacting with various LLM APIs"""
    
    # 支持的提供商映射
    PROVIDER_MAP = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
        "gemini": GeminiProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "vllm": VLLMProvider
    }
    
    def __init__(self, 
                 api_key: str = None, 
                 model: str = None,
                 provider: str = "openai",
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        Initialize LLM client
        
        Args:
            api_key: API key (defaults to environment variable)
            model: LLM model to use (defaults to provider's default)
            provider: API provider ("openai", "deepseek", "gemini", "anthropic", "ollama", or "vllm")
            api_base: Base URL for API (optional, for custom endpoints)
            **kwargs: Additional provider-specific parameters
        """
        # 验证提供商
        if provider not in self.PROVIDER_MAP:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {', '.join(self.PROVIDER_MAP.keys())}")
        
        self.provider_name = provider
        self.api_key = api_key
        
        # 如果没有指定模型，使用提供商的默认模型
        provider_class = self.PROVIDER_MAP[provider]
        if not model:
            model = provider_class.get_default_model()
        
        # 初始化提供商实例
        self.provider = provider_class(
            api_key=api_key,
            model=model,
            api_base=api_base,
            **kwargs
        )
        
        # 记录初始化信息
        logger.info(f"Initialized LLM client with provider: {provider}, model: {self.provider.model}")
    
    def chat_with_history(self, 
                         user_input: str, 
                         history: List[Dict[str, str]] = None, 
                         system_prompt: str = None) -> str:
        """
        与LLM进行对话，支持历史记录和系统提示
        
        Args:
            user_input: 用户输入
            history: 对话历史记录
            system_prompt: 系统提示
            
        Returns:
            str: 模型回复
        """
        try:
            # 检查是否应该停止所有操作
            if should_stop_operations():
                return "操作已被用户停止，跳过API调用。"
            
            # 准备消息列表
            messages = []
            
            # 添加系统提示（如果有）
            if system_prompt:
                messages.append(self.provider.format_system_prompt(system_prompt))
            
            # 添加历史记录（如果有）
            if history:
                messages.extend(history)
            
            # 添加当前用户输入
            messages.append(self.provider.format_user_prompt(user_input))
            
            # 使用进度条显示
            with Progress() as progress:
                task = progress.add_task("[cyan]thinking...", total=None)
                
                # 再次检查是否应该停止所有操作
                if should_stop_operations():
                    return "Operation stopped by user, skipping API call."
                
                # 调用提供商的API
                response = self.provider.call_with_messages(messages)
                
                progress.update(task, completed=100)
                return response
                
        except Exception as e:
            console.print(f"[bold red]调用 {self.provider_name.upper()} API 时出错: {str(e)}[/bold red]")
            # 打印更详细的错误信息以便调试
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return f"处理您的请求时遇到错误: {str(e)}"
    
    def call_llm_with_prompt(self, prompt: str) -> str:
        """
        使用完整格式化的提示字符串调用LLM API
        这是一个更简单的版本，不处理对话历史或系统提示
        
        Args:
            prompt: 发送给LLM的完整格式化提示
            
        Returns:
            str: 模型的响应文本
            
        Raises:
            ConnectionError: 与API连接出现问题时
        """
        try:
            # 检查是否应该停止所有操作
            if should_stop_operations():
                return "操作已被用户停止，跳过API调用。"
            
            # 使用进度条显示
            with Progress() as progress:
                task = progress.add_task("[cyan]thinking...", total=None)
                
                # 再次检查是否应该停止所有操作
                if should_stop_operations():
                    return "Operation stopped by user, skipping API call."
                
                # 调用提供商的API
                response = self.provider.call_with_prompt(prompt)
                
                progress.update(task, completed=100)
                return response
                
        except ConnectionError:
            # 直接重新抛出连接错误，不进行处理
            raise
        except Exception as e:
            console.print(f"[bold red]调用 {self.provider_name.upper()} API 时出错: {str(e)}[/bold red]")
            # 打印更详细的错误信息以便调试
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return f"处理您的请求时遇到错误: {str(e)}"
    
    def call_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM API with a list of messages
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: The model's response text
            
        Raises:
            ConnectionError: When there's a connection issue with the API
        """
        try:
            # Check if operations should be stopped
            if should_stop_operations():
                return "Operation stopped by user, skipping API call."
            
            # Display thinking progress bar
            with Progress() as progress:
                task = progress.add_task("[cyan]Thinking...", total=None)
                
                try:
                    # Call the API
                    response = self.provider.call_with_messages(messages)
            
                    
                    # Complete the progress bar
                    progress.update(task, completed=100)
                    
                    return response
                    
                except Exception as e:
                    # Complete the progress bar (with error)
                    progress.update(task, completed=100)
                    raise e
                
        except ConnectionError as e:
            logger.error(f"Connection error with LLM API: {str(e)}")
            raise ConnectionError(f"Unable to connect to LLM API: {str(e)}")
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            raise
    
    def extract_commands(self, text: str) -> List[str]:
        """
        从文本中提取命令
        
        Args:
            text: 包含命令的文本
            
        Returns:
            List[str]: 提取的命令列表
        """
        # 获取命令上下文管理器
        cmd_ctx = get_command_context()
        
        # 检查是否有 is_valid_command 方法，如果没有则使用 is_command 方法或默认实现
        def is_valid_command(cmd: str) -> bool:
            if hasattr(cmd_ctx, "is_valid_command"):
                return cmd_ctx.is_valid_command(cmd)
            elif hasattr(cmd_ctx, "is_command"):
                return cmd_ctx.is_command(cmd)
            else:
                # 默认实现：非空命令都视为有效
                return bool(cmd.strip())
        
        # 尝试从特定标记中提取命令
        commands = []
        
        # 匹配<command>标记中的命令
        command_matches = re.findall(r'<command>([^<]+)</command>', text)
        for cmd in command_matches:
            cmd = cmd.strip()
            if cmd and is_valid_command(cmd):
                commands.append(cmd)
        
        # 如果没有找到标记的命令，尝试从代码块中提取
        if not commands:
            code_blocks = re.findall(r'```(?:bash|shell|sh)?\s*([\s\S]*?)```', text)
            for block in code_blocks:
                lines = block.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and is_valid_command(line):
                        commands.append(line)
        
        # 如果仍然没有找到命令，尝试查找行内命令
        if not commands:
            inline_commands = re.findall(r'(?:^|\n)\s*[$>]\s*(.*?)(?:\n|$)', text)
            for cmd in inline_commands:
                cmd = cmd.strip()
                if cmd and is_valid_command(cmd):
                    commands.append(cmd)
        
        return commands
    
    def extract_json_from_text(self, text: str) -> Dict:
        """
        从文本中提取 JSON 对象。
        
        支持从以下格式提取 JSON：
        1. Markdown 代码块（```json ... ```）
        2. 纯文本中的 JSON 对象（{...}）
        3. 带有 "Thought:" 前缀的 JSON
        
        Args:
            text: 包含 JSON 的文本
            
        Returns:
            提取的 JSON 对象（字典）
        """
        if not text:
            return {}
        
        # 如果输入已经是字典，直接返回
        if isinstance(text, dict):
            return text
        
        # 记录原始输入（用于调试）
        logger.debug(f"提取 JSON 的原始输入: {text[:100]}...")
        
        # 尝试从 Markdown 代码块中提取 JSON
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        json_matches = re.findall(code_block_pattern, text)
        
        # 如果没有找到代码块，尝试直接从文本中提取 JSON
        if not json_matches:
            # 处理可能带有 "Thought:" 前缀的 JSON
            if text.strip().startswith("Thought:"):
                text = text.strip()[len("Thought:"):].strip()
            
            # 尝试找到最外层的大括号对
            try:
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and start < end:
                    json_matches = [text[start:end+1]]
            except Exception:
                pass
        
        # 处理找到的每个 JSON 匹配项
        for json_str in json_matches:
            # 移除 JavaScript 风格的注释
            # 移除单行注释 (// ...)
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
            # 移除多行注释 (/* ... */)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # 基本清理
            json_str = json_str.strip()
            
            # 修复常见的 JSON 格式问题
            
            # 1. 修复缺少引号的键
            json_str = re.sub(r'([\{\,]\s*)([a-zA-Z0-9_-]+)(\s*:)', r'\1"\2"\3', json_str)
            
            # 2. 将单引号替换为双引号
            # 处理键
            json_str = re.sub(r'\'([^\']*?)\'(\s*:)', r'"\1"\2', json_str)
            # 处理值
            json_str = re.sub(r':\s*\'([^\']*?)\'', r': "\1"', json_str)
            # 处理数组中的值
            json_str = re.sub(r'\[\s*\'([^\']*?)\'', r'["\1"', json_str)
            json_str = re.sub(r'\'([^\']*?)\'\s*\]', r'"\1"]', json_str)
            json_str = re.sub(r'\'([^\']*?)\'\s*,', r'"\1",', json_str)
            
            # 3. 修复尾部多余的逗号
            json_str = re.sub(r',(\s*[\}\]])', r'\1', json_str)
            
            # 4. 处理多行字符串，将多个空格替换为单个空格
            multiline_pattern = r'"([^"]*?\n[^"]*?)"'
            for match in re.finditer(multiline_pattern, json_str):
                original = match.group(0)
                content = match.group(1)
                # 将多行字符串中的换行和多余空格替换为单个空格
                cleaned = re.sub(r'\s+', ' ', content).strip()
                json_str = json_str.replace(original, f'"{cleaned}"')
            
            # 尝试解析 JSON
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                # 如果解析失败，尝试处理特殊情况（如脚本内容）
                try:
                    # 查找脚本内容
                    script_pattern = r'"content":\s*"(.*?)",'
                    script_match = re.search(script_pattern, json_str, re.DOTALL)
                    
                    if script_match:
                        # 找到了脚本内容
                        script_content = script_match.group(1)
                        # 创建占位符
                        placeholder = "__SCRIPT_CONTENT__"
                        # 替换脚本内容为占位符
                        json_with_placeholder = json_str.replace(script_match.group(1), placeholder)
                        
                        # 尝试解析替换后的 JSON
                        try:
                            parsed = json.loads(json_with_placeholder)
                            if isinstance(parsed, dict) and "action" in parsed and "input" in parsed["action"]:
                                # 恢复脚本内容
                                parsed["action"]["input"]["content"] = script_content
                                return parsed
                        except json.JSONDecodeError:
                            pass
                except Exception as script_err:
                    pass
        
        # 如果所有尝试都失败，返回空字典
        logger.warning("无法从文本中提取有效的JSON")
        return {}
    
    @property
    def model(self) -> str:
        """获取当前使用的模型名称"""
        return self.provider.model
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """获取所有可用的提供商列表"""
        return list(cls.PROVIDER_MAP.keys())
    
    def get_available_models(self) -> List[str]:
        """获取当前提供商的可用模型列表"""
        return self.provider.get_available_models()

# 为向后兼容性保留
GPTClient = LLMClient
