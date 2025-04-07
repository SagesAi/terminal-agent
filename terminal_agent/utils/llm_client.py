#!/usr/bin/env python3
"""
LLM API client for Terminal Agent
Supporting OpenAI and DeepSeek APIs
"""

import os
from typing import List, Dict, Any, Optional, Literal, Union, Tuple
from openai import OpenAI
import httpx
from rich.console import Console
import re
import json
import logging
from rich.progress import Progress
from terminal_agent.utils.logging_config import get_logger

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
        from .command_context import command_context as ctx
        _command_context = ctx
    return _command_context

class LLMClient:
    """Client for interacting with various LLM APIs"""
    
    def __init__(self, 
                 api_key: str = None, 
                 model: str = "gpt-4",
                 provider: Literal["openai", "deepseek"] = "openai",
                 api_base: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            api_key: API key (defaults to environment variable)
            model: LLM model to use
            provider: API provider ("openai" or "deepseek")
            api_base: Base URL for API (optional, for custom endpoints)
        """
        self.provider = provider
        self.model = model
        
        # Set up API key based on provider
        if provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
            self.client = OpenAI(api_key=self.api_key, base_url=api_base)
        elif provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API key not provided and not found in environment variables")
            self.api_base = api_base or "https://api.deepseek.com/v1"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'deepseek'")
    
    def call_llm(self, user_input: str, system_prompt: str, 
                conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Call the LLM API with the given input and system prompt
        
        Args:
            user_input: The user's query or command
            system_prompt: The system prompt to guide the model's response
            conversation_history: Optional list of previous exchanges
            
        Returns:
            The model's response text
        """
        try:
            # 检查是否应该停止所有操作
            from terminal_agent.utils.command_executor import should_stop_operations
            if should_stop_operations():
                return "操作已被用户停止，跳过API调用。"
                
            # Prepare conversation history for context
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history for context if provided
            if conversation_history:
                for exchange in conversation_history:
                    # 确保键名正确，适配不同API
                    user_content = exchange.get("user", exchange.get("content", ""))
                    assistant_content = exchange.get("assistant", exchange.get("content", ""))
                    
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": assistant_content})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Call the appropriate API based on provider
            with Progress() as progress:
                task = progress.add_task("[cyan]Thinking...", total=None)
                
                # 再次检查是否应该停止所有操作
                if should_stop_operations():
                    return "操作已被用户停止，跳过API调用。"
                
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=2000
                    )
                    progress.update(task, completed=100)
                    return response.choices[0].message.content
                
                elif self.provider == "deepseek":
                    # Use httpx for DeepSeek API
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # 为DeepSeek API准备正确的消息格式
                    deepseek_messages = []
                    for msg in messages:
                        # 确保消息格式正确
                        if msg["role"] == "system":
                            # DeepSeek可能使用不同的系统消息格式
                            deepseek_messages.append({
                                "role": "system",
                                "content": msg["content"]
                            })
                        else:
                            deepseek_messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                    
                    payload = {
                        "model": self.model,
                        "messages": deepseek_messages,
                        "temperature": 0.2,
                        "max_tokens": 2000
                    }
                    
                    # 再次检查是否应该停止所有操作
                    if should_stop_operations():
                        return "操作已被用户停止，跳过API调用。"
                    
                    # 记录API请求以便调试
                    #console.print("[dim]Sending request to DeepSeek API...[/dim]")
                    
                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(
                            f"{self.api_base}/chat/completions",
                            headers=headers,
                            json=payload
                        )
                        
                        # 如果请求失败，打印详细错误信息
                        if response.status_code != 200:
                            console.print(f"[bold red]DeepSeek API Error: Status {response.status_code}[/bold red]")
                            console.print(f"[bold red]Response: {response.text}[/bold red]")
                            response.raise_for_status()
                            
                        data = response.json()
                    
                    progress.update(task, completed=100)
                    return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            console.print(f"[bold red]Error calling {self.provider.upper()} API: {str(e)}[/bold red]")
            # 打印更详细的错误信息以便调试
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def call_llm_with_prompt(self, prompt: str) -> str:
        """
        Call the LLM API with a fully formatted prompt string.
        This is a simpler version that doesn't handle conversation history or system prompts.
        
        Args:
            prompt: The complete, formatted prompt to send to the LLM
            
        Returns:
            The model's response text
            
        Raises:
            ConnectionError: When there's a connection issue with the API
        """
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Thinking...", total=None)
                
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=2000
                    )
                    progress.update(task, completed=100)
                    return response.choices[0].message.content
                
                elif self.provider == "deepseek":
                    # Use httpx for DeepSeek API
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 2000
                    }
                    
                    # 记录API请求以便调试
                    #console.print("[dim]Sending request to DeepSeek API...[/dim]")
                    
                    try:
                        with httpx.Client(timeout=60.0) as client:
                            response = client.post(
                                f"{self.api_base}/chat/completions",
                                headers=headers,
                                json=payload
                            )
                            
                            # 如果请求失败，打印详细错误信息
                            if response.status_code != 200:
                                console.print(f"[bold red]DeepSeek API Error: Status {response.status_code}[/bold red]")
                                console.print(f"[bold red]Response: {response.text}[/bold red]")
                                response.raise_for_status()
                                
                            data = response.json()
                        
                        progress.update(task, completed=100)
                        return data["choices"][0]["message"]["content"]
                    except httpx.ConnectError as e:
                        # 连接错误，这是一个严重的错误，应该直接退出 React loop
                        logger.error(f"Connection error with {self.provider.upper()} API: {str(e)}")
                        console.print(f"[bold red]Connection error with {self.provider.upper()} API: {str(e)}[/bold red]")
                        # 抛出特定的连接错误异常
                        raise ConnectionError(f"Unable to connect to {self.provider.upper()} API: {str(e)}")
                
        except ConnectionError:
            # 直接重新抛出连接错误，不进行处理
            raise
        except Exception as e:
            console.print(f"[bold red]Error calling {self.provider.upper()} API: {str(e)}[/bold red]")
            # 打印更详细的错误信息以便调试
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return f"I encountered an error while processing your request: {str(e)}"
    
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
        
        # 尝试从特定标记中提取命令
        commands = []
        
        # 匹配<command>标记中的命令
        command_matches = re.findall(r'<command>([^<]+)</command>', text)
        for cmd in command_matches:
            cmd = cmd.strip()
            if cmd and cmd_ctx.is_valid_command(cmd):
                commands.append(cmd)
        
        # 如果没有找到标记的命令，尝试从代码块中提取
        if not commands:
            code_blocks = re.findall(r'```(?:bash|shell|sh)?\s*([\s\S]*?)```', text)
            for block in code_blocks:
                lines = block.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and cmd_ctx.is_valid_command(line):
                        commands.append(line)
        
        # 如果仍然没有找到命令，尝试查找行内命令
        if not commands:
            inline_commands = re.findall(r'(?:^|\n)\s*[$>]\s*(.*?)(?:\n|$)', text)
            for cmd in inline_commands:
                cmd = cmd.strip()
                if cmd and cmd_ctx.is_valid_command(cmd):
                    commands.append(cmd)
        
        return commands
    
    def _is_valid_command(self, text: str) -> bool:
        """
        判断文本是否是有效的命令，而不是输出或文件路径
        
        Args:
            text: 要检查的文本
            
        Returns:
            bool: 是否是有效的命令
        """
        # 使用命令上下文管理器进行判断
        cmd_ctx = get_command_context()
        return cmd_ctx.is_valid_command(text)


# For backward compatibility
GPTClient = LLMClient
