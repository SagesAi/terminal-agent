#!/usr/bin/env python3
"""
Command translation module for Terminal Agent

This is a compatibility layer that uses ReAct Agent internally.
It maintains the same API but delegates the actual work to ReAct Agent.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from rich.console import Console

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_executor import should_stop_operations, reset_stop_flag
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.modules.react_module import ReActModule

# 初始化Rich控制台
console = Console()

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommandTranslatorModule:
    """
    Module for translating natural language to shell commands
    
    This is a compatibility layer that uses ReAct Agent internally.
    """
    
    def __init__(self, llm_client: LLMClient, system_info: Dict[str, Any]):
        """
        Initialize command translator module
        
        Args:
            llm_client: LLM client for API interactions
            system_info: Dictionary containing system information
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.command_analyzer = CommandAnalyzer(llm_client, system_info)
        
        # 使用ReActModule作为内部实现
        self.react_module = ReActModule(llm_client, system_info)
        
        # 记录日志
        logger.info("CommandTranslatorModule initialized (using ReAct Agent internally)")
    
    def translate_command(self, natural_language_request: str, 
                         conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Translate natural language request to shell command
        
        Args:
            natural_language_request: Natural language request from user
            conversation_history: Optional conversation history for context
            
        Returns:
            Response text with command and explanation
        """
        # 重置停止标志，以便开始新的翻译序列
        reset_stop_flag()
        
        # 检查是否应该停止操作
        if should_stop_operations():
            return "操作已被用户停止，跳过命令翻译。"
        
        # 使用ReActModule处理查询
        return self.react_module.process_query(natural_language_request, conversation_history)
    
    def execute_commands(self, commands: List[str], user_goal: str) -> Dict[str, Any]:
        """
        执行命令列表并返回结果
        
        Args:
            commands: 要执行的命令列表
            user_goal: 用户目标
            
        Returns:
            Dict: 包含执行结果的字典
        """
        # 简单地将请求传递给ReActModule
        prompt = f"Execute the following commands to achieve this goal: {user_goal}\nCommands: {', '.join(commands)}"
        self.react_module.process_query(prompt)
        
        # 返回一个简单的成功响应
        return {
            "success": True,
            "commands": commands,
            "outputs": {}
        }
    
    def _handle_suggested_commands(self, suggested_commands: List[str], 
                                  conversation_history: List[Dict[str, str]] = None,
                                  user_goal: str = "") -> None:
        """
        处理建议的命令列表
        
        Args:
            suggested_commands: 建议的命令列表
            conversation_history: 可选的对话历史记录
            user_goal: 用户目标
        """
        if not suggested_commands or should_stop_operations():
            return
            
        # 将请求传递给ReActModule
        prompt = f"Execute one of these suggested commands to achieve this goal: {user_goal}\nCommands: {', '.join(suggested_commands)}"
        self.react_module.process_query(prompt, conversation_history)
    
    def _analyze_user_intent(self, user_request: str, 
                           conversation_history: List[Dict[str, str]] = None) -> tuple:
        """
        分析用户请求的意图和目标
        
        Args:
            user_request: 用户请求
            conversation_history: 可选的对话历史记录
            
        Returns:
            tuple: (意图描述, 目标描述)
        """
        # 简单地返回一个默认值，实际处理由ReActModule完成
        return "Execute command", "Command execution completed"
    
    def _check_goal_achieved(self, command: str, output: str, return_code: int, goal: str) -> bool:
        """
        检查是否达到了用户目标
        
        Args:
            command: 执行的命令
            output: 命令输出
            return_code: 返回代码
            goal: 用户目标
            
        Returns:
            bool: 是否达到目标
        """
        # 简单地检查命令是否成功执行
        return return_code == 0
    
    def _is_valid_command(self, text: str) -> bool:
        """
        判断文本是否是有效的命令，而不是输出或文件路径
        
        Args:
            text: 要检查的文本
            
        Returns:
            bool: 是否是有效的命令
        """
        # 简单地检查文本是否非空
        return bool(text and text.strip())
