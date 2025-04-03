#!/usr/bin/env python3
"""
Diagnostics module for Terminal Agent

This is a compatibility layer that uses ReAct Agent internally.
It maintains the same API but delegates the actual work to ReAct Agent.
"""

import logging
from typing import Dict, List, Any, Optional
from rich.console import Console

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_executor import should_stop_operations, reset_stop_flag
from terminal_agent.utils.command_analyzer import CommandAnalyzer
from terminal_agent.modules.react_module import ReActModule

# Initialize Rich console
console = Console()

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiagnosticsModule:
    """
    Module for system diagnostics
    
    This is a compatibility layer that uses ReAct Agent internally.
    """
    
    def __init__(self, llm_client: LLMClient, system_info: Dict[str, str]):
        """
        Initialize diagnostics module
        
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
        logger.info("DiagnosticsModule initialized (using ReAct Agent internally)")
    
    def run_diagnostics(self, user_issue: str, 
                      conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Run diagnostics based on user-described issue
        
        Args:
            user_issue: Description of the issue from user
            conversation_history: Optional conversation history for context
            
        Returns:
            Response text with diagnostic results
        """
        # 重置停止标志，以便开始新的诊断序列
        reset_stop_flag()
        
        # 检查是否应该停止操作
        if should_stop_operations():
            return "操作已被用户停止，跳过系统诊断。"
        
        # 使用ReActModule处理诊断请求
        return self.react_module.process_query(user_issue, conversation_history)
    
    def _analyze_user_intent(self, user_issue: str, 
                           conversation_history: List[Dict[str, str]] = None) -> tuple:
        """
        分析用户问题的意图和目标
        
        Args:
            user_issue: 用户描述的问题
            conversation_history: 可选的对话历史记录
            
        Returns:
            tuple: (意图描述, 目标描述)
        """
        # 简单地返回一个默认值，实际处理由ReActModule完成
        return "Diagnose issue", "Identify and resolve the problem"
    
    def _check_goal_achieved(self, command: str, output: str, return_code: int, goal: str) -> bool:
        """
        检查是否达到了诊断目标
        
        Args:
            command: 执行的命令
            output: 命令输出
            return_code: 返回代码
            goal: 诊断目标
            
        Returns:
            bool: 是否达到目标
        """
        # 简单地检查命令是否成功执行
        return return_code == 0
    
    def _generate_conclusion(self, user_issue: str, results: List[Dict[str, Any]], goal: str) -> str:
        """
        生成诊断结论
        
        Args:
            user_issue: 用户描述的问题
            results: 诊断结果列表
            goal: 诊断目标
            
        Returns:
            str: 诊断结论
        """
        # 如果没有结果，返回默认消息
        if not results:
            return "没有执行诊断命令，无法生成结论。"
        
        # 检查最后一个命令是否成功执行
        last_result = results[-1]
        if last_result.get("return_code", 1) == 0:
            return f"诊断已完成。最后执行的命令 '{last_result.get('command', '')}' 成功执行，可能已解决问题。"
        else:
            return f"诊断未能完全解决问题。最后执行的命令 '{last_result.get('command', '')}' 返回了错误。可能需要进一步调查。"
