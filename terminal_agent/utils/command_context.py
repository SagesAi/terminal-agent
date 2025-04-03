#!/usr/bin/env python3
"""
命令上下文管理模块 - 跟踪命令执行上下文

这是一个简化的兼容层，大部分功能现在由ReAct Agent处理。
"""

import logging
from typing import List, Dict, Set, Optional
import re
from rich.console import Console

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化Rich控制台
console = Console()

class CommandContextManager:
    """
    命令上下文管理器 - 跟踪和管理命令执行上下文
    
    这是一个简化的兼容层，大部分功能现在由ReAct Agent处理。
    """
    
    def __init__(self):
        """初始化命令上下文管理器"""
        # 记录已执行的命令
        self.executed_commands: List[str] = []
        # 记录命令的输出
        self.command_outputs: Dict[str, str] = {}
        # 当前会话中识别的有效命令前缀
        self.valid_command_prefixes: Set[str] = set()
        # 当前会话中识别的输出模式
        self.output_patterns: List[re.Pattern] = []
        # 命令执行状态
        self.in_command_execution = False
        # 最后执行的命令
        self.last_command: Optional[str] = None
        
        # 记录日志
        logger.info("CommandContextManager initialized (simplified compatibility layer)")
        console.print("[dim]CommandContextManager已初始化(简化兼容层)[/dim]", style="dim")
    
    def _initialize_common_prefixes(self):
        """初始化常见命令前缀"""
        # 简化版本不初始化前缀
        pass
    
    def _initialize_output_patterns(self):
        """初始化常见输出模式"""
        # 简化版本不初始化输出模式
        pass
    
    def add_command(self, command: str) -> None:
        """
        添加命令到执行历史
        
        Args:
            command: 执行的命令
        """
        self.executed_commands.append(command)
        self.last_command = command
        self.in_command_execution = True
    
    def add_output(self, command: str, output: str) -> None:
        """
        添加命令输出
        
        Args:
            command: 执行的命令
            output: 命令输出
        """
        self.command_outputs[command] = output
        self.in_command_execution = False
    
    def is_command(self, text: str) -> bool:
        """
        判断文本是否是命令
        
        Args:
            text: 要判断的文本
            
        Returns:
            bool: 是否是命令
        """
        # 简化版本总是返回True
        return True
    
    def get_command_history(self, limit: int = 10) -> List[str]:
        """
        获取命令执行历史
        
        Args:
            limit: 最大返回数量
            
        Returns:
            List[str]: 命令历史列表
        """
        return self.executed_commands[-limit:] if self.executed_commands else []
    
    def get_output(self, command: str) -> Optional[str]:
        """
        获取命令的输出
        
        Args:
            command: 命令
            
        Returns:
            Optional[str]: 命令输出，如果不存在则返回None
        """
        return self.command_outputs.get(command)
    
    def clear_history(self) -> None:
        """清除命令历史"""
        self.executed_commands = []
        self.command_outputs = {}
        self.last_command = None
        self.in_command_execution = False

# 创建全局命令上下文管理器实例
command_context = CommandContextManager()
