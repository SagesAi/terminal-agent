#!/usr/bin/env python3
"""
Software installer module for Terminal Agent

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


class SoftwareInstallerModule:
    """
    Module for handling software installation requests
    
    This is a compatibility layer that uses ReAct Agent internally.
    """
    
    def __init__(self, llm_client: LLMClient, system_info: Dict[str, str]):
        """
        Initialize software installer module
        
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
        logger.info("SoftwareInstallerModule initialized (using ReAct Agent internally)")
    
    def install_software(self, software_request: str, 
                       conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Process software installation request
        
        Args:
            software_request: Natural language request for software installation
            conversation_history: Optional conversation history for context
            
        Returns:
            Response text with installation results
        """
        # 重置停止标志，以便开始新的安装序列
        reset_stop_flag()
        
        # 检查是否应该停止操作
        if should_stop_operations():
            return "操作已被用户停止，跳过软件安装。"
        
        # 使用ReActModule处理安装请求
        return self.react_module.process_query(software_request, conversation_history)
    
    def _analyze_software_request(self, software_request: str, 
                                conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        分析软件安装请求
        
        Args:
            software_request: 软件安装请求
            conversation_history: 可选的对话历史记录
            
        Returns:
            Dict: 包含软件名称、版本等信息的字典
        """
        # 简单地返回一个默认值，实际处理由ReActModule完成
        return {
            "software_name": "unknown",
            "version": "latest",
            "package_manager": "apt" if "ubuntu" in self.system_info.get("distribution", "").lower() else "unknown"
        }
    
    def _determine_package_manager(self) -> str:
        """
        确定系统使用的包管理器
        
        Returns:
            str: 包管理器名称
        """
        # 根据系统信息确定包管理器
        distribution = self.system_info.get("distribution", "").lower()
        
        if "ubuntu" in distribution or "debian" in distribution:
            return "apt"
        elif "fedora" in distribution or "centos" in distribution or "rhel" in distribution:
            return "dnf" if "fedora" in distribution else "yum"
        elif "arch" in distribution:
            return "pacman"
        elif "darwin" in self.system_info.get("os", "").lower():
            return "brew"
        else:
            return "unknown"
    
    def _check_installation_success(self, software_name: str) -> bool:
        """
        检查软件是否成功安装
        
        Args:
            software_name: 软件名称
            
        Returns:
            bool: 是否成功安装
        """
        # 简单地返回成功，实际检查由ReActModule完成
        return True
