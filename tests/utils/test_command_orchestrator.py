#!/usr/bin/env python3
"""
测试命令编排器功能 (简化版)

这个测试文件已经简化，只测试兼容层的基本功能。
"""

import pytest
from unittest.mock import MagicMock
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from terminal_agent.utils.command_orchestrator import CommandOrchestrator


class TestCommandOrchestrator:
    """测试CommandOrchestrator类的功能 (简化版)"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """创建模拟的LLM客户端"""
        mock_client = MagicMock()
        mock_client.call_llm.return_value = '{"dependencies": {}}'
        return mock_client
    
    @pytest.fixture
    def system_info(self):
        """创建系统信息字典"""
        return {
            "os": "Linux",
            "distribution": "Ubuntu",
            "version": "20.04",
            "kernel": "5.4.0",
            "architecture": "x86_64"
        }
    
    @pytest.fixture
    def orchestrator(self, mock_llm_client, system_info):
        """创建CommandOrchestrator实例"""
        return CommandOrchestrator(mock_llm_client, system_info)
    
    def test_initialization(self, orchestrator, mock_llm_client, system_info):
        """测试初始化"""
        assert orchestrator.llm_client == mock_llm_client
        assert orchestrator.system_info == system_info
    
    def test_analyze_and_optimize(self, orchestrator):
        """测试命令分析和优化"""
        commands = ["apt-get update", "apt-get install -y python3"]
        user_goal = "安装Python"
        
        # 调用方法
        result = orchestrator.analyze_and_optimize(commands, user_goal)
        
        # 验证结果 - 简化版本应该直接返回原始命令
        assert result == commands
    
    def test_check_command_safety(self, orchestrator):
        """测试命令安全检查"""
        # 调用方法
        result = orchestrator.check_command_safety("rm -rf /tmp/test")
        
        # 验证结果
        assert result["is_safe"] is True
        assert "risk_level" in result
        assert "explanation" in result
    
    def test_suggest_command_improvements(self, orchestrator):
        """测试命令改进建议"""
        # 调用方法
        result = orchestrator.suggest_command_improvements("apt-get intall python3", "E: Invalid operation intall", 1)
        
        # 验证结果 - 简化版本应该返回空列表
        assert result == []
