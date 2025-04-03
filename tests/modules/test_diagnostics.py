#!/usr/bin/env python3
"""
测试诊断模块功能 (简化版)

这个测试文件已经简化，只测试兼容层的基本功能。
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from terminal_agent.modules.diagnostics import DiagnosticsModule


class TestDiagnosticsModule:
    """测试DiagnosticsModule类的功能 (简化版)"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """创建模拟的LLM客户端"""
        mock_client = MagicMock()
        mock_client.call_llm.return_value = "模拟的LLM响应"
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
    def diagnostics(self, mock_llm_client, system_info):
        """创建DiagnosticsModule实例"""
        with patch('terminal_agent.modules.diagnostics.ReActModule') as mock_react_class:
            mock_react_class.return_value = MagicMock()
            mock_react_class.return_value.process_query.return_value = "模拟的ReAct响应"
            diagnostics = DiagnosticsModule(mock_llm_client, system_info)
            return diagnostics
    
    def test_initialization(self, diagnostics, mock_llm_client, system_info):
        """测试初始化"""
        assert diagnostics.llm_client == mock_llm_client
        assert diagnostics.system_info == system_info
        assert hasattr(diagnostics, 'react_module')
    
    def test_run_diagnostics(self, diagnostics):
        """测试运行诊断"""
        # 设置模拟返回值
        diagnostics.react_module.process_query.return_value = "模拟的诊断结果"
        
        # 调用方法
        result = diagnostics.run_diagnostics("我的磁盘空间不足")
        
        # 验证结果
        assert result == "模拟的诊断结果"
        diagnostics.react_module.process_query.assert_called_once_with("我的磁盘空间不足", None)
    
    def test_check_goal_achieved(self, diagnostics):
        """测试目标达成检查"""
        # 调用方法
        result = diagnostics._check_goal_achieved("df -h", "输出内容", 0, "检查磁盘空间")
        
        # 验证结果
        assert result is True
