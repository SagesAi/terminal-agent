#!/usr/bin/env python3
"""
测试命令翻译模块功能 (简化版)

这个测试文件已经简化，只测试兼容层的基本功能。
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from terminal_agent.modules.command_translator import CommandTranslatorModule


class TestCommandTranslatorModule:
    """测试CommandTranslatorModule类的功能 (简化版)"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """创建模拟的LLM客户端"""
        mock_client = MagicMock()
        mock_client.call_llm.return_value = "模拟的LLM响应"
        return mock_client
    
    @pytest.fixture
    def mock_react_module(self):
        """创建模拟的ReActModule"""
        mock_react = MagicMock()
        mock_react.process_query.return_value = "模拟的ReAct响应"
        return mock_react
    
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
    def translator(self, mock_llm_client, system_info):
        """创建CommandTranslatorModule实例"""
        with patch('terminal_agent.modules.command_translator.ReActModule') as mock_react_class:
            mock_react_class.return_value = MagicMock()
            mock_react_class.return_value.process_query.return_value = "模拟的ReAct响应"
            translator = CommandTranslatorModule(mock_llm_client, system_info)
            return translator
    
    def test_initialization(self, translator, mock_llm_client, system_info):
        """测试初始化"""
        assert translator.llm_client == mock_llm_client
        assert translator.system_info == system_info
        assert hasattr(translator, 'react_module')
    
    def test_translate_command(self, translator):
        """测试命令翻译"""
        # 设置模拟返回值
        translator.react_module.process_query.return_value = "模拟的翻译结果"
        
        # 调用方法
        result = translator.translate_command("查找大文件")
        
        # 验证结果
        assert result == "模拟的翻译结果"
        translator.react_module.process_query.assert_called_once_with("查找大文件", None)
    
    def test_execute_commands(self, translator):
        """测试命令执行"""
        # 调用方法
        result = translator.execute_commands(["ls -la", "df -h"], "查看系统状态")
        
        # 验证结果
        assert result["success"] is True
        assert result["commands"] == ["ls -la", "df -h"]
        assert "outputs" in result
