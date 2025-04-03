#!/usr/bin/env python3
"""
测试软件安装模块功能 (简化版)

这个测试文件已经简化，只测试兼容层的基本功能。
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from terminal_agent.modules.software_installer import SoftwareInstallerModule


class TestSoftwareInstallerModule:
    """测试SoftwareInstallerModule类的功能 (简化版)"""
    
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
    def installer(self, mock_llm_client, system_info):
        """创建SoftwareInstallerModule实例"""
        with patch('terminal_agent.modules.software_installer.ReActModule') as mock_react_class:
            mock_react_class.return_value = MagicMock()
            mock_react_class.return_value.process_query.return_value = "模拟的ReAct响应"
            installer = SoftwareInstallerModule(mock_llm_client, system_info)
            return installer
    
    def test_initialization(self, installer, mock_llm_client, system_info):
        """测试初始化"""
        assert installer.llm_client == mock_llm_client
        assert installer.system_info == system_info
        assert hasattr(installer, 'react_module')
    
    def test_install_software(self, installer):
        """测试软件安装"""
        # 设置模拟返回值
        installer.react_module.process_query.return_value = "模拟的安装结果"
        
        # 调用方法
        result = installer.install_software("安装Python 3.9")
        
        # 验证结果
        assert result == "模拟的安装结果"
        installer.react_module.process_query.assert_called_once_with("安装Python 3.9", None)
    
    def test_determine_package_manager(self, installer):
        """测试包管理器确定"""
        # 调用方法
        result = installer._determine_package_manager()
        
        # 验证结果
        assert result == "apt"  # 因为系统信息中设置了Ubuntu
