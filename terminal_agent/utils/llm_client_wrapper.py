#!/usr/bin/env python3
"""
LLM 客户端包装器

这个模块提供了 LLMClient 的导出，以保持向后兼容性。
现在我们已经完成了 LLMClient 的重构，不再需要区分新旧实现。
"""

import logging
from typing import List, Dict, Any, Optional, Literal, Union

# 获取日志记录器
logger = logging.getLogger(__name__)

# 直接导入当前的 LLMClient 实现
try:
    # 使用绝对导入路径
    from terminal_agent.utils.llm_client import LLMClient, GPTClient, should_stop_operations
    
    logger.info("Using current LLMClient implementation")
    
except Exception as e:
    logger.error(f"Error importing LLMClient: {e}")
    raise

# 导出符号
__all__ = ["LLMClient", "GPTClient", "should_stop_operations"]
