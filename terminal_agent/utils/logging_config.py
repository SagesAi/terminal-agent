#!/usr/bin/env python3
"""
日志配置模块，提供集中的日志系统配置功能
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional


def configure_logging(log_level_str: str = "WARNING", enable_file_logging: bool = True) -> str:
    """
    配置全局日志系统
    
    Args:
        log_level_str: 日志级别字符串 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        enable_file_logging: 是否启用文件日志
        
    Returns:
        str: 日志文件路径（如果启用了文件日志）
    """
    # 将日志级别字符串转换为对应的常量
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = log_level_map.get(log_level_str.upper(), logging.WARNING)
    
    # 创建基本处理器列表
    handlers = [logging.StreamHandler(sys.stdout)]  # 始终添加控制台处理器
    
    # 如果启用文件日志，创建文件处理器
    log_file = None
    if enable_file_logging:
        # 创建日志目录
        log_dir = os.path.join(os.path.expanduser("~"), ".terminal_agent", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名，包含日期和时间
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"terminal_agent_{log_timestamp}.log")
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)  # 文件处理器使用相同的日志级别
        
        # 设置格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到处理器列表
        handlers.append(file_handler)
    
    # 配置根日志记录器
    # 注意：这会重置之前的配置，但我们确保这是应用程序中第一个调用的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # 设置根日志记录器级别和处理器
    logging.root.setLevel(log_level)
    for handler in handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
    
    # 配置特定模块的日志级别
    # 例如，如果某些模块需要不同的日志级别，可以在这里设置
    # logging.getLogger("some_module").setLevel(logging.DEBUG)
    
    # 记录初始日志消息，确认配置成功
    logging.info(f"日志系统已配置，级别: {log_level_str}, 文件: {log_file if enable_file_logging else '禁用'}")
    
    return log_file if enable_file_logging else ""


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return logging.getLogger(name)
