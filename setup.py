#!/usr/bin/env python3
"""
Setup script for Terminal Agent

A modern, intelligent terminal assistant that uses natural language to help with
command-line tasks, diagnostics, and system operations.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 核心依赖
requirements = [
    "openai>=1.12.0",
    "httpx>=0.26.0",
    "rich>=13.7.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "psutil>=5.9.6",
    "colorama>=0.4.6",
    "pyfiglet>=1.0.2",
    "typer>=0.9.0",
    "prompt-toolkit>=3.0.43",
    # 网页爬取工具依赖
    "requests>=2.28.0",
    "markdownify>=0.11.6",
    "readabilipy>=0.2.0",
    # 其他 LLM 提供商支持
    "google-generativeai>=0.3.0",  # Gemini 支持
    "anthropic>=0.5.0",  # Claude 支持
    # 远程执行支持
    "paramiko>=2.7.0",  # SSH 连接
]

# 可选依赖 - 已经全部移到核心依赖中
extras_require = {}

setup(
    name="terminal-agent",
    version="0.2.0",
    author="Terminal Agent Team",
    author_email="info@terminalagent.ai",
    description="An intelligent terminal assistant using ReAct architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SagesAi/terminal-agent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: System :: Shells",
        "Topic :: Utilities",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "terminal-agent=terminal_agent.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "terminal_agent": [
            "react/templates/*.txt",
        ],
    },
)
