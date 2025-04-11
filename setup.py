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
    "openai>=1.0.0",
    "httpx>=0.24.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "psutil>=5.9.0",
    "colorama>=0.4.6",
]

setup(
    name="terminal-agent",
    version="0.2.0",
    author="Terminal Agent Team",
    author_email="info@terminalagent.ai",
    description="An intelligent terminal assistant using ReAct architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terminalagent/terminal_agent",
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
