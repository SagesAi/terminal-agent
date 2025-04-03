#!/usr/bin/env python3
"""
Command output analyzer for Terminal Agent
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from terminal_agent.utils.llm_client import LLMClient
from terminal_agent.utils.command_executor import should_stop_operations
from terminal_agent.utils.file_editor import FileEditor

# 初始化Rich控制台
console = Console()

# 延迟导入command_context，避免循环导入
_command_context = None

def get_command_context():
    """获取命令上下文管理器实例，延迟导入避免循环依赖"""
    global _command_context
    if _command_context is None:
        from .command_context import command_context as ctx
        _command_context = ctx
    return _command_context

class CommandAnalyzer:
    """Analyze command outputs and provide intelligent suggestions"""
    
    def __init__(self, llm_client: LLMClient, system_info: Dict[str, str]):
        """
        Initialize command analyzer
        
        Args:
            llm_client: LLM client for API interactions
            system_info: Dictionary containing system information
        """
        self.llm_client = llm_client
        self.system_info = system_info
        self.file_editor = FileEditor()
        # 保存分析历史，用于上下文理解
        self.analysis_history = []
    
    def analyze_output(self, command: str, output: str, return_code: int,
                      conversation_history: List[Dict[str, str]] = None,
                      user_goal: Optional[str] = None) -> Tuple[str, Optional[List[str]]]:
        """
        Analyze command output and suggest next actions
        
        Args:
            command: The command that was executed
            output: The output of the command
            return_code: The return code of the command
            conversation_history: Optional conversation history for context
            user_goal: Optional user goal for context
            
        Returns:
            Tuple of (analysis, suggested_commands)
        """
        # 检查是否应该停止所有操作
        if should_stop_operations():
            return "操作已被用户停止，跳过分析。", None
            
        # 首先进行快速分析，检查常见模式
        quick_analysis = self._quick_analyze(command, output, return_code)
        if quick_analysis:
            # 如果快速分析发现了明确的结果，直接返回
            analysis, suggested_commands = quick_analysis
            # 保存分析结果到历史
            self.analysis_history.append({
                "command": command,
                "output_summary": output[:100] + "..." if len(output) > 100 else output,
                "analysis": analysis,
                "suggestions": suggested_commands
            })
            return analysis, suggested_commands
        
        # 准备分析上下文
        context = ""
        if self.analysis_history:
            context = "Previous command analysis:\n"
            for i, item in enumerate(self.analysis_history[-3:]):  # 只使用最近的3个分析
                context += f"{i+1}. Command: {item['command']}\n"
                context += f"   Analysis: {item['analysis'][:100]}...\n"
        
        # 准备用户目标上下文
        goal_context = ""
        if user_goal:
            goal_context = f"\nUser's goal: {user_goal}\n"
            goal_context += "Evaluate if this command output indicates the goal has been achieved or if further actions are needed.\n"
        
        # Prepare system prompt for analysis
        system_prompt = f"""
        You are an expert Linux command output analyzer. Analyze the output of the command
        and provide insights and suggestions for next steps.
        
        Current system information:
        - OS: {self.system_info['os']}
        - Distribution: {self.system_info['distribution']}
        - Version: {self.system_info['version']}
        - Architecture: {self.system_info['architecture']}
        
        {context}
        
        Command executed: {command}
        Return code: {return_code} (0 means success, non-zero indicates error)
        {goal_context}
        
        Follow these guidelines:
        1. Explain what the command output shows in simple terms
        2. Identify any errors or warnings in the output
        3. Suggest 2-3 follow-up commands that would be helpful based on this output
        4. For each suggested command, explain why it would be useful
        5. If the output indicates a system issue, suggest commands to diagnose or fix it
        6. Format your suggested commands in code blocks using ```bash
        
        Be concise but thorough in your analysis.
        """
        
        # 再次检查是否应该停止所有操作
        if should_stop_operations():
            return "操作已被用户停止，跳过分析。", None
            
        # 限制输出长度，避免超出token限制
        max_output_length = 2000
        truncated_output = output
        if len(output) > max_output_length:
            truncated_output = output[:max_output_length] + f"\n... (output truncated, total length: {len(output)} characters)"
        
        # Prepare input for the LLM
        input_text = f"""
        Command: {command}
        Return Code: {return_code}
        
        Output:
        {truncated_output}
        """
        
        # Get analysis from LLM
        analysis = self.llm_client.call_llm(
            input_text, 
            system_prompt, 
            conversation_history
        )
        
        # 确保分析结果是字符串
        if not isinstance(analysis, str):
            analysis = str(analysis)
        
        # Extract suggested commands from the analysis
        suggested_commands = self.llm_client.extract_commands(analysis)
        
        # 确保suggested_commands是列表
        if suggested_commands is None:
            suggested_commands = []
        
        # 保存分析结果到历史
        self.analysis_history.append({
            "command": command,
            "output_summary": output[:100] + "..." if len(output) > 100 else output,
            "analysis": analysis,
            "suggestions": suggested_commands
        })
        
        # 如果历史记录太长，只保留最近的10个
        if len(self.analysis_history) > 10:
            self.analysis_history = self.analysis_history[-10:]
        
        return analysis, suggested_commands
    
    def _quick_analyze(self, command: str, output: str, return_code: int) -> Optional[Tuple[str, List[str]]]:
        """
        快速分析命令输出，检查常见模式
        
        Args:
            command: 执行的命令
            output: 命令输出
            return_code: 返回代码
            
        Returns:
            Optional[Tuple[str, List[str]]]: 如果能快速分析，返回(分析, 建议命令)，否则返回None
        """
        # 分割命令以获取基本命令
        cmd_parts = command.split()
        base_command = cmd_parts[0] if cmd_parts else ""
        
        # 检查命令是否存在
        if "command not found" in output or "not found" in output and return_code != 0:
            # 提取可能的命令名称
            cmd_name = base_command
            analysis = f"""
            ### Command Not Found
            
            The command `{cmd_name}` was not found on your system. This usually means the program is not installed.
            """
            
            # 根据不同的系统提供安装建议
            suggested_commands = []
            if self.system_info['os'] == 'Linux':
                if self.system_info['distribution'].lower() in ['ubuntu', 'debian']:
                    suggested_commands = [
                        f"apt update && apt install {cmd_name}",
                        f"apt search {cmd_name}"
                    ]
                    analysis += f"\nYou can try installing it using apt:"
                elif self.system_info['distribution'].lower() in ['fedora', 'centos', 'rhel']:
                    suggested_commands = [
                        f"yum install {cmd_name}",
                        f"dnf search {cmd_name}"
                    ]
                    analysis += f"\nYou can try installing it using yum or dnf:"
            elif self.system_info['os'] == 'Darwin':  # macOS
                suggested_commands = [
                    f"brew install {cmd_name}",
                    f"brew search {cmd_name}"
                ]
                analysis += f"\nYou can try installing it using Homebrew:"
            
            return analysis, suggested_commands
            
        # 检查权限错误
        if "permission denied" in output.lower() and return_code != 0:
            analysis = """
            ### Permission Denied
            
            You don't have sufficient permissions to run this command.
            """
            
            suggested_commands = [f"sudo {command}"]
            analysis += "\nYou can try running the command with sudo:"
            
            # 如果是Docker命令，提供添加用户到docker组的建议
            if base_command == "docker":
                suggested_commands.append("sudo usermod -aG docker $USER && newgrp docker")
                analysis += "\nFor Docker commands, you can either use sudo or add your user to the docker group:"
            
            return analysis, suggested_commands
        
        # 检查Docker特定错误
        if base_command == "docker" and return_code != 0:
            # Docker守护进程未运行
            if "cannot connect to the docker daemon" in output.lower() or "is the docker daemon running" in output.lower():
                analysis = """
                ### Docker Daemon Not Running
                
                The Docker daemon is not running or you cannot connect to it.
                """
                
                suggested_commands = [
                    "sudo systemctl start docker",
                    "sudo service docker start",
                    "docker info"
                ]
                
                analysis += "\nYou can try starting the Docker daemon:"
                
                return analysis, suggested_commands
                
            # NVIDIA容器工具包错误
            if "could not select device driver" in output.lower() and "nvidia" in command.lower():
                analysis = """
                ### NVIDIA Container Toolkit Error
                
                Docker cannot select the NVIDIA device driver. This typically happens when trying to use NVIDIA GPUs with Docker but the NVIDIA Container Toolkit is not properly installed or configured.
                """
                
                suggested_commands = [
                    "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit",
                    "sudo systemctl restart docker",
                    "nvidia-smi",
                    "docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"
                ]
                
                analysis += "\nYou can try installing and configuring the NVIDIA Container Toolkit:"
                
                return analysis, suggested_commands
                
            # 镜像不存在
            if "no such image" in output.lower() or "not found locally" in output.lower():
                # 尝试提取镜像名称
                image_name = ""
                for i, part in enumerate(cmd_parts):
                    if part == "run" and i+1 < len(cmd_parts):
                        image_name = cmd_parts[i+1]
                        break
                
                analysis = f"""
                ### Docker Image Not Found
                
                The Docker image was not found locally.
                """
                
                suggested_commands = []
                if image_name:
                    suggested_commands = [
                        f"docker pull {image_name}",
                        f"docker images | grep {image_name.split(':')[0]}"
                    ]
                    analysis += f"\nYou can try pulling the image '{image_name}':"
                else:
                    suggested_commands = [
                        "docker images",
                        "docker pull <image_name>"
                    ]
                    analysis += "\nYou can list available images and pull the one you need:"
                
                return analysis, suggested_commands
        
        # 检查网络连接错误
        if ("could not resolve" in output.lower() or "name resolution" in output.lower() or 
            "network is unreachable" in output.lower() or "connection refused" in output.lower()) and return_code != 0:
            analysis = """
            ### Network Connection Error
            
            There seems to be a network connectivity issue.
            """
            
            suggested_commands = [
                "ping -c 4 8.8.8.8",
                "ping -c 4 google.com",
                "ip addr show",
                "ifconfig",
                "nmcli dev status"
            ]
            
            analysis += "\nYou can check your network connection with these commands:"
            
            return analysis, suggested_commands
        
        # 检查磁盘空间不足
        if "no space left on device" in output.lower() and return_code != 0:
            analysis = """
            ### Disk Space Error
            
            You've run out of disk space on your device.
            """
            
            suggested_commands = [
                "df -h",
                "du -sh /* | sort -hr | head -10"
            ]
            
            analysis += "\nYou can check your disk usage and free up space:"
            
            return analysis, suggested_commands
        
        # 检查文件或目录不存在
        if "no such file or directory" in output.lower() and return_code != 0:
            analysis = """
            ### File or Directory Not Found
            
            The specified file or directory does not exist.
            """
            
            suggested_commands = [
                "ls -la",
                "pwd"
            ]
            
            analysis += "\nYou can check the current directory and files:"
            
            return analysis, suggested_commands
            
        # 检查包管理器错误
        if base_command in ["apt", "apt-get", "yum", "dnf"] and return_code != 0:
            if "unable to locate package" in output.lower() or "no package" in output.lower():
                # 尝试提取包名
                package_name = ""
                if "install" in cmd_parts:
                    install_index = cmd_parts.index("install")
                    if install_index + 1 < len(cmd_parts):
                        package_name = cmd_parts[install_index + 1]
                
                analysis = f"""
                ### Package Not Found
                
                The package manager could not find the requested package.
                """
                
                suggested_commands = []
                if base_command in ["apt", "apt-get"]:
                    suggested_commands = [
                        "sudo apt update",
                        f"apt search {package_name}" if package_name else "apt search <package_name>"
                    ]
                    analysis += f"\nYou can update your package lists and search for the package:"
                elif base_command in ["yum", "dnf"]:
                    suggested_commands = [
                        f"yum search {package_name}" if package_name else "yum search <package_name>",
                        "yum repolist"
                    ]
                    analysis += f"\nYou can search for the package and check your enabled repositories:"
                
                return analysis, suggested_commands
            
            if "could not get lock" in output.lower() or "another process is using" in output.lower():
                analysis = """
                ### Package Manager Lock Error
                
                Another process is currently using the package management system.
                """
                
                suggested_commands = []
                if base_command in ["apt", "apt-get"]:
                    suggested_commands = [
                        "ps aux | grep -i apt",
                        "sudo killall apt apt-get",
                        "sudo rm /var/lib/apt/lists/lock",
                        "sudo rm /var/cache/apt/archives/lock",
                        "sudo rm /var/lib/dpkg/lock*"
                    ]
                    analysis += "\nYou can check for and kill any running apt processes, then remove the lock files:"
                elif base_command in ["yum", "dnf"]:
                    suggested_commands = [
                        "ps aux | grep -i yum",
                        "sudo rm -f /var/run/yum.pid"
                    ]
                    analysis += "\nYou can check for and kill any running yum processes, then remove the lock file:"
                
                return analysis, suggested_commands
                
        # 检查Git错误
        if base_command == "git" and return_code != 0:
            if "fatal: not a git repository" in output.lower():
                analysis = """
                ### Not a Git Repository
                
                The current directory is not a Git repository.
                """
                
                suggested_commands = [
                    "git init",
                    "git clone <repository_url>"
                ]
                
                analysis += "\nYou can initialize a new Git repository or clone an existing one:"
                
                return analysis, suggested_commands
            
            if "fatal: remote origin already exists" in output.lower():
                analysis = """
                ### Remote Already Exists
                
                The remote 'origin' already exists in this Git repository.
                """
                
                suggested_commands = [
                    "git remote -v",
                    "git remote remove origin",
                    "git remote add origin <repository_url>"
                ]
                
                analysis += "\nYou can view, remove, and add remotes:"
                
                return analysis, suggested_commands
        
        # 如果没有匹配到任何快速分析模式，返回None，让LLM进行完整分析
        return None
        
    def analyze_commands(self, commands: List[str]) -> List[Dict[str, Any]]:
        """分析命令列表，识别每个命令的类型和特性
        
        Args:
            commands: 要分析的命令列表
            
        Returns:
            List[Dict]: 每个命令的分析结果
        """
        results = []
        
        # 获取命令上下文管理器
        cmd_ctx = get_command_context()
        
        # 过滤掉非命令文本
        valid_commands = [cmd for cmd in commands if cmd_ctx.is_valid_command(cmd)]
        
        if not valid_commands:
            logging.warning("没有有效的命令需要分析")
            return []
        
        for command in valid_commands:
            # 识别命令类型
            cmd_type = self._identify_command_type(command)
            
            # 基本分析结果
            analysis = {
                "original_command": command,
                "type": cmd_type,
                "primary_function": "Primary function of the command",
                "resources": ["Required resources"],
                "side_effects": ["Potential side effects"]
            }
            
            # 根据命令类型进行特定分析
            if cmd_type == "file_edit":
                self._analyze_file_edit_command(command, analysis)
            elif cmd_type == "windows_cmd":
                self._analyze_windows_command(command, analysis)
            elif cmd_type == "package_manager":
                self._analyze_package_manager_command(command, analysis)
            elif cmd_type == "service_control":
                self._analyze_service_command(command, analysis)
            
            results.append(analysis)
        
        return results
    
    def is_valid_command(self, text: str) -> bool:
        """
        判断文本是否是有效的命令，而不是输出或文件路径
        
        Args:
            text: 要检查的文本
            
        Returns:
            bool: 是否是有效的命令
        """
        # 使用命令上下文管理器进行判断
        cmd_ctx = get_command_context()
        return cmd_ctx.is_valid_command(text)
    
    def _identify_command_type(self, command: str) -> str:
        """
        识别命令类型
        
        Args:
            command: 命令字符串
            
        Returns:
            命令类型
        """
        # 检查是否为文件编辑命令
        if self._is_file_edit_command(command):
            return "file_edit"
        
        # 检查是否为Windows CMD命令
        if command.strip().startswith("cmd") or "netsh" in command:
            return "windows_cmd"
        
        # 检查是否为包管理器命令
        if command.strip().startswith(("apt", "apt-get", "yum", "dnf", "pacman", "brew")):
            return "package_manager"
        
        # 检查是否为服务控制命令
        if command.strip().startswith(("systemctl", "service")):
            return "service_control"
        
        # 默认为普通命令
        return "normal"
    
    def _is_file_edit_command(self, command: str) -> bool:
        """
        判断是否为文件编辑命令
        
        Args:
            command: 命令字符串
            
        Returns:
            是否为文件编辑命令
        """
        # 编辑器命令
        editor_patterns = [
            r'(sudo\s+)?(nano|vim|vi|emacs|gedit|code)\s+\S+',
            r'echo\s+.+\s*>\s*\S+',
            r'cat\s+.+\s*>\s*\S+'
        ]
        
        return any(re.search(pattern, command) for pattern in editor_patterns)
    
    def _extract_file_edit_info(self, command: str) -> Tuple[Optional[str], Optional[str]]:
        """
        从文件编辑命令中提取文件路径和内容
        
        Args:
            command: 命令字符串
            
        Returns:
            (文件路径, 文件内容)元组，如果无法提取则返回(None, None)
        """
        # 提取编辑器命令中的文件路径
        editor_match = re.search(r'(sudo\s+)?(nano|vim|vi|emacs|gedit|code)\s+(\S+)', command)
        if editor_match:
            return editor_match.group(3), None
        
        # 提取重定向命令中的文件路径和内容
        redirect_match = re.search(r'echo\s+"?([^">]+)"?\s*>\s*(\S+)', command)
        if redirect_match:
            content = redirect_match.group(1)
            file_path = redirect_match.group(2)
            return file_path, content
        
        # 提取cat重定向命令中的文件路径
        cat_redirect_match = re.search(r'cat\s+.+\s*>\s*(\S+)', command)
        if cat_redirect_match:
            return cat_redirect_match.group(1), None
        
        return None, None
    
    def _needs_sudo(self, file_path: str) -> bool:
        """
        判断编辑文件是否需要sudo权限
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否需要sudo权限
        """
        # 系统目录通常需要sudo权限
        system_paths = [
            "/etc/",
            "/var/",
            "/usr/",
            "/boot/",
            "/root/"
        ]
        
        return any(file_path.startswith(path) for path in system_paths)
    
    def _parse_windows_cmd(self, command: str) -> str:
        """
        解析Windows CMD命令
        
        Args:
            command: 命令字符串
            
        Returns:
            解析后的命令
        """
        # 处理cmd命令
        cmd_match = re.match(r'cmd\s+(.+)', command)
        if cmd_match:
            cmd_content = cmd_match.group(1)
            # 如果是netsh命令，直接提取
            if "netsh" in cmd_content:
                netsh_match = re.search(r'(netsh\s+.+)', cmd_content)
                if netsh_match:
                    return netsh_match.group(1)
        
        return command
    
    def _analyze_file_edit_command(self, command: str, analysis: Dict[str, Any]):
        """
        分析文件编辑命令
        
        Args:
            command: 要分析的命令
            analysis: 分析结果字典，将被修改
        """
        file_path, content = self._extract_file_edit_info(command)
        if file_path:
            analysis["file_path"] = file_path
            analysis["content"] = content
            analysis["needs_sudo"] = self._needs_sudo(file_path)
    
    def _analyze_windows_command(self, command: str, analysis: Dict[str, Any]):
        """
        分析Windows命令
        
        Args:
            command: 要分析的命令
            analysis: 分析结果字典，将被修改
        """
        parsed_cmd = self._parse_windows_cmd(command)
        if parsed_cmd != command:
            analysis["parsed_command"] = parsed_cmd
    
    def _analyze_package_manager_command(self, command: str, analysis: Dict[str, Any]):
        """
        分析包管理器命令
        
        Args:
            command: 要分析的命令
            analysis: 分析结果字典，将被修改
        """
        # 提取包管理器类型
        if command.startswith("apt") or command.startswith("apt-get"):
            analysis["package_manager"] = "apt"
        elif command.startswith("yum"):
            analysis["package_manager"] = "yum"
        elif command.startswith("dnf"):
            analysis["package_manager"] = "dnf"
        elif command.startswith("pacman"):
            analysis["package_manager"] = "pacman"
        elif command.startswith("brew"):
            analysis["package_manager"] = "brew"
        
        # 提取操作类型（安装、卸载等）
        if "install" in command:
            analysis["operation"] = "install"
        elif "remove" in command or "uninstall" in command:
            analysis["operation"] = "remove"
        elif "update" in command:
            analysis["operation"] = "update"
        elif "upgrade" in command:
            analysis["operation"] = "upgrade"
        
        # 尝试提取包名
        parts = command.split()
        if len(parts) > 2 and parts[1] in ["install", "remove", "uninstall"]:
            analysis["packages"] = parts[2:]
    
    def _analyze_service_command(self, command: str, analysis: Dict[str, Any]):
        """
        分析服务控制命令
        
        Args:
            command: 要分析的命令
            analysis: 分析结果字典，将被修改
        """
        # 提取服务名和操作
        parts = command.split()
        if len(parts) >= 3:
            if parts[0] == "systemctl":
                analysis["service_manager"] = "systemd"
                analysis["operation"] = parts[1]  # start, stop, restart, etc.
                analysis["service_name"] = parts[2]
            elif parts[0] == "service":
                analysis["service_manager"] = "sysvinit"
                analysis["service_name"] = parts[1]
                analysis["operation"] = parts[2]  # start, stop, restart, etc.
    
    def display_analysis(self, analysis: str, suggested_commands: Optional[List[str]] = None) -> Optional[str]:
        """
        显示分析结果和建议命令
        
        Args:
            analysis: 分析结果文本
            suggested_commands: 可选的建议命令列表
            
        Returns:
            Optional[str]: 如果用户选择执行修复命令，返回选择的命令；否则返回None
        """
        # 显示分析结果
        console.print(Panel(analysis, title="[bold blue]分析结果[/bold blue]", expand=False))
        
        # 如果有建议的命令，显示并询问用户是否要执行
        if suggested_commands and len(suggested_commands) > 0:
            console.print("\n[bold green]建议的修复命令:[/bold green]")
            
            # 显示命令选项
            for i, cmd in enumerate(suggested_commands):
                console.print(f"  [bold cyan]{i+1}.[/bold cyan] {cmd}")
            
            # 询问用户是否要执行修复命令
            console.print("\n[bold yellow]您想执行哪个修复命令？[/bold yellow]")
            console.print("[dim](输入命令编号执行对应命令，输入'n'跳过，输入'q'退出)[/dim]")
            
            choice = input("> ")
            
            if choice.lower() == 'q':
                # 用户选择退出
                return "quit"
            elif choice.lower() == 'n':
                # 用户选择跳过
                return None
            else:
                try:
                    # 尝试将用户输入转换为整数
                    cmd_index = int(choice) - 1
                    if 0 <= cmd_index < len(suggested_commands):
                        # 返回用户选择的命令
                        return suggested_commands[cmd_index]
                    else:
                        console.print("[bold red]无效的选择[/bold red]")
                except ValueError:
                    console.print("[bold red]无效的输入[/bold red]")
            
        return None
    
    def clear_history(self):
        """清除分析历史"""
        self.analysis_history = []
