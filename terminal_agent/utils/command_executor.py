#!/usr/bin/env python3
"""
Command execution utilities for Terminal Agent
"""

import re
import subprocess
import sys
import signal
import threading
import time
import os
import queue
from typing import Tuple, Optional, Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeRemainingColumn
from rich.markdown import Markdown
import logging

# Initialize Rich console
console = Console()

# Initialize logger
logger = logging.getLogger(__name__)

# 全局变量，用于跟踪当前正在运行的进程和执行状态
current_process: Dict[str, Any] = {
    "process": None,
    "command": "",
    "running": False
}

# 全局标志，用于指示是否应该停止所有操作
stop_all_operations = False


def terminate_current_command() -> bool:
    """
    终止当前正在运行的命令
    
    Returns:
        bool: 是否成功终止命令
    """
    global current_process, stop_all_operations
    
    # 设置停止所有操作的标志
    stop_all_operations = True
    
    if not current_process["running"] or current_process["process"] is None:
        console.print("[yellow]没有正在运行的命令[/yellow]")
        return False
    
    try:
        # 获取进程组ID
        pgid = os.getpgid(current_process["process"].pid)
        
        # 向整个进程组发送SIGTERM信号
        os.killpg(pgid, signal.SIGTERM)
        
        # 等待进程结束
        process = current_process["process"]  # 保存一个引用，防止在循环中被设置为None
        if process is not None:
            for _ in range(5):  # 最多等待5秒
                if process.poll() is not None:
                    break
                time.sleep(1)
            
            # 如果进程仍在运行，发送SIGKILL信号
            if process.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
        
        console.print(f"[bold red]命令已终止: {current_process['command']}[/bold red]")
        console.print("[bold red]已停止所有操作[/bold red]")
        
        # 重置进程状态
        current_process["running"] = False
        current_process["process"] = None
        current_process["command"] = ""
        return True
    except Exception as e:
        console.print(f"[bold red]终止命令时出错: {str(e)}[/bold red]")
        
        # 即使出错，也重置进程状态，防止状态不一致
        current_process["running"] = False
        current_process["process"] = None
        current_process["command"] = ""
        return False


def reset_stop_flag():
    """重置停止所有操作的标志"""
    global stop_all_operations
    stop_all_operations = False


def set_stop_flag():
    """设置停止所有操作的标志"""
    global stop_all_operations
    stop_all_operations = True


def should_stop_operations() -> bool:
    """
    检查是否应该停止所有操作
    
    Returns:
        bool: 是否应该停止所有操作
    """
    return stop_all_operations


def execute_command(command: str, module_name: str = "Command", check_success: bool = False, 
                  need_confirmation: bool = True, auto_confirm: bool = False, 
                  show_output: bool = True, env: Optional[Dict[str, str]] = None,
                  timeout: Optional[int] = None) -> Tuple[int, str, bool]:
    """
    执行命令并返回结果
    
    Args:
        command: 要执行的命令
        module_name: 模块名称，用于显示错误信息
        check_success: 是否检查命令执行成功
        need_confirmation: 是否需要用户确认
        auto_confirm: 是否自动确认（用于自动化测试）
        show_output: 是否显示命令输出
        env: 环境变量字典，用于设置命令执行环境
        timeout: 命令执行超时时间（秒），如果为None则不设置超时
        
    Returns:
        Tuple[int, str, bool]: 返回代码，输出，是否用户主动取消
    """
    try:
        # 显示要执行的命令
        console.print(f"\n>>> Command to execute: {command}")
        console.print("(输入 'stop' 可以终止当前命令和所有后续操作)")
        
        # 如果需要确认，询问用户是否要执行
        if need_confirmation and not auto_confirm:
            try:
                user_choice = input("Execute this command? (y/n): ")
                
                # 如果用户选择不执行，返回错误代码
                if user_choice.lower() not in ["y", "yes"]:
                    # 设置停止标志，防止后续操作（包括LLM API调用）
                    set_stop_flag()
                    return 1, "Command execution skipped by user", True
            except (KeyboardInterrupt, EOFError) as e:
                # 处理用户中断（如Ctrl+C或Ctrl+D）
                console.print("\n[bold red]用户中断了命令确认[/bold red]")
                set_stop_flag()
                return 1, f"Command confirmation interrupted by user: {str(e)}", True
        
        # 执行命令并显示进度
        try:
            return_code, output, user_stopped = execute_command_single(
                command, 
                show_output, 
                need_confirmation=False, 
                auto_confirm=auto_confirm,
                timeout=timeout,
                env=env  # 传递环境变量
            )
        except Exception as e:
            # 捕获执行命令过程中的任何异常
            error_msg = f"Error executing command: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]{error_msg}[/bold red]")
            return 1, error_msg, False
        
        # 初始化建议的修复命令列表
        suggested_commands = []
        
        # 如果没有提供分析器，直接返回结果
        if not check_success:
            return return_code, output, user_stopped
        
        # 如果用户主动放弃命令，不进行分析，直接返回
        if user_stopped:
            return return_code, output, user_stopped
            
        # 如果命令执行失败，进行错误分析和修复
        if return_code != 0:
            console.print(f"[yellow]{module_name}执行未成功完成[/yellow]")
            
            try:
                # 分析命令输出，提供错误诊断
                success, error_message = analyze_command_success(command, output, return_code)
                
                # 如果没有提供错误消息，使用默认消息
                if error_message is None:
                    error_message = "命令执行失败，请查看输出了解详情"
                
                # 显示分析结果，帮助用户理解错误，并询问是否执行修复命令
                console.print("[bold yellow]进行问题定位...[/bold yellow]")
                selected_fix = display_analysis(error_message, suggested_commands, command, output, return_code, auto_mode=auto_confirm)
                
                # 处理用户选择
                if selected_fix == "quit":
                    console.print("[bold red]用户选择退出[/bold red]")
                    return return_code, output, True
                elif selected_fix:
                    console.print(f"[bold green]执行修复命令: {selected_fix}[/bold green]")
                    try:
                        fix_return_code, fix_output, fix_user_stopped = execute_command(selected_fix, module_name=module_name, need_confirmation=True)
                        
                        # 分析修复命令的执行结果
                        if fix_return_code == 0:
                            console.print("[bold green]修复命令执行成功[/bold green]")
                            
                            try:
                                # 询问是否重试原命令
                                retry = input("修复可能已解决问题，是否重试原命令? (y/n): ")
                                if retry.lower() in ["y", "yes"]:
                                    console.print(f"[bold cyan]重试命令: {command}[/bold cyan]")
                                    return_code, output, user_stopped = execute_command(command, module_name=module_name, need_confirmation=True)
                                    
                                    # 如果重试成功，更新分析结果
                                    if return_code == 0:
                                        console.print("[bold green]命令执行成功[/bold green]")
                                        success, error_message = analyze_command_success(command, output, return_code)
                                        display_analysis(error_message, suggested_commands, command, output, return_code, auto_mode=auto_confirm)
                            except (KeyboardInterrupt, EOFError) as e:
                                # 处理用户中断
                                console.print("\n[bold red]用户中断了重试确认[/bold red]")
                                return return_code, output, True
                        else:
                            console.print("[bold red]修复命令执行失败[/bold red]")
                    except Exception as e:
                        # 捕获执行修复命令过程中的任何异常
                        error_msg = f"Error executing fix command: {str(e)}"
                        logger.error(error_msg)
                        console.print(f"[bold red]{error_msg}[/bold red]")
            except Exception as e:
                # 捕获分析和修复过程中的任何异常
                error_msg = f"Error during analysis and fix: {str(e)}"
                logger.error(error_msg)
                console.print(f"[bold red]{error_msg}[/bold red]")
        else:
            try:
                # 命令成功执行，分析命令输出
                success, error_message = analyze_command_success(command, output, return_code)
                
                # 显示分析结果
                display_analysis(error_message, suggested_commands, command, output, return_code, auto_mode=auto_confirm)
            except Exception as e:
                # 捕获分析过程中的任何异常
                error_msg = f"Error during success analysis: {str(e)}"
                logger.error(error_msg)
                console.print(f"[bold red]{error_msg}[/bold red]")
        
        return return_code, output, user_stopped
    except Exception as e:
        # 捕获整个函数中的任何未处理异常
        error_msg = f"Unexpected error in execute_command: {str(e)}"
        logger.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        return 1, error_msg, False


def execute_command_single(command: str, show_output: bool, need_confirmation: bool = False, auto_confirm: bool = False, timeout: Optional[int] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, bool]:
    """
    执行命令并返回结果
    
    Args:
        command: 要执行的命令
        show_output: 是否显示命令输出
        need_confirmation: 是否需要用户确认（如果在execute_command中已经确认过，则不需要再次确认）
        auto_confirm: 是否自动确认（用于自动化测试）
        timeout: 命令执行超时时间（秒），如果为None则不设置超时
        env: 环境变量字典，用于设置命令执行环境
        
    Returns:
        Tuple[int, str, bool]: 返回代码，输出，是否用户主动取消
    """
    # 如果需要确认，显示命令并询问用户是否要执行
    if need_confirmation and not auto_confirm:
        console.print(f"\n>>> Command to execute: {command}")
        console.print("(输入 'stop' 可以终止当前命令和所有后续操作)")
        
        user_choice = input("Execute this command? (y/n): ")
        
        # 如果用户输入stop，设置停止标志
        if user_choice.lower() == "stop":
            set_stop_flag()
            return 1, "Command execution stopped by user", True
        
        # 如果用户选择不执行，返回错误代码
        if user_choice.lower() not in ["y", "yes"]:
            # 设置停止标志，防止后续操作（包括LLM API调用）
            set_stop_flag()
            return 1, "Command execution skipped by user", True
    
    # 创建进程
    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
        transient=False  # 改为非临时进度条，这样可以看到历史输出
    ) as progress:
        task = progress.add_task("[cyan]Executing command...", total=100)  # 设置一个具体的总量
        
        # 创建一个事件来通知线程停止
        stop_event = threading.Event()
        
        # 创建一个队列来存储命令输出
        output_queue = queue.Queue()
        
        # 创建一个线程来执行命令
        def execute_in_thread():
            try:
                # 执行命令并捕获输出
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # 行缓冲
                    universal_newlines=True,
                    preexec_fn=os.setsid,  # 在新的进程组中运行命令
                    env=env  # 传递环境变量
                )
                
                # 更新全局变量，跟踪当前进程
                global current_process
                current_process["process"] = process
                current_process["command"] = command
                current_process["running"] = True
                
                # 创建线程来读取stdout和stderr
                def read_output(pipe, prefix=""):
                    for line in iter(pipe.readline, ""):
                        if line:
                            output_queue.put(f"{prefix}{line}")
                
                # 创建并启动读取线程
                stdout_thread = threading.Thread(target=read_output, args=(process.stdout, ""))
                stderr_thread = threading.Thread(target=read_output, args=(process.stderr, ""))
                
                stdout_thread.daemon = True
                stderr_thread.daemon = True
                
                # 启动线程
                stdout_thread.start()
                stderr_thread.start()
                
                # 计算超时时间点
                timeout_time = time.time() + timeout if timeout else None
                
                # 等待进程完成或者被用户中断
                while process.poll() is None:
                    # 检查是否超时
                    if timeout_time and time.time() > timeout_time:
                        try:
                            # 终止整个进程组
                            pgid = os.getpgid(process.pid)
                            os.killpg(pgid, signal.SIGTERM)
                            
                            # 等待进程终止
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                # 如果超时，发送SIGKILL
                                os.killpg(pgid, signal.SIGKILL)
                            
                            output_queue.put("\nCommand execution timed out after {} seconds".format(timeout))
                            break
                        except Exception as e:
                            output_queue.put(f"\nError terminating process: {str(e)}")
                    
                    if stop_event.is_set():
                        # 用户中断了命令
                        try:
                            # 终止整个进程组
                            pgid = os.getpgid(process.pid)
                            os.killpg(pgid, signal.SIGTERM)
                            
                            # 等待进程终止
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                # 如果超时，发送SIGKILL
                                os.killpg(pgid, signal.SIGKILL)
                        except Exception as e:
                            output_queue.put(f"\nError terminating process: {str(e)}")
                        
                        output_queue.put("\nCommand execution interrupted by user")
                        break
                    
                    # 短暂休眠，避免CPU占用过高
                    time.sleep(0.1)
                
                # 等待读取线程完成
                stdout_thread.join(1)
                stderr_thread.join(1)
                
                # 获取进程返回代码
                return_code = process.returncode
                
                # 更新全局变量，表示进程已完成
                current_process["running"] = False
                current_process["process"] = None
                
                # 将返回代码放入队列，但使用特殊格式，以便后续处理时能够识别并提取
                output_queue.put(f"__RETURN_CODE__:{return_code}")
            except Exception as e:
                output_queue.put(f"\nError executing command: {str(e)}")
                return_code = 1  # 设置一个非零的返回代码表示错误
                output_queue.put(f"__RETURN_CODE__:{return_code}")
        
        # 创建并启动执行线程
        execution_thread = threading.Thread(target=execute_in_thread)
        execution_thread.daemon = True
        execution_thread.start()
        
        # 收集输出
        output_lines = []
        return_code_received = False
        return_code = 0  # 默认为成功
        
        # 用于进度条更新的计数器
        progress_counter = 0
        
        # 检查用户是否想要停止命令执行
        start_time = time.time()
        while execution_thread.is_alive():
            # 检查是否有新的输出
            output_received = False
            while not output_queue.empty():
                item = output_queue.get()
                output_received = True
                # 检查是否是返回代码
                if item.startswith("__RETURN_CODE__:"):
                    return_code = int(item.split(":")[1])
                    return_code_received = True
                else:
                    output_lines.append(item)
                    # 如果需要显示输出，则打印
                    if show_output:
                        console.print(item, end="")
            
            # 更新进度条 - 使用时间和输出作为进度指示
            elapsed_time = time.time() - start_time
            if output_received or elapsed_time > 0.5:  # 每0.5秒或有新输出时更新
                progress_counter = (progress_counter + 5) % 100  # 循环更新进度
                progress.update(task, completed=progress_counter)
                start_time = time.time()  # 重置时间计数器
            
            # 检查是否应该停止所有操作
            if should_stop_operations():
                stop_event.set()
                execution_thread.join(timeout=10)
                output_lines.append("\nCommand execution stopped by user")
                return 1, "".join(output_lines), True
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.05)
        
        # 命令执行完成，将进度条设为100%
        progress.update(task, completed=100)
        
        # 最后一次检查队列中的输出
        while not output_queue.empty():
            item = output_queue.get()
            # 检查是否是返回代码
            if item.startswith("__RETURN_CODE__:"):
                return_code = int(item.split(":")[1])
                return_code_received = True
            else:
                output_lines.append(item)
                # 如果需要显示输出，则打印
                if show_output:
                    console.print(item, end="")
        
        # 返回结果
        return return_code, "".join(output_lines), False


def analyze_command_success(command: str, output: str, return_code: int) -> Tuple[bool, Optional[str]]:
    """
    分析命令是否成功实现了其目标
    
    Args:
        command: 执行的命令
        output: 命令输出
        return_code: 返回代码
        
    Returns:
        Tuple[bool, Optional[str]]: (命令是否成功, 错误消息)
    """
    # 如果返回代码为0，通常表示命令成功
    if return_code == 0:
        # 检查常见的错误模式，即使返回代码为0
        if re.search(r'\b(error|exception|failure|failed)\b', output.lower()):
            # 有些命令即使出错也可能返回0
            # 但我们需要确保这是真正的错误，而不是输出中偶然包含这些词
            error_lines = [line for line in output.lower().split('\n') 
                          if re.search(r'\b(error|exception|failure|failed)\b', line)]
            
            # 如果有明确的错误行，则认为命令失败
            if error_lines and not any(ignore_pattern in ' '.join(error_lines).lower() 
                                      for ignore_pattern in ["no error", "no exception", "0 error"]):
                error_message = extract_error_message(command, output)
                if error_message:
                    return False, error_message
        
        # 如果没有检测到错误，则认为命令成功
        # 为不同类型的命令生成适当的成功消息
        if command.strip().startswith("df"):
            return True, "命令执行成功，显示了文件系统的磁盘使用情况"
        elif command.strip().startswith("ls") or command.strip().startswith("dir"):
            return True, "命令执行成功，列出了目录内容"
        elif command.strip().startswith("ps"):
            return True, "命令执行成功，显示了当前运行的进程"
        elif "grep" in command or "find" in command:
            return True, "命令执行成功，完成了搜索操作"
        else:
            return True, "命令执行成功"
    
    # 返回代码非0，通常表示命令失败
    # 尝试从输出中提取错误信息
    error_message = extract_error_message(command, output)
    return False, error_message


def extract_error_message(command: str, output: str) -> Optional[str]:
    """
    从命令输出中提取错误信息
    
    Args:
        command: 执行的命令
        output: 命令输出
        
    Returns:
        Optional[str]: 提取的错误信息，如果没有找到则返回None
    """
    # 分割命令以获取基本命令
    base_command = command.split()[0] if command else ""
    
    # 通用错误模式
    error_patterns = [
        r"(?:Error|ERROR)[^:]*:(.*?)(?:\n|$)",
        r"(?:Exception|EXCEPTION)[^:]*:(.*?)(?:\n|$)",
        r"(?:Failed|FAILED)[^:]*:(.*?)(?:\n|$)",
        r"(?:fatal|FATAL)[^:]*:(.*?)(?:\n|$)"
    ]
    
    # 尝试匹配通用错误模式
    for pattern in error_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 检查一些常见的错误情况
    if "permission denied" in output.lower():
        return "权限不足，可能需要使用sudo或检查文件权限"
    
    if "command not found" in output.lower():
        return "命令未找到，请检查命令是否正确或是否已安装"
    
    if "no such file or directory" in output.lower():
        return "文件或目录不存在，请检查路径是否正确"
    
    if "connection refused" in output.lower() or "network is unreachable" in output.lower():
        return "网络连接问题，请检查网络连接或目标服务是否可达"
    
    # 检查特定命令的常见错误模式（保持通用性的同时不丢失特定错误识别能力）
    
    # Docker常见错误
    if base_command == "docker":
        if "could not select device driver" in output:
            return "Docker无法选择设备驱动程序，可能需要安装nvidia-container-toolkit"
        if "cannot connect to the docker daemon" in output.lower():
            return "无法连接到Docker守护进程，请检查Docker服务是否运行"
        if "no such image" in output.lower():
            return "找不到指定的Docker镜像，请检查镜像名称或拉取镜像"
    
    # 包管理器常见错误
    if base_command in ["apt", "apt-get", "yum", "dnf"]:
        if "unable to locate package" in output.lower():
            return "找不到指定的软件包，尝试更新软件源或检查包名是否正确"
        if "could not resolve" in output.lower():
            return "无法解析软件源地址，请检查网络连接和软件源配置"
    
    # Git常见错误
    if base_command == "git":
        if "not a git repository" in output.lower():
            return "当前目录不是Git仓库，请检查路径或初始化仓库"
        if "remote origin already exists" in output.lower():
            return "远程仓库'origin'已存在，请先移除或使用其他名称"
    
    # 如果没有找到具体错误但返回代码非0，提供一个通用消息
    return "命令执行失败，请查看输出了解详情"


def display_analysis(analysis: str, suggested_commands: List[str], command: str = None, 
                    output: str = None, return_code: int = None, 
                    auto_mode: bool = False) -> Optional[str]:
    """
    显示分析结果
    
    Args:
        analysis: 分析结果
        suggested_commands: 建议的修复命令
        command: 执行的命令
        output: 命令输出
        return_code: 返回代码
        auto_mode: 是否在自动模式下运行（不请求用户输入）
        
    Returns:
        Optional[str]: 用户选择的修复命令，如果用户选择退出则返回"quit"
    """
    # 显示命令执行结果（如果提供）
    if command and output is not None and return_code is not None:
        console.print("\n[bold cyan]命令执行详情:[/bold cyan]")
        console.print(f"[bold]命令:[/bold] {command}")
        console.print(f"[bold]返回代码:[/bold] {return_code}")
        
        # 创建一个可折叠的面板来显示输出
        from rich.panel import Panel
        from rich.text import Text
        
        # 限制输出长度，避免过长
        max_output_length = 500
        if len(output) > max_output_length:
            truncated_output = output[:max_output_length] + "...\n[dim](输出已截断，显示前500个字符)[/dim]"
        else:
            truncated_output = output
            
        output_text = Text(truncated_output)
        console.print(Panel(output_text, title="命令输出", expand=False))
        
        # 提供查看完整输出的选项
        if len(output) > max_output_length and not auto_mode:
            show_full = input("查看完整输出? (y/n): ")
            if show_full.lower() in ["y", "yes"]:
                console.print(Panel(Text(output), title="完整命令输出", expand=False))
    
    # 显示分析结果
    console.print("\n[bold yellow]分析结果:[/bold yellow]")
    
    # 使用Markdown渲染分析结果
    if analysis is not None:
        console.print(Markdown(analysis))
    else:
        # 如果返回代码为0，显示成功消息
        if return_code == 0:
            console.print("[bold green]命令执行成功[/bold green]")
        else:
            console.print("[bold red]命令执行失败，但没有详细的错误信息[/bold red]")
    
    if suggested_commands and not auto_mode:
        console.print("\n[bold yellow]建议的修复命令:[/bold yellow]")
        for i, cmd in enumerate(suggested_commands):
            console.print(f"{i+1}. {cmd}")
        
        choice = input("请选择修复命令（输入命令序号或'quit'退出）：")
        if choice.lower() == "quit":
            console.print("[bold red]用户选择退出[/bold red]")
            return "quit"
        try:
            choice = int(choice)
            if 1 <= choice <= len(suggested_commands):
                return suggested_commands[choice-1]
        except ValueError:
            pass
    elif suggested_commands:
        console.print("\n[bold yellow]建议的修复命令:[/bold yellow]")
        for i, cmd in enumerate(suggested_commands):
            console.print(f"{i+1}. {cmd}")
    
    return None


def execute_command_with_fix(command: str, 
                           analyzer: Optional[Any] = None,
                           conversation_history: Optional[List[Dict]] = None,
                           user_goal: Optional[str] = None,
                           module_name: str = "Command") -> Tuple[int, str, bool]:
    """
    执行命令并在失败时提供修复选项
    
    Args:
        command: 要执行的命令
        analyzer: 命令分析器
        conversation_history: 对话历史
        user_goal: 用户目标
        module_name: 模块名称，用于显示错误信息
        
    Returns:
        Tuple[int, str, bool]: 返回代码，输出，是否继续执行
    """
    # 执行命令
    return_code, output, user_stopped = execute_command(command, module_name=module_name, need_confirmation=True, auto_confirm=False, show_output=True)
    
    # 如果没有提供分析器，直接返回结果
    if analyzer is None:
        return return_code, output, True
    
    # 如果用户主动放弃命令，不进行分析，直接返回
    if user_stopped:
        return return_code, output, False
    
    # 如果命令执行失败，进行错误分析和修复
    if return_code != 0:
        console.print(f"[yellow]{module_name}执行未成功完成[/yellow]")
        
        # 获取修复建议
        fix_command = get_fix_suggestion(
            command, output, return_code, 
            analyzer, conversation_history, user_goal
        )
        
        # 如果用户选择了退出，则返回
        if fix_command == "quit":
            return return_code, output, False
        
        # 如果有修复命令，执行它
        if fix_command:
            console.print(f"[bold green]执行修复命令: {fix_command}[/bold green]")
            fix_return_code, fix_output, fix_user_stopped = execute_command(
                fix_command, 
                module_name=f"{module_name}修复",
                need_confirmation=True,
                auto_confirm=False,
                show_output=True
            )
            
            # 如果修复命令成功，返回修复结果
            if fix_return_code == 0:
                console.print("[bold green]修复命令执行成功[/bold green]")
                
                # 询问是否重试原命令
                retry = input("修复可能已解决问题，是否重试原命令? (y/n): ")
                if retry.lower() in ["y", "yes"]:
                    console.print(f"[bold cyan]重试命令: {command}[/bold cyan]")
                    return_code, output, user_stopped = execute_command(command, module_name=module_name, need_confirmation=True)
                    
                    # 如果重试成功，更新分析结果
                    if return_code == 0:
                        console.print("[bold green]命令执行成功[/bold green]")
                        success, error_message = analyze_command_success(command, output, return_code)
                        display_analysis(error_message, [], command, output, return_code, auto_mode=False)
            else:
                console.print("[bold red]修复命令执行失败[/bold red]")
    else:
        # 命令成功执行，直接返回结果
        return return_code, output, True


def get_fix_suggestion(command: str, output: str, return_code: int, 
                      analyzer: Optional[Any] = None,
                      conversation_history: Optional[List[Dict]] = None,
                      user_goal: Optional[str] = None,
                      auto_mode: bool = False) -> Optional[str]:
    """
    获取修复建议
    
    Args:
        command: 执行的命令
        output: 命令输出
        return_code: 返回代码
        analyzer: 命令分析器
        conversation_history: 对话历史
        user_goal: 用户目标
        auto_mode: 是否自动模式
        
    Returns:
        Optional[str]: 修复命令或None
    """
    # 如果没有提供分析器，直接返回None
    if analyzer is None:
        return None
    
    # 分析命令输出，提供错误诊断
    analysis_result = analyzer.analyze_output(
        command, output, return_code, conversation_history, user_goal
    )
    
    # 确保分析结果是一个元组，包含分析文本和建议命令列表
    if isinstance(analysis_result, tuple) and len(analysis_result) == 2:
        analysis, suggested_commands = analysis_result
    else:
        # 如果返回值不符合预期，使用默认值
        analysis = str(analysis_result)
        suggested_commands = []
    
    # 显示分析结果
    if not auto_mode:
        from rich.panel import Panel
        from rich.text import Text
        
        # 限制输出长度，避免过长
        max_output_length = 500
        if len(output) > max_output_length:
            truncated_output = output[:max_output_length] + "...\n[dim](输出已截断，显示前500个字符)[/dim]"
        else:
            truncated_output = output
            
        output_text = Text(truncated_output)
        console.print(Panel(output_text, title="命令输出", expand=False))
        
        # 提供查看完整输出的选项
        if len(output) > max_output_length and not auto_mode:
            show_full = input("查看完整输出? (y/n): ")
            if show_full.lower() in ["y", "yes"]:
                console.print(Panel(Text(output), title="完整命令输出", expand=False))
    
    # 显示分析结果
    console.print("\n[bold yellow]分析结果:[/bold yellow]")
    
    # 使用Markdown渲染分析结果
    if analysis is not None:
        console.print(Markdown(analysis))
    else:
        # 如果返回代码为0，显示成功消息
        if return_code == 0:
            console.print("[bold green]命令执行成功[/bold green]")
        else:
            console.print("[bold red]命令执行失败，但没有详细的错误信息[/bold red]")
    
    if suggested_commands and not auto_mode:
        console.print("\n[bold yellow]建议的修复命令:[/bold yellow]")
        for i, cmd in enumerate(suggested_commands):
            console.print(f"{i+1}. {cmd}")
        
        choice = input("请选择修复命令（输入命令序号或'quit'退出）：")
        if choice.lower() == "quit":
            console.print("[bold red]用户选择退出[/bold red]")
            # 设置全局停止标志，确保后续操作不会继续
            set_stop_flag()
            return "quit"
        try:
            choice = int(choice)
            if 1 <= choice <= len(suggested_commands):
                return suggested_commands[choice-1]
        except ValueError:
            pass
    elif suggested_commands:
        console.print("\n[bold yellow]建议的修复命令:[/bold yellow]")
        for i, cmd in enumerate(suggested_commands):
            console.print(f"{i+1}. {cmd}")
    
    return None


def execute_commands_batch(commands: List[str], need_confirmation: bool = True, auto_confirm: bool = False) -> Tuple[List[Dict[str, Any]], bool]:
    """
    批量执行多个命令，并返回每个命令的执行结果
    
    Args:
        commands: 要执行的命令列表
        need_confirmation: 是否需要用户确认
        auto_confirm: 是否自动确认（用于自动化测试）
        
    Returns:
        Tuple[List[Dict[str, Any]], bool]: 命令执行结果列表，是否用户主动取消
    """
    results = []
    
    # 预处理命令列表，确保每个命令都是单独的
    processed_commands = []
    for cmd in commands:
        # 分割可能包含多个命令的字符串（按换行符分割）
        cmd_lines = [c.strip() for c in cmd.split('\n') if c.strip()]
        processed_commands.extend(cmd_lines)
    
    # 显示要执行的命令列表
    console.print("\n[bold cyan]即将执行以下命令:[/bold cyan]")
    for i, cmd in enumerate(processed_commands):
        console.print(f"{i+1}. {cmd}")
    
    # 如果需要确认，询问用户是否要执行
    if need_confirmation and not auto_confirm:
        user_choice = input("执行这些命令? (y/n): ")
        
        # 如果用户选择不执行，返回空结果
        if user_choice.lower() not in ["y", "yes"]:
            return [], True
    
    # 执行每个命令
    for i, cmd in enumerate(processed_commands):
        # 检查是否应该停止所有操作
        if should_stop_operations():
            console.print("[bold red]已停止执行剩余命令[/bold red]")
            return results, True
            
        console.print(f"\n[bold cyan]执行命令 ({i+1}/{len(processed_commands)}): {cmd}[/bold cyan]")
        
        # 执行命令
        return_code, output, user_stopped = execute_command_single(cmd, show_output=True, need_confirmation=False, auto_confirm=auto_confirm)
        
        # 收集结果
        results.append({
            "command": cmd,
            "output": output,
            "return_code": return_code
        })
        
        # 如果用户主动放弃命令，停止执行后续命令
        if user_stopped:
            return results, True
            
        # 如果命令执行失败且不是最后一个命令，询问用户是否继续执行
        if return_code != 0 and i < len(processed_commands) - 1 and not auto_confirm:
            continue_execution = input("命令执行失败，是否继续执行剩余命令? (y/n): ")
            if continue_execution.lower() not in ["y", "yes"]:
                console.print("[yellow]已停止执行剩余命令[/yellow]")
                return results, True
    
    return results, False
