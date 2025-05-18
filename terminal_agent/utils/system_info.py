#!/usr/bin/env python3
"""
System information utilities for Terminal Agent
"""

import os
import re
import platform
import subprocess
import logging
from typing import Dict

# Import command forwarder for remote execution
from terminal_agent.utils.command_forwarder import forwarder


def get_system_info() -> Dict[str, str]:
    """
    Gather system information for better command contextualization.
    When remote execution is enabled, gets information from the remote system.
    Otherwise, gets information from the local system.
    
    Returns:
        Dictionary containing system information
    """
    # Check if remote execution is enabled
    if hasattr(forwarder, 'remote_enabled') and forwarder.remote_enabled:
        return _get_remote_system_info()
    else:
        return _get_local_system_info()


def _get_local_system_info() -> Dict[str, str]:
    """
    Gather system information from the local system
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "os": platform.system(),
        "distribution": "",
        "version": platform.version(),
        "architecture": platform.machine(),
        "remote": False
    }
    
    # Get Linux distribution details if on Linux
    if info["os"] == "Linux":
        try:
            # Try to get distribution info from /etc/os-release
            with open("/etc/os-release", "r") as f:
                os_release = f.read()
                
            # Extract distribution name
            name_match = re.search(r'NAME="?([^"\n]+)"?', os_release)
            if name_match:
                info["distribution"] = name_match.group(1)
                
            # Extract version
            version_match = re.search(r'VERSION_ID="?([^"\n]+)"?', os_release)
            if version_match:
                info["version"] = version_match.group(1)
        except FileNotFoundError:
            # Fallback method
            try:
                result = subprocess.run(["lsb_release", "-a"], 
                                       capture_output=True, 
                                       text=True, 
                                       check=False)
                if result.returncode == 0:
                    distro_match = re.search(r'Distributor ID:\s+(.+)', result.stdout)
                    if distro_match:
                        info["distribution"] = distro_match.group(1)
                    
                    version_match = re.search(r'Release:\s+(.+)', result.stdout)
                    if version_match:
                        info["version"] = version_match.group(1)
            except FileNotFoundError:
                pass
    
    # For macOS
    elif info["os"] == "Darwin":
        info["distribution"] = "macOS"
        mac_version = subprocess.run(["sw_vers", "-productVersion"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=False)
        if mac_version.returncode == 0:
            info["version"] = mac_version.stdout.strip()
            
    # For Windows
    elif info["os"] == "Windows":
        info["distribution"] = "Windows"
        
    return info


def _get_remote_system_info() -> Dict[str, str]:
    """
    Gather system information from the remote system
    
    Returns:
        Dictionary containing system information
    """
    logger = logging.getLogger(__name__)
    info = {
        "os": "",
        "distribution": "",
        "version": "",
        "architecture": "",
        "remote": True
    }
    logger.debug(f"get remote info")
    try:
        # Get OS type
        exit_code, stdout, stderr = forwarder.forward_command("uname -s")
        if exit_code == 0:
            info["os"] = stdout.strip()
        
        # Get architecture
        exit_code, stdout, stderr = forwarder.forward_command("uname -m")
        if exit_code == 0:
            info["architecture"] = stdout.strip()
        
        # Handle different OS types
        if info["os"] == "Linux":
            # Try to get distribution info from /etc/os-release
            exit_code, stdout, stderr = forwarder.forward_command("cat /etc/os-release")
            if exit_code == 0:
                os_release = stdout
                
                # Extract distribution name
                name_match = re.search(r'NAME="?([^"\n]+)"?', os_release)
                if name_match:
                    info["distribution"] = name_match.group(1)
                    
                # Extract version
                version_match = re.search(r'VERSION_ID="?([^"\n]+)"?', os_release)
                if version_match:
                    info["version"] = version_match.group(1)
            else:
                # Fallback method
                exit_code, stdout, stderr = forwarder.forward_command("lsb_release -a")
                if exit_code == 0:
                    distro_match = re.search(r'Distributor ID:\s+(.+)', stdout)
                    if distro_match:
                        info["distribution"] = distro_match.group(1)
                    
                    version_match = re.search(r'Release:\s+(.+)', stdout)
                    if version_match:
                        info["version"] = version_match.group(1)
        
        # For macOS
        elif info["os"] == "Darwin":
            info["distribution"] = "macOS"
            exit_code, stdout, stderr = forwarder.forward_command("sw_vers -productVersion")
            if exit_code == 0:
                info["version"] = stdout.strip()
    
    except Exception as e:
        logger.error(f"Error getting remote system info: {str(e)}")
        # Fallback to basic info
        info["os"] = "Unknown"
        info["distribution"] = "Unknown"
        
    return info


def get_package_manager(system_info: Dict[str, str]) -> str:
    """
    Determine the appropriate package manager for the system.
    Works with both local and remote systems.
    
    Args:
        system_info: Dictionary containing system information
        
    Returns:
        String representing the detected package manager
    """
    # Check if we're dealing with a remote system
    is_remote = system_info.get("remote", False)
    
    if is_remote:
        return _get_remote_package_manager(system_info)
    else:
        return _get_local_package_manager(system_info)


def _get_local_package_manager(system_info: Dict[str, str]) -> str:
    """
    Determine the appropriate package manager for the local system
    
    Args:
        system_info: Dictionary containing system information
        
    Returns:
        String representing the detected package manager
    """
    if system_info["os"] == "Linux":
        # Debian/Ubuntu based
        if os.path.exists("/usr/bin/apt") or os.path.exists("/usr/bin/apt-get"):
            return "apt"
        # Red Hat/Fedora based
        elif os.path.exists("/usr/bin/dnf"):
            return "dnf"
        elif os.path.exists("/usr/bin/yum"):
            return "yum"
        # Arch based
        elif os.path.exists("/usr/bin/pacman"):
            return "pacman"
        # SUSE based
        elif os.path.exists("/usr/bin/zypper"):
            return "zypper"
        # Try snap as fallback
        elif os.path.exists("/usr/bin/snap"):
            return "snap"
    elif system_info["os"] == "Darwin":
        if os.path.exists("/usr/local/bin/brew") or os.path.exists("/opt/homebrew/bin/brew"):
            return "brew"
        elif os.path.exists("/usr/local/bin/port"):
            return "port"
    
    return "unknown"


def _get_remote_package_manager(system_info: Dict[str, str]) -> str:
    """
    Determine the appropriate package manager for the remote system
    
    Args:
        system_info: Dictionary containing system information
        
    Returns:
        String representing the detected package manager
    """
    logger = logging.getLogger(__name__)
    
    try:
        if system_info["os"] == "Linux":
            # Check for common package managers
            package_managers = [
                ("apt", "which apt || which apt-get"),
                ("dnf", "which dnf"),
                ("yum", "which yum"),
                ("pacman", "which pacman"),
                ("zypper", "which zypper"),
                ("snap", "which snap")
            ]
            
            for pm_name, pm_cmd in package_managers:
                exit_code, stdout, stderr = forwarder.forward_command(pm_cmd)
                if exit_code == 0 and stdout.strip():
                    return pm_name
                    
        elif system_info["os"] == "Darwin":
            # Check for Homebrew
            exit_code, stdout, stderr = forwarder.forward_command("which brew")
            if exit_code == 0 and stdout.strip():
                return "brew"
                
            # Check for MacPorts
            exit_code, stdout, stderr = forwarder.forward_command("which port")
            if exit_code == 0 and stdout.strip():
                return "port"
    
    except Exception as e:
        logger.error(f"Error detecting remote package manager: {str(e)}")
    
    return "unknown"
