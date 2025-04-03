#!/usr/bin/env python3
"""
System information utilities for Terminal Agent
"""

import os
import re
import platform
import subprocess
from typing import Dict


def get_system_info() -> Dict[str, str]:
    """
    Gather system information for better command contextualization
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "os": platform.system(),
        "distribution": "",
        "version": platform.version(),
        "architecture": platform.machine(),
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


def get_package_manager(system_info: Dict[str, str]) -> str:
    """
    Determine the appropriate package manager for the system
    
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
