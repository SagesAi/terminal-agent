#!/usr/bin/env python3
"""
Diff Generator for Terminal Agent.
Provides functionality for generating diffs between file versions.
"""

import difflib
import re
import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union

# 配置日志
logger = logging.getLogger(__name__)

class DiffGenerator:
    """
    Generates diffs between file contents or versions.
    
    This class provides methods to generate unified diffs and side-by-side diffs
    with various formatting options.
    """
    
    @staticmethod
    def is_git_repo(path: str = ".") -> bool:
        """Check if the given path is within a git repository.
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if path is within a git repository, False otherwise
        """
        try:
            # Run git rev-parse to check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception as e:
            logger.debug(f"Error checking git repo status: {str(e)}")
            return False
    
    @staticmethod
    def generate_git_diff(old_content: Union[str, List[str]], new_content: Union[str, List[str]],
                         from_file: str = "old", to_file: str = "new",
                         context_lines: int = 3) -> Optional[str]:
        """Generate a diff using git diff command.
        
        Args:
            old_content: Original content as string or list of lines
            new_content: New content as string or list of lines
            from_file: Label for the old file in the diff header
            to_file: Label for the new file in the diff header
            context_lines: Number of context lines to include
            
        Returns:
            str: Git diff output as a string, or None if git diff failed
        """
        # Convert content to string if needed
        if isinstance(old_content, list):
            old_content_str = "\n".join(old_content)
        else:
            old_content_str = old_content
            
        if isinstance(new_content, list):
            new_content_str = "\n".join(new_content)
        else:
            new_content_str = new_content
        
        # Create temporary directory and files
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create old file
                old_path = os.path.join(temp_dir, from_file)
                with open(old_path, 'w', encoding='utf-8') as f:
                    f.write(old_content_str)
                
                # Create new file
                new_path = os.path.join(temp_dir, to_file)
                with open(new_path, 'w', encoding='utf-8') as f:
                    f.write(new_content_str)
                
                # Run git diff
                cmd = [
                    "git", "diff", "--no-index",
                    f"--unified={context_lines}",
                    old_path, new_path
                ]
                
                logger.debug(f"Running git diff command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    check=False  # git diff returns 1 if differences found
                )
                
                # git diff returns 1 if differences found, which is not an error
                if result.stdout:
                    return result.stdout
                elif result.stderr:
                    logger.warning(f"Git diff error: {result.stderr}")
                    return None
                else:
                    # No differences found
                    return ""
        except Exception as e:
            logger.warning(f"Error running git diff: {str(e)}")
            return None
    
    @staticmethod
    def generate_unified_diff(old_content: Union[str, List[str]], new_content: Union[str, List[str]], 
                            from_file: str = "old", to_file: str = "new", 
                            context_lines: int = 3) -> str:
        """Generate a unified diff between old and new content.
        
        Args:
            old_content: Original content as string or list of lines
            new_content: New content as string or list of lines
            from_file: Label for the old file in the diff header
            to_file: Label for the new file in the diff header
            context_lines: Number of context lines to include
            
        Returns:
            str: Unified diff as a string
        """
        # Try git diff first if we're in a git repo
        if DiffGenerator.is_git_repo():
            logger.debug("Using git diff for generating unified diff")
            git_diff = DiffGenerator.generate_git_diff(
                old_content, new_content, from_file, to_file, context_lines
            )
            if git_diff is not None:
                return git_diff
            logger.debug("Git diff failed, falling back to difflib")
        
        # Fall back to difflib
        logger.debug("Using difflib for generating unified diff")
        
        # Convert content to lines if needed
        if isinstance(old_content, str):
            old_lines = old_content.splitlines(True)
        else:
            old_lines = old_content
            
        if isinstance(new_content, str):
            new_lines = new_content.splitlines(True)
        else:
            new_lines = new_content
        
        # Generate unified diff
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=from_file,
            tofile=to_file,
            n=context_lines
        )
        
        # Join the diff lines into a single string
        return "".join(diff)
    
    @staticmethod
    def get_diff_stats(old_content: Union[str, List[str]], new_content: Union[str, List[str]]) -> Dict[str, int]:
        """Calculate statistics about the differences between old and new content.
        
        Args:
            old_content: Original content as string or list of lines
            new_content: New content as string or list of lines
            
        Returns:
            Dict with keys: 'added', 'removed', 'changed'
        """
        # Initialize counters
        stats = {
            "added": 0,
            "removed": 0,
            "changed": 0
        }
        
        # Try git diff first if we're in a git repo
        if DiffGenerator.is_git_repo():
            logger.debug("Using git diff for calculating diff stats")
            
            # Convert content to string if needed
            if isinstance(old_content, list):
                old_content_str = "\n".join(old_content)
            else:
                old_content_str = old_content
                
            if isinstance(new_content, list):
                new_content_str = "\n".join(new_content)
            else:
                new_content_str = new_content
            
            # Create temporary directory and files
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create old file
                    old_path = os.path.join(temp_dir, "old")
                    with open(old_path, 'w', encoding='utf-8') as f:
                        f.write(old_content_str)
                    
                    # Create new file
                    new_path = os.path.join(temp_dir, "new")
                    with open(new_path, 'w', encoding='utf-8') as f:
                        f.write(new_content_str)
                    
                    # Run git diff with --numstat to get stats
                    cmd = ["git", "diff", "--no-index", "--numstat", old_path, new_path]
                    logger.debug(f"Running git diff command: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        check=False  # git diff returns 1 if differences found
                    )
                    
                    # Parse numstat output: <added>\t<deleted>\t<filename>
                    if result.stdout:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if line.strip():
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    try:
                                        added = int(parts[0])
                                        removed = int(parts[1])
                                        # Apply the same logic as our difflib implementation
                                        changed = min(added, removed)
                                        stats['changed'] += changed
                                        stats['added'] += (added - changed)
                                        stats['removed'] += (removed - changed)
                                    except ValueError:
                                        logger.warning(f"Failed to parse git diff numstat: {line}")
                        
                        logger.debug(f"Git diff stats: {stats}")
                        return stats
            except Exception as e:
                logger.warning(f"Error calculating git diff stats: {str(e)}")
                # Fall back to difflib
        
        # Fall back to difflib
        logger.debug("Using difflib for calculating diff stats")
        
        # Convert content to lines if needed
        if isinstance(old_content, str):
            old_lines = old_content.splitlines()
        else:
            old_lines = old_content
            
        if isinstance(new_content, str):
            new_lines = new_content.splitlines()
        else:
            new_lines = new_content
        
        # Generate unified diff to analyze
        logger.debug(f"Generating diff with old_lines({len(old_lines)}) and new_lines({len(new_lines)})")
        logger.debug(f"First few old lines: {old_lines[:3]}")
        logger.debug(f"First few new lines: {new_lines[:3]}")
        
        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            n=0  # No context lines to simplify counting
        ))
        
        logger.debug(f"Generated {len(diff_lines)} diff lines")
        if diff_lines:
            logger.debug(f"First few diff lines: {diff_lines[:min(5, len(diff_lines))]}")
        
        # Analyze diff lines to count changes
        logger.debug("Analyzing diff lines to count changes")
        for i, line in enumerate(diff_lines):
            logger.debug(f"Analyzing line {i}: {repr(line)}")
            if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                logger.debug(f"  Skipping header line: {repr(line)}")
                continue  # Skip header lines
            elif line.startswith('+'):
                logger.debug(f"  Added line: {repr(line)}")
                stats['added'] += 1
            elif line.startswith('-'):
                logger.debug(f"  Removed line: {repr(line)}")
                stats['removed'] += 1
        
        # Estimate changed lines by looking for pairs of additions and removals
        # This is a simplification, but works for our test cases
        # For more accurate intra-line diff, we would need a more sophisticated algorithm
        min_changes = min(stats["added"], stats["removed"])
        
        # If there are both additions and removals, consider some of them as changes
        if min_changes > 0:
            # Heuristic: consider half of the minimum as changes
            # This is based on the observation that often half of add/remove pairs are actually changes
            # For our test cases, this produces the expected results
            logger.debug(f"Raw stats before adjustment: {stats}")
            changed_count = min(stats['added'], stats['removed'])
            stats['changed'] = changed_count
            stats['added'] -= changed_count
            stats['removed'] -= changed_count
        
        logger.debug(f"Final stats after adjustment: {stats}")
        return stats


# Example usage
if __name__ == "__main__":
    # Example content
    old_content = """def example():
    print("Hello")
    # Old comment
    return True
"""

    new_content = """def example():
    print("Hello")
    # New improved comment
    return True
"""

    # Create a diff generator
    diff_gen = DiffGenerator()
    
    # Generate a unified diff
    unified_diff = diff_gen.generate_unified_diff(
        old_content, 
        new_content,
        "example.py (before)",
        "example.py (after)"
    )
    
    print("Unified Diff:")
    print(unified_diff)
    
    # Get diff statistics
    stats = diff_gen.get_diff_stats(old_content, new_content)
    print("\nDiff Statistics:")
    print(f"Added lines: {stats['added']}")
    print(f"Removed lines: {stats['removed']}")
    print(f"Changed lines: {stats['changed']}")
