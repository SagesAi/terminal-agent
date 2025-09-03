from typing import List
from terminal_agent.react.agent import shell_command_tool
from terminal_agent.react.tools.script_tool import script_tool
from terminal_agent.react.agent import message_tool
import json


SHELL_TOOL_DESCRIPTION = """
Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first verify that "foo" exists and is the intended parent directory

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command.
   - Capture the output of the command.

Usage notes:
  - The command argument is required.
  - If the output exceeds 3000 characters, output will be truncated before being returned to you.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings).
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.
    <good-example>
    pytest /foo/bar/tests
    </good-example>
    <bad-example>
    cd /foo/bar && pytest tests
    </bad-example>


### Committing changes with git

When the user asks you to create a new git commit, follow these steps carefully:

1. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following bash commands in parallel, each using the Bash tool:
  - Run a git status command to see all untracked files.
  - Run a git diff command to see both staged and unstaged changes that will be committed.
  - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.
2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
  - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. "add" means a wholly new feature, "update" means an enhancement to an existing feature, "fix" means a bug fix, etc.).
  - Check for any sensitive information that shouldn't be committed
  - Draft a concise (1-2 sentences) commit message that focuses on the "why" rather than the "what"
  - Ensure it accurately reflects the changes and their purpose
3. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following commands in parallel:
   - Add relevant untracked files to the staging area.
   - Create the commit with a message ending with:
   🤖 Generated with [Terminal Agent](https://github.com/sagesai/terminal-agent)
   - Run git status to make sure the commit succeeded.
4. If the commit fails due to pre-commit hook changes, retry the commit ONCE to include these automated changes. If it fails again, it usually means a pre-commit hook is preventing the commit. If the commit succeeds but you notice that files were modified by the pre-commit hook, you MUST amend your commit to include them.

Important notes:
- NEVER update the git config
- NEVER run additional commands to read or explore code, besides git bash commands
- DO NOT push to the remote repository unless the user explicitly asks you to do so
- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.
- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit
- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:
<example>
git commit -m "$(cat <<'EOF'
   Commit message here.

   🤖 Generated with [Terminal Agent](https://github.com/sagesai/terminal-agent)
   EOF
   )"
</example>

### Creating pull requests
Use the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases. If given a Github URL use the gh command to get the information needed.

IMPORTANT: When the user asks you to create a pull request, follow these steps carefully:

1. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following bash commands in parallel using the Bash tool, in order to understand the current state of the branch since it diverged from the main branch:
   - Run a git status command to see all untracked files
   - Run a git diff command to see both staged and unstaged changes that will be committed
   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote
   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)
2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request summary
3. You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. ALWAYS run the following commands in parallel:
   - Create new branch if needed
   - Push to remote with -u flag if needed
   - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.
<example>
gh pr create --title "the pr title" --body "$(cat <<'EOF'
#### Summary
<1-3 bullet points>

#### Test plan
[Checklist of TODOs for testing the pull request...]

🤖 Generated with [Terminal Agent](https://github.com/sagesai/terminal-agent)
EOF
)"
</example>

"""

SHELL_TOOL_SCHEMA = {
     "type": "function",
     "function": {
         "name": "shell",
         "description": SHELL_TOOL_DESCRIPTION,
         "parameters": {    
             "type": "object",
             "properties": {
                 "command": {
                     "type": "string",
                     "description": "The command to execute"
                 },
                 "background": {
                     "type": "boolean",
                     "description": "Whether to run the command in the background"
                 }
             },
             "required": ["command"],
             "additionalProperties": False
         }
     }
}


MESSAGE_TOOL_DESCRIPTION = """
Use this when you need to ask the user a question and wait for their response. This tool is essential when:
- You need clarification on ambiguous requirements
- You need to confirm before proceeding with important actions
- You need additional information to complete a task
- You want to offer options and request user preference
- You need to validate assumptions critical to task success
- You are uncertain about the user's instructions or the user's instructions are unclear

Important Notes:
- Use this tool ONLY when user input is essential to proceed. Always provide clear context in your questions, explaining why the information is needed and what impact different answers might have. Be specific about what information you need and provide options when applicable. This tool BLOCKS execution until the user responds, so use it judiciously.
- Keep questions concise and focused on a single issue
- For complex choices, present numbered or bulleted options
- When appropriate, suggest a default or recommended option
- Avoid open-ended questions when specific information is needed
- Validate critical user inputs by confirming understanding before proceeding with important actions
"""

MESSAGE_TOOL_SCHEMA = {
     "type": "function",
     "function": {
         "name": "message",
         "description": MESSAGE_TOOL_DESCRIPTION,
         "parameters": {    
             "type": "object",
             "properties": {
                 "question": {
                     "type": "string",
                     "description": "The question to ask the user"
                 }
             },
             "required": ["question"],
             "additionalProperties": False
         }
     }
}

# Tool function implementations (wrappers around existing tools)
def shell_function(command: str, background: bool = False) -> str:
    """Wrapper for shell command execution"""
    return shell_command_tool(json.dumps(command))


def script_function(action: str, filename: str, content: str = None, 
                   interpreter: str = "bash", args: List[str] = None, 
                   timeout: int = None) -> str:
    """Wrapper for script operations"""
    
    params = {
        "action": action,
        "filename": filename,
        "interpreter": interpreter,
        "args": args or []
    }
    if content:
        params["content"] = content
    if timeout:
        params["timeout"] = timeout
    return script_tool(json.dumps(params))




def message_function(question: str) -> str:
    """Wrapper for user messaging"""
   
    params = {"question": question}
    return message_tool(json.dumps(params))

