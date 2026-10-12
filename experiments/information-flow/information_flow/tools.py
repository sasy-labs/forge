"""
Tools for the information flow example.

These allow the agent to read and list files and send emails.
The multi-agent path uses Langroid's sub-task mechanism + the
built-in ``RecipientTool`` for routing to the research-assistant
helper, so dependency edges between the helper's reads and the
main agent's later actions are captured by SASY's observability
graph (see ``main._build_research_assistant_task``).
"""

from langroid.agent.tool_message import ToolMessage
from information_flow.filesystem import read_file, SecurityLevel


class ReadFileTool(ToolMessage):
    """Tool to read a file from the simulated filesystem."""

    request: str = "read_file"
    purpose: str = """Read a file from the filesystem.
    Available directories:
    - /documents/public/ (public files)
    - /documents/internal/ (internal company files)
    - /documents/restricted/ (restricted access)
    - /documents/secret/ (top secret)"""

    path: str

    def handle(self) -> str:
        """Read the file and return its contents."""
        content, level = read_file(self.path)
        if content is None:
            return f"Error: File not found: {self.path}"

        return f"[File: {self.path}]\n\n{content}"


class SendEmailTool(ToolMessage):
    """Tool to send an email (simulated - just logs the email)."""

    request: str = "send_email"
    purpose: str = """Send an email to a recipient.
    The email will be checked against security policies before sending."""

    to: str
    subject: str
    body: str

    def handle(self) -> str:
        """Send the email (simulated)."""
        # In a real implementation, this would actually send email
        # The policy engine will intercept and check if this is allowed
        print(f"\n{'='*60}EMAIL SENT{'='*60}")
        print(f"To: {self.to}")
        print(f"Subject: {self.subject}")
        print(f"Body:\n{self.body}")
        print(f"{'='*60}\n")

        return f"Email sent successfully to {self.to}"


class ListFilesTool(ToolMessage):
    """Tool to list available files."""

    request: str = "list_files"
    purpose: str = "List available files in the filesystem up to a given security level."
    max_security_level: SecurityLevel

    def handle(self) -> str:
        """List all files."""
        from .filesystem import list_files

        files = list_files()
        result = "Available files:\n\n"
        for path, level in sorted(files):
            if level <= self.max_security_level:
                result += f"  {path}\n"

        return result
