# ovat/tui/__init__.py
"""The agent TUI (`ovat agent`): chat-first, live tools, subagents, A2A.

A front end for ovat/agentic/session.AgentSession and nothing more: every
rule (permissions, subagents, cancel, shell) lives in the session, and this
package only draws what the session reports. Needs the [tui] extra. Bare
`ovat` and `ovat tui` open it too.

This file imports no UI toolkit, so importing `ovat.tui` is cheap and works
without the [tui] extra; `runner()` is what needs it.
"""
from typing import Callable


def runner() -> Callable[..., None]:
    """Return the agent TUI runner.

    Raises ImportError when the [tui] extra is not installed.
    """
    from ovat.tui.agent_app import run as run_agent
    return run_agent


def run(workflow: str | None = None) -> None:
    """Run `ovat agent` on `workflow` (default: ./workflow.yml)."""
    runner()(workflow)
