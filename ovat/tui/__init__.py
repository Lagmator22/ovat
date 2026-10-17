# ovat/tui/__init__.py
"""The agent TUI (`ovat agent`): chat-first, live tools, subagents, A2A.

A front end for ovat/agentic/session.AgentSession and nothing more: every
rule (permissions, subagents, cancel, shell) lives in the session, and this
package only draws what the session reports. Needs the [tui] extra. Bare
`ovat` and `ovat tui` open it too. The older launcher (ovat/cli/tui.py) runs
only with OVAT_LAUNCHER=classic, until it is removed.

Two apps draw it. The default (agent_app.py) is the copied terminal client
in vibe/ (see NOTICE) running on OVAT's own app server (server/), which
drives OVAT's agent; `OVAT_AGENT_UI=classic` runs the first one (app.py,
OVAT's own) instead, for as long as both are kept.

This file imports no UI toolkit, so importing `ovat.tui` is cheap and works
without the [tui] extra; `runner()` is what needs it.
"""
import os
from typing import Callable

CLASSIC = "classic"


def classic_requested() -> bool:
    """OVAT_AGENT_UI=classic: run the first app instead of the new one."""
    return os.environ.get("OVAT_AGENT_UI", "").strip().lower() == CLASSIC


def runner() -> Callable[..., None]:
    """The function that runs `ovat agent`: the new app, or the classic one
    when OVAT_AGENT_UI=classic. Raises ImportError when the [tui] extra is
    not installed (the caller prints the hint)."""
    if classic_requested():
        from ovat.tui.app import run as run_classic
        return run_classic
    from ovat.tui.agent_app import run as run_agent
    return run_agent


def run(workflow: str | None = None) -> None:
    """Run `ovat agent` on `workflow` (default: ./workflow.yml)."""
    runner()(workflow)
