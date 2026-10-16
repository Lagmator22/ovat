# ovat/tui/__init__.py
"""The agent TUI (`ovat agent`): chat-first, live tools, subagents, A2A.

A front end for ovat/agentic/session.AgentSession and nothing more: every
rule (permissions, subagents, cancel, shell) lives in the session, and this
package only draws what the session reports. Needs the [tui] extra. The
older launcher (`ovat tui`) is untouched and still the default for bare
`ovat`.

Two apps draw it. The default (agent_app.py) is the copied Vibe app running
on OVAT's own app server (server/); `OVAT_AGENT_UI=classic` runs the first
one (app.py) instead, unchanged, for as long as both are kept.

This file imports no UI toolkit, so importing `ovat.tui` is cheap and works
without the [tui] extra; `runner()` is what needs it.

Design reference: Mistral Vibe's TUI (Apache-2.0), studied for its layout
and behaviour (banner in the transcript, a loading line above a borderless
input, folding tool summaries, an approval panel in place of the input).
The code here is OVAT's own.
"""
import os
from typing import Callable

CLASSIC = "classic"


def runner() -> Callable[[str | None], None]:
    """The function that runs `ovat agent`: the new app, or the classic one
    when OVAT_AGENT_UI=classic. Raises ImportError when the [tui] extra is
    not installed (the caller prints the hint)."""
    if os.environ.get("OVAT_AGENT_UI", "").strip().lower() == CLASSIC:
        from ovat.tui.app import run as run_classic
        return run_classic
    from ovat.tui.agent_app import run as run_agent
    return run_agent


def run(workflow: str | None = None) -> None:
    """Run `ovat agent` on `workflow` (default: ./workflow.yml)."""
    runner()(workflow)
