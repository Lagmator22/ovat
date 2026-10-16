# ovat/agentic/rules.py
"""The tools a person said "always allow" to: `.ovat/permissions.json`.

Note to myself: a workflow decides what a tool may do (permission: allow, ask
or deny). "Always allow" in the approval panel is a person's decision about
ONE project, so it is kept in the project folder, next to the saved
conversations, and not in the workflow (which is shared and reviewed).

    {"format": 1, "always_allow": ["describe_image", "delegate"]}

Who reads it: only the ApprovalBroker of an interactive session
(ovat/agentic/session.py hands the list to it). `ovat run`, a bench worker
and an A2A server have no broker, so a file here never makes them send
anything out unasked: with no one to ask, `ask` stays no.

To take a tool back, delete its line, or delete the file. Owner-only (0600)
like the A2A token and the saved conversations. This file never imports
textual.
"""
from __future__ import annotations

import json
from pathlib import Path

from ovat.agentic import atomic

RULES_FILE = "permissions.json"
FORMAT = 1


def rules_path(cwd: str) -> Path:
    return Path(cwd) / ".ovat" / RULES_FILE


def load(cwd: str) -> list[str]:
    """The tools allowed for good in this project; [] when there are none or
    the file is odd (a hand-edited file must not stop the app)."""
    try:
        with open(rules_path(cwd), encoding="utf-8") as handle:
            data = json.load(handle)
        tools = data["always_allow"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [t for t in tools if isinstance(t, str)] if isinstance(
        tools, list) else []


def add(cwd: str, tool: str) -> None:
    """Remember `tool`. Raises OSError when the disk says no."""
    tools = load(cwd)
    if tool not in tools:
        _save(cwd, [*tools, tool])


def _save(cwd: str, tools: list[str]) -> None:
    atomic.write_text(rules_path(cwd), json.dumps(
        {"format": FORMAT, "always_allow": tools}, indent=2))
