# ovat/agentic/rules.py
"""The tools a person said "always allow" to, per project.

Note to myself: a workflow decides what a tool may do (permission: allow, ask
or deny). "Always allow" in the approval panel is a person's decision about
ONE project. It is kept in the person's OWN folder, keyed by the project's
full path, never in the project: a repository could otherwise ship a
.ovat/permissions.json granting shell and write_file, and whoever opened it
would have them run without being asked.

    ~/.ovat/permissions.json
    {"format": 2, "projects": {"C:\\\\Users\\\\me\\\\proj": ["describe_image"]}}

Who reads it: only the ApprovalBroker of an interactive session
(ovat/agentic/session.py hands the list to it). `ovat run`, a bench worker
and an A2A server have no broker, so a grant here never makes them send
anything out unasked: with no one to ask, `ask` stays no.

To take a tool back, delete it from the list, or delete the file.
Owner-only (0600) like the A2A token. This file never imports textual.
"""
from __future__ import annotations

import json
from pathlib import Path

from ovat import userdata
from ovat.agentic import atomic

FORMAT = 2


def store_path() -> Path:
    return userdata.folder() / "permissions.json"


def _projects() -> dict:
    """Every project's list; {} when there is none or the file is odd (a
    hand-edited file must not stop the app)."""
    try:
        with open(store_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        projects = data["projects"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return projects if isinstance(projects, dict) else {}


def load(cwd: str) -> list[str]:
    """The tools allowed for good in this project."""
    tools = _projects().get(userdata.project_key(cwd))
    return [t for t in tools if isinstance(t, str)] if isinstance(
        tools, list) else []


def add(cwd: str, tool: str) -> None:
    """Remember `tool` for this project. Raises OSError when the disk says
    no."""
    projects = _projects()
    tools = load(cwd)
    if tool in tools:
        return
    projects[userdata.project_key(cwd)] = [*tools, tool]
    atomic.write_text(store_path(), json.dumps(
        {"format": FORMAT, "projects": projects}, indent=2))
