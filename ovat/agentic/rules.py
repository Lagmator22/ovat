# ovat/agentic/rules.py
"""Per-user, per-project permanent tool grants stored in ``~/.ovat/permissions.json``.

Keeping permanent grants in the user's home directory keyed by canonical project
path prevents untrusted repositories from shipping pre-approved ``.ovat/`` tool
grants:

    ~/.ovat/permissions.json
    {"format": 2, "projects": {"/path/to/project": ["describe_image"]}}

Only interactive sessions wire saved grants into ``ApprovalBroker``; headless
entry points (``ovat run``, benchmarks, A2A) ignore them. Stored with user-only
permissions (``0600`` on POSIX).
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
