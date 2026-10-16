# ovat/agentic/prefs.py
"""The one UI preferences file: `.ovat/chat_prefs.json` in the project.

Note to myself: the launcher's /chat already keeps the last workflow and model
folder here, so the agent app's theme goes in the SAME file, never a second
one. Each writer changes only its own keys and leaves the others as they are
(`/chat` must not forget the theme, `/theme` must not forget the model path).

    config       the workflow the launcher's /chat used last
    model_path   the model folder it used last
    theme        the Textual theme chosen with /theme in the agent app

This file never imports textual.
"""
from __future__ import annotations

import json
from pathlib import Path

from ovat.agentic import atomic

PREFS_FILE = "chat_prefs.json"


def prefs_path(cwd: str) -> Path:
    return Path(cwd) / ".ovat" / PREFS_FILE


def load(cwd: str) -> dict:
    """The saved preferences; {} when there are none or the file is odd."""
    try:
        with open(prefs_path(cwd), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def update(cwd: str, **changes) -> None:
    """Set these keys and keep every other one; a value of None removes its
    key. Raises OSError when the disk says no: the caller decides how to tell
    the person."""
    merged = {**load(cwd), **changes}
    atomic.write_text(prefs_path(cwd), json.dumps(
        {k: v for k, v in merged.items() if v is not None}, indent=2))
