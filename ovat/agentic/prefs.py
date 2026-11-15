# ovat/agentic/prefs.py
"""Shared per-project UI preferences stored in ``.ovat/chat_prefs.json``.

Maintains independent keys with read-modify-write updates so separate UI views
preserve each other's settings:

    config       last selected workflow path
    model_path   last local model folder selected
    theme        selected Textual theme name
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
