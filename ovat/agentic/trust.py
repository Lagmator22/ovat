# ovat/agentic/trust.py
"""Per-user workspace trust decisions stored in ``~/.ovat/trusted.json``.

Kept in the user's home directory rather than inside a project's ``.ovat/``
folder so untrusted repositories cannot mark themselves as trusted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from ovat import userdata
from ovat.agentic import atomic

TRUST_FILE = "trusted.json"


def trust_path() -> Path:
    """Return the path to the user-owned workspace trust registry."""
    return userdata.folder() / TRUST_FILE


def _load() -> dict[str, list[str]]:
    try:
        data = json.loads(trust_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"trusted": [], "untrusted": []}
    if not isinstance(data, dict):
        return {"trusted": [], "untrusted": []}
    trusted = [str(p) for p in data.get("trusted", []) if isinstance(p, str)]
    untrusted = [str(p) for p in data.get("untrusted", []) if isinstance(p, str)]
    return {"trusted": trusted, "untrusted": untrusted}


def repo_root_for(cwd: str | os.PathLike) -> Path | None:
    """Return the enclosing git repository root for ``cwd``, or ``None``."""
    root = Path(cwd).resolve()
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def detected_config_files(cwd: str | os.PathLike) -> list[str]:
    """List local OVAT configuration entries present inside ``cwd``."""
    root = Path(cwd).resolve()
    found: list[str] = []
    if (root / ".ovat").exists():
        found.append(".ovat")
    if (root / "workflow.yml").is_file():
        found.append("workflow.yml")
    return found


def declined(cwd: str | os.PathLike) -> bool:
    """True when the user explicitly declined to trust ``cwd``."""
    return userdata.project_key(cwd) in _load()["untrusted"]


def status(cwd: str | os.PathLike) -> Literal["trusted", "untrusted"]:
    """Return whether ``cwd`` (or its non-declined git repo) is trusted."""
    data = _load()
    key = userdata.project_key(cwd)
    if key in data["untrusted"]:
        return "untrusted"
    if key in data["trusted"]:
        return "trusted"
    repo = repo_root_for(cwd)
    if repo is not None:
        repo_key = userdata.project_key(repo)
        if repo_key in data["trusted"] and repo_key not in data["untrusted"]:
            return "trusted"
    return "untrusted"


def decide(cwd: str | os.PathLike, decision: str) -> Literal["trusted", "untrusted"]:
    """Record a trust decision (``trust_cwd``, ``trust_repo``, or ``decline``)."""
    data = _load()
    trusted = [p for p in data["trusted"]]
    untrusted = [p for p in data["untrusted"]]
    key = userdata.project_key(cwd)
    repo = repo_root_for(cwd)
    repo_key = userdata.project_key(repo) if repo is not None else key

    if decision == "trust_cwd":
        untrusted = [p for p in untrusted if p != key]
        if key not in trusted:
            trusted.append(key)
    elif decision == "trust_repo":
        untrusted = [p for p in untrusted if p not in (repo_key, key)]
        if repo_key not in trusted:
            trusted.append(repo_key)
    elif decision == "decline":
        trusted = [p for p in trusted if p != key]
        if key not in untrusted:
            untrusted.append(key)

    atomic.write_json(trust_path(), {"trusted": trusted, "untrusted": untrusted})
    return status(cwd)


def details(cwd: str | os.PathLike) -> dict:
    """Build ``WorkspaceTrustDetails`` field values for ``cwd``."""
    root = Path(cwd).resolve()
    repo = repo_root_for(root)
    repo_key = userdata.project_key(repo) if repo is not None else None
    data = _load()
    decisions = ["trust_cwd", "decline"]
    if repo is not None and repo != root:
        decisions.insert(0, "trust_repo")
    return {
        "cwd": str(root),
        "repo_root": str(repo) if repo is not None else None,
        "detected_files": detected_config_files(root),
        "repo_detected_files": detected_config_files(repo) if repo is not None else [],
        "repo_explicitly_untrusted": bool(repo_key and repo_key in data["untrusted"]),
        "settings_path": str(trust_path()),
        "available_decisions": decisions,
    }


def not_loaded_reason(workflow: str) -> str:
    """Why `workflow` was not loaded on its own; callers add what to do."""
    return (f"{os.path.basename(workflow)} in this folder was not loaded: the "
            f"folder is not trusted, and loading it would start the tools it "
            f"lists.")


def untrusted_config_dirs(cwd: str | os.PathLike) -> tuple[list[str], str]:
    """Return ``(dirs, settings_path)`` for untrusted folders containing OVAT config."""
    root = Path(cwd).resolve()
    path_str = str(trust_path())
    if status(root) != "trusted" and detected_config_files(root):
        return [str(root)], path_str
    return [], path_str
