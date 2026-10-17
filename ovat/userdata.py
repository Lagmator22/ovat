# ovat/userdata.py
"""The person's own OVAT folder, ~/.ovat, for what a project must not set.

Note to myself: some state belongs to the PERSON, not to the project they
opened: the "always allow" grants and the A2A server's generated token. A
project folder is whatever a repository shipped, so a file there could grant
a tool or hand its author a working token. Both live here instead, keyed by
the project's full path.

Top level because both ovat/agentic/ and ovat/a2a/ need it and neither may
import the other. Tests point folder() at a temporary one (conftest).
"""
import hashlib
import os
from pathlib import Path


def folder() -> Path:
    return Path.home() / ".ovat"


def project_key(cwd: str | os.PathLike = ".") -> str:
    """One string per project folder. normcase: on Windows C:\\Proj and
    c:\\proj are the same folder."""
    return os.path.normcase(str(Path(cwd).resolve()))


def project_id(cwd: str | os.PathLike = ".") -> str:
    """project_key, short and safe as a file name."""
    return hashlib.sha256(project_key(cwd).encode("utf-8")).hexdigest()[:16]
