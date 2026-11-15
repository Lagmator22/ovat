# ovat/userdata.py
"""Per-user state directory (``~/.ovat``) separated from project repositories.

Stores user-owned trust/auth state such as permanent tool grants and generated
A2A bearer tokens, keyed by project path so untrusted repositories cannot
pre-populate sensitive state under ``.ovat/``. Tests redirect ``folder()``.
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
