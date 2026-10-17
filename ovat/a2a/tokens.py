# ovat/a2a/tokens.py
"""Bearer tokens for `ovat a2a serve`, and which caller each one belongs to.

A token is required even on 127.0.0.1. Any local process can reach a
loopback port, and so can a web page open in the user's browser (DNS
rebinding). The token is what stops both.

Two ways to configure callers:
- nothing in the YAML: one random token is generated into .ovat/a2a/token
  (owner-only permissions on POSIX), for a caller named "local";
- a2a.server.tokens: {planner: OVAT_A2A_PLANNER}: one caller per entry,
  each token read from the NAMED environment variable. One name per token
  is what lets the server keep each caller's tasks apart (spec 13.1) and
  lets the audit log say who asked.
"""
import os
import secrets
from pathlib import Path

DEFAULT_TOKEN_FILE = Path(".ovat") / "a2a" / "token"
LOCAL_CALLER = "local"


def load_or_create(path: Path = DEFAULT_TOKEN_FILE) -> str:
    """The generated token, made on first use. Never empty."""
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return rotate(path)


def rotate(path: Path = DEFAULT_TOKEN_FILE) -> str:
    """Replace the generated token. Old callers stop working at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    # Created owner-only from the start rather than chmod-ed afterwards, so
    # there is no moment when another user could read it. On Windows the
    # mode is ignored and the user profile's own permissions apply.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    return token


def callers(tokens: dict[str, str], token_file: Path = DEFAULT_TOKEN_FILE,
            environ=os.environ) -> dict[str, str]:
    """{token: caller name} for the server's auth check.

    Raises ValueError, naming the variable, when a configured one is unset
    or empty: a server that silently accepted no one would look like a
    network problem from the caller's side.
    """
    if not tokens:
        return {load_or_create(token_file): LOCAL_CALLER}
    table = {}
    for caller, env_name in tokens.items():
        value = environ.get(env_name, "").strip()
        if not value:
            raise ValueError(
                f"a2a.server.tokens.{caller} names the environment variable "
                f"{env_name}, which is not set. Set it to the token that "
                f"caller will send.")
        if value in table:
            raise ValueError(
                f"callers {table[value]} and {caller} have the same token, "
                f"so the server could not tell them apart. Give each its own.")
        table[value] = caller
    return table
