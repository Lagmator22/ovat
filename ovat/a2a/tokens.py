# ovat/a2a/tokens.py
"""Bearer tokens for `ovat a2a serve`, and which caller each one belongs to.

A token is required even on 127.0.0.1. Any local process can reach a
loopback port, and so can a web page open in the user's browser (DNS
rebinding). The token is what stops both.

Two ways to configure callers:
- nothing in the YAML: one random token is generated for the project, for a
  caller named "local". It is kept in the person's own folder
  (~/.ovat/a2a/<project id>.token, owner-only on POSIX), never in the
  project: a repository could ship a token file there and its author would
  then hold a working token for whoever serves that project;
- a2a.server.tokens: {planner: OVAT_A2A_PLANNER}: one caller per entry,
  each token read from the NAMED environment variable. One name per token
  is what lets the server keep each caller's tasks apart (spec 13.1) and
  lets the audit log say who asked.
"""
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from ovat import userdata
from ovat.agent.permissions import is_loopback

LOCAL_CALLER = "local"


def token_file(cwd: str | os.PathLike = ".") -> Path:
    """Where the project in `cwd` keeps its generated token."""
    return userdata.folder() / "a2a" / f"{userdata.project_id(cwd)}.token"


def load_or_create(path: Path | None = None) -> str:
    """The generated token, made on first use. Never empty."""
    path = path or token_file()
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return rotate(path)


def rotate(path: Path | None = None) -> str:
    """Replace the generated token. Old callers stop working at once."""
    path = path or token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    # Created owner-only from the start rather than chmod-ed afterwards, so
    # there is no moment when another user could read it. On Windows the
    # mode is ignored and the user profile's own permissions apply.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token + "\n")
    return token


def callers(tokens: dict[str, str], path: Path | None = None,
            environ=os.environ) -> dict[str, str]:
    """{token: caller name} for the server's auth check.

    Raises ValueError, naming the variable, when a configured one is unset
    or empty: a server that silently accepted no one would look like a
    network problem from the caller's side.
    """
    if not tokens:
        return {load_or_create(path): LOCAL_CALLER}
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


def same_origin(url: str, configured: str) -> bool:
    """True when `url` goes where `configured` points: same scheme, port
    and host. localhost and 127.0.0.1 count as one host, because OVAT's own
    card names 127.0.0.1 while a workflow may say localhost."""
    a, b = urlsplit(url), urlsplit(configured)
    if a.scheme != b.scheme:
        return False
    try:
        if a.port != b.port:
            return False
    except ValueError:                  # a port that is not a number
        return False
    host_a, host_b = (a.hostname or "").lower(), (b.hostname or "").lower()
    return host_a == host_b or (is_loopback(url) and is_loopback(configured))


def peer_token(url: str, token_env: str | None,
               path: Path | None = None,
               environ=os.environ) -> str:
    """The bearer token to send to the agent at `url`, or "" for none.

    A named variable always decides. With none named, an agent on THIS
    machine gets the token `ovat a2a serve` generated for this project, so
    serving and calling from one project needs no setup. The file never
    goes to another host.
    """
    if token_env:
        return environ.get(token_env, "").strip()
    if not is_loopback(url):
        return ""
    try:
        return (path or token_file()).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
