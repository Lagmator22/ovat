# ovat/tools/project.py
"""What the five coding tools share: the project folder fence and `Shown`.

Note to myself: read_file, write_file, edit, grep and shell let the model
touch the user's own files, so two things have to be the same everywhere.

1. The fence. The project folder is the working directory when the agent was
   built. A path is resolved (symlinks included) BEFORE it is compared, so a
   link inside the project that points outside is refused, and so is "../x".
2. What a tool hands back. The model reads a plain string, but the TUI wants
   structure (a diff, an exit code). `Shown` is a str that also carries a
   `display` dict. Every engine just sees a str; only events.observed() looks
   for `.display`.

Stdlib only, like events.py: the CLI imports this and must stay light.
"""
from pathlib import Path


# The five coding tools. exposure.py never offers them to a remote caller;
# permissions.py makes the three that change things ask by default.
CODING_TOOLS = frozenset({"read_file", "write_file", "edit", "grep", "shell"})
# The web tools send the model's words (a URL, a query) off this machine.
WEB_TOOLS = frozenset({"web_fetch", "web_search"})
ASK_BY_DEFAULT = frozenset({"write_file", "edit", "shell"}) | WEB_TOOLS
# Never offered to a remote A2A caller, with the reason in one phrase.
NEVER_REMOTE = {
    **{name: "it works on this machine's files and shell"
       for name in CODING_TOOLS},
    **{name: "it would send requests out from this machine on a "
             "stranger's say-so" for name in WEB_TOOLS},
    "ask_user": "it would put questions in front of this machine's user "
                "on a remote caller's behalf",
}


class Shown(str):
    """The text the model reads, plus a `display` dict for the TUI."""

    display: dict

    def __new__(cls, text: str, display: dict | None = None):
        obj = super().__new__(cls, text)
        obj.display = display or {}
        return obj


def fail(tool: str, message: str) -> Shown:
    """A failure the model can act on. The TUI shows it as a failed call."""
    text = f"Error: {message}"
    return Shown(text, {"kind": tool, "failed": True, "error": text})


def project_root() -> Path:
    """The folder the agent works in: the working directory, right now."""
    return Path.cwd().resolve()


def resolve_inside(raw: str, root: Path) -> tuple[Path | None, str | None]:
    """(resolved path, None) or (None, the sentence that says why not).

    Relative paths are taken from the project folder, absolute ones as they
    are. resolve() follows symlinks, and does not need the file to exist yet
    (write_file makes new ones).
    """
    raw = (raw or "").strip()
    if not raw:
        return None, "the path is empty."
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"cannot resolve {raw}: {exc}"
    if resolved != root and not resolved.is_relative_to(root):
        return None, (f"{raw} is outside the project folder ({root}). "
                      f"I only work on files inside it.")
    return resolved, None


def shown_path(path: Path, root: Path) -> str:
    """The path as the person knows it: relative to the project if inside."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
