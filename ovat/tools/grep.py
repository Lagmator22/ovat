# ovat/tools/grep.py
"""The grep tool: search project files by regular expression.

Uses ripgrep when available and falls back to a pure-Python directory walk
so searches work without external binaries across Windows, Linux, and macOS.
Applies smart case (all-lowercase patterns ignore case), directory/file
exclusions, match/byte caps, and formats output as relative `path:line:text`.
"""
import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from ovat.tools.project import Shown, fail, is_sensitive_path, resolve_inside

DEFAULT_MAX_MATCHES = 100
MAX_OUTPUT_CHARS = 64_000
SEARCH_TIMEOUT = 60
MAX_LINE_CHARS = 300             # avoid exhausting the output budget on minified lines
MAX_FILE_BYTES = 2_000_000       # skip very large files during the Python walk

# Default excluded directories and files, checked directories-first.
SKIP_DIRS = {".venv", "venv", ".env", "env", "node_modules", ".git", ".ovat",
             "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".nox",
             "htmlcov", "dist", "build", ".idea", ".vscode"}
SKIP_FILES = ("*.pyc", "*.pyo", "*.pyd", ".DS_Store", "Thumbs.db",
              ".coverage")
SKIP_SUFFIX = ".egg-info"


def _smart_case(pattern: str) -> bool:
    """True when the search should ignore case: no capital letter in it."""
    return pattern == pattern.lower()


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(SKIP_SUFFIX)


def _matches_glob(rel: str, glob: str | None) -> bool:
    if not glob:
        return True
    name = rel.rsplit("/", 1)[-1]
    return fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(name, glob)


def _walk(target: Path, root: Path, regex, glob, limit: int):
    """Yield `rel:line:text` for matches in the Python fallback."""
    files = [target] if target.is_file() else []
    if target.is_dir():
        for folder, dirs, names in os.walk(target):
            dirs[:] = sorted(d for d in dirs if not _skip_dir(d))
            files.extend(Path(folder) / n for n in sorted(names)
                         if not any(fnmatch.fnmatch(n, p) for p in SKIP_FILES))
    count = 0
    for file in files:
        # os.walk does not follow symlinked FOLDERS, but a symlinked FILE is
        # opened through its link: one pointing out of the project (say at
        # ~/.ssh/id_rsa) would be read past the fence. Symlinked files are
        # skipped, as ripgrep does, and the path checked is the exact one
        # opened (not resolve_inside, which also expands ~ in user input).
        if file.is_symlink():
            continue
        real = file.resolve()
        if real != root and not real.is_relative_to(root):
            continue
        rel = file.relative_to(root).as_posix() if file.is_relative_to(root) \
            else str(file)
        if not _matches_glob(rel, glob) or is_sensitive_path(rel,
                                                             for_read=True):
            continue
        try:
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            with open(real, "rb") as f:
                head = f.read(8192)
                if b"\0" in head:
                    continue
                f.seek(0)
                for number, raw in enumerate(f, start=1):
                    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if regex.search(text):
                        yield f"{rel}:{number}:{text[:MAX_LINE_CHARS]}"
                        count += 1
                        if count > limit:
                            return
        except OSError:
            continue


def _ripgrep(rg: str, pattern: str, target: Path, root: Path, glob, limit):
    # --null ends each file name with a NUL, so a ':' inside a folder name
    # cannot shift where the path stops.
    cmd = [rg, "--no-config", "--no-follow", "--line-number", "--no-heading",
           "--with-filename", "--null", "--no-binary", "--max-columns",
           str(MAX_LINE_CHARS)]
    if _smart_case(pattern):
        cmd.append("--ignore-case")
    for name in sorted(SKIP_DIRS):
        cmd += ["--glob", f"!{name}/"]
    for name in SKIP_FILES:
        cmd += ["--glob", f"!{name}"]
    if glob:
        cmd += ["--glob", glob]
    # cwd is the project and the target is relative to it, so ripgrep prints
    # project-relative paths by itself.
    cmd += ["-e", pattern, "--", os.path.relpath(target, root)]
    proc = subprocess.Popen(cmd, cwd=root, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = []
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            # Searching "." makes ripgrep print "./a.py" (".\\a.py" on Windows).
            name, _, rest = line.partition("\0")
            name = name[2:] if name[:2] in ("./", ".\\") else name
            # .env and friends are files, so the folder globs above miss
            # them; read_file would ask first, so grep leaves them out.
            if is_sensitive_path(name, for_read=True):
                continue
            lines.append(f"{name}:{rest}")
            if len(lines) > limit:
                break               # enough: do not read a whole monorepo
    finally:
        proc.kill()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()
    return lines, err


def grep_impl(pattern: str, root: Path, path: str = ".",
              glob: str | None = None, max_matches: int | None = None) -> str:
    if not (pattern or "").strip():
        return fail("grep", "the search pattern is empty.")
    target, why = resolve_inside(path or ".", root)
    if why:
        return fail("grep", why)
    if not target.exists():
        return fail("grep", f"path does not exist: {path}")
    limit = max_matches if max_matches and max_matches > 0 \
        else DEFAULT_MAX_MATCHES
    try:
        regex = re.compile(pattern,
                           re.IGNORECASE if _smart_case(pattern) else 0)
    except re.error as exc:
        return fail("grep", f"the pattern is not a valid regex: {exc}")

    rg = shutil.which("rg")
    try:
        if rg:
            lines, err = _ripgrep(rg, pattern, target, root, glob, limit)
            if not lines and err.strip():
                return fail("grep", f"ripgrep said: {err.strip()}")
        else:
            lines = list(_walk(target, root, regex, glob, limit))
    except (OSError, subprocess.SubprocessError) as exc:
        return fail("grep", f"could not search: {exc}")

    truncated = len(lines) > limit
    lines = lines[:limit]
    body = "\n".join(lines)
    if len(body) > MAX_OUTPUT_CHARS:
        body, truncated = body[:MAX_OUTPUT_CHARS], True
    parsed = []
    for line in body.splitlines():
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            parsed.append({"path": str(root / parts[0]),
                           "line": int(parts[1])})
    text = body or f"No matches for {pattern!r}."
    if truncated:
        text += (f"\n\n[results cut at {len(lines)} matches; narrow the "
                 f"pattern, path or glob]")
    return Shown(text, {
        "kind": "grep", "pattern": pattern, "matches": body,
        "match_count": len(lines), "was_truncated": truncated,
        "parsed_matches": parsed})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": "Search the user's project files for a regular "
                       "expression and return matching lines as "
                       "path:line:text. It skips .git, virtual environments "
                       "and node_modules. A pattern in all lowercase ignores "
                       "case.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string",
                            "description": "regular expression to look for"},
                "path": {"type": "string",
                         "description": "file or folder to search, relative "
                                        "to the project folder",
                         "default": "."},
                "glob": {"type": "string",
                         "description": "only search files whose name matches "
                                        "this pattern, for example *.py"},
                "max_matches": {"type": "integer",
                                "description": "maximum matches to return",
                                "default": DEFAULT_MAX_MATCHES},
            },
            "required": ["pattern"],
        },
    },
}
