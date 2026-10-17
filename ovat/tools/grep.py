# ovat/tools/grep.py
"""The grep tool: find a regex in the project's files.

Mirrors Mistral Vibe's grep.py (vibe/core/tools/builtins): ripgrep when it is
installed, the same exclude list, smart case (all lowercase pattern ignores
case), a match cap and a byte cap. Vibe falls back to GNU grep; OVAT falls
back to a plain Python walk instead, because Windows has neither and a tool
that needs a second program installed is a tool that fails on the demo PC.
Output lines are `path:line:text`, relative to the project folder.
"""
import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from ovat.tools.project import Shown, fail, resolve_inside

DEFAULT_MAX_MATCHES = 100        # Vibe's default_max_matches
MAX_OUTPUT_CHARS = 64_000        # Vibe's max_output_bytes
SEARCH_TIMEOUT = 60              # Vibe's default_timeout
MAX_LINE_CHARS = 300             # one minified line must not eat the cap
MAX_FILE_BYTES = 2_000_000       # the Python walk skips bigger files

# Vibe's exclude_patterns, folders first.
SKIP_DIRS = {".venv", "venv", ".env", "env", "node_modules", ".git",
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
        if not _matches_glob(rel, glob):
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
    cmd = [rg, "--line-number", "--no-heading", "--with-filename",
           "--no-binary", "--max-columns", str(MAX_LINE_CHARS)]
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
            lines.append(line[2:] if line[:2] in ("./", ".\\") else line)
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
