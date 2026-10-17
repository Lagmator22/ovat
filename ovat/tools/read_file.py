# ovat/tools/read_file.py
"""The read_file tool: numbered lines of a text file inside the project.

Mirrors Mistral Vibe's read_file.py (vibe/core/tools/builtins), minus its
AGENTS.md discovery, sensitive-file patterns and IO port. Same contract:
1-indexed offset, a line limit, a byte cap, line numbers in front.
"""
from pathlib import Path

from ovat.tools.project import Shown, fail, resolve_inside, shown_path

DEFAULT_LINE_LIMIT = 2000        # Vibe's DEFAULT_LINE_LIMIT
MAX_BYTES = 50 * 1024            # Vibe's MAX_BYTES


def read_file_impl(file_path: str, root: Path, offset: int | None = None,
                   limit: int = DEFAULT_LINE_LIMIT) -> str:
    path, why = resolve_inside(file_path, root)
    if why:
        return fail("read_file", why)
    if not path.exists():
        return fail("read_file", f"file not found: {file_path}")
    if path.is_dir():
        return fail("read_file", f"{file_path} is a folder, not a file.")
    start = max(1, offset or 1)
    limit = limit if limit and limit > 0 else DEFAULT_LINE_LIMIT
    try:
        with open(path, "rb") as f:
            # A NUL byte early on is the usual tell for "not text".
            if b"\0" in f.read(8192):
                return fail("read_file", f"{file_path} looks like a binary "
                                         f"file, not text. I cannot read it.")
            f.seek(0)
            picked, size, total = [], 0, 0
            for raw in f:
                total += 1
                if total < start or len(picked) >= limit:
                    continue
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                size += len(line.encode("utf-8")) + 1
                if size > MAX_BYTES and picked:
                    # Over the byte cap: stop collecting, keep counting lines
                    # so the note below can say how many there are.
                    limit = len(picked)
                    continue
                picked.append(line)
    except PermissionError:
        return fail("read_file", f"permission denied: {file_path}")
    except OSError as exc:
        return fail("read_file", f"cannot read {file_path}: {exc}")

    last = start + len(picked) - 1
    if picked:
        body = "\n".join(f"{n:>9}→{line}"
                         for n, line in enumerate(picked, start=start))
    elif total == 0:
        body = "(the file exists but is empty)"
    else:
        body = (f"(the file has {total} lines, which is shorter than the "
                f"offset {start})")
    cut = bool(picked) and last < total
    text = body
    if cut:
        text += (f"\n\n[showing lines {start}-{last} of {total}; "
                 f"call read_file again with offset {last + 1} for more]")
    return Shown(text, {
        "kind": "read_file", "file_path": shown_path(path, root),
        "content": body, "num_lines": len(picked), "start_line": start,
        "requested_offset": offset, "requested_limit": limit,
        "total_lines": total, "was_truncated": cut})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file in the user's project and return its "
                       "lines with line numbers. Use it before editing a file. "
                       "For a big file, pass offset and limit to read it in "
                       "parts.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string",
                              "description": "path to the file, relative to "
                                             "the project folder or absolute"},
                "offset": {"type": "integer",
                           "description": "line number to start from "
                                          "(1 is the first line)"},
                "limit": {"type": "integer",
                          "description": "maximum number of lines to read",
                          "default": DEFAULT_LINE_LIMIT},
            },
            "required": ["file_path"],
        },
    },
}
