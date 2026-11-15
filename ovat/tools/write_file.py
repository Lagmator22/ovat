# ovat/tools/write_file.py
"""The write_file tool: create or overwrite a file inside the project sandbox.

Enforces a 64,000-byte size limit and creates missing parent directories.
Overwrites are permitted (gated by the ``ask`` default permission), and the TUI
displays an inline diff against previous file contents when updating an
existing file.
"""
from pathlib import Path

from ovat.tools.project import Shown, fail, resolve_inside, save_text, shown_path

MAX_WRITE_BYTES = 64_000


def write_file_impl(file_path: str, content: str, root: Path) -> str:
    path, why = resolve_inside(file_path, root)
    if why:
        return fail("write_file", why)
    content = content if isinstance(content, str) else str(content)
    size = len(content.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        return fail("write_file", f"the content is {size} bytes; the limit "
                                  f"is {MAX_WRITE_BYTES}. Write it in "
                                  f"smaller files.")
    if path.is_dir():
        return fail("write_file", f"{file_path} is a folder, not a file.")
    existed, previous = path.exists(), None
    try:
        if existed:
            try:
                previous = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                previous = None           # binary or unreadable: no diff
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" keeps the text byte for byte: no CRLF surprise on Windows.
        save_text(path, content)
    except PermissionError:
        return fail("write_file", f"permission denied: {file_path}")
    except OSError as exc:
        return fail("write_file", f"cannot write {file_path}: {exc}")
    shown = shown_path(path, root)
    verb = "Overwrote" if existed else "Created"
    return Shown(f"{verb} {shown} ({size} bytes).", {
        "kind": "write_file", "file_path": shown, "content": content,
        "file_existed": existed, "previous_content": previous})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create a file in the user's project with the given "
                       "content, or replace an existing file completely. "
                       "Parent folders are created. To change part of a file, "
                       "use edit instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string",
                              "description": "path to the file, relative to "
                                             "the project folder or absolute"},
                "content": {"type": "string",
                            "description": "the full text of the file"},
            },
            "required": ["file_path", "content"],
        },
    },
}
