# ovat/tools/edit.py
"""The edit tool: replace one exact piece of text in a file.

Mirrors Mistral Vibe's edit.py (vibe/core/tools/builtins) and its
line_contexts() from vibe/utils/text.py: same checks, same wording, and the
same `occurrences` (start_line, old lines, new lines) the diff card draws.
Left out: its checkpoints, atomic_replace and file locks.
"""
from pathlib import Path

from ovat.tools.project import Shown, fail, resolve_inside, shown_path


def line_contexts(content: str, snippet: str) -> list[tuple[int, str, str]]:
    """(start_line, prefix, suffix) per match, completing it to whole lines."""
    if not snippet.strip("\n"):
        return []
    results = []
    pos = content.find(snippet)
    while pos != -1:
        start_line = content.count("\n", 0, pos) + 1
        line_start = content.rfind("\n", 0, pos) + 1
        prefix = content[line_start:pos]
        match_end = pos + len(snippet)
        # A match that ends on a line break has no partial last line.
        if match_end > 0 and content[match_end - 1] == "\n":
            suffix = ""
        else:
            line_end = content.find("\n", match_end)
            if line_end == -1:
                line_end = len(content)
            elif line_end > match_end and content[line_end - 1] == "\r":
                line_end -= 1
            suffix = content[match_end:line_end]
        results.append((start_line, prefix, suffix))
        pos = content.find(snippet, match_end)      # like str.replace
    return results


def edit_impl(file_path: str, old_string: str, new_string: str, root: Path,
              replace_all: bool = False) -> str:
    if not old_string:
        return fail("edit", "old_string cannot be empty. Use write_file to "
                            "create new files.")
    if old_string == new_string:
        return fail("edit", "no changes to make: old_string and new_string "
                            "are identical.")
    path, why = resolve_inside(file_path, root)
    if why:
        return fail("edit", why)
    if not path.exists():
        return fail("edit", f"file does not exist: {file_path}")
    if not path.is_file():
        return fail("edit", f"{file_path} is not a file.")
    try:
        # newline="" so a CRLF file stays CRLF after the edit.
        with open(path, "r", encoding="utf-8", newline="") as f:
            original = f.read()
    except UnicodeDecodeError:
        return fail("edit", f"{file_path} is not valid UTF-8 text.")
    except PermissionError:
        return fail("edit", f"permission denied: {file_path}")
    except OSError as exc:
        return fail("edit", f"cannot read {file_path}: {exc}")

    if old_string not in original:
        return fail("edit", f"the string to replace was not found in "
                            f"{file_path}. Read the file again and copy the "
                            f"text exactly, spaces and line breaks included.")
    count = original.count(old_string)
    if count > 1 and not replace_all:
        return fail("edit", f"found {count} matches of the string to "
                            f"replace, but replace_all is false. Set "
                            f"replace_all to true to change all of them, or "
                            f"add more surrounding text to pick one.")
    contexts = line_contexts(original, old_string)
    if not replace_all:
        contexts = contexts[:1]
    modified = original.replace(old_string, new_string,
                                -1 if replace_all else 1)
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(modified)
    except OSError as exc:
        return fail("edit", f"cannot write {file_path}: {exc}")
    shown = shown_path(path, root)
    done = count if replace_all else 1
    return Shown(
        f"Edited {shown}: replaced {done} occurrence"
        f"{'' if done == 1 else 's'}.",
        {"kind": "edit", "file": shown, "old_string": old_string,
         "new_string": new_string,
         "occurrences": [
             {"start_line": start, "old_text": prefix + old_string + suffix,
              "new_text": prefix + new_string + suffix}
             for start, prefix, suffix in contexts]})


SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Change a file in the user's project by replacing an "
                       "exact piece of text with new text. Read the file "
                       "first and copy old_string exactly. It must match "
                       "once, unless replace_all is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string",
                              "description": "path to the file, relative to "
                                             "the project folder or absolute"},
                "old_string": {"type": "string",
                               "description": "the exact text to replace"},
                "new_string": {"type": "string",
                               "description": "the text to put in its place "
                                              "(must differ from old_string)"},
                "replace_all": {"type": "boolean",
                                "description": "replace every occurrence",
                                "default": False},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
}
