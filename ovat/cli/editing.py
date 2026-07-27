# ovat/cli/editing.py
"""Input-line conveniences a terminal user already expects: history and paste.

Textual's Input has neither. It is a text box, not a shell prompt, so Up and
Down move nothing and Ctrl-V depends entirely on whether the terminal turns
it into a bracketed-paste sequence. Both gaps are felt immediately by anyone
who has used a shell.

Kept free of any textual import so the logic is unit-testable on its own and
so nothing here ties the CLI to the [tui] extra.
"""
import subprocess
import sys


class InputHistory:
    """Up/Down recall for one input line, the way a shell does it.

    The part that is easy to get wrong is the draft. If you type half a
    question, press Up to check what you asked before, then press Down again,
    a shell gives you your half-finished line back. Losing it is infuriating,
    so the draft is stashed on the way up and restored on the way past the
    end.
    """

    def __init__(self, limit: int = 200):
        self._entries: list = []
        self._limit = limit
        self._index = None       # None means "not browsing"
        self._draft = ""

    @property
    def entries(self) -> list:
        return list(self._entries)

    def add(self, text: str) -> None:
        """Record a submitted line and stop browsing."""
        text = text.strip()
        if not text:
            return
        if self._entries and self._entries[-1] == text:
            # Asking the same thing twice in a row should not need two
            # presses of Up to get past.
            self._entries.pop()
        self._entries.append(text)
        if len(self._entries) > self._limit:
            del self._entries[:-self._limit]
        self.reset()

    def reset(self) -> None:
        self._index = None
        self._draft = ""

    def previous(self, current: str):
        """The entry before this one, or None when there is no history."""
        if not self._entries:
            return None
        if self._index is None:
            self._draft = current          # stash the half-typed line
            self._index = len(self._entries)
        if self._index == 0:
            return self._entries[0]        # already at the oldest; stay put
        self._index -= 1
        return self._entries[self._index]

    def next(self):
        """The entry after this one, then the stashed draft, then None."""
        if self._index is None:
            return None                    # not browsing: Down does nothing
        if self._index >= len(self._entries) - 1:
            self._index = None
            draft, self._draft = self._draft, ""
            return draft                   # back to what you were typing
        self._index += 1
        return self._entries[self._index]


def _paste_command() -> list:
    if sys.platform == "win32":
        return ["powershell", "-NoProfile", "-Command", "Get-Clipboard"]
    if sys.platform == "darwin":
        return ["pbpaste"]
    return ["xclip", "-selection", "clipboard", "-o"]


def read_clipboard(_run=subprocess.run) -> str:
    """The system clipboard as text, or "" if it cannot be read.

    Terminals give an app no portable way to READ the clipboard: OSC 52 is
    write-mostly and usually disabled for reads, which is why Textual offers
    copy_to_clipboard and no counterpart. Shelling out to the platform's own
    tool is the reliable route and needs no new dependency.

    Never raises. A missing xclip or a locked clipboard is a reason for a
    paste to do nothing, not for the TUI to fall over.
    """
    try:
        result = _run(_paste_command(), capture_output=True, text=True,
                      timeout=5)
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    # Get-Clipboard adds a trailing newline; a single-line input does not
    # want it, and neither does it want to become multi-line by accident.
    return result.stdout.replace("\r\n", "\n").rstrip("\n")
