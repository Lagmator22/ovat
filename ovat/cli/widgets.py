# ovat/cli/widgets.py
"""Small widget subclasses shared by the TUI screens."""
from textual.widgets import Input

from ovat.cli.editing import read_clipboard


class PasteInput(Input):
    """An Input whose Ctrl-V reaches the SYSTEM clipboard.

    Textual's own Input.action_paste reads app.clipboard, which is Textual's
    internal buffer and only ever holds what the app itself copied with
    copy_to_clipboard. Text copied from a browser, an editor, or the
    terminal's own selection was therefore impossible to paste in: the key
    was bound, the action ran, and nothing happened, which is worse than not
    being bound at all.

    The system clipboard wins when it has anything; app.clipboard is the
    fallback, so pasting straight back what /copy just put there still works
    even on a machine with no pbpaste/xclip.
    """

    def action_paste(self) -> None:
        text = read_clipboard() or self.app.clipboard
        if not text:
            return
        start, end = self.selection
        self.replace(text, start, end)
