# ovat/tui/server/safe.py
"""Showing model-written text to a person who is about to decide something.

Both the app server's approval callbacks and the classic agent TUI
(ovat/tui/app.py) put a tool's arguments in front of the person, so the
cleaning lives in one place.
"""

_APPROVAL_ARG_MAX = 600


def approval_safe(value):
    """A model-written argument as shown in the approval panel.

    The person deciding must see what will be sent, but the text must not
    be able to dress up as part of the panel: a "task" containing
    "\\n1. Allow once" would otherwise print lines that look like the
    panel's own options, or push the real ones out of view. So control
    characters go, every newline shows as a visible mark on ONE line, and
    long text is cut (with the length said)."""
    if isinstance(value, dict):
        return {str(k): approval_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [approval_safe(v) for v in value]
    if not isinstance(value, str):
        return value
    text = "".join(ch if ch.isprintable() or ch == "\n" else "?"
                   for ch in value)
    text = text.replace("\r\n", "\n").replace("\n", " \u21b5 ")
    if len(text) > _APPROVAL_ARG_MAX:
        text = (text[:_APPROVAL_ARG_MAX]
                + f" ... [{len(value)} characters in all]")
    return text
