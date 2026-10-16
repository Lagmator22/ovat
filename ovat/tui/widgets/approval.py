# ovat/tui/widgets/approval.py
# Look and grace period adapted from Mistral Vibe
# (https://github.com/mistralai/mistral-vibe, vibe/cli/textual_ui/widgets/
# approval_app.py and app.tcss #approval-app), Copyright Mistral AI, Apache
# License 2.0. Changed for OVAT: no "always (permanent)" option, OVAT's
# broker and wording.
"""Asking the person whether a tool may run, in place of the input box.

    ⚠ Allow subagent cloud?
      sends the task to https://api.mistral.ai/v1
      task: Summarise recent public research on NPU KV-cache sizing.
    › 1 Yes, once
      2 Yes, for this session
      3 No  (esc)

It takes the input's place rather than opening a modal, so the
conversation stays visible while deciding (Mistral Vibe, OpenCode and Codex
all do this). Keys are ignored for the first half second after it appears:
someone typing a sentence when the request pops up must not approve it
with their next Enter. There is no "always" option: a permanent rule
belongs in the workflow (`permission: allow`), where it can be reviewed.
"""
import json
import time

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from ovat.agent.permissions import ONCE, REFUSE, SESSION
from ovat.cli import ui

GRACE_S = 0.5
_OPTIONS = ((ONCE, "Yes, once"), (SESSION, "Yes, for this session"),
            (REFUSE, "No  (esc)"))


def _args_text(arguments: dict, width: int = 400) -> str:
    if set(arguments) == {"task"}:
        text = str(arguments["task"])
    else:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    return text if len(text) <= width else text[:width - 1] + "…"


class ApprovalPanel(Widget, can_focus=True):
    DEFAULT_CSS = """
    ApprovalPanel { height: auto; max-height: 70vh; width: 100%;
                    padding: 0 1; background: transparent;
                    border: solid $foreground 35%; }
    ApprovalPanel:focus { border: solid $foreground 35%; }
    ApprovalPanel Static { height: auto; }
    """

    class Decided(Message):
        def __init__(self, request_id: int, decision: str) -> None:
            super().__init__()
            self.request_id = request_id
            self.decision = decision

    def __init__(self, request, clock=time.monotonic, **kwargs) -> None:
        super().__init__(**kwargs)
        self.request = request
        self.index = 0
        self._clock = clock
        self._shown_at = clock()
        self._decided = False

    def compose(self):
        yield Static(self._render_body(), id="approval-body")

    def _render_body(self) -> Text:
        r = self.request
        out = Text()
        out.append(f"{r.tool} wants to run", style=f"bold {ui.YELLOW}")
        if r.reason:
            out.append(f"\n⚠ {r.reason}", style=ui.YELLOW)
        if r.arguments:
            out.append(f"\n{_args_text(r.arguments)}", style=ui.DIM)
        out.append("\n")
        for i, (decision, label) in enumerate(_OPTIONS):
            selected = i == self.index
            mark = "› " if selected else "  "
            if selected and decision != REFUSE:
                style = f"bold {ui.GREEN}"
            elif selected:
                style = "bold"
            else:
                style = ""
            out.append(f"\n{mark}{i + 1}. {label}", style=style)
        out.append("\n↑↓ choose · enter confirm · 1-3 · esc no",
                   style=ui.DIM)
        return out

    def _redraw(self) -> None:
        self.query_one("#approval-body", Static).update(self._render_body())

    def decide(self, decision: str) -> None:
        if self._decided:
            return
        self._decided = True
        self.post_message(self.Decided(self.request.id, decision))

    def on_key(self, event: events.Key) -> None:
        event.stop()
        event.prevent_default()
        if self._clock() - self._shown_at < GRACE_S:
            return                       # typing that was meant for the input
        key = event.key
        if key in ("1", "y"):
            self.decide(ONCE)
        elif key == "2":
            self.decide(SESSION)
        elif key in ("3", "n", "escape"):
            self.decide(REFUSE)
        elif key == "up":
            self.index = (self.index - 1) % len(_OPTIONS)
            self._redraw()
        elif key == "down":
            self.index = (self.index + 1) % len(_OPTIONS)
            self._redraw()
        elif key == "enter":
            self.decide(_OPTIONS[self.index][0])
