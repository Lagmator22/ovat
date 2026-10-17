# ovat/tui/widgets/tools.py
"""All of one answer's tool calls, folded into one live summary line.

    ■ Searching docs, delegating to researcher · researcher: calling search_docs
    ⏵ Searched docs, delegated to researcher  (3 calls, 4.2s)

Present tense with a pulsing square while the answer is being worked on,
past tense with a fold arrow once it is done; click (or Ctrl-O) to open the
per-call rows. A failed call is shown in grey and only turns the summary
red if the ANSWER failed: OVAT recovers from many tool errors (a misspelt
path, a retry), and red for a recovered error is noise. Layout idea from
Mistral Vibe's tool groups; the wording and the events are OVAT's.
"""
import json

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static

from ovat.agent.events import (REMOTE_STATUS, SUBAGENT_STATUS, TOOL_ENDED,
                               TOOL_STARTED)
from ovat.cli import ui

_PULSE = ("■", "■", "■", "□", "□")

# (running, done) wording per built-in tool.
_WORDS = {
    "search_docs": ("searching docs", "searched docs"),
    "transcribe": ("transcribing audio", "transcribed audio"),
    "describe_image": ("looking at an image", "looked at an image"),
}


def _phrase(name: str, kind: str, args: dict, running: bool) -> str:
    if name in _WORDS:
        return _WORDS[name][0 if running else 1]
    if kind == "subagent":
        who = args.get("agent", "a subagent")
        return f"delegating to {who}" if running else f"delegated to {who}"
    if kind == "a2a":
        return f"asking {name}" if running else f"asked {name}"
    return f"calling {name}" if running else f"called {name}"


def _preview(args: dict, width: int = 70) -> str:
    if not args:
        return ""
    if set(args) == {"agent", "task"}:
        text = f"{args['agent']}: {args['task']}"
    elif len(args) == 1:
        text = str(next(iter(args.values())))
    else:
        text = json.dumps(args, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


class _Call:
    def __init__(self, name, kind, args):
        self.name, self.kind, self.args = name, kind, args
        self.status, self.duration = "running", None


class ToolGroup(Vertical):
    DEFAULT_CSS = """
    ToolGroup { height: auto; width: 100%; margin-top: 1; }
    ToolGroup .head { height: auto; width: 100%; }
    ToolGroup .body { height: auto; padding-left: 2; color: $text-muted;
                      display: none; }
    ToolGroup.open .body { display: block; }
    """

    def __init__(self, kinds: dict[str, str] | None = None) -> None:
        super().__init__()
        self._kinds = kinds or {}
        self._calls: dict[str, _Call] = {}
        self._order: list[str] = []
        self._notes: list[tuple[str, str]] = []     # (call_id or "", text)
        self._live = ""
        self._step = 0
        self._working = True
        self._failed = False
        self._elapsed = 0.0
        self._timer = None
        # Made here, not in compose(): events can arrive before the group
        # has finished mounting, and they must still land somewhere.
        self._head = Static(self._header(), classes="head")
        self._body_widget = Static("", classes="body")

    def compose(self):
        yield self._head
        yield self._body_widget

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.15, self._pulse)

    # ---- events ------------------------------------------------------------

    def handle(self, event) -> None:
        d = event.data
        if event.type == TOOL_STARTED:
            name = d.get("name", "?")
            call = _Call(name, self._kinds.get(name, "builtin"),
                         d.get("arguments") or {})
            self._calls[d.get("call_id", name)] = call
            self._order.append(d.get("call_id", name))
            self._live = ""
        elif event.type == TOOL_ENDED:
            call = self._calls.get(d.get("call_id"))
            if call is not None:
                call.status = d.get("status", "ok")
                call.duration = d.get("duration_s")
        elif event.type in (REMOTE_STATUS, SUBAGENT_STATUS):
            self._live = f"{d.get('agent')}: {d.get('text')}"
            self._notes.append((self._order[-1] if self._order else "",
                                self._live))
        self._redraw()

    def finish(self, failed: bool, elapsed: float) -> None:
        self._working, self._failed, self._elapsed = False, failed, elapsed
        self._live = ""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._redraw()

    def toggle(self) -> None:
        self.toggle_class("open")
        self._redraw()

    def on_click(self) -> None:
        if not self._working:
            self.toggle()

    @property
    def call_count(self) -> int:
        return len(self._order)

    # ---- drawing -----------------------------------------------------------

    def _pulse(self) -> None:
        self._step += 1
        self._head.update(self._header(), layout=False)

    def _redraw(self) -> None:
        self._head.update(self._header())
        self._body_widget.update(self._body())

    def _summary(self) -> str:
        phrases = []
        for cid in self._order:
            c = self._calls[cid]
            phrase = _phrase(c.name, c.kind, c.args, self._working)
            if phrase not in phrases:
                phrases.append(phrase)
        text = ", ".join(phrases) or ("working" if self._working else "done")
        return text[:1].upper() + text[1:]

    def _header(self) -> Text:
        head = Text()
        if self._working:
            glyph = _PULSE[self._step % len(_PULSE)]
            head.append(glyph + " ", style=f"bold {ui.CYAN}")
            head.append(self._summary(), style="bold")
            if self._live:
                head.append(f"  ·  {self._live}", style=ui.DIM)
            return head
        arrow = "⏷ " if self.has_class("open") else "⏵ "
        errors = any(c.status not in ("ok", "running")
                     for c in self._calls.values())
        colour = ui.RED if self._failed and errors else ui.DIM
        head.append(arrow, style=colour)
        head.append(self._summary(), style=colour)
        n = len(self._order)
        head.append(f"  ({n} call{'s' if n != 1 else ''}, "
                    f"{self._elapsed:.1f}s)", style=ui.DIM)
        return head

    def _body(self) -> Text:
        out = Text()
        for i, cid in enumerate(self._order):
            c = self._calls[cid]
            if i:
                out.append("\n")
            if c.status == "running":
                out.append("◌ ", style=ui.CYAN)
            elif c.status == "ok":
                out.append("✓ ", style=ui.GREEN)
            elif c.status == "denied":
                out.append("⊘ ", style=ui.YELLOW)
            else:
                out.append("✕ ", style=ui.DIM)
            out.append(c.name, style="bold")
            if c.duration is not None:
                out.append(f"  {c.duration:.1f}s", style=ui.DIM)
            if c.status == "denied":
                out.append("  not allowed", style=ui.YELLOW)
            preview = _preview(c.args)
            if preview:
                out.append(f"  {preview}", style=ui.DIM)
            for owner, note in self._notes:
                if owner == cid:
                    out.append(f"\n    ↳ {note}", style=ui.DIM)
        return out
