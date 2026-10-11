# ovat/tui/widgets/banner.py
"""The welcome box at the top of the transcript.

The launcher's lettering (cli/tui.py: the same FIGlet faces and gradients,
OVAT over "powered by OpenVINO + OVMS") in a purple box only 8 rows tall,
with every fact about the session beside it; on a narrow terminal the facts
move under the lettering instead of being cut, and on a very narrow one the
lettering is dropped. It scrolls away with the chat: this screen is for
talking to the agent. The Intel animation plays in the intro
(widgets/intro.py), not here.
"""
from importlib.metadata import PackageNotFoundError, version

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from ovat.cli import ui
from ovat.tui.widgets.loading import wave_text

# Wide enough for the lettering (35 + 2 + 56) plus a readable facts column.
_SIDE_BY_SIDE_MIN = 140
_LETTERING_MIN = 100

WHATS_NEW = ("subagents (local, A2A, cloud)", "A2A serve and call",
             "approvals", "!shell", "live NPU")


def _version() -> str:
    try:
        return version("ovat")
    except PackageNotFoundError:
        return "dev"


def lettering() -> tuple[Text, Text]:
    """(OVAT, "powered by" over "OpenVINO + OVMS"), in the launcher's faces."""
    from ovat.cli import tui as launcher
    ovat = launcher._wordmark_text()
    credit = launcher._credit_wordmark("powered by")
    credit.append_text(launcher._attribution_wordmark("OpenVINO + OVMS"))
    ovat.rstrip()
    credit.rstrip()
    return ovat, credit


def session_lines(info: dict) -> list[Text]:
    """The facts, one per line, once the agent is ready."""
    model = Text()
    model.append("● ", style=ui.GREEN)
    model.append(info.get("model", "?"), style="bold")
    model.append(" on ", style=ui.DIM)
    model.append(info.get("device", "?"), style=f"bold {ui.CYAN}")
    if info.get("provider") == "genai":
        model.append("  in-process", style=ui.DIM)

    tools = [t for t in info.get("tools", []) if t["kind"] != "subagent"]
    subs = info.get("subagents", [])
    engine = Text(style=ui.DIM)
    engine.append(f"{info.get('engine', '?')} engine  ·  {len(tools)} tool"
                  f"{'s' if len(tools) != 1 else ''}")
    agents = Text(style=ui.DIM)
    if subs:
        agents.append("subagents: ")
        agents.append(", ".join(s["name"] for s in subs), style=ui.PURPLE)
    else:
        agents.append("no subagents")
    where = Text(info.get("workflow", ""), style=ui.DIM)
    return [model, engine, agents, where]


def _header() -> Text:
    line = Text()
    line.append("OVAT agent", style=f"bold {ui.CYAN}")
    line.append(f"  v{_version()}", style=ui.DIM)
    return line


def _news() -> Text:
    line = Text()
    line.append("new: ", style=f"bold {ui.PURPLE}")
    line.append("  ·  ".join(WHATS_NEW), style=ui.DIM)
    return line


def _help() -> Text:
    line = Text(style=ui.DIM)
    line.append("/help", style=f"bold {ui.PURPLE}")
    line.append(" commands  ")
    line.append("!cmd", style=f"bold {ui.PURPLE}")
    line.append(" shell  ")
    line.append("esc", style=f"bold {ui.PURPLE}")
    line.append(" stop")
    return line


class Banner(Vertical):
    DEFAULT_CSS = """
    Banner { height: auto; width: 100%; margin-top: 1;
             border: round $secondary; padding: 0 1; }
    Banner #art { height: auto; width: auto; }
    Banner #ovat-mark { width: auto; height: auto; margin-right: 2; }
    Banner #credit { width: auto; height: auto; }
    Banner #facts { height: auto; width: 1fr; padding-left: 2;
                    border-left: solid $secondary 50%; }
    Banner #body { height: auto; }
    Banner.narrow #body { layout: vertical; }
    Banner.narrow #facts { border-left: none; padding-left: 0;
                           margin-top: 1; }
    Banner.tiny #art { display: none; }
    Banner Static { height: auto; }
    """

    def __init__(self, animate: bool = True, **kwargs) -> None:
        # `animate` is kept for the app's signature; the lettering is still.
        super().__init__(**kwargs)
        self._step = 0
        self._timer = None
        self._model = Static(wave_text("loading the agent…", 0))
        self._lines = [Static("") for _ in range(3)]

    def compose(self):
        ovat, credit = lettering()
        with Horizontal(id="body"):
            with Horizontal(id="art"):
                yield Static(ovat, id="ovat-mark")
                yield Static(credit, id="credit")
            with Vertical(id="facts"):
                yield Static(_header())
                yield self._model
                yield from self._lines
                yield Static(_news())
                yield Static(_help())

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._wave)
        self._fit(self.app.size.width)

    def on_resize(self, event) -> None:
        self._fit(event.size.width)

    def _fit(self, width: int) -> None:
        self.set_class(width < _SIDE_BY_SIDE_MIN, "narrow")
        self.set_class(width < _LETTERING_MIN, "tiny")

    def _wave(self) -> None:
        self._step += 1
        self._model.update(wave_text("loading the agent…", self._step),
                           layout=False)

    def show_session(self, info: dict) -> None:
        self._stop_wave()
        model, *rest = session_lines(info)
        self._model.update(model)
        for widget, line in zip(self._lines, rest):
            widget.update(line)

    def show_problem(self, text: str) -> None:
        self._stop_wave()
        self._model.update(Text("● no agent loaded", style=f"bold {ui.YELLOW}"))
        self._lines[0].update(Text(text, style=ui.DIM))

    def _stop_wave(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
