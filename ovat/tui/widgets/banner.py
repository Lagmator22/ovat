# ovat/tui/widgets/banner.py
# Layout adapted from Mistral Vibe (https://github.com/mistralai/mistral-vibe,
# vibe/cli/textual_ui/widgets/banner/banner.py), Copyright Mistral AI, Apache
# License 2.0. Changed for OVAT: OVAT's chip mark instead of Vibe's mascot,
# OVAT's facts (model, device, engine, tools, subagents), no Vibe state.
"""The welcome lines at the top of the transcript.

    ⠀⣿⣿⣿⣿⣿⡇   OVAT v1.2.0 · Qwen3.5-4B-int4-ov on GPU · native
    ⠀⣿⣿⣿⣿⣿⡇   1 tool · 3 subagents · workflow.yml
    ⠀⣿⣿⣿⣿⣿⡇   Type /help for more information

Three lines and a small animated mark: the screen is for the conversation.
"""
from importlib.metadata import PackageNotFoundError, version

from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalGroup
from textual.widgets import Static

from ovat.cli import ui
from ovat.tui.widgets.braille import ChipMark


# Blue -> cyan -> purple: the wave's stops.
WAVE = (ui.BLUE, "#1E90E6", ui.CYAN, "#7AA2FF", ui.PURPLE)


def wave_text(text: str, step: int) -> Text:
    """`text` with the colour wave at position `step`."""
    out = Text()
    span = len(WAVE)
    for i, ch in enumerate(text):
        # A smooth triangle wave over the stops, moving right each tick.
        pos = (step - i) % (2 * span - 2)
        index = pos if pos < span else 2 * span - 2 - pos
        out.append(ch, style=f"bold {WAVE[index]}")
    return out


def _version() -> str:
    try:
        return version("ovat")
    except PackageNotFoundError:
        return "dev"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def model_line(info: dict) -> Text:
    line = Text()
    line.append(info.get("model", "?"), style="bold")
    line.append(" on ", style=ui.DIM)
    line.append(info.get("device", "?"), style=f"bold {ui.CYAN}")
    line.append(f" · {info.get('engine', '?')}")
    if info.get("provider") == "genai":
        line.append(" · in-process", style=ui.DIM)
    return line


def counts_line(info: dict) -> Text:
    tools = [t for t in info.get("tools", []) if t["kind"] != "subagent"]
    subs = info.get("subagents", [])
    parts = [_plural(len(tools), "tool"), _plural(len(subs), "subagent")]
    if info.get("workflow"):
        parts.append(info["workflow"])
    return Text(" · ".join(parts))


class Banner(Static):
    DEFAULT_CSS = """
    Banner { height: auto; width: 100%; }
    Banner #banner-container { align: left top; padding: 1 1 0 0;
                               width: auto; height: auto; }
    Banner #banner-info { width: auto; height: auto; margin-top: 0;
                          margin-left: 1; }
    Banner .banner-line { width: auto; height: 1; }
    Banner #banner-brand { color: $accent; text-style: bold; width: auto; }
    Banner .banner-meta { width: auto; }
    Banner .banner-cmd { color: $secondary; width: auto; }
    """

    def __init__(self, animate: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.can_focus = False
        self.mark = ChipMark(animate=animate, classes="banner-mark")
        self._model = Static(wave_text("loading the agent…", 0),
                             classes="banner-meta")
        self._counts = Static("", classes="banner-meta")
        self._step = 0
        self._timer = None

    def compose(self):
        with Horizontal(id="banner-container"):
            yield self.mark
            with Vertical(id="banner-info"):
                with Horizontal(classes="banner-line"):
                    yield Static("OVAT", id="banner-brand")
                    yield Static(f" v{_version()} · ", classes="banner-meta")
                    yield self._model
                with Horizontal(classes="banner-line"):
                    yield self._counts
                with Horizontal(classes="banner-line"):
                    yield Static("Type ", classes="banner-meta")
                    yield Static("/help", classes="banner-cmd")
                    yield Static(" for more information",
                                 classes="banner-meta")

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._wave)

    def _wave(self) -> None:
        self._step += 1
        self._model.update(wave_text("loading the agent…", self._step))

    def show_session(self, info: dict) -> None:
        self._stop_wave()
        self._model.update(model_line(info))
        self._counts.update(counts_line(info))

    def show_problem(self, text: str) -> None:
        self._stop_wave()
        self._model.update(Text("no agent loaded", style=f"bold {ui.YELLOW}"))
        self._counts.update(Text(text, style=ui.DIM))

    def _stop_wave(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
