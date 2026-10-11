# ovat/tui/widgets/banner.py
"""The welcome block at the top of the transcript.

Left: the Intel mark, played once from assets/intel.gif and settling on the
Intel "e", by the SAME renderer the launcher uses (cli/tui.py
_StartupIntelAnimation), only smaller. Right: what this session is (model,
device, engine, tools, subagents), filled in when the agent is built,
with a waving "loading" line until then.

It scrolls away with the conversation, like Mistral Vibe's banner: it is
part of the transcript, not chrome that eats rows forever.
"""
from importlib.metadata import PackageNotFoundError, version

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from ovat.cli import ui
from ovat.tui.widgets.loading import wave_text

MARK_COLUMNS = 24


def _version() -> str:
    try:
        return version("ovat")
    except PackageNotFoundError:
        return "dev"


def brand_line() -> Text:
    line = Text()
    line.append("OVAT", style=f"bold {ui.CYAN}")
    line.append(" agent", style=f"bold {ui.PURPLE}")
    line.append(f"  v{_version()}", style=ui.DIM)
    return line


def session_lines(info: dict) -> list[Text]:
    """The lines under the brand once the agent is ready."""
    model = Text()
    model.append("● ", style=ui.GREEN)
    model.append(info.get("model", "?"), style="bold")
    model.append(" on ", style=ui.DIM)
    model.append(info.get("device", "?"), style=f"bold {ui.CYAN}")
    model.append(f"  ·  {info.get('engine', '?')} engine", style=ui.DIM)
    if info.get("provider") == "genai":
        model.append("  ·  in-process", style=ui.DIM)

    # `delegate` is how subagents are reached, not a tool of its own.
    tools = [t for t in info.get("tools", []) if t["kind"] != "subagent"]
    subs = info.get("subagents", [])
    counts = Text(style=ui.DIM)
    counts.append(f"{len(tools)} tool{'s' if len(tools) != 1 else ''}")
    if subs:
        counts.append(f"  ·  {len(subs)} subagent"
                      f"{'s' if len(subs) != 1 else ''}: "
                      + ", ".join(s["name"] for s in subs))
    if info.get("workflow"):
        counts.append(f"  ·  {info['workflow']}")
    return [model, counts]


def help_line() -> Text:
    line = Text(style=ui.DIM)
    line.append("Type ")
    line.append("/help", style=f"bold {ui.PURPLE}")
    line.append(" for commands  ·  ")
    line.append("!cmd", style=f"bold {ui.PURPLE}")
    line.append(" runs a shell command  ·  ")
    line.append("esc", style=f"bold {ui.PURPLE}")
    line.append(" stops")
    return line


class Banner(Horizontal):
    DEFAULT_CSS = """
    Banner { height: auto; width: 100%; margin: 1 0 0 0; }
    Banner #mark { width: 24; height: 8; margin-right: 2; }
    Banner #info { width: 1fr; height: auto; padding-top: 1; }
    Banner .line { height: 1; width: 100%; }
    Banner .gap { height: 1; }
    Banner #counts, Banner #model { height: auto; }
    """

    def __init__(self, animate: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._play_intro = animate
        self._step = 0
        self._timer = None

    def compose(self):
        if self._play_intro:
            # Imported here: cli/tui.py is the launcher module, and only the
            # animation is shared, not anything else of it.
            from ovat.cli.tui import _StartupIntelAnimation
            yield _StartupIntelAnimation(columns=MARK_COLUMNS, id="mark")
        else:
            yield Static("", id="mark")
        with Vertical(id="info"):
            yield Static(brand_line(), classes="line")
            yield Static(Text("OpenVINO Agentic Toolkit  ·  agents on your "
                              "AI PC", style=ui.DIM), classes="line")
            yield Static("", classes="gap")
            yield Static(wave_text("Loading the agent…", 0), id="model",
                         classes="line")
            yield Static("", id="counts", classes="line")
            yield Static("", classes="gap")
            yield Static(help_line(), classes="line")

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._wave)

    def _wave(self) -> None:
        self._step += 1
        self.query_one("#model", Static).update(
            wave_text("Loading the agent…", self._step), layout=False)

    def show_session(self, info: dict) -> None:
        self._stop_wave()
        model, counts = session_lines(info)
        self.query_one("#model", Static).update(model)
        self.query_one("#counts", Static).update(counts)

    def show_problem(self, text: str) -> None:
        self._stop_wave()
        self.query_one("#model", Static).update(
            Text("● " + "no agent loaded", style=f"bold {ui.YELLOW}"))
        self.query_one("#counts", Static).update(Text(text, style=ui.DIM))

    def _stop_wave(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
