# ovat/cli/tui.py
"""The OVAT terminal UI: a full-screen, OpenCode-style launcher.

Textual runs this in the terminal's alternate screen buffer, so it behaves like
a separate window inside the terminal (its own scroll regions, restored cleanly
on exit) instead of smearing into the normal scrollback. That is what keeps the
UI from glitching on scroll or resize.

The view is intentionally thin. Every command routes through ovat.cli.commands,
which runs the real toolkit, so nothing here is a mock. Adding a command means
registering it there; this file just shows results.
"""
from rich.console import Group
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static

from ovat.cli import ui
from ovat.cli.commands import cmd_help, dispatch, match_commands

# The wordmark, drawn in blocks and shaded top-to-bottom from cyan to Intel blue.
_WORDMARK = [
    "██████ ██  ██  ████  ██████",
    "██  ██ ██  ██ ██  ██   ██  ",
    "██  ██ ██  ██ ██████   ██  ",
    "██  ██  ████  ██  ██   ██  ",
    "██████   ██   ██  ██   ██  ",
]
_WORDMARK_SHADE = ["#00C7FD", "#16B2F2", "#1199DC", "#0B82CA", "#0068B5"]


def _banner() -> Group:
    """Build the header: the OVAT wordmark, the full product name, and a tagline.

    The wordmark is the short brand mark; the full name spells out what OVAT
    stands for so a first-time user is never left guessing: OpenVINO Agentic
    Toolkit (OV - A - T).
    """
    mark = Text()
    for row, color in zip(_WORDMARK, _WORDMARK_SHADE):
        mark.append(row + "\n", style=f"bold {color}")
    name = Text()
    name.append("\nOpenVINO ", style=f"bold {ui.CYAN}")
    name.append("Agentic ", style=f"bold {ui.CYAN}")
    name.append("Toolkit", style=f"bold {ui.CYAN}")
    tagline = Text("one YAML  +  one command", style=ui.PURPLE)
    return Group(mark, name, tagline)


class OvatTUI(App):
    """The OVAT launcher app."""

    TITLE = "OVAT"

    CSS = """
    Screen { background: #0b0e14; }
    #banner { padding: 1 2 0 2; height: auto; }
    #output {
        height: 1fr;
        border: round #0068B5;
        background: #0d1117;
        padding: 0 1;
        margin: 1 2;
    }
    #palette {
        display: none;
        height: auto;
        padding: 0 3;
    }
    #prompt {
        margin: 0 2 1 2;
        border: round #0068B5;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(_banner(), id="banner")
        yield RichLog(id="output", highlight=False, markup=False, wrap=True)
        yield Static("", id="palette")
        yield Input(placeholder="Type a /command, e.g. /doctor  (Tab completes)",
                    id="prompt")

    def on_mount(self) -> None:
        log = self.query_one("#output", RichLog)
        log.write(cmd_help([]).renderable)
        log.write(Text("Pick a command above, then add a path if it needs one. "
                       "/help anytime, /exit to leave.", style=ui.CYAN))
        self.query_one("#prompt", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show the live slash menu while the command word is being typed."""
        palette = self.query_one("#palette", Static)
        value = event.value
        head, _, _ = value.partition(" ")
        typing_args = " " in value
        if value.startswith("/") and not typing_args:
            matches = match_commands(head)
            if matches:
                menu = Text()
                for c in matches:
                    menu.append(f"  {c.usage:<26}", style=f"bold {ui.PURPLE}")
                    menu.append(f"{c.description}\n", style=ui.DIM)
                palette.update(menu)
                palette.display = True
                return
        palette.display = False

    def on_key(self, event) -> None:
        """Tab completes the partial command to the first matching name."""
        if event.key != "tab":
            return
        inp = self.query_one("#prompt", Input)
        value = inp.value
        if value.startswith("/") and " " not in value:
            matches = match_commands(value)
            if matches:
                inp.value = f"/{matches[0].name} "
                inp.cursor_position = len(inp.value)
                event.prevent_default()
                event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value
        inp = self.query_one("#prompt", Input)
        log = self.query_one("#output", RichLog)
        inp.value = ""
        self.query_one("#palette", Static).display = False

        result = dispatch(line)
        if result is None:
            return
        # Echo what was run, then the result, so the log reads like a transcript.
        log.write(Text(f"› {line}", style=ui.DIM))
        if result.action == "clear":
            log.clear()
            return
        log.write(result.renderable)
        if result.action == "exit":
            self.exit()


def run_tui() -> None:
    """Launch the TUI. Called when someone runs `ovat` with no subcommand."""
    OvatTUI().run()
