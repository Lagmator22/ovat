# ovat/cli/tui.py
"""The OVAT terminal UI: a styled, full-screen wrapper around a real shell.

Textual runs this in the terminal's alternate screen buffer, so it behaves like
a separate window inside the terminal and never smears into the scrollback.

The model is deliberately simple and unlimited: anything you type runs as a real
command in the venv (ovat ..., python -m pytest ..., ls, git status, anything).
Typing "/" opens a dropdown of OVAT command templates you can pick and complete,
so you get shortcuts without losing the ability to run any command at all.
"""
import os

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from ovat.cli import shell, ui

# A dependency-free fallback, used only if pyfiglet is somehow unavailable.
_FALLBACK_WORDMARK = [
    "██████ ██  ██  ████  ██████",
    "██  ██ ██  ██ ██  ██   ██  ",
    "██  ██ ██  ██ ██████   ██  ",
    "██  ██  ████  ██  ██   ██  ",
    "██████   ██   ██  ██   ██  ",
]
_CYAN_RGB = (0, 199, 253)
_BLUE_RGB = (0, 104, 181)


def _lerp_hex(a: tuple, b: tuple, t: float) -> str:
    """Blend two RGB colours by fraction t in 0..1 and return a hex string."""
    return "#{:02X}{:02X}{:02X}".format(
        *(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    )


def _wordmark_lines() -> list:
    """Render OVAT with a real FIGlet font (ansi_shadow), the stylish block look."""
    try:
        import pyfiglet
        lines = [ln for ln in pyfiglet.figlet_format("OVAT", font="ansi_shadow").split("\n")
                 if ln.strip()]
        if lines:
            return lines
    except Exception:
        pass
    return _FALLBACK_WORDMARK


def _wordmark_text() -> Text:
    """The wordmark as Rich Text, shaded top-to-bottom from cyan to Intel blue."""
    lines = _wordmark_lines()
    text = Text()
    last = max(len(lines) - 1, 1)
    for i, line in enumerate(lines):
        text.append(line + "\n", style=f"bold {_lerp_hex(_CYAN_RGB, _BLUE_RGB, i / last)}")
    return text


def _banner() -> Text:
    """Header: the FIGlet OVAT wordmark, the full product name, and a tagline."""
    out = _wordmark_text()
    out.append("\nOpenVINO Agentic Toolkit\n", style=f"bold {ui.CYAN}")
    out.append("one YAML  +  one command", style=ui.PURPLE)
    return out


def _option(template) -> Option:
    """Build one dropdown row: the slash name in purple, its help in grey."""
    label = Text()
    label.append(f"{template.name:<9} ", style=f"bold {ui.PURPLE}")
    label.append(template.description, style=ui.DIM)
    # ids cannot contain a slash cleanly, so I key options by the bare name.
    return Option(label, id=template.name.lstrip("/"))


class OvatTUI(App):
    """The OVAT launcher: a styled terminal that knows OVAT's commands."""

    TITLE = "OVAT"

    CSS = """
    Screen { background: #0b0e14; }
    #banner { padding: 1 2 0 2; height: auto; }
    #output {
        height: 1fr;
        border: round #0068B5;
        background: #0d1117;
        padding: 0 1;
        margin: 1 2 0 2;
    }
    #palette {
        display: none;
        height: auto;
        max-height: 9;
        margin: 0 2;
        border: round #8F5CFF;
        background: #0d1117;
    }
    #palette > .option-list--option-highlighted {
        background: #0068B5;
        color: #ffffff;
        text-style: bold;
    }
    #prompt {
        margin: 0 2 1 2;
        border: round #0068B5;
    }
    """

    def __init__(self):
        super().__init__()
        # Commands run from where the user launched the TUI, so relative paths
        # (./docs, examples/workflow.yml) mean what they expect.
        self._cwd = os.getcwd()
        self._proc = None        # the currently running subprocess, if any

    def compose(self) -> ComposeResult:
        yield Static(_banner(), id="banner")
        yield RichLog(id="output", highlight=False, markup=False, wrap=True)
        yield OptionList(id="palette")
        yield Input(placeholder="Run any command (e.g. ovat doctor)  ·  type / for shortcuts",
                    id="prompt")

    def on_mount(self) -> None:
        log = self.query_one("#output", RichLog)
        log.write(Text("Type any command and press Enter; it runs in the venv.",
                       style=ui.CYAN))
        log.write(Text("Type /  for OVAT shortcuts  ·  Esc cancels a running "
                       "command  ·  /exit to quit.", style=ui.DIM))
        self.query_one("#prompt", Input).focus()

    # The live slash dropdown

    def on_input_changed(self, event: Input.Changed) -> None:
        palette = self.query_one("#palette", OptionList)
        value = event.value
        if value.startswith("/") and " " not in value:
            matches = shell.match_templates(value)
            palette.clear_options()
            if matches:
                palette.add_options([_option(t) for t in matches])
                palette.display = True
                return
        palette.display = False

    def on_key(self, event) -> None:
        inp = self.query_one("#prompt", Input)
        palette = self.query_one("#palette", OptionList)

        if event.key == "escape":
            if self._proc is not None:          # cancel a running command first
                self._proc.terminate()
                event.stop()
            elif palette.display:
                palette.display = False
                inp.focus()
                event.stop()
            return

        if event.key == "down" and palette.display and self.focused is inp:
            palette.focus()                     # step into the dropdown
            if palette.option_count:
                palette.highlighted = 0
            event.stop()
            return

        if event.key == "tab" and inp.value.startswith("/") and " " not in inp.value:
            matches = shell.match_templates(inp.value)
            if matches:
                self._apply_template(matches[0])
                event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        template = shell.TEMPLATES_BY_NAME.get("/" + (event.option.id or ""))
        if template:
            self._apply_template(template)

    def _apply_template(self, template) -> None:
        """Act on a chosen template: run UI actions, or fill the input to edit."""
        palette = self.query_one("#palette", OptionList)
        inp = self.query_one("#prompt", Input)
        palette.display = False
        if template.insert is None:             # /clear or /exit are UI actions
            self._do_action(template.name)
            return
        inp.value = template.insert
        inp.cursor_position = len(template.insert)
        inp.focus()

    # Running commands

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        inp = self.query_one("#prompt", Input)
        inp.value = ""
        self.query_one("#palette", OptionList).display = False
        if not line:
            return

        if line in ("/clear", "/exit"):
            self._do_action(line)
            return

        # A bare slash command typed directly (not expanded) still works: map it
        # to its real command, keeping any arguments the user added.
        if line.startswith("/"):
            head, _, rest = line.partition(" ")
            template = shell.TEMPLATES_BY_NAME.get(head)
            if template is None:
                self.query_one("#output", RichLog).write(
                    Text(f"Unknown shortcut {head}. Type / to see them, or just "
                         f"type a normal command.", style=ui.YELLOW))
                return
            if template.insert is None:
                self._do_action(head)
                return
            line = (template.insert + rest).strip() if rest else template.insert.strip()

        self._start_command(line)

    def _do_action(self, name: str) -> None:
        if name == "/clear":
            self.query_one("#output", RichLog).clear()
        elif name == "/exit":
            self.exit()

    def _start_command(self, cmd: str) -> None:
        if self._proc is not None:
            self.query_one("#output", RichLog).write(
                Text("A command is still running. Press Esc to cancel it first.",
                     style=ui.YELLOW))
            return
        cols = max(80, self.size.width - 8)
        self._run_command(cmd, cols)

    @work(thread=True, exclusive=True, group="cmd")
    def _run_command(self, cmd: str, cols: int) -> None:
        """Run a shell command in the venv, streaming its output into the log.

        Runs on a worker thread so a slow command never freezes the UI. I push
        every line back to the main thread with call_from_thread, and convert
        ANSI colour codes so coloured tools (ovat doctor, pytest) keep their
        colours inside the log.
        """
        log = self.query_one("#output", RichLog)
        self.call_from_thread(log.write, Text(f"$ {cmd}", style="bold #FFFFFF"))
        try:
            proc = shell.spawn(cmd, self._cwd, shell.venv_env(columns=cols))
        except Exception as exc:
            self.call_from_thread(log.write, Text(f"failed to start: {exc}", style=ui.RED))
            return
        self._proc = proc
        try:
            for line in proc.stdout:
                self.call_from_thread(log.write, Text.from_ansi(line.rstrip("\n")))
        finally:
            if proc.stdout:
                proc.stdout.close()
            code = proc.wait()
            self._proc = None
        color = ui.GREEN if code == 0 else ui.RED
        self.call_from_thread(log.write, Text(f"[exit {code}]", style=color))


def run_tui() -> None:
    """Launch the TUI. Called when someone runs `ovat` with no subcommand."""
    OvatTUI().run()
