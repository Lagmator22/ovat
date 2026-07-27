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
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Iterable, Sequence

from PIL import Image, ImageSequence, UnidentifiedImageError
from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual import work

from textual.app import App, ComposeResult, RenderResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Footer, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from ovat.cli import shell, ui
from ovat.cli.editing import InputHistory
from ovat.cli.widgets import PasteInput
from ovat.cli.theme import OVAT_THEME

# A dependency-free fallback, used only if pyfiglet is somehow unavailable.
_FALLBACK_WORDMARK = [
    "██████ ██  ██  ████  ██████",
    "██  ██ ██  ██ ██  ██   ██  ",
    "██  ██ ██  ██ ██████   ██  ",
    "██  ██  ████  ██  ██   ██  ",
    "██████   ██   ██  ██   ██  ",
]
# How long a first Ctrl-C stays "armed" for the confirming second press.
# Long enough to read the hint, short enough that an unrelated Ctrl-C minutes
# later is not treated as the second half of a quit.
QUIT_CONFIRM_S = 3.0

_CYAN_RGB = (0, 199, 253)
_BLUE_RGB = (0, 104, 181)
_INTEL_PHASE_SOURCES: Final = (
    Path(__file__).resolve().parent.parent / "assets" / "intel-phase-1.png",
    Path(__file__).resolve().parent.parent / "assets" / "intel-phase-2.png",
    Path(__file__).resolve().parent.parent / "assets" / "intel-phase-3-static.png",
)
_INTEL_GIF_SOURCE: Final = Path(__file__).resolve().parent.parent / "assets" / "intel.gif"
_INTEL_GIF_CROP: Final = (200, 170, 600, 430)


@dataclass(frozen=True)
class _StartupFrame:
    """One cached startup image and the delay before its successor."""

    renderable: Text
    duration: float | None


class _StartupIntelAnimation(Widget):
    """A compact, one-shot three-phase Intel startup mark for the launcher.

    This class intentionally stays in ``tui.py``: it is part of the optional
    Textual launcher, has no process or CLI side effects, and cannot be imported
    by a normal ``ovat`` command. The bundled GIF and three boards are decoded
    once at mount time. Widget-owned timers swap cached Rich renderables, then
    the final board is left untouched for the rest of the TUI session.
    """

    DEFAULT_CSS = """
    _StartupIntelAnimation {
        width: 46;
        height: 15;
        content-align: center middle;
    }
    """
    _frame_cache: ClassVar[dict[tuple[Path, tuple[Path, ...], int, float],
                                tuple[_StartupFrame, ...]]] = {}

    def __init__(self, sources: Sequence[str | Path] = _INTEL_PHASE_SOURCES,
                 *, intro_source: str | Path = _INTEL_GIF_SOURCE, columns: int = 46,
                 phase_seconds: float = 0.7,
                 name: str | None = None, id: str | None = None,
                 classes: str | None = None, disabled: bool = False) -> None:
        if len(sources) != 3:
            raise ValueError("the Intel startup mark needs exactly three phases")
        if columns < 2:
            raise ValueError("columns must be at least 2")
        if phase_seconds <= 0:
            raise ValueError("phase_seconds must be positive")
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._sources = tuple(Path(source) for source in sources)
        self._intro_source = Path(intro_source)
        self._columns = columns
        self._phase_seconds = phase_seconds
        self._frames: tuple[_StartupFrame, ...] = ()
        self._frame_index = 0
        self._timer: Timer | None = None
        self._error: Text | None = None

    @property
    def frame_index(self) -> int:
        """The visible cached GIF or board frame."""
        return self._frame_index

    @property
    def frame_count(self) -> int:
        """Number of cached GIF and board frames, zero if loading failed."""
        return len(self._frames)

    @property
    def is_animating(self) -> bool:
        """Whether another cached frame transition remains scheduled."""
        return self._timer is not None

    def on_mount(self) -> None:
        """Decode and rasterize each source exactly once before scheduling."""
        try:
            cache_key = (self._intro_source, self._sources, self._columns,
                         self._phase_seconds)
            self._frames = self._frame_cache.get(cache_key, ())
            if not self._frames:
                self._frames = self._decode_frames()
                self._frame_cache[cache_key] = self._frames
        except (OSError, UnidentifiedImageError) as error:
            self._error = Text(f"Intel startup mark unavailable: {error}", style="red")
            self.refresh()
            return
        self._frame_index = 0
        self.refresh()
        self._schedule_advance()

    def on_unmount(self) -> None:
        """A removed launcher must not retain a timer or Pillow image state."""
        self._cancel_timer()

    def restart(self) -> None:
        """Replay the GIF and three boards without re-decoding cached frames."""
        if not self._frames:
            return
        self._cancel_timer()
        self._frame_index = 0
        self.refresh()
        self._schedule_advance()

    def render(self) -> RenderResult:
        """Return a cached frame only; this method never opens an image file."""
        if self._frames:
            return self._frames[self._frame_index].renderable
        return self._error or Text("")

    def _schedule_advance(self) -> None:
        if self._frame_index < len(self._frames) - 1:
            duration = self._frames[self._frame_index].duration
            if duration is not None:
                self._timer = self.set_timer(duration, self._advance)

    def _advance(self) -> None:
        self._timer = None
        if self._frame_index >= len(self._frames) - 1:
            return
        self._frame_index += 1
        self.refresh()
        self._schedule_advance()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _decode_frames(self) -> tuple[_StartupFrame, ...]:
        """Decode the original GIF, then append the three requested boards."""
        frames: list[_StartupFrame] = []
        with Image.open(self._intro_source) as image:
            default_duration_ms = image.info.get("duration", 100)
            for raw_frame in ImageSequence.Iterator(image):
                duration_ms = raw_frame.info.get("duration", default_duration_ms)
                frames.append(_StartupFrame(
                    self._to_braille_renderable(raw_frame.crop(_INTEL_GIF_CROP)),
                    max(float(duration_ms) / 1000, 0.001),
                ))
        for source in self._sources[:-1]:
            with Image.open(source) as image:
                frames.append(_StartupFrame(
                    self._to_braille_renderable(image), self._phase_seconds
                ))
        with Image.open(self._sources[-1]) as image:
            # The non-dotted quadrant glyphs give the static Intel "e" a clean
            # opening instead of turning its white stroke into Braille specks.
            frames.append(_StartupFrame(self._to_quadrant_renderable(image), None))
        return tuple(frames)

    def _to_braille_renderable(self, image: Image.Image) -> Text:
        """Resample one source into a high-density two-colour Braille image.

        A terminal cell has a foreground and background colour. Choosing the
        nearest of two per-cell palette colours preserves the Intel artwork's
        blue and white details much better than a global-background threshold.
        """
        rows = max(1, round(self._columns * image.height / image.width / 2))
        scaled = image.convert("RGB").resize(
            (self._columns * 2, rows * 4), Image.Resampling.LANCZOS
        )
        pixels = list(scaled.get_flattened_data())

        text = Text(no_wrap=True)
        for row in range(rows):
            for column in range(self._columns):
                cell = self._cell_pixels(pixels, row, column, vertical_pixels=4)
                dark, light, mask = self._two_colour_mask(
                    cell, (0x01, 0x08, 0x02, 0x10, 0x04, 0x20, 0x40, 0x80)
                )
                text.append(chr(0x2800 + mask), Style(
                    color=Color.from_rgb(*light), bgcolor=Color.from_rgb(*dark)
                ))
            if row + 1 < rows:
                text.append("\n")
        return text

    def _to_quadrant_renderable(self, image: Image.Image) -> Text:
        """Render the settled board with contiguous 2×2 blocks, not dots."""
        rows = max(1, round(self._columns * image.height / image.width / 2))
        scaled = image.convert("RGB").resize(
            (self._columns * 2, rows * 2), Image.Resampling.LANCZOS
        )
        pixels = list(scaled.get_flattened_data())
        glyphs = (" ", "▘", "▝", "▀", "▖", "▌", "▞", "▛",
                  "▗", "▚", "▐", "▜", "▄", "▙", "▟", "█")
        text = Text(no_wrap=True)
        for row in range(rows):
            for column in range(self._columns):
                cell = self._cell_pixels(pixels, row, column, vertical_pixels=2)
                dark, light, mask = self._two_colour_mask(cell, (0x01, 0x02, 0x04, 0x08))
                text.append(glyphs[mask], Style(
                    color=Color.from_rgb(*light), bgcolor=Color.from_rgb(*dark)
                ))
            if row + 1 < rows:
                text.append("\n")
        return text

    def _cell_pixels(self, pixels: list[tuple[int, int, int]], row: int, column: int,
                     *, vertical_pixels: int) -> list[tuple[int, int, int]]:
        """Return the source pixels corresponding to one terminal cell."""
        return [pixels[(row * vertical_pixels + y) * self._columns * 2 + column * 2 + x]
                for y in range(vertical_pixels) for x in range(2)]

    @classmethod
    def _two_colour_mask(cls, pixels: list[tuple[int, int, int]], bits: Sequence[int]) -> tuple[
            tuple[int, int, int], tuple[int, int, int], int]:
        """Cluster a cell into its two representative colours and dot mask."""
        dark, light = cls._cell_palette(pixels)
        mask = 0
        for pixel, bit in zip(pixels, bits):
            if cls._distance(pixel, light) < cls._distance(pixel, dark):
                mask |= bit
        return dark, light, mask

    @classmethod
    def _cell_palette(cls, pixels: list[tuple[int, int, int]]) -> tuple[
            tuple[int, int, int], tuple[int, int, int]]:
        """Use two small k-means passes instead of picking colour outliers."""
        luminance = lambda pixel: 299 * pixel[0] + 587 * pixel[1] + 114 * pixel[2]
        dark, light = min(pixels, key=luminance), max(pixels, key=luminance)
        for _ in range(2):
            dark_group, light_group = [], []
            for pixel in pixels:
                (light_group if cls._distance(pixel, light) < cls._distance(pixel, dark)
                 else dark_group).append(pixel)
            if not dark_group or not light_group:
                break
            dark = tuple(sum(pixel[channel] for pixel in dark_group) // len(dark_group)
                         for channel in range(3))
            light = tuple(sum(pixel[channel] for pixel in light_group) // len(light_group)
                          for channel in range(3))
        return (dark, light) if luminance(dark) <= luminance(light) else (light, dark)

    @staticmethod
    def _distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> int:
        return sum((first[channel] - second[channel]) ** 2 for channel in range(3))


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


def _small_figlet_text(label: str, *, style: str, font: str = "mini") -> Text:
    """Render a compact FIGlet label, or readable text if FIGlet is absent."""
    try:
        import pyfiglet
        lines = pyfiglet.figlet_format(label, font=font).rstrip("\n").split("\n")
        if lines:
            out = Text()
            for line in lines:
                out.append(line + "\n", style=style)
            return out
    except Exception:
        pass
    return Text(label + "\n", style=f"bold {style}")


def _wordmark_text() -> Text:
    """The wordmark as Rich Text, shaded top-to-bottom from cyan to Intel blue."""
    lines = _wordmark_lines()
    text = Text()
    last = max(len(lines) - 1, 1)
    for i, line in enumerate(lines):
        text.append(line + "\n", style=f"bold {_lerp_hex(_CYAN_RGB, _BLUE_RGB, i / last)}")
    return text


def _banner() -> Text:
    """Header: the OVAT wordmark and a compact OpenVINO / OVMS lockup."""
    out = _wordmark_text()
    out.append_text(_small_figlet_text("Powered by", style=ui.PURPLE, font="term"))
    out.append_text(_small_figlet_text("OpenVINO + OVMS", style=ui.BLUE))
    return out


def _startup_updates() -> Text:
    """A deliberately empty board, ready for future toolkit updates."""
    return _small_figlet_text("Updates", style=ui.PURPLE)


def _option(template) -> Option:
    """Build one dropdown row. Styling is shared with the doctor screen's menu
    via ui.slash_label, so the two menus cannot drift apart."""
    # ids cannot contain a slash cleanly, so I key options by the bare name.
    return Option(ui.slash_label(template.name, template.description),
                  id=template.name.lstrip("/"))


class OvatTUI(App):
    """The OVAT launcher: a styled terminal that knows OVAT's commands."""

    TITLE = "OVAT"

    # Textual's default Ctrl-C quits the whole app instantly; mid-command
    # that throws away a run the user only wanted to interrupt. priority=True
    # wins over the built-in binding; the action decides: busy -> cancel the
    # child (terminal muscle memory), idle -> quit like before.
    #
    # alt+p is a SECOND way into the palette, not a replacement: Textual's own
    # COMMAND_PALETTE_BINDING stays ctrl+p, which is the one that always works.
    # Alt bindings depend on the terminal sending a real Meta key, and macOS
    # Terminal composes Option+p into "π" unless "Use Option as Meta" is on, so
    # ctrl+p remains the documented one.
    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", show=False, priority=True),
        # No ctrl+p binding here. Textual already provides one via
        # COMMAND_PALETTE_BINDING and the Footer shows it, so declaring our
        # own listed "^p Palette" TWICE, once at each end of the footer.
    ]

    def get_system_commands(self, screen) -> Iterable[SystemCommand]:
        """App-wide palette entries, on top of Textual's own.

        Only things that make sense from anywhere. Screen-specific commands
        live on the screens themselves (Screen.COMMANDS), so the palette shows
        "Copy the last answer" while you are in a chat and not before.

        Nothing here re-implements a built-in: Textual already provides Theme,
        Screenshot, Quit and Keys, and an earlier version of this shipped its
        own worse copies of the first two.
        """
        yield from super().get_system_commands(screen)
        yield SystemCommand("Doctor",
                            "run the environment and workflow checks",
                            lambda: self._open_doctor(""))
        yield SystemCommand("Chat",
                            "chat with your indexed documents",
                            lambda: self._open_chat(""))
        # Textual's own Theme command switches; this one comes back. Several
        # of the 21 built-in themes fight OVAT's own greens and blues, and
        # finding "ovat" again in an alphabetical list of 22 is a chore.
        if self.theme != OVAT_THEME.name:
            yield SystemCommand("Reset theme",
                                "back to OVAT's own colours",
                                self.action_reset_theme)
        if self.theme != "ansi-dark":
            # A real "no theme": ansi-dark maps everything onto the sixteen
            # colours your terminal already defines, so OVAT inherits your
            # terminal's scheme instead of imposing one. The right answer for
            # anyone who has already themed their terminal and wants OVAT to
            # stop arguing with it.
            yield SystemCommand("No theme",
                                "use your terminal's own colours",
                                self.action_no_theme)

    def action_no_theme(self) -> None:
        self.theme = "ansi-dark"

    def action_reset_theme(self) -> None:
        self.theme = OVAT_THEME.name

    def action_cancel_or_quit(self) -> None:
        """Busy: cancel the child. Idle: quit, but only on a SECOND press.

        One stray Ctrl-C used to close the whole app, which in a terminal is
        the keystroke people hit reflexively to interrupt something. Now the
        first press says what a second one will do, and the intent has to be
        repeated within QUIT_CONFIRM_S to count.
        """
        if self._busy:
            self._cancel_running()
            return
        now = time.monotonic()
        if now - self._last_quit_press < QUIT_CONFIRM_S:
            self.exit()
            return
        self._last_quit_press = now
        # notify(), not a write to #output. App.query_one searches the whole
        # DOM, so writing to the launcher's log SUCCEEDED from the chat and
        # doctor screens and put the warning somewhere the user could not see:
        # first press looked like nothing happened, second press quit. A toast
        # appears on whatever screen is actually in front.
        self.notify("Press Ctrl-C again to quit  ·  /exit works too",
                    severity="warning", timeout=QUIT_CONFIRM_S)

    CSS = """
    Screen { background: $background; }
    #masthead {
        height: 17;
        margin: 1 2 0 2;
        border: round $primary;
        background: $surface;
    }
    #brand-panel {
        width: 32%;
        height: 1fr;
        padding: 0 1;
        border-right: tall $primary;
        content-align: left middle;
    }
    #updates-panel {
        width: 1fr;
        height: 1fr;
        padding: 1 1;
        border-right: tall $primary;
        content-align: left top;
    }
    #intel-panel {
        width: 29%;
        height: 1fr;
        padding: 0;
        content-align: center middle;
    }
    #output {
        height: 1fr;
        border: round $primary;
        background: $surface;
        padding: 0 1;
        margin: 1 2 0 2;
    }
    #palette {
        display: none;
        height: auto;
        max-height: 9;
        margin: 0 2;
        border: round $secondary;
        background: $surface;
    }
    /* The pointer says "this is clickable" before you click it. */
    #palette > .option-list--option-hover { background: $primary 30%; }
    #palette > .option-list--option-highlighted {
        background: $primary;
        color: $foreground;
        text-style: bold;
    }
    #prompt {
        margin: 0 2 1 2;
        border: round $primary;
    }
    """

    def __init__(self):
        super().__init__()
        # Commands run from where the user launched the TUI, so relative paths
        # (./docs, examples/workflow.yml) mean what they expect.
        self._cwd = os.getcwd()
        self._proc = None        # the currently running subprocess, if any
        # The submission gate. Flipped on the MAIN thread BEFORE the worker is
        # scheduled, so a double-Enter can never pass the guard; _proc alone
        # could not do that job, because the worker assigns it only after
        # spawn() returns (a race window the old code had).
        self._busy = False
        self._last_quit_press = 0.0    # when Ctrl-C was last pressed while idle
        self._history = InputHistory() # Up/Down recall, like the shell it wraps

    def compose(self) -> ComposeResult:
        with Horizontal(id="masthead"):
            yield Static(_banner(), id="brand-panel")
            yield Static(_startup_updates(), id="updates-panel")
            with Vertical(id="intel-panel"):
                yield _StartupIntelAnimation(id="intel-animation")
        output = RichLog(id="output", highlight=False, markup=False, wrap=True)
        output.border_title = "output"
        output.border_subtitle = "Ctrl-P palette · / shortcuts"
        yield output
        yield OptionList(id="palette")
        yield PasteInput(placeholder="Run any command (e.g. ovat doctor)"
                                     "  ·  type / for shortcuts", id="prompt")
        # Textual's Footer lists the active bindings, including ^p for the
        # command palette. Without it the palette existed but nothing on
        # screen ever said so.
        yield Footer()

    def on_mount(self) -> None:
        # Register the brand palette AS a Textual theme and select it, so every
        # widget can be written against $primary/$success and still come out in
        # OVAT colours, while a theme picked from the command palette actually
        # recolours them.
        self.register_theme(OVAT_THEME)
        self.theme = OVAT_THEME.name
        log = self.query_one("#output", RichLog)
        log.write(Text("Type any command and press Enter; it runs in the venv.",
                       style=ui.CYAN))
        log.write(Text("Type /  for OVAT shortcuts  ·  Ctrl-P  for the command "
                       "palette  ·  Esc cancels a running command.",
                       style=ui.DIM))
        self.query_one("#prompt", Input).focus()
        # The wordmark fades up rather than snapping in. 250ms: slow enough
        # to register as an entrance, fast enough not to delay the prompt,
        # which is focused and usable throughout.
        banner = self.query_one("#brand-panel", Static)
        banner.styles.opacity = 0.0
        banner.styles.animate("opacity", value=1.0, duration=0.25)

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
        # This handler belongs to the LAUNCHER. When another screen is pushed
        # (the chat screen), its widgets are not in the active screen and the
        # queries below would raise NoMatches on every keypress; breaking the
        # pushed screen's own key handling. Stand down; the screen owns keys.
        if len(self.screen_stack) > 1:
            return
        inp = self.query_one("#prompt", Input)
        palette = self.query_one("#palette", OptionList)

        if event.key == "escape":
            if self._busy:                      # cancel a running command first
                self._cancel_running()
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

        # This is a shell front-end, so Up/Down recall what you ran, exactly
        # as they would at the prompt underneath it.
        if event.key in ("up", "down") and not palette.display \
                and self.focused is inp:
            recalled = (self._history.previous(inp.value) if event.key == "up"
                        else self._history.next())
            if recalled is not None:
                inp.value = recalled
                self.call_after_refresh(setattr, inp, "cursor_position",
                                        len(recalled))
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
        inp.focus()
        # Land the cursor where the next keystrokes belong (e.g. the config
        # path gap in `ovat run | --input ""`), not blindly at the end.
        # call_after_refresh because Input moves its own cursor to the end
        # asynchronously after a programmatic value change; a synchronous
        # assignment here would be overwritten a moment later.
        target = (template.cursor if template.cursor is not None
                  else len(template.insert))
        self.call_after_refresh(setattr, inp, "cursor_position", target)

    # Running commands

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        inp = self.query_one("#prompt", Input)
        inp.value = ""
        self.query_one("#palette", OptionList).display = False
        if not line:
            return

        self._history.add(line)

        if line in ("/clear", "/exit"):
            self._do_action(line)
            return

        # A bare slash command typed directly (not expanded) still works: map it
        # to its real command, keeping any arguments the user added.
        if line.startswith("/"):
            head, _, rest = line.partition(" ")
            template = shell.TEMPLATES_BY_NAME.get(head)
            if template is not None:
                if template.insert is None:
                    self._do_action(head, rest.strip())
                    return
                line = ((template.insert + rest).strip() if rest
                        else template.insert.strip())
            elif head.count("/") == 1:
                # One slash and no others: "/doctr" is a shortcut they typo'd,
                # so say so. More slashes means an absolute PATH like
                # /usr/bin/env or /opt/homebrew/bin/foo, and this promised to
                # run any command you type; it used to swallow every one of
                # them as an unknown shortcut instead. Fall through and run it.
                self.query_one("#output", RichLog).write(
                    Text(f"Unknown shortcut {head}. Type / to see them, or just "
                         f"type a normal command.", style=ui.YELLOW))
                return

        if self._maybe_cd(line):
            return
        self._start_command(line)

    def _maybe_cd(self, line: str) -> bool:
        """Handle `cd` myself so the directory persists between commands.

        Each command runs in its own subprocess, so a `cd` inside one would not
        affect the next. I intercept `cd <dir>`, update the directory I launch
        commands from, and report it, which makes the TUI feel like a real shell.
        """
        parts = line.split(maxsplit=1)
        if parts[0] != "cd":
            return False
        target = os.path.expanduser(parts[1].strip()) if len(parts) > 1 else os.path.expanduser("~")
        if not os.path.isabs(target):
            target = os.path.join(self._cwd, target)
        target = os.path.normpath(target)
        log = self.query_one("#output", RichLog)
        if os.path.isdir(target):
            self._cwd = target
            log.write(Text(f"cwd → {target}", style=ui.CYAN))
        else:
            log.write(Text(f"cd: no such directory: {target}", style=ui.RED))
        return True

    def _do_action(self, name: str, args: str = "") -> None:
        if name == "/clear":
            self.query_one("#output", RichLog).clear()
        elif name == "/exit":
            self.exit()
        elif name == "/chat":
            self._open_chat(args)
        elif name == "/doctor":
            self._open_doctor(args)

    def _open_doctor(self, args: str) -> None:
        """Push the in-app doctor screen: `/doctor [config]`.

        Imported here, not at module scope, for the same reason as the chat
        screen: this module must stay importable without the [tui] extra.
        """
        from ovat.cli.doctor_screen import DoctorScreen

        if isinstance(self.screen, DoctorScreen):
            # The palette works from every screen, so "Doctor" while already
            # on it used to stack a second copy that Esc had to unwind twice.
            self.screen.action_refresh_checks()
            return
        config = args.strip()
        if config and not os.path.isabs(config):
            config = os.path.normpath(os.path.join(self._cwd, config))
        self.push_screen(DoctorScreen(config_path=config or None))

    def _open_chat(self, args: str) -> None:
        """Push the native chat screen: `/chat [config] [model-path]`.

        Both arguments are remembered in .ovat/chat_prefs.json after the first
        successful load, so from then on a bare `/chat` just works. Relative
        paths resolve against the TUI's cd-tracked working directory.
        """
        from ovat.cli.chat_screen import ChatScreen, load_prefs

        if isinstance(self.screen, ChatScreen):
            # Same as the doctor screen, but this one also costs a second
            # model load (~30s and the memory to go with it).
            self.notify("Already in a chat.")
            return
        log = self.query_one("#output", RichLog)
        prefs = load_prefs(self._cwd)
        parts = args.split()
        config = parts[0] if parts else prefs.get("config", "workflow.yml")
        model = parts[1] if len(parts) > 1 else prefs.get("model_path")
        if not model:
            # Friendliest path last: AUTO-DETECT a local text LLM instead of
            # demanding a path a new user does not know.
            from ovat.core.model_scout import find_models, pick_chat_llm
            choice, llms = pick_chat_llm()
            if choice is not None:
                model = choice["path"]
                log.write(Text(f"auto-detected local LLM: {choice['name']}  "
                               f"({choice['path']})", style=ui.CYAN))
                if len(llms) > 1:
                    log.write(Text("also available: " +
                                   ", ".join(m["name"] for m in llms
                                             if m is not choice) +
                                   "  ·  choose with /chat <config> <path>",
                                   style=ui.DIM))
            else:
                log.write(Text(
                    "No local text LLM found (scanned OVAT_MODELS, ./models, "
                    "~/models).", style=ui.YELLOW))
                others = find_models()
                if others:
                    log.write(Text("Found, but wrong kind for chat: " +
                                   ", ".join(f"{m['name']} ({m['kind']})"
                                             for m in others), style=ui.DIM))
                log.write(Text(
                    "Fix: /chat [config] [model-path], or set OVAT_MODELS to "
                    "your models folder. I remember your choice afterwards.",
                    style=ui.DIM))
                return

        def _resolve(p: str) -> str:
            return p if os.path.isabs(p) else os.path.normpath(
                os.path.join(self._cwd, p))

        self.push_screen(ChatScreen(_resolve(config), _resolve(model),
                                    cwd=self._cwd))

    def _start_command(self, cmd: str) -> None:
        # Gate on _busy, which only the main thread flips, BEFORE scheduling
        # the worker. The old check on _proc raced: the worker assigned it
        # only after spawn(), so a fast double-Enter started two processes;
        # and exclusive=True then cancelled the first worker THREAD while its
        # OS process kept running, orphaned, with the handle overwritten.
        if self._busy:
            self.query_one("#output", RichLog).write(
                Text("A command is still running. Press Esc to cancel it first.",
                     style=ui.YELLOW))
            return
        self._busy = True
        # The placeholder doubles as the status line: while a command runs
        # (serve can sit silent for two minutes) the empty prompt says so.
        self.query_one("#prompt", Input).placeholder = \
            "running…  Esc cancels  ·  output streams above"
        # Width the CHILD is told to render at. Taken from the log's own
        # content box minus the scrollbar, not from the app width: once
        # output overflows, Textual's scrollbar OVERLAYS the last two
        # columns, so a rich Panel drawn to the full width had its right
        # edge and bottom corner swallowed and the box came apart. Reserving
        # for the scrollbar up front means boxes fit whether it is there or
        # not, at the cost of two unused columns before it appears.
        log = self.query_one("#output", RichLog)
        usable = log.content_size.width - log.scrollbar_size_vertical
        cols = max(40, usable)
        self._run_command(cmd, cols)

    def _cancel_running(self) -> None:
        """Cancel the running command: ask politely, then force.

        Local ref first: the worker nulls self._proc from its thread, so
        reading it twice could hit None mid-flight. Escalation runs on a tiny
        daemon thread because the main thread must never block on wait().
        """
        proc = self._proc
        if proc is None:
            return
        self.query_one("#output", RichLog).write(
            Text("cancelling…", style=ui.YELLOW))
        try:
            proc.terminate()                    # SIGTERM: a request
        except ProcessLookupError:
            return                              # already gone; worker cleans up

        def _force():
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()                     # SIGKILL: not a request

        threading.Thread(target=_force, daemon=True).start()

    def _mark_idle(self) -> None:
        self._busy = False
        self.query_one("#prompt", Input).placeholder = \
            "Run any command (e.g. ovat doctor)  ·  type / for shortcuts"

    @work(thread=True, group="cmd")
    def _run_command(self, cmd: str, cols: int) -> None:
        """Run a shell command in the venv, streaming its output into the log.

        Runs on a worker thread so a slow command never freezes the UI. I push
        every line back to the main thread with call_from_thread, and convert
        ANSI colour codes so coloured tools (ovat doctor, pytest) keep their
        colours inside the log. No exclusive=True here on purpose: cancelling
        a worker thread does not kill its subprocess, so exclusivity is
        enforced by the _busy gate instead.
        """
        log = self.query_one("#output", RichLog)
        self.call_from_thread(log.write, Text(f"$ {cmd}", style=f"bold {ui.CYAN}"))
        try:
            try:
                proc = shell.spawn(cmd, self._cwd, shell.venv_env(columns=cols))
            except Exception as exc:
                self.call_from_thread(log.write,
                                      Text(f"failed to start: {exc}", style=ui.RED))
                return
            self._proc = proc
            try:
                # iter_display_lines also tames \r progress bars (pip, tqdm)
                # that would otherwise stall then flood the append-only log.
                for line in shell.iter_display_lines(proc.stdout):
                    self.call_from_thread(log.write, Text.from_ansi(line))
            finally:
                if proc.stdout:
                    proc.stdout.close()
                code = proc.wait()
                self._proc = None
            color = ui.GREEN if code == 0 else ui.RED
            self.call_from_thread(log.write, Text(f"[exit {code}]", style=color))
        finally:
            # Always reopen the gate, even when spawn failed; otherwise one
            # typo'd command would lock the TUI forever.
            self.call_from_thread(self._mark_idle)


def run_tui() -> None:
    """Launch the TUI. Called when someone runs `ovat` with no subcommand.

    Refuses when stdout is not a real terminal: a full-screen app in a pipe
    (`ovat | tee`, a subprocess, CI) can only print escape-code garbage and
    then hang waiting for a terminal that is not there. This is the root-cause
    guard; the OVAT_TUI env check in the CLI is the friendly early layer.
    """
    import sys
    if not sys.stdout.isatty():
        print("ovat: the TUI needs a real terminal (stdout is not a TTY). "
              "Use a subcommand instead, e.g. `ovat doctor` or `ovat --help`.")
        return
    OvatTUI().run()
