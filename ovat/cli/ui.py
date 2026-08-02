# ovat/cli/ui.py
"""Shared look and feel for the OVAT command line.

One Console, one Theme, one banner, used by every command so the toolkit feels
like a single product instead of a pile of scripts. The palette leans on Intel
blue with a purple accent, which is also what makes `ovat doctor` read at a
glance: green is fine, yellow is a heads-up, red needs fixing.

Nothing here is decorative-only. The styles below are referenced by name from
the real command output, so changing a colour in one place restyles the whole
CLI.
"""
import re
import sys

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme


def enable_windows_utf8() -> None:
    """Let a Windows console print OVAT's output instead of crashing on it.

    Almost everything OVAT prints is non-ASCII: doctor's table borders, the
    status glyphs (✓ ! ✗), the FIGlet wordmark, the → in the routing row, the
    · separators. A Windows console still defaults to a legacy code page
    (cp1252 in most locales), and encoding those characters for it raises
    UnicodeEncodeError, which kills the command part-way through a table.

    Both halves are needed and they do different jobs:
      * reconfigure() makes PYTHON encode its output as UTF-8;
      * SetConsoleOutputCP(65001) makes the CONSOLE decode it as UTF-8.
    With only the first, python stops raising but the console renders mojibake;
    with only the second, python still refuses to encode.

    Deliberately NOT done here, both of which appear in the usual recipe:
    SetConsoleCP, which changes the INPUT code page and belongs to whatever
    else is reading that console; and os.system(""), the trick for switching on
    VT sequences, which spawns an entire shell as an import side effect. rich
    already enables VT on Windows itself.

    Never raises. A console we cannot reconfigure is a reason for plainer
    output, not a reason for the whole CLI to fail to start.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            # No reconfigure (a captured or replaced stream), or the console
            # refused. Either way rich falls back to what it can render.
            pass


# Before the Console below is built, so its first write already has a console
# that can accept the glyphs.
enable_windows_utf8()

# The raw brand palette, hex by ROLE: the single source of truth. The rich
# theme below derives from it, and the Textual TUI reads the aliases further
# down for its widgets, so terminal and TUI can never drift.
PALETTE = {
    "blue":   "#0068B5",   # Intel energy blue, the primary brand colour
    "cyan":   "#00C7FD",   # bright accent for highlights
    "purple": "#8F5CFF",   # the purple half of the requested scheme
    "ok":     "#3DD68C",   # a check passed / success
    "warn":   "#FFC107",   # heads-up, not a failure
    "fail":   "#FF5C5C",   # failed, needs attention
    "dim":    "#7A8CA0",   # secondary text
}

# Plain-hex aliases for the Textual TUI. Its widgets and CSS cannot read rich
# theme names, only raw colours; but these are DERIVED, never redefined, so
# there is still exactly one palette.
BLUE = PALETTE["blue"]
CYAN = PALETTE["cyan"]
PURPLE = PALETTE["purple"]
GREEN = PALETTE["ok"]
YELLOW = PALETTE["warn"]
RED = PALETTE["fail"]
DIM = PALETTE["dim"]

# Named styles derived from the palette. Two kinds of names on purpose:
#   ovat.*  : role names new code should use ("ovat.ok", not a colour).
#   green/red/yellow/dim/cyan: remaps of the GENERIC names the existing
#   commands already use in markup like "[green]Indexed[/green]". Remapping
#   them here restyles every command into the brand palette at once, through
#   the one shared console, without editing every message string.
OVAT_THEME = Theme({
    "ovat.blue": PALETTE["blue"],
    "ovat.cyan": PALETTE["cyan"],
    "ovat.purple": PALETTE["purple"],
    "ovat.ok": f"bold {PALETTE['ok']}",
    "ovat.warn": f"bold {PALETTE['warn']}",
    "ovat.fail": f"bold {PALETTE['fail']}",
    "ovat.dim": PALETTE["dim"],
    # rich cannot combine "bold" with a theme name in one string, so where I
    # want a bold brand colour I bake the weight into its own named style.
    "ovat.brand": f"bold {PALETTE['cyan']}",   # the OVAT wordmark
    "ovat.header": f"bold {PALETTE['blue']}",  # table headers
    "green": PALETTE["ok"],
    "yellow": PALETTE["warn"],
    "red": PALETTE["fail"],
    "dim": PALETTE["dim"],
    "cyan": PALETTE["cyan"],
})

# A single shared console. Importing this everywhere keeps styling consistent.
console = Console(theme=OVAT_THEME)

# The glyph + style I show per check status, kept in one place so doctor and any
# future status output agree on what "ok" looks like.
STATUS_GLYPH = {
    "ok": ("✓", "ovat.ok"),
    "warn": ("!", "ovat.warn"),
    "fail": ("✗", "ovat.fail"),
}


def esc(value) -> str:
    """Neutralise rich markup in a value OVAT did not write. Use it on ALL of them.

    rich reads [something] in a printed string as a style tag, exactly the way
    printf reads % in its format string. So a value must never BE the format
    string. In C++ terms: this is printf("%s", data), and rprint(f"...{data}")
    without it is printf(data), with the same class of consequence.

    Two things went wrong before this existed, both on ordinary output:
      * a Llama-family model ending its answer with [/INST] crashed `ovat run`
        with a MarkupError traceback (breaking the "errors are for users" rule),
      * a model writing [bold] had it silently swallowed as a style tag, so the
        user was shown text the model never wrote.

    Model answers, exception text, file paths, model folder names, tool names
    imported from an MCP server and `ovms` subprocess output are all DATA. They
    come through here on the way to the console. Escaping happens at the output
    edge, not at the source, so the checks and providers stay free to return
    plain strings.
    """
    return escape(str(value))


def _hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _lerp_hex(a: tuple, b: tuple, t: float) -> str:
    """Blend two RGB colours by fraction t in 0..1 and return a hex string."""
    return "#{:02X}{:02X}{:02X}".format(
        *(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    )


def wordmark(text: str = "OVAT") -> Text:
    """A big FIGlet wordmark, shaded top-to-bottom cyan → Intel blue.

    Shared by every front-end that wants the 'big sign' look (the TUI
    banner, `ovat doctor`). pyfiglet ships with the [tui] extra; without it
    this degrades gracefully to the plain bold brand style instead of
    crashing a CLI-only install.
    """
    lines: list = []
    try:
        import pyfiglet
        lines = [ln for ln in
                 pyfiglet.figlet_format(text, font="ansi_shadow").split("\n")
                 if ln.strip()]
    except Exception:
        pass
    out = Text()
    if not lines:
        out.append(text, style="ovat.brand")
        return out
    top, bottom = _hex_to_rgb(PALETTE["cyan"]), _hex_to_rgb(PALETTE["blue"])
    last = max(len(lines) - 1, 1)
    for i, line in enumerate(lines):
        out.append(line + "\n",
                   style=f"bold {_lerp_hex(top, bottom, i / last)}")
    return out


def banner(subtitle: str | None = None) -> None:
    """Print the OVAT header panel in the brand colours."""
    title = Text()
    title.append("OVAT", style="ovat.brand")
    title.append("  OpenVINO Agentic Toolkit", style="ovat.blue")
    body = Text("one YAML  +  one command", style="ovat.purple")
    if subtitle:
        body.append(f"\n{subtitle}", style="ovat.dim")
    console.print(Panel(body, title=title, border_style="ovat.blue", expand=False))


def slash_label(name: str, description: str, width: int = 9) -> Text:
    """One row of a slash-command dropdown: the name in purple, help in grey.

    Shared by every screen that offers a "/" menu (the launcher and the doctor
    screen) so the menus cannot drift apart in look. Raw hex, not theme names:
    these render inside Textual widgets, against the app's console rather than
    the CLI's themed one, where a name like "ovat.purple" means nothing.
    """
    label = Text()
    label.append(f"{name:<{width}} ", style=f"bold {PURPLE}")
    label.append(description, style=DIM)
    return label


def status_text(status: str) -> Text:
    """Build the coloured glyph + label for a check status (ok/warn/fail)."""
    glyph, style = STATUS_GLYPH.get(status, ("?", "ovat.dim"))
    return Text(f"{glyph} {status}", style=style)


# Reasoning models narrate before they answer. Qwen3 and the DeepSeek-R1
# family wrap that narration in <think>…</think>; some exports spell it
# <thinking>. Both spellings, any case, across newlines.
#
# This lives HERE, not in chat_screen.py where it started, because the plain
# CLI needs it too and chat_screen imports textual. Moving it kept the
# isolation contract intact: `ovat run` must work with no TUI installed.
_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>",
                          re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop the reasoning, leaving the answer.

    Interesting once, clutter every time after: a single Qwen3 answer can
    spend fifteen lines deciding what the question meant before saying
    anything. This affects DISPLAY only; the session keeps the raw text, so
    /thinking on can bring it back and /save never loses it.

    Three shapes, because models produce all three:

    1. A complete <think>…</think> block.
    2. An UNCLOSED <think>: generation stopped mid-thought (Esc, or the token
       cap ran out). Everything from the opening tag on is reasoning.
    3. A DANGLING </think> with no opening tag. Qwen3's chat template inserts
       the opening tag itself, so the model only ever emits the terminator --
       which meant every `ovat run` answer on the AI PC ended up carrying a
       stray </think>. Everything BEFORE it is reasoning, so it goes too.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    match = _THINK_OPEN.search(cleaned)
    if match:
        cleaned = cleaned[:match.start()]
    # Case 3 runs last, so a well-formed block has already been removed and
    # cannot be mistaken for a dangling terminator.
    closing = _THINK_CLOSE.search(cleaned)
    if closing:
        cleaned = cleaned[closing.end():]
    return cleaned.strip()
