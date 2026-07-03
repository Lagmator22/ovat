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
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# The raw brand palette, hex by ROLE — the single source of truth. The rich
# theme below derives from it, and the Textual TUI (separate branch) reads
# these same entries for its widgets, so terminal and TUI can never drift.
PALETTE = {
    "blue":   "#0068B5",   # Intel energy blue, the primary brand colour
    "cyan":   "#00C7FD",   # bright accent for highlights
    "purple": "#8F5CFF",   # the purple half of the requested scheme
    "ok":     "#3DD68C",   # a check passed / success
    "warn":   "#FFC107",   # heads-up, not a failure
    "fail":   "#FF5C5C",   # failed, needs attention
    "dim":    "#7A8CA0",   # secondary text
}

# Named styles derived from the palette. Two kinds of names on purpose:
#   ovat.*  — role names new code should use ("ovat.ok", not a colour).
#   green/red/yellow/dim/cyan — remaps of the GENERIC names the existing
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


def banner(subtitle: str | None = None) -> None:
    """Print the OVAT header panel in the brand colours."""
    title = Text()
    title.append("OVAT", style="ovat.brand")
    title.append("  OpenVINO Agentic Toolkit", style="ovat.blue")
    body = Text("one YAML  +  one command", style="ovat.purple")
    if subtitle:
        body.append(f"\n{subtitle}", style="ovat.dim")
    console.print(Panel(body, title=title, border_style="ovat.blue", expand=False))


def status_text(status: str) -> Text:
    """Build the coloured glyph + label for a check status (ok/warn/fail)."""
    glyph, style = STATUS_GLYPH.get(status, ("?", "ovat.dim"))
    return Text(f"{glyph} {status}", style=style)
