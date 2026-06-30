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

# Intel-inspired palette. I name the styles by role, not by colour, so a command
# asks for "ovat.ok" and never hardcodes a hex value.
OVAT_THEME = Theme({
    "ovat.blue": "#0068B5",        # Intel energy blue, the primary brand colour
    "ovat.cyan": "#00C7FD",        # bright accent for highlights
    "ovat.purple": "#8F5CFF",      # the purple half of the requested scheme
    "ovat.ok": "bold #3DD68C",     # a check passed
    "ovat.warn": "bold #FFC107",   # a check is a heads-up, not a failure
    "ovat.fail": "bold #FF5C5C",   # a check failed and needs attention
    "ovat.dim": "#7A8CA0",         # secondary text
    # rich cannot combine "bold" with a theme name in one string, so where I
    # want a bold brand colour I bake the weight into its own named style.
    "ovat.brand": "bold #00C7FD",  # the OVAT wordmark
    "ovat.header": "bold #0068B5", # table headers
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
