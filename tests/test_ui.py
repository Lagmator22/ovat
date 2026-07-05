# tests/test_ui.py
"""Tests for the shared look and feel (ovat/cli/ui.py).

Note to myself: rendering is hard to eyeball in tests, so I test the CONTRACT:
the theme derives from the one PALETTE dict (single source of truth), the
generic colour names commands already use are remapped into the brand palette,
and the helpers produce the glyphs doctor relies on.
"""
from rich.style import Style

from ovat.cli import ui


def test_theme_styles_derive_from_the_palette():
    # If someone edits a hex in OVAT_THEME directly instead of PALETTE, the
    # single source of truth is broken; this pins the derivation.
    assert ui.OVAT_THEME.styles["ovat.blue"] == Style.parse(ui.PALETTE["blue"])
    assert ui.OVAT_THEME.styles["ovat.ok"] == Style.parse(f"bold {ui.PALETTE['ok']}")


def test_generic_markup_names_are_remapped_to_brand_colours():
    # Commands write "[green]...[/green]"; the theme must turn that into the
    # brand ok-colour, not the terminal's stock green.
    for generic, role in [("green", "ok"), ("yellow", "warn"),
                          ("red", "fail"), ("dim", "dim"), ("cyan", "cyan")]:
        assert ui.OVAT_THEME.styles[generic] == Style.parse(ui.PALETTE[role])


def test_status_text_glyphs():
    assert ui.status_text("ok").plain == "✓ ok"
    assert ui.status_text("warn").plain == "! warn"
    assert ui.status_text("fail").plain == "✗ fail"
    assert ui.status_text("mystery").plain == "? mystery"   # unknown stays calm


def test_banner_prints_wordmark_and_subtitle(capsys):
    ui.banner("hello subtitle")
    out = capsys.readouterr().out
    assert "OVAT" in out
    assert "hello subtitle" in out
