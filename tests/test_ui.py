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


def test_esc_lets_any_model_output_print(capsys):
    """A model's words are data, so rendering must PRINT them, not parse them."""
    # [/INST] is what Llama/Mistral-family models emit. Unescaped it raised
    # MarkupError and killed `ovat run` with a traceback.
    ui.console.print(ui.esc("Sure![/INST]"))
    # A style tag used to be swallowed silently: the user saw text nobody wrote.
    ui.console.print(ui.esc("Use [bold] tags"))
    out = capsys.readouterr().out
    assert "Sure![/INST]" in out
    assert "Use [bold] tags" in out


def test_esc_round_trips_any_value_not_just_strings(capsys):
    """Exceptions, paths and lists reach the console too, so esc() takes them.

    The contract is what the user SEES, not how it is escaped: rich only reads
    [a-z#/@...] as a tag, so "[1]" was never at risk while "[/INST]" and
    "[bold]" were. Asserting the round-trip covers both without pinning
    rich's tag syntax.
    """
    for value in [ValueError("bad [tag]"), ["notes[1].md"], "[/INST]", "[bold]x"]:
        ui.console.print(ui.esc(value))
    out = capsys.readouterr().out
    assert "bad [tag]" in out
    assert "notes[1].md" in out
    assert "[/INST]" in out
    assert "[bold]x" in out


def test_windows_utf8_setup_is_a_noop_everywhere_else(monkeypatch):
    """It must not touch stdout on macOS/Linux, where the encoding is fine."""
    touched = []

    class Stream:
        def reconfigure(self, **kwargs):
            touched.append(kwargs)

    monkeypatch.setattr(ui.sys, "platform", "darwin")
    monkeypatch.setattr(ui.sys, "stdout", Stream())
    monkeypatch.setattr(ui.sys, "stderr", Stream())
    ui.enable_windows_utf8()
    assert touched == []


def test_windows_utf8_setup_reconfigures_both_streams(monkeypatch):
    touched = []

    class Stream:
        def reconfigure(self, **kwargs):
            touched.append(kwargs)

    monkeypatch.setattr(ui.sys, "platform", "win32")
    monkeypatch.setattr(ui.sys, "stdout", Stream())
    monkeypatch.setattr(ui.sys, "stderr", Stream())
    ui.enable_windows_utf8()          # ctypes.windll is absent here: must not raise
    assert touched == [{"encoding": "utf-8"}, {"encoding": "utf-8"}]


def test_windows_utf8_setup_survives_a_console_that_refuses(monkeypatch):
    """A locked-down console means plainer output, never a CLI that cannot start."""
    class Hostile:
        def reconfigure(self, **kwargs):
            raise OSError("console refused")

    class NoReconfigure:
        pass                          # e.g. a captured or replaced stream

    monkeypatch.setattr(ui.sys, "platform", "win32")
    monkeypatch.setattr(ui.sys, "stdout", Hostile())
    monkeypatch.setattr(ui.sys, "stderr", NoReconfigure())
    ui.enable_windows_utf8()          # must return normally


def test_banner_prints_wordmark_and_subtitle(capsys):
    ui.banner("hello subtitle")
    out = capsys.readouterr().out
    assert "OVAT" in out
    assert "hello subtitle" in out
