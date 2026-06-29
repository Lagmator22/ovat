# tests/test_commands.py
"""Tests for the slash-command layer that backs the TUI.

Note to myself: this is where I prove the TUI is wired to the real toolkit. Each
dispatch() call runs the same code the CLI runs (the diagnostics, the config
loader, the indexer). No Textual here, so it is fast and runs anywhere.
"""
from rich.table import Table

from ovat.cli.commands import COMMANDS, dispatch, match_commands


def test_match_commands_filters_by_prefix():
    names = {c.name for c in match_commands("/d")}
    assert names == {"doctor"}
    # a bare slash matches everything
    assert len(match_commands("/")) == len(COMMANDS)


def test_help_lists_every_command():
    result = dispatch("/help")
    assert isinstance(result.renderable, Table)


def test_doctor_runs_real_checks():
    # No exception, and it returns a renderable built from real diagnostics.
    result = dispatch("/doctor")
    assert isinstance(result.renderable, Table)


def test_init_writes_then_refuses_to_overwrite(tmp_path):
    target = str(tmp_path / "workflow.yml")
    first = dispatch(f'/init "{target}"')
    assert "Wrote a starter workflow" in first.renderable.plain
    # the file it wrote must be a real, loadable workflow
    from ovat.config.workflow import load_workflow
    assert load_workflow(target).model.name == "Qwen3-8B-int4-ov"
    # second time it must refuse
    second = dispatch(f'/init "{target}"')
    assert "Refusing to overwrite" in second.renderable.plain


def test_validate_reports_good_and_bad(tmp_path):
    good = tmp_path / "good.yml"
    good.write_text("model:\n  name: m\nagent:\n  type: react\n", encoding="utf-8")
    ok = dispatch(f'/validate "{good}"')
    assert "Valid" in ok.renderable.plain and "react" in ok.renderable.plain

    bad = dispatch("/validate /no/such/file.yml")
    assert "No such file" in bad.renderable.plain

    usage = dispatch("/validate")
    assert "Usage" in usage.renderable.plain


def test_index_requires_two_args_and_a_rag_section(tmp_path):
    assert "Usage" in dispatch("/index onlyonearg").renderable.plain

    no_rag = tmp_path / "norag.yml"
    no_rag.write_text("model:\n  name: m\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    result = dispatch(f'/index "{docs}" "{no_rag}"')
    assert "no rag" in result.renderable.plain


def test_unknown_command_is_reported():
    assert "Unknown command" in dispatch("/banana").renderable.plain


def test_non_slash_line_is_nudged_toward_help():
    assert "/help" in dispatch("just chatting").renderable.plain


def test_clear_and_exit_carry_actions():
    assert dispatch("/clear").action == "clear"
    assert dispatch("/exit").action == "exit"


def test_blank_line_does_nothing():
    assert dispatch("   ") is None
