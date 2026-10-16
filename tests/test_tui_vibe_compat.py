# tests/test_tui_vibe_compat.py
"""The stand-ins in ovat/tui/vibe/_compat.py behave like what they replace.

They exist so the copied Mistral Vibe code runs on Python 3.10, and for two
small third-party helpers (humanize, tomli_w) OVAT does not depend on.
"""
import asyncio
from datetime import timedelta
from enum import auto

import pytest

from ovat.tui.vibe import _compat
from ovat.tui.vibe._compat import (
    StrEnum,
    UTC,
    fromisoformat,
    naturaldelta,
    naturalsize,
    task_cancelling,
    toml_dumps,
    tomllib,
)


def test_str_enum_is_a_str_and_auto_is_the_lower_cased_name():
    class Colour(StrEnum):
        RED = auto()
        DARK_BLUE = auto()

    assert Colour.RED == "red"
    assert Colour.DARK_BLUE.value == "dark_blue"
    assert str(Colour.RED) == "red"
    assert f"{Colour.RED}" == "red"
    assert Colour("red") is Colour.RED


def test_utc_is_offset_zero():
    assert UTC.utcoffset(None) == timedelta(0)


@pytest.mark.parametrize("text", [
    "2026-05-01T12:00:00Z",
    "2026-05-01T12:00:00+00:00",
    "2026-05-01T12:00:00.123Z",
    "2026-05-01T12:00:00.123456789Z",     # 3.10 refuses this one
])
def test_fromisoformat_takes_what_vibe_session_timestamps_look_like(text):
    parsed = fromisoformat(text)
    assert parsed.utcoffset() == timedelta(0)
    assert (parsed.year, parsed.month, parsed.day, parsed.hour) == (2026, 5, 1, 12)


def test_toml_round_trip():
    data = {
        "top": 3,
        "greeting": {"last_shown_at": 1700000000, "ok": True, "name": 'a "quoted"\nline'},
        "update": {"latest": "1.2.3", "ratio": 0.5, "seen": ["a", "b"], "nested": {"x": 1}},
        "odd key": {"k": "v"},
    }
    assert tomllib.loads(toml_dumps(data)) == data


def test_naturalsize_matches_humanize_for_the_values_the_tui_shows():
    assert naturalsize(10 * 1024 * 1024, binary=True) == "10.0 MiB"
    assert naturalsize(1536, binary=True) == "1.5 KiB"
    assert naturalsize(512, binary=True) == "512 Bytes"
    assert naturalsize(1, binary=True) == "1 Byte"


def test_naturaldelta_names_the_snooze_period():
    assert naturaldelta(604800) == "7 days"
    assert naturaldelta(3600) == "an hour"
    assert naturaldelta(90) == "a minute"


def test_task_cancelling_is_zero_for_a_plain_task_and_counts_on_new_pythons():
    async def go():
        async def sleeper():
            await asyncio.sleep(10)

        task = asyncio.create_task(sleeper())
        await asyncio.sleep(0)
        before = task_cancelling(task)
        task.cancel()
        after = task_cancelling(task)
        with pytest.raises(asyncio.CancelledError):
            await task
        return before, after

    before, after = asyncio.run(go())
    assert before == 0
    assert after in (0, 1)        # 3.11+ counts the request; 3.10 has no count


def test_the_compat_module_is_the_only_place_that_names_a_newer_python():
    # A copied file that imports StrEnum / UTC / tomllib / assert_never from
    # the stdlib would import fine on 3.13 and die on 3.10.
    import ast
    from pathlib import Path

    banned = {("enum", "StrEnum"), ("datetime", "UTC"), ("typing", "assert_never"),
              ("typing", "Self"), ("typing", "override")}
    root = Path(_compat.__file__).parent
    bad = []
    for path in root.rglob("*.py"):
        if path.name == "_compat.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if (node.module, alias.name) in banned:
                        bad.append(f"{path.relative_to(root)}:{node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tomllib":
                        bad.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not bad, f"import these from ovat.tui.vibe._compat instead: {bad}"
