# tests/conftest.py
"""Shared test helpers for the agent tests.

Note to myself: I do not want my unit tests to need a real LLM or a GPU. So I
build a FakeLLMProvider that returns replies I script in advance. Because my
loop only depends on the LLMProvider contract, swapping in this fake tests the
entire loop logic in milliseconds, on any machine, Mac or AI PC.
"""
import json
import os
import shlex
import sys
from types import SimpleNamespace

import pytest

from ovat.providers.base import LLMProvider


def py_command(code: str) -> str:
    """A shell command string that runs `code` in THIS interpreter, anywhere.

    The TUI's shell layer takes a command STRING, so a test has to name a real
    command, and the obvious ones are POSIX-only: cmd.exe has no sleep, no pwd
    and no printf. Routing through sys.executable gives one command that
    behaves identically on both platforms, quoted for whichever shell will
    parse it. Using THIS interpreter also means the child is the venv python,
    the same one venv_env() puts first on PATH.

    Keep `code` free of double quotes: cmd.exe cannot nest them, so use
    chr(13)/chr(10) rather than escaped literals.
    """
    if os.name == "nt":
        return f'"{sys.executable}" -c "{code}"'
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


@pytest.fixture(autouse=True)
def _plain_console_output():
    """Assert on WHAT the CLI printed, never on how it was coloured.

    When rich has colour enabled its highlighter styles text it thinks looks
    like a python repr, and it does so INSIDE otherwise plain output: brackets
    get bold and bare numbers get cyan, so "[/INST]" is captured as
    "\\x1b[1m[\\x1b[0m/INST\\x1b[1m]\\x1b[0m" and "Llama-3.2-3B" gets escape
    codes in the middle of the name. Exact substring assertions then fail.

    Not hypothetical, and not someone's stray shell setting: OVAT's own TUI
    exports FORCE_COLOR=1 into every child it runs (shell.venv_env), so typing
    `pytest` inside the TUI failed four tests on any platform. Neutralising it
    in one place beats asking every assertion to remember, and colour is not
    what any of them are testing.

    Two levers, because the obvious ones do not work. Dropping the env var is
    too late: rich resolves the colour system when the shared Console is BUILT,
    at ovat.cli.ui import, long before any fixture runs. And no_color=True only
    strips colour, leaving the bold that wraps the brackets. Clearing
    _color_system is what actually silences it; the env vars are handled too so
    a Console built DURING a test starts out plain as well.
    """
    from ovat.cli import ui

    saved_env = {k: os.environ.pop(k) for k in ("FORCE_COLOR", "CLICOLOR_FORCE")
                 if k in os.environ}
    saved_system = ui.console._color_system
    ui.console._color_system = None
    try:
        yield
    finally:
        ui.console._color_system = saved_system
        os.environ.update(saved_env)


class FakeLLMProvider(LLMProvider):
    """A stand in for OVMS that hands back replies I prepared in advance."""

    def __init__(self, scripted_replies: list[dict]):
        # I hand back one reply per chat() call, in order.
        self._replies = list(scripted_replies)
        # I record every call so my tests can assert what the loop sent.
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], tools=None) -> dict:
        self.calls.append({"messages": [m.copy() for m in messages], "tools": tools})
        return self._replies.pop(0)


def make_tool_call(call_id: str, name: str, arguments: dict):
    """I build a fake tool_call shaped exactly like the OpenAI SDK objects.

    Note to myself: the real provider gives me objects with attribute access
    (call.function.name) and arguments as a JSON string. My fake matches that
    shape so the loop cannot tell the difference.
    """
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def reply(finish_reason: str, content=None, tool_calls=None) -> dict:
    """I build a reply dict shaped like what LLMProvider.chat() returns."""
    return {
        "finish_reason": finish_reason,
        "content": content,
        "tool_calls": tool_calls,
        "raw": None,
    }
