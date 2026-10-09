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

# BEFORE importing anything from ovat. `ovat.cli.ui` builds its themed Console
# at import time, and Rich fixes a console's width when it is constructed, so a
# fixture that sets COLUMNS later is already too late -- which is exactly what
# happened: the fixture ran, and both Ubuntu jobs still failed on a path split
# mid-word as "workflo\nw.yml".
#
# Pinning it here means every Rich console in the test session, however it is
# built, agrees on the width: pytest capture, a CliRunner, an xdist worker, a
# narrow CI runner. Assertions then measure the code instead of the window.
os.environ.setdefault("COLUMNS", "200")
os.environ.setdefault("LINES", "50")

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


# The three framework engines on a scripted model, for the token-usage tests.
#
# Each fake replies twice, the way OVMS does on a one-tool question: first a
# call to add_note, then the answer. `usage` is one (prompt, completion) pair
# per reply, or None for a server that sends no usage block. Shared here so
# the engine test files and test_bench drive the very same scripts. Every
# framework import is inside a function: the frameworks are optional and
# this file is loaded by every test.

ANSWER = "noted and answered"


def note_tools() -> tuple[dict, list]:
    """One tool shaped like a real OVAT tool, and the list of its calls."""
    calls = []

    def add_note(text: str) -> str:
        calls.append(text)
        return f"noted {text}"

    tools = {"add_note": {"function": add_note, "schema": {
        "type": "function", "function": {
            "name": "add_note", "description": "Record a note.",
            "parameters": {"type": "object",
                           "properties": {"text": {"type": "string"}},
                           "required": ["text"]}}}}}
    return tools, calls


def _react_agent(config, tools, usage):
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel)
    from langchain_core.messages import AIMessage

    from ovat.agent.langchain_agent import build_react_agent

    class ToolModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def meta(i):
        if usage is None:
            return None
        p, c = usage[i]
        return {"input_tokens": p, "output_tokens": c,
                "total_tokens": p + c}

    replies = iter([
        AIMessage(content="", usage_metadata=meta(0),
                  tool_calls=[{"name": "add_note", "args": {"text": "x"},
                               "id": "c1"}]),
        AIMessage(content=ANSWER, usage_metadata=meta(1)),
    ])
    return build_react_agent(config, tools, llm=ToolModel(messages=replies))


def _llamaindex_agent(config, tools, usage):
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.llms import MockFunctionCallingLLM
    from llama_index.core.llms.llm import ToolSelection
    from pydantic import PrivateAttr

    from ovat.agent.llamaindex_agent import build_llamaindex_agent

    def raw(i):
        # What OpenAILike hands over: the reply's model_dump(), whose
        # "usage" is None when the server sent none.
        if usage is None:
            return {"usage": None}
        p, c = usage[i]
        return {"usage": {"prompt_tokens": p, "completion_tokens": c,
                          "total_tokens": p + c}}

    class ScriptedLLM(MockFunctionCallingLLM):
        _replies: list = PrivateAttr(default_factory=list)

        async def achat(self, messages, **kwargs):
            return self._replies.pop(0)

    llm = ScriptedLLM(is_chat_model=True)
    llm._replies = [
        ChatResponse(message=ChatMessage(
            role="assistant", content="",
            additional_kwargs={"tool_calls": [ToolSelection(
                tool_id="c1", tool_name="add_note",
                tool_kwargs={"text": "x"})]}), raw=raw(0)),
        ChatResponse(message=ChatMessage(role="assistant", content=ANSWER),
                     raw=raw(1)),
    ]
    return build_llamaindex_agent(config, tools, llm=llm)


def _openai_agents_agent(config, tools, usage):
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (ResponseFunctionToolCall,
                                        ResponseOutputMessage,
                                        ResponseOutputText)

    from ovat.agent.openai_agents_agent import build_openai_agents_agent

    def use(i):
        # What OpenAIChatCompletionsModel records for a reply with no usage
        # block: one request, every count zero.
        if usage is None:
            return Usage(requests=1)
        p, c = usage[i]
        return Usage(requests=1, input_tokens=p, output_tokens=c,
                     total_tokens=p + c)

    outputs = [
        [ResponseFunctionToolCall(type="function_call", call_id="c1",
                                  name="add_note",
                                  arguments='{"text": "x"}')],
        [ResponseOutputMessage(
            id="m1", type="message", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text=ANSWER,
                                        annotations=[])])],
    ]

    class ScriptedModel(Model):
        def __init__(self):
            self._i = 0

        async def get_response(self, *args, **kwargs):
            i, self._i = self._i, self._i + 1
            return ModelResponse(output=outputs[i], usage=use(i),
                                 response_id=None)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    return build_openai_agents_agent(config, tools, model=ScriptedModel())


def scripted_framework_agent(engine: str, usage):
    """Build `engine` on a model that calls add_note once, then answers.

    Returns (agent, calls): `calls` lists every add_note that really ran.
    """
    from ovat.config.workflow import WorkflowConfig

    config = WorkflowConfig(model={"name": "test-model"},
                            agent={"type": engine})
    tools, calls = note_tools()
    build = {"react": _react_agent, "llamaindex": _llamaindex_agent,
             "openai-agents": _openai_agents_agent}[engine]
    return build(config, tools, usage), calls
