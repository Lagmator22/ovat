# tests/conftest.py
"""Shared test helpers for the agent tests.

Note to myself: I do not want my unit tests to need a real LLM or a GPU. So I
build a FakeLLMProvider that returns replies I script in advance. Because my
loop only depends on the LLMProvider contract, swapping in this fake tests the
entire loop logic in milliseconds, on any machine, Mac or AI PC.
"""
import json
from types import SimpleNamespace

from ovat.providers.base import LLMProvider


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
