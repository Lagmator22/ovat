# tests/test_a2a_policy.py
"""Admission: what the server refuses before any model work starts.

Pinned here: every limit refuses, and says which setting to change. Not
pinned, on purpose: the ORDER of the checks and whether a refused message
counts against the caller's rate. Those are policy choices; the docstring
of Admission.check explains them.
"""
import pytest

from ovat.a2a.policy import Admission, Ask
from ovat.config.workflow import A2AServerConfig


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _admission(**settings):
    clock = Clock()
    return Admission(A2AServerConfig(**settings), "me", clock=clock), clock


def test_an_ordinary_message_is_accepted():
    admission, _ = _admission()
    assert admission.check(Ask(caller="a", text="How many leave days?")) is None


def test_an_empty_message_is_refused():
    admission, _ = _admission()
    assert admission.check(Ask(caller="a", text="   "))


@pytest.mark.parametrize("ask,setting", [
    (Ask(caller="a", text="x" * 101), "max_message_chars"),
    (Ask(caller="a", text="hi", queued=2), "max_queue"),
    (Ask(caller="a", text="hi", hops=3), "max_hops"),
    (Ask(caller="a", text="hi", turns=5), "max_turns_per_context"),
])
def test_each_limit_refuses_and_names_its_setting(ask, setting):
    admission, _ = _admission(max_message_chars=100, max_queue=2,
                              max_hops=3, max_turns_per_context=5)
    refusal = admission.check(ask)
    assert refusal and setting in refusal


def test_a_request_that_already_passed_through_here_is_a_cycle():
    admission, _ = _admission()
    refusal = admission.check(Ask(caller="a", text="hi", hops=1,
                                  chain=["other", "me"]))
    assert refusal and "cycle" in refusal.lower()


def test_the_rate_limit_is_per_caller_and_per_minute():
    admission, clock = _admission(rate_limit_per_min=3)
    for _ in range(3):
        assert admission.check(Ask(caller="a", text="hi")) is None
    refusal = admission.check(Ask(caller="a", text="hi"))
    assert refusal and "rate_limit_per_min" in refusal
    assert admission.check(Ask(caller="b", text="hi")) is None  # others fine
    clock.now += 61
    assert admission.check(Ask(caller="a", text="hi")) is None  # a new minute
