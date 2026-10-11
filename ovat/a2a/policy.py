# ovat/a2a/policy.py
"""Admission: whether the server takes on a task at all.

Every refusal is one plain sentence naming the limit and its value, sent
back as a REJECTED task, so the calling agent (and the person behind it)
knows what to change. Nothing here touches the model: a refusal costs the
server microseconds, which is the point of checking first.

Note to myself: one OVMS on one iGPU serves every caller, so a request
that is refused late has already cost everyone else seconds of GPU time.
"""
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ovat.config.workflow import A2AServerConfig

_WINDOW_S = 60.0


@dataclass
class Ask:
    """The facts about one incoming message that admission looks at."""

    caller: str
    text: str
    hops: int = 0                 # agents this request already went through
    chain: list[str] = field(default_factory=list)   # their server ids
    queued: int = 0               # tasks already waiting for the model
    turns: int = 0                # messages already in this conversation


class Admission:
    def __init__(self, settings: A2AServerConfig, server_id: str,
                 clock=time.monotonic):
        self.s = settings
        self.server_id = server_id
        self._clock = clock
        self._lock = threading.Lock()
        # caller -> times of their recent messages, oldest first
        self._recent: dict[str, deque] = defaultdict(deque)

    def _rate_exceeded(self, caller: str) -> bool:
        """True when `caller` already sent rate_limit_per_min messages in
        the last minute. Forgets anything older as it goes."""
        now = self._clock()
        with self._lock:
            times = self._recent[caller]
            while times and now - times[0] >= _WINDOW_S:
                times.popleft()
            return len(times) >= self.s.rate_limit_per_min

    def _record(self, caller: str) -> None:
        """Count one message from `caller` against the rate limit."""
        with self._lock:
            self._recent[caller].append(self._clock())

    def check(self, ask: Ask) -> str | None:
        """None to accept, or the sentence that explains the refusal.

        The limits, all in self.s (A2AServerConfig):
          empty text                     nothing to answer
          len(ask.text) > max_message_chars
          ask.queued >= max_queue        the model is busy enough already
          rate                           self._rate_exceeded(ask.caller)
          ask.hops >= max_hops           too deep a chain of agents
          self.server_id in ask.chain    a cycle: this server already
                                         handled this request upstream
          ask.turns >= max_turns_per_context
        and self._record(ask.caller) counts a message against the rate.

        Order: loop protection first, so a runaway chain of agents is always
        named for what it is; then what the message itself is; then whether
        there is room for it. Only ACCEPTED messages count against the rate:
        an honest caller who sent one message too long has not used up a
        slot of their minute.
        """
        s = self.s
        if self.server_id in ask.chain:
            return ("this request already passed through this agent (a "
                    "cycle between agents), so it was not answered again")
        if ask.hops >= s.max_hops:
            return (f"the request came through {ask.hops} agents already; "
                    f"the limit is {s.max_hops} (a2a.server.max_hops)")
        if not ask.text.strip():
            return "the message was empty"
        if len(ask.text) > s.max_message_chars:
            return (f"message longer than {s.max_message_chars} characters "
                    f"(a2a.server.max_message_chars)")
        if ask.turns >= s.max_turns_per_context:
            return (f"this conversation reached {s.max_turns_per_context} "
                    f"messages (a2a.server.max_turns_per_context); start a "
                    f"new one")
        if ask.queued >= s.max_queue:
            return (f"{ask.queued} tasks are already waiting for the model "
                    f"(a2a.server.max_queue); try again shortly")
        if self._rate_exceeded(ask.caller):
            return (f"more than {s.rate_limit_per_min} messages in a minute "
                    f"(a2a.server.rate_limit_per_min)")
        self._record(ask.caller)
        return None
